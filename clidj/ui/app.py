"""cli-dj: snippet-driven, transport-quantized live-coding DJ TUI.

Textual is the only Textual-aware module in this project -- transport,
library, snippets, lanes, scheduler and interpreter are all plain Python
with no UI dependency, so a real audio engine could drive the same
Transport/Scheduler/Lane stack from a different front end later.

No audio is decoded or played. Everything here -- waveforms, playback,
BPM -- is a visual simulation.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import DataTable, Input, RichLog, Static, TabbedContent, TabPane

from ..config import Config, Paths
from ..interpreter import Interpreter
from ..lanes import LANE_NAMES, Lane, check_warnings
from ..library import SAMPLES_PER_BEAT, Track
from ..session import Session
from ..snippets import BEATS_PER_BAR as SNIPPET_BEATS_PER_BAR
from ..snippets import Snippet
from ..transport import Transport

BAR_CHARS = " ▁▂▃▄▅▆▇█"  # " ▁▂▃▄▅▆▇█"
QUARTER_COLS_PER_BEAT = 4  # 1 character column ~= 1/4 beat in the zoomed lane view
LANE_ACCENTS = {"L1": "cyan", "L2": "magenta", "L3": "yellow", "L4": "green"}
MIN_WIDTH = 20
LANE_GUTTER = 4  # columns left of the waveform area: lane name on info rows, blank on the rest
SETS_DIR = Path(__file__).resolve().parents[2] / "sets"


class HistoryInput(Input):
    """An Input with up/down arrow command history, Textual has no built-in
    equivalent for a single-line REPL prompt."""

    BINDINGS = [
        Binding("up", "history_prev", "Previous command", show=False),
        Binding("down", "history_next", "Next command", show=False),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.history: list[str] = []
        self._history_index = 0

    def remember(self, text: str) -> None:
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text)
        self._history_index = len(self.history)

    def action_history_prev(self) -> None:
        if not self.history:
            return
        self._history_index = max(0, self._history_index - 1)
        self.value = self.history[self._history_index]
        self.cursor_position = len(self.value)

    def action_history_next(self) -> None:
        if not self.history:
            return
        self._history_index = min(len(self.history), self._history_index + 1)
        self.value = self.history[self._history_index] if self._history_index < len(self.history) else ""
        self.cursor_position = len(self.value)


def _meter(value: float, width: int = 4) -> str:
    value = max(0.0, min(1.0, value))
    filled = value * width
    chars = []
    for i in range(width):
        level = max(0.0, min(1.0, filled - i))
        chars.append(BAR_CHARS[round(level * (len(BAR_CHARS) - 1))])
    return "".join(chars)


def _loop_display(lane: Lane, transport: Transport) -> str:
    if lane.snippet is None:
        return ""
    local = lane.local_beat(transport)
    if local is None:
        return ""
    total_bars = lane.snippet.length_beats / transport.beats_per_bar
    if lane.snippet.loop:
        current_bar = int(local // transport.beats_per_bar) + 1
        return f"loop {current_bar}/{int(round(total_bars))}"
    remaining_bars = (lane.snippet.length_beats - local) / transport.beats_per_bar
    return f"{remaining_bars:.1f} left"


@dataclass(frozen=True)
class WaveAxis:
    """The one column <-> beat mapping shared by the ruler and every lane's
    waveform rows, so the ▼ marker, bar lines and waveform columns can't
    drift apart.

    Columns [left, left + width) are the waveform area; the playhead sits at
    `center`. Column `c` covers beats [beat(c) - 1/8, beat(c) + 1/8) around
    `beat(c) = position + (c - center) / QUARTER_COLS_PER_BEAT`, and a beat
    `b` is drawn in the column whose range contains it.
    """

    left: int
    width: int

    @property
    def center(self) -> int:
        return self.left + self.width // 2

    def columns(self) -> range:
        return range(self.left, self.left + self.width)

    def offset_beats(self, col: int) -> float:
        return (col - self.center) / QUARTER_COLS_PER_BEAT

    def col_for_offset(self, offset_beats: float) -> int:
        return self.center + math.floor(offset_beats * QUARTER_COLS_PER_BEAT + 0.5)

    @classmethod
    def for_width(cls, width: int) -> "WaveAxis":
        return cls(left=LANE_GUTTER, width=max(4, width - LANE_GUTTER))


def _bar_head_columns(axis: WaveAxis, position: float, beats_per_bar: int) -> dict[int, str]:
    """col -> "bar" | "beat" for every beat head visible on the axis."""
    lo = math.floor(position + axis.offset_beats(axis.left)) - 1
    hi = math.ceil(position + axis.offset_beats(axis.left + axis.width)) + 1
    heads: dict[int, str] = {}
    for beat_n in range(lo, hi + 1):
        col = axis.col_for_offset(beat_n - position)
        if axis.left <= col < axis.left + axis.width:
            heads[col] = "bar" if beat_n % beats_per_bar == 0 else "beat"
    return heads


def _column_amplitude(snippet: Snippet, local_lo: float, local_hi: float) -> Optional[float]:
    """Peak waveform amplitude over [local_lo, local_hi) beats of the
    snippet, or None where a non-looping snippet has nothing to show."""
    length = snippet.length_beats
    first = math.floor(local_lo * SAMPLES_PER_BEAT)
    last = max(first + 1, math.floor(local_hi * SAMPLES_PER_BEAT))
    peak: Optional[float] = None
    for index in range(first, last):
        local = index / SAMPLES_PER_BEAT
        if snippet.loop:
            local %= length
        elif not 0.0 <= local < length:
            continue
        amp = snippet.track.amplitude_at_beat(snippet.start_beat + local)
        peak = amp if peak is None else max(peak, amp)
    return peak


class LanesView(Static):
    """Transport status, a shared beat ruler, and per lane an info row plus
    a two-row zoomed, center-locked waveform -- all drawn on one WaveAxis so
    beat heads line up vertically across the ruler and every lane."""

    def __init__(self, transport: Transport, lanes: dict[str, Lane], quant_getter, **kwargs) -> None:
        super().__init__(**kwargs)
        self.transport = transport
        self.lanes = lanes
        self._quant_getter = quant_getter

    def refresh_view(self) -> None:
        self.update(self.build_text(max(self.content_size.width, MIN_WIDTH)))

    def build_text(self, width: int) -> Text:
        transport = self.transport
        axis = WaveAxis.for_width(width)
        heads = _bar_head_columns(axis, transport.position_beats, transport.beats_per_bar)

        status = "● RUN" if transport.running else "■ STOP"
        header = (
            f"TRANSPORT  {transport.display}  {status}  {transport.bpm:.1f} BPM  "
            f"{transport.beats_per_bar}/4  q:{self._quant_getter()}"
        )
        rows: list[Text] = [Text(header, style="bold"), self._ruler(axis, heads)]
        for name in LANE_NAMES:
            rows.extend(self._lane_rows(self.lanes[name], axis, heads, width))
        return Text("\n").join(rows)

    def _ruler(self, axis: WaveAxis, heads: dict[int, str]) -> Text:
        row = [" "] * (axis.left + axis.width)
        for col, kind in heads.items():
            row[col] = "|" if kind == "bar" else "·"
        row[axis.center] = "▼"
        return Text("".join(row), style="grey62")

    def _lane_rows(self, lane: Lane, axis: WaveAxis, heads: dict[int, str], width: int) -> list[Text]:
        accent = LANE_ACCENTS.get(lane.name, "white")
        warnings: list[str] = []
        if lane.snippet is not None:
            others = [other for name, other in self.lanes.items() if name != lane.name]
            warnings = check_warnings(lane, others, self.transport.bpm)

        if lane.snippet is None:
            label, role, key = "—", "", ""
        else:
            label, role, key = lane.snippet.name, lane.snippet.role, lane.snippet.key

        info = Text()
        info.append(f"{lane.name:<{LANE_GUTTER}}", style="bold yellow" if warnings else f"bold {accent}")
        info.append(f"{label:<12.12} {role:<6.6} {key:<4.4}", style="grey70")
        info.append(f" gain {_meter(lane.gain)} ", style="grey62")
        info.append(f"{_loop_display(lane, self.transport):<11.11}", style="grey62")
        if warnings:
            info.append(" " + warnings[0], style="yellow")
        info.truncate(width)

        top, bottom = self._wave_rows(lane, axis, heads, accent)
        return [info, top, bottom]

    def _wave_rows(self, lane: Lane, axis: WaveAxis, heads: dict[int, str], accent: str) -> tuple[Text, Text]:
        """Two rows stacked into one bar per column: the bottom row fills up
        to half amplitude, the top row shows the rest."""
        top = Text(" " * axis.left)
        bottom = Text(" " * axis.left)
        snippet = lane.snippet
        local_now = None if snippet is None else lane.local_beat(self.transport)
        half_col = 0.5 / QUARTER_COLS_PER_BEAT

        for col in axis.columns():
            tint = {"bar": " on grey15", "beat": ""}.get(heads.get(col, ""), "")
            if snippet is None or local_now is None:
                top.append(" ", style=f"grey37{tint}" if tint else "")
                bottom.append(" ", style=f"grey37{tint}" if tint else "")
                continue
            local = local_now + axis.offset_beats(col)
            amp = _column_amplitude(snippet, local - half_col, local + half_col)
            base = f"bold {accent}" if col < axis.center else "grey50"
            style = base + tint
            if amp is None:
                top.append(" ", style=style)
                bottom.append(" ", style=style)
                continue
            levels = len(BAR_CHARS) - 1
            top.append(BAR_CHARS[round(max(0.0, min(1.0, amp * 2 - 1)) * levels)], style=style)
            bottom.append(BAR_CHARS[round(max(0.0, min(1.0, amp * 2)) * levels)], style=style)
        return top, bottom


TRACK_STATUS_GLYPHS = {"demo": " ", "new": "·", "analyzing": "…", "ready": "✓", "failed": "✗", "missing": "?"}
PREP_GLYPHS = {"none": "·", "pending": "…", "ready": "✓", "failed": "✗"}


class DJApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "cli-dj"

    def __init__(
        self,
        set_path: Path | None = None,
        *,
        demo: bool = False,
        paths: Paths | None = None,
        config: Config | None = None,
        jobs=None,
        audio: bool = False,
    ) -> None:
        super().__init__()
        self._set_path = set_path
        self._pending_log: list[tuple[str, str]] = []
        self.paths = paths or Paths.default()
        self.config = config or Config.load(self.paths)
        self.session = Session(self.paths, self.config, self._log, demo=demo, jobs=jobs, prepare=audio)
        self.library: list[Track] = self.session.tracks
        self.transport = self.session.transport
        self.lanes: dict[str, Lane] = self.session.lanes
        self.scheduler = self.session.scheduler
        self.interp = Interpreter(
            self.transport,
            self.scheduler,
            self.lanes,
            self.library,
            self._log,
            SETS_DIR,
            on_clear=self._clear_log,
            session=self.session,
        )
        self._last_tick = time.monotonic()
        self._tracks_version = -1
        self._snips_signature: tuple = ()

    def compose(self) -> ComposeResult:
        with Horizontal(id="root"):
            with Vertical(id="library-pane"):
                with TabbedContent(id="library-tabs"):
                    with TabPane("TRACKS", id="tracks-tab"):
                        yield DataTable(id="tracks-table")
                    with TabPane("SNIPS", id="snips-tab"):
                        yield DataTable(id="snips-table")
            with Vertical(id="right-pane"):
                with Vertical(id="console-pane"):
                    yield RichLog(id="console-log", markup=True, wrap=True)
                    yield Static(id="queue-view")
                    yield HistoryInput(
                        id="command-input",
                        placeholder='start()  L1 << kick  at(33, xf(L1, L2, 8))  help()',
                    )
                yield LanesView(
                    self.transport, self.lanes, lambda: self.interp.quant_mode, id="lanes-view"
                )

    def on_mount(self) -> None:
        self.query_one("#library-pane").border_title = "LIBRARY"
        self.query_one("#console-pane").border_title = "CONSOLE"
        self.query_one("#lanes-view").border_title = "TRANSPORT / LANES"

        tracks_table = self.query_one("#tracks-table", DataTable)
        tracks_table.cursor_type = "row"
        # Column widths + 2 cells of padding per column must fit the pane's
        # inner width minus a vertical scrollbar (see #library-pane in app.tcss).
        tracks_table.add_column("#", width=3)
        tracks_table.add_column("", width=1)  # analysis status glyph
        tracks_table.add_column("Title", width=13)
        tracks_table.add_column("Artist", width=8)
        tracks_table.add_column("BPM", width=5)
        tracks_table.add_column("Key", width=3)
        tracks_table.add_column("Time", width=5)

        snips_table = self.query_one("#snips-table", DataTable)
        snips_table.cursor_type = "row"
        snips_table.add_column("", width=1)  # preparation status glyph
        snips_table.add_column("Name", width=8)
        snips_table.add_column("Track", width=13)
        snips_table.add_column("Bars", width=4)
        snips_table.add_column("Role", width=5)
        snips_table.add_column("Key", width=3)
        snips_table.add_column("Loop", width=4)

        mode = "demo library" if self.session.demo else f"library: {len(self.library)} track(s)"
        prep = "snippets are pre-rendered" if self.session.preparing else "--no-audio: visual only"
        self.query_one("#console-log", RichLog).write(
            f"[bold]cli-dj[/] — snippet-driven live coding ({mode}; {prep}). type help() and press Enter"
        )
        for message, level in self._pending_log:
            self._log(message, level)
        self._pending_log.clear()
        if not self.session.demo and not self.config.library_folders:
            self._log(f"no library folders yet: add them to {self.paths.config_file}, then scan()", "warn")

        if self._set_path is not None:
            self._load_set_path(self._set_path)

        self.set_interval(1 / 30, self._on_tick)
        self.query_one("#command-input", HistoryInput).focus()
        self._refresh_all()

    def on_unmount(self) -> None:
        self.session.close()

    def _load_set_path(self, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            self._log(f"could not read {path}: {exc}", "error")
            return
        self._log(f"loading {path}", "info")
        self.interp.run(text)

    def _on_tick(self) -> None:
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now
        self.session.tick(dt)
        self._refresh_all()

    def _refresh_all(self) -> None:
        try:
            lanes_view = self.query_one("#lanes-view", LanesView)
        except NoMatches:  # the interval timer can fire once more while the app shuts down
            return
        lanes_view.refresh_view()
        self._refresh_tracks_table()
        self._refresh_snips_table()
        self._refresh_queue()

    def _refresh_tracks_table(self) -> None:
        pane = self.query_one("#library-pane")
        pane.border_subtitle = self.session.activity or ""
        version = self.session.library_version
        if version == self._tracks_version:
            return
        self._tracks_version = version
        table = self.query_one("#tracks-table", DataTable)
        table.clear()
        for t in self.library:
            m, s = divmod(int(t.duration), 60)
            bpm = f"{t.bpm:.1f}" if t.bpm > 0 else "--"
            table.add_row(
                str(t.id), TRACK_STATUS_GLYPHS.get(t.status, " "), t.title, t.artist, bpm, t.key or "--",
                f"{m}:{s:02d}",
            )

    def _refresh_snips_table(self) -> None:
        table = self.query_one("#snips-table", DataTable)
        entries = [(name, value) for name, value in self.interp.env.items() if isinstance(value, Snippet)]
        bpm = self.session.pending_bpm or self.transport.bpm
        states = [self.session.prep_state(snippet, bpm) if self.session.preparing else "" for _, snippet in entries]
        signature = tuple((name, id(snippet), state) for (name, snippet), state in zip(entries, states))
        if signature == self._snips_signature:
            return
        self._snips_signature = signature
        table.clear()
        for (name, snippet), state in zip(entries, states):
            bars = snippet.length_beats / SNIPPET_BEATS_PER_BAR
            table.add_row(
                PREP_GLYPHS.get(state, " "), name, snippet.track.title, f"{bars:g}", snippet.role, snippet.key,
                "yes" if snippet.loop else "no",
            )

    def _refresh_queue(self) -> None:
        view = self.query_one("#queue-view", Static)
        items = self.scheduler.pending(limit=5)
        lines = ["[bold]QUEUE[/]"]
        if not items:
            lines.append("(empty)")
        else:
            for event_id, beat, desc in items:
                lines.append(escape(f"#{event_id} @{self.transport.display_at(beat)} {desc}"))
        view.update("\n".join(lines))

    def _log(self, message: str, level: str = "info") -> None:
        if not self.is_mounted:
            self._pending_log.append((message, level))
            return
        safe = escape(message)
        if level == "warn":
            rendered = f"[yellow]{safe}[/]"
        elif level == "error":
            rendered = f"[red]{safe}[/]"
        else:
            rendered = safe
        try:
            self.query_one("#console-log", RichLog).write(rendered)
        except NoMatches:
            self._pending_log.append((message, level))

    def _clear_log(self) -> None:
        self.query_one("#console-log", RichLog).clear()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if isinstance(event.input, HistoryInput):
            event.input.remember(text)
        self.query_one("#console-log", RichLog).write(f"[bold cyan]> {escape(text)}[/]")
        self.interp.run(text)
        self._refresh_all()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m clidj", description="cli-dj: live-coding DJ TUI")
    parser.add_argument("--set", dest="set_path", default=None, help="path to a .djs set file to load at startup")
    parser.add_argument("--demo", action="store_true", help="use the built-in demo library instead of your music")
    parser.add_argument("--no-audio", action="store_true", help="visual only: no rendering, no audio device")
    args = parser.parse_args(argv)
    set_path = Path(args.set_path) if args.set_path else None
    DJApp(set_path=set_path, demo=args.demo, audio=not args.no_audio).run()


if __name__ == "__main__":
    main()
