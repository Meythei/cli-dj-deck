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

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import DataTable, Input, RichLog, Static, TabbedContent, TabPane

from interpreter import Interpreter
from lanes import LANE_NAMES, Lane, check_warnings
from library import DEMO_LIBRARY, SAMPLES_PER_BEAT, Track
from scheduler import Scheduler
from snippets import BEATS_PER_BAR as SNIPPET_BEATS_PER_BAR
from snippets import Snippet
from transport import Transport

BAR_CHARS = " ▁▂▃▄▅▆▇█"  # " ▁▂▃▄▅▆▇█"
QUARTER_COLS_PER_BEAT = 4  # 1 character column ~= 1/4 beat in the zoomed lane view
LANE_ACCENTS = {"L1": "cyan", "L2": "magenta", "L3": "yellow", "L4": "green"}
MIN_WIDTH = 20
LANE_GUTTER = 4  # columns left of the waveform area: lane name on info rows, blank on the rest
SETS_DIR = Path(__file__).parent / "sets"


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


class DJApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "cli-dj"

    def __init__(self, set_path: Path | None = None) -> None:
        super().__init__()
        self._set_path = set_path
        self.library: list[Track] = DEMO_LIBRARY
        self.transport = Transport()
        self.lanes: dict[str, Lane] = {name: Lane(name) for name in LANE_NAMES}
        self.scheduler = Scheduler(self.transport, self._log)
        self.interp = Interpreter(
            self.transport,
            self.scheduler,
            self.lanes,
            self.library,
            self._log,
            SETS_DIR,
            on_clear=self._clear_log,
        )
        self._last_tick = time.monotonic()

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
        # inner width minus a vertical scrollbar (see LIBRARY_PANE_WIDTH).
        tracks_table.add_column("#", width=3)
        tracks_table.add_column("Title", width=14)
        tracks_table.add_column("Artist", width=10)
        tracks_table.add_column("BPM", width=5)
        tracks_table.add_column("Key", width=3)
        tracks_table.add_column("Time", width=5)
        for t in self.library:
            m, s = divmod(int(t.duration), 60)
            tracks_table.add_row(str(t.id), t.title, t.artist, f"{t.bpm:.0f}", t.key, f"{m}:{s:02d}")

        snips_table = self.query_one("#snips-table", DataTable)
        snips_table.cursor_type = "row"
        snips_table.add_column("Name", width=8)
        snips_table.add_column("Track", width=14)
        snips_table.add_column("Bars", width=4)
        snips_table.add_column("Role", width=6)
        snips_table.add_column("Key", width=3)
        snips_table.add_column("Loop", width=5)

        self.query_one("#console-log", RichLog).write(
            "[bold]cli-dj[/] — snippet-driven live coding, visual only. type help() and press Enter"
        )

        if self._set_path is not None:
            self._load_set_path(self._set_path)

        self.set_interval(1 / 30, self._on_tick)
        self.query_one("#command-input", HistoryInput).focus()
        self._refresh_all()

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
        self.scheduler.tick(dt)
        for lane in self.lanes.values():
            lane.update(self.transport)
        self._refresh_all()

    def _refresh_all(self) -> None:
        try:
            lanes_view = self.query_one("#lanes-view", LanesView)
        except NoMatches:  # the interval timer can fire once more while the app shuts down
            return
        lanes_view.refresh_view()
        self._refresh_snips_table()
        self._refresh_queue()

    def _refresh_snips_table(self) -> None:
        table = self.query_one("#snips-table", DataTable)
        names = [name for name, value in self.interp.env.items() if isinstance(value, Snippet)]
        if table.row_count != len(names):
            table.clear()
            for name in names:
                snippet = self.interp.env[name]
                bars = snippet.length_beats / SNIPPET_BEATS_PER_BAR
                table.add_row(
                    name, snippet.track.title, f"{bars:g}", snippet.role, snippet.key,
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
                lines.append(f"#{event_id} @{self.transport.display_at(beat)} {desc}")
        view.update("\n".join(lines))

    def _log(self, message: str, level: str = "info") -> None:
        if level == "warn":
            rendered = f"[yellow]{message}[/]"
        elif level == "error":
            rendered = f"[red]{message}[/]"
        else:
            rendered = message
        self.query_one("#console-log", RichLog).write(rendered)

    def _clear_log(self) -> None:
        self.query_one("#console-log", RichLog).clear()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if isinstance(event.input, HistoryInput):
            event.input.remember(text)
        self.query_one("#console-log", RichLog).write(f"[bold cyan]> {text}[/]")
        self.interp.run(text)
        self._refresh_all()


def main() -> None:
    parser = argparse.ArgumentParser(description="cli-dj: visual-only live-coding DJ TUI")
    parser.add_argument("--set", dest="set_path", default=None, help="path to a .djs set file to load at startup")
    args = parser.parse_args()
    set_path = Path(args.set_path) if args.set_path else None
    DJApp(set_path=set_path).run()


if __name__ == "__main__":
    main()
