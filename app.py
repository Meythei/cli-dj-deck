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
import time
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Input, RichLog, Static, TabbedContent, TabPane

from interpreter import Interpreter
from lanes import LANE_NAMES, Lane, check_warnings
from library import DEMO_LIBRARY, Track
from scheduler import Scheduler
from snippets import BEATS_PER_BAR as SNIPPET_BEATS_PER_BAR
from snippets import Snippet
from transport import Transport

BAR_CHARS = " ▁▂▃▄▅▆▇█"  # " ▁▂▃▄▅▆▇█"
QUARTER_COLS_PER_BEAT = 4  # 1 character column ~= 1/4 beat in the zoomed lane view
LANE_ACCENTS = {"L1": "cyan", "L2": "magenta", "L3": "yellow", "L4": "green"}
MIN_WIDTH = 20
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


class LanesView(Static):
    """Transport status, a shared beat ruler, and one zoomed, center-locked
    waveform row per lane -- all sharing the same beat axis so beat heads
    line up vertically across lanes."""

    def __init__(self, transport: Transport, lanes: dict[str, Lane], quant_getter, **kwargs) -> None:
        super().__init__(**kwargs)
        self.transport = transport
        self.lanes = lanes
        self._quant_getter = quant_getter

    def refresh_view(self) -> None:
        width = max(self.content_size.width, MIN_WIDTH)
        transport = self.transport

        status = "● RUN" if transport.running else "■ STOP"
        header = (
            f"TRANSPORT  {transport.display}  {status}  {transport.bpm:.1f} BPM  "
            f"{transport.beats_per_bar}/4  q:{self._quant_getter()}"
        )

        text = Text(header + "\n", style="bold")
        text.append(self._ruler(width) + "\n", style="grey62")
        for name in LANE_NAMES:
            self._append_lane_row(text, self.lanes[name], width)
        self.update(text)

    def _ruler(self, width: int) -> str:
        transport = self.transport
        row = [" "] * width
        center = width / 2.0
        base = transport.position_beats
        beat_lo = int(base - center / QUARTER_COLS_PER_BEAT) - 1
        beat_hi = int(base + (width - center) / QUARTER_COLS_PER_BEAT) + 1
        for beat_n in range(beat_lo, beat_hi + 1):
            col = round(center + (beat_n - base) * QUARTER_COLS_PER_BEAT)
            if 0 <= col < width:
                row[col] = "|" if beat_n % transport.beats_per_bar == 0 else "·"
        playhead_col = round(center)
        if 0 <= playhead_col < width:
            row[playhead_col] = "▼"
        return "".join(row)

    def _append_lane_row(self, text: Text, lane: Lane, width: int) -> None:
        accent = LANE_ACCENTS.get(lane.name, "white")
        warnings: list[str] = []
        if lane.snippet is not None:
            others = [other for name, other in self.lanes.items() if name != lane.name]
            warnings = check_warnings(lane, others, self.transport.bpm)
        name_style = "bold yellow" if warnings else f"bold {accent}"

        if lane.snippet is None:
            label, role, key = "—", "", ""
        else:
            label, role, key = lane.snippet.name, lane.snippet.role, lane.snippet.key

        prefix = f"{lane.name} "
        info = f"{label:<9.9}{role:<7.7}{key:<4.4}"
        suffix = f" gain {_meter(lane.gain)} {_loop_display(lane, self.transport):<11.11}"
        wave_width = max(4, width - len(prefix) - len(info) - len(suffix))

        text.append(prefix, style=name_style)
        text.append(info, style="grey70")
        text.append_text(self._wave_segment(lane, wave_width, accent))
        text.append(suffix, style="grey62")
        text.append("\n")

    def _wave_segment(self, lane: Lane, width: int, accent: str) -> Text:
        segment = Text()
        snippet = lane.snippet
        local_now = None if snippet is None else lane.local_beat(self.transport)
        if snippet is None or local_now is None:
            segment.append(" " * width, style="grey37")
            return segment

        center = width // 2
        for x in range(width):
            offset_beats = (x - center) / QUARTER_COLS_PER_BEAT
            local = local_now + offset_beats
            if snippet.loop:
                local_wrapped = local % snippet.length_beats
                visible = True
            else:
                visible = 0.0 <= local < snippet.length_beats
                local_wrapped = local
            if not visible:
                segment.append(" ")
                continue
            track_beat = snippet.start_beat + local_wrapped
            amp = snippet.track.amplitude_at_beat(track_beat)
            level = round(amp * (len(BAR_CHARS) - 1))
            style = f"bold {accent}" if x < center else "grey50"
            segment.append(BAR_CHARS[level], style=style)
        return segment


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
        tracks_table.add_column("#", width=3)
        tracks_table.add_column("Title", width=16)
        tracks_table.add_column("Artist", width=14)
        tracks_table.add_column("BPM", width=5)
        tracks_table.add_column("Key", width=4)
        tracks_table.add_column("Time", width=6)
        for t in self.library:
            m, s = divmod(int(t.duration), 60)
            tracks_table.add_row(str(t.id), t.title, t.artist, f"{t.bpm:.0f}", t.key, f"{m}:{s:02d}")

        snips_table = self.query_one("#snips-table", DataTable)
        snips_table.cursor_type = "row"
        snips_table.add_column("Name", width=8)
        snips_table.add_column("Track", width=16)
        snips_table.add_column("Bars", width=5)
        snips_table.add_column("Role", width=7)
        snips_table.add_column("Key", width=4)
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
        self.query_one("#lanes-view", LanesView).refresh_view()
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
