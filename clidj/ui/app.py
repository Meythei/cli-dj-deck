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
import gc
import sys
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import DataTable, Input, RichLog, Static, TabbedContent, TabPane

from ..config import Config, ConfigError, Paths, install_dir
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
SETS_DIR = install_dir() / "sets"


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
class _Clock:
    """The bits of a Transport the lane drawing reads, at a given position."""

    position_beats: float
    beats_per_bar: int


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


def _column_peaks(snippet: Snippet, waveform: np.ndarray, local_now: float, axis: "WaveAxis") -> np.ndarray:
    """Peak waveform amplitude per waveform column: the max over the column's
    [beat - 1/8, beat + 1/8) range of the snippet, NaN where a non-looping
    snippet has nothing to show. Vectorised: this runs for every lane on
    every frame."""
    cols = np.arange(axis.left, axis.left + axis.width)
    local = local_now + (cols - axis.center) / QUARTER_COLS_PER_BEAT
    half = 0.5 / QUARTER_COLS_PER_BEAT
    first = np.floor((local - half) * SAMPLES_PER_BEAT).astype(np.int64)
    last = np.maximum(first + 1, np.floor((local + half) * SAMPLES_PER_BEAT).astype(np.int64))
    span = int((last - first).max())
    index = first[:, None] + np.arange(span)[None, :]
    covered = index < last[:, None]
    sample_local = index / SAMPLES_PER_BEAT
    if snippet.loop:
        sample_local = np.mod(sample_local, snippet.length_beats)
        valid = covered
    else:
        valid = covered & (sample_local >= 0.0) & (sample_local < snippet.length_beats)
    if len(waveform) == 0:
        amps = np.zeros(index.shape)
    else:
        track_index = np.clip(((snippet.start_beat + sample_local) * SAMPLES_PER_BEAT).astype(np.int64),
                              0, len(waveform) - 1)
        amps = waveform[track_index]
    peaks = np.where(valid, amps, -np.inf).max(axis=1)
    peaks[np.isneginf(peaks)] = np.nan
    return peaks


def _append_runs(text: Text, chars: list[str], styles: list[str]) -> None:
    """Append characters grouped into runs of equal style (one span per run
    instead of per cell keeps Rich fast)."""
    start = 0
    for i in range(1, len(chars) + 1):
        if i == len(chars) or styles[i] != styles[start]:
            text.append("".join(chars[start:i]), style=styles[start] or None)
            start = i


class LanesView(Static):
    """Transport status, a shared beat ruler, and per lane an info row plus
    a two-row zoomed, center-locked waveform -- all drawn on one WaveAxis so
    beat heads line up vertically across the ruler and every lane."""

    def __init__(self, transport: Transport, lanes: dict[str, Lane], quant_getter, session=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.transport = transport
        self.lanes = lanes
        self._quant_getter = quant_getter
        self.session = session
        self._waveforms: dict[int, tuple[list, np.ndarray]] = {}

    def _waveform_array(self, track: Track) -> np.ndarray:
        track.amplitude_at_beat(0.0)  # loads a lazily computed waveform
        cached = self._waveforms.get(id(track))
        if cached is None or cached[0] is not track.waveform:
            cached = (track.waveform, np.asarray(track.waveform, dtype=np.float64))
            self._waveforms[id(track)] = cached
        return cached[1]

    def refresh_view(self) -> None:
        self.update(self.build_text(max(self.content_size.width, MIN_WIDTH)))

    def build_text(self, width: int) -> Text:
        transport = self.transport
        # Draw what is being *heard*: the engine is ahead of the speakers by
        # its output latency plus the limiter's lookahead.
        position = self.session.heard_beats if self.session is not None else transport.position_beats
        clock = _Clock(position, transport.beats_per_bar)
        axis = WaveAxis.for_width(width)
        heads = _bar_head_columns(axis, position, transport.beats_per_bar)

        status = "● RUN" if transport.running else "■ STOP"
        header = Text(
            f"TRANSPORT  {transport.display_at(position)}  {status}  {transport.bpm:.1f} BPM  "
            f"{transport.beats_per_bar}/4  q:{self._quant_getter()}",
            style="bold",
        )
        rows: list[Text] = [header]
        if self.session is not None:
            note = self.session.tempo_note()
            if note:
                header.append(f"  {note}", style="bold yellow")
            rows.append(self._audio_row(width))
        rows.append(self._ruler(axis, heads))
        for name in LANE_NAMES:
            rows.extend(self._lane_rows(self.lanes[name], axis, heads, width, clock))
        return Text("\n").join(rows)

    def _audio_row(self, width: int) -> Text:
        summary = self.session.audio_summary()
        row = Text("AUDIO      ", style="bold")
        if summary["mode"] == "off":
            row.append("off (--no-audio: visual only)", style="grey62")
            return row
        if not summary["alive"]:
            row.append("ENGINE DOWN -- no sound; restart cli-dj", style="bold white on red")
            return row
        # Problems first, so a narrow pane never truncates them away.
        row.append(f"xrun {summary['underruns']}", style="bold red" if summary["underruns"] else "grey70")
        late_style = "bold white on red" if summary["late_recent"] else ("bold red" if summary["late"] else "grey70")
        late_text = f"  late {summary['late']}"
        if summary["late"]:
            late_text += f" (max {summary['late_max_ms']:.0f}ms)"
        row.append(late_text, style=late_style)
        if summary["errors"] or summary["missing"]:
            row.append(f"  err {summary['errors']} missing {summary['missing']}", style="bold red")
        # Warn on the smoothed load (about the last second) reaching half the
        # block time; a one-off peak only matters if it overran a whole block.
        load, load_max = summary["load"], summary["load_max"]
        row.append(f"  cpu {load:.0%}", style="bold yellow" if load >= 0.5 else "grey70")
        row.append(f" (max {load_max:.0%})", style="bold yellow" if load_max >= 1.0 else "grey70")
        row.append(f"  lat {summary['latency_ms']:.0f}ms  {summary['samplerate'] / 1000:g}k/{summary['blocksize']}",
                   style="grey70")
        device = summary["device"] if summary["mode"] != "null" else "null backend (no sound)"
        hostapi = summary["hostapi"].replace("Windows ", "")
        row.append(f"  {device}" + (f" [{hostapi}]" if hostapi else ""), style="grey50")
        row.truncate(width)
        return row

    def _ruler(self, axis: WaveAxis, heads: dict[int, str]) -> Text:
        row = [" "] * (axis.left + axis.width)
        for col, kind in heads.items():
            row[col] = "|" if kind == "bar" else "·"
        row[axis.center] = "▼"
        return Text("".join(row), style="grey62")

    def _lane_rows(self, lane: Lane, axis: WaveAxis, heads: dict[int, str], width: int, clock=None) -> list[Text]:
        clock = clock or self.transport
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
        eq_flat = lane.lo == lane.mid == lane.hi == 1.0
        info.append("eq" + "".join(_meter(v, 1) for v in (lane.lo, lane.mid, lane.hi)) + " ",
                    style="grey62" if eq_flat else "bold cyan")
        if lane.muted:
            info.append("MUTE ", style="bold red")
        info.append(f"{_loop_display(lane, clock):<11.11}", style="grey62")
        if self.session is not None:
            for note, style in self.session.lane_notes(lane.name):
                info.append(f" {note}", style=style)
        if warnings:
            info.append(" " + warnings[0], style="yellow")
        info.truncate(width)

        top, bottom = self._wave_rows(lane, axis, heads, accent, clock)
        return [info, top, bottom]

    def _wave_rows(self, lane: Lane, axis: WaveAxis, heads: dict[int, str], accent: str,
                   clock=None) -> tuple[Text, Text]:
        """Two rows stacked into one bar per column: the bottom row fills up
        to half amplitude, the top row shows the rest."""
        top = Text(" " * axis.left)
        bottom = Text(" " * axis.left)
        snippet = lane.snippet
        local_now = None if snippet is None else lane.local_beat(clock or self.transport)
        columns = axis.columns()
        tints = [" on grey15" if heads.get(col) == "bar" else "" for col in columns]

        if snippet is None or local_now is None:
            styles = [f"grey37{tint}" if tint else "" for tint in tints]
            blank = [" "] * len(columns)
            _append_runs(top, blank, styles)
            _append_runs(bottom, blank, styles)
            return top, bottom

        peaks = _column_peaks(snippet, self._waveform_array(snippet.track), local_now, axis)
        levels = len(BAR_CHARS) - 1
        present = ~np.isnan(peaks)
        filled = np.nan_to_num(peaks)
        top_levels = np.rint(np.clip(filled * 2 - 1, 0.0, 1.0) * levels).astype(int)
        bottom_levels = np.rint(np.clip(filled * 2, 0.0, 1.0) * levels).astype(int)
        top_chars = [BAR_CHARS[level] if ok else " " for level, ok in zip(top_levels, present)]
        bottom_chars = [BAR_CHARS[level] if ok else " " for level, ok in zip(bottom_levels, present)]
        past, future = f"bold {accent}", "grey50"
        styles = [(past if col < axis.center else future) + tint for col, tint in zip(columns, tints)]
        _append_runs(top, top_chars, styles)
        _append_runs(bottom, bottom_chars, styles)
        return top, bottom


TRACK_STATUS_GLYPHS = {"demo": " ", "new": "·", "analyzing": "…", "ready": "✓", "failed": "✗", "missing": "?"}
PREP_GLYPHS = {"none": "·", "pending": "…", "ready": "✓", "failed": "✗"}


class DJApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "cli-dj"
    # Ctrl+C quits (and so shuts the engine process down) instead of Textual's
    # default "press ctrl+q" hint.
    BINDINGS = [Binding("ctrl+c", "quit", "Quit", show=False, priority=True)]

    def __init__(
        self,
        set_path: Path | None = None,
        *,
        demo: bool = False,
        paths: Paths | None = None,
        config: Config | None = None,
        jobs=None,
        audio: bool = False,
        backend: str = "sounddevice",
        device=None,
        engine=None,
    ) -> None:
        super().__init__()
        self._set_path = set_path
        self._pending_log: list[tuple[str, str]] = []
        self.paths = paths or Paths.default()
        self.config = config or Config.load(self.paths)
        if engine is None and audio:
            engine = self._start_engine(backend, device)
            audio = engine is not None
        self.session = Session(self.paths, self.config, self._log, demo=demo, jobs=jobs, prepare=audio, engine=engine)
        self.library: list[Track] = self.session.tracks
        self.transport = self.session.transport
        self.lanes: dict[str, Lane] = self.session.lanes
        self.scheduler = self.session.scheduler
        self.interp = Interpreter(self.session, SETS_DIR, log=self._log, on_clear=self._clear_log,
                                  on_quit=self.exit)
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
                    yield RichLog(id="console-log", markup=True, wrap=True, max_lines=4000)
                    yield Static(id="queue-view")
                    yield HistoryInput(
                        id="command-input",
                        placeholder='start()  L1 << kick  at(33, xf(L1, L2, 8))  help()',
                    )
                yield LanesView(
                    self.transport, self.lanes, lambda: self.interp.quant_mode, session=self.session,
                    id="lanes-view",
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
        # Startup objects (Textual, numpy, the library) live as long as the app:
        # keep them out of the cyclic GC's full passes, which otherwise stall
        # frames (and so delay reservations) by tens of milliseconds.
        gc.collect()
        gc.freeze()

    def _start_engine(self, backend: str, device):
        """Start the engine process; on failure log why and return None so the
        app runs visual-only instead of not at all."""
        from ..engine.host import HostConfig, RealtimeEngineClient

        config = self.config
        client = RealtimeEngineClient(HostConfig(
            backend=backend, samplerate=config.samplerate, blocksize=config.blocksize, bpm=128.0,
            device=device if device is not None else config.device, hostapi=config.hostapi,
            limiter_ceiling_db=config.limiter_ceiling_db, master_gain=config.master_gain,
            log_path=str(self.paths.log_dir / "engine.log"),
        ))
        try:
            info = client.start()
        except RuntimeError as exc:
            self._pending_log.append((f"audio engine failed to start ({exc}); running visual-only", "error"))
            return None
        latency_ms = info.get("output_latency_s", 0.0) * 1000
        self._pending_log.append((
            f"audio: {info['device']} {info.get('hostapi', '')} -- {info['samplerate']} Hz, "
            f"{info['blocksize']} samples, output latency {latency_ms:.0f} ms", "info"))
        if info.get("fallback_reason"):
            self._pending_log.append((
                f"could not open the audio device ({info['fallback_reason']}); the engine runs without sound", "error"))
        return client

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m clidj", description="cli-dj: live-coding DJ TUI")
    parser.add_argument("--set", dest="set_path", default=None, help="path to a .djs set file to load at startup")
    parser.add_argument("--demo", action="store_true", help="use the built-in demo library instead of your music")
    parser.add_argument("--no-audio", action="store_true", help="visual only: no rendering, no audio device")
    parser.add_argument("--device", default=None, help="output device: index or part of its name (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true", help="list audio output devices and exit")
    parser.add_argument("--null-audio", action="store_true",
                        help="run the realtime engine without a device (timing and load, no sound)")
    args = parser.parse_args(argv)
    if args.list_devices:
        from ..engine.backends import list_output_devices

        for dev in list_output_devices():
            marker = "*" if dev.is_default else " "
            print(f"{marker} {dev.index:>3}  {dev.hostapi:<22} {dev.channels}ch {dev.default_samplerate:>7.0f} Hz  {dev.name}")
        print("(* = default output of its host API; pass the number or part of the name to --device)")
        return 0
    set_path = Path(args.set_path) if args.set_path else None
    try:
        app = DJApp(set_path=set_path, demo=args.demo, audio=not args.no_audio,
                    backend="null" if args.null_audio else "sounddevice", device=args.device)
    except ConfigError as exc:
        print(f"cli-dj: {exc}", file=sys.stderr)
        return 2
    app.run()
    return 0


if __name__ == "__main__":
    main()
