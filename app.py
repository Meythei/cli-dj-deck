"""cli-dj: a visual-only TUI prototype for a terminal DJ mixer.

Ghostty-style pane layout: library on the left, a command console top-right,
two beat-synced deck waveforms bottom-right. No audio is decoded or played --
tracks, waveforms and playback are all simulated so the layout and command
language can be iterated on before wiring up a real audio engine.
"""
from __future__ import annotations

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import DataTable, Input, RichLog, Static

from commands import CommandError, execute
from library import DEMO_LIBRARY, Track
from models import Deck

BAR_CHARS = " ▁▂▃▄▅▆▇█"  # " ▁▂▃▄▅▆▇█"
MIN_WIDTH = 20


class DeckView(Static):
    def __init__(self, deck: Deck, accent: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.deck = deck
        self.accent = accent

    def refresh_view(self) -> None:
        deck = self.deck
        width = max(self.content_size.width, MIN_WIDTH)

        if not deck.track:
            self.update(f"{deck.name}  — empty deck —\n(load with {deck.name.lower()}.load(n))")
            return

        track = deck.track
        icon = "▶" if deck.playing else ("■" if deck.position == 0 else "❚❚")
        sync = f" sync→{deck.synced_to}" if deck.synced_to else ""
        pos_m, pos_s = divmod(int(deck.position), 60)
        dur_m, dur_s = divmod(int(track.duration), 60)
        header = (
            f"{deck.name} {icon}  {track.title} — {track.artist}   "
            f"{deck.bpm:.1f} BPM{sync}   {track.key}   "
            f"{pos_m:02d}:{pos_s:02d}/{dur_m:02d}:{dur_s:02d}"
        )

        played_ratio = (deck.position / track.duration) if track.duration else 0.0
        played_col = min(width - 1, int(played_ratio * width))

        loop_cols = None
        if deck.loop:
            loop_start, loop_end = deck.loop
            loop_cols = (
                max(0, min(width - 1, int(loop_start / track.duration * width))),
                max(0, min(width - 1, int(loop_end / track.duration * width))),
            )

        marker_line = [" "] * width
        for cue_n, cue_t in track.cues.items():
            col = max(0, min(width - 1, int(cue_t / track.duration * width)))
            marker_line[col] = str(cue_n)
        if loop_cols:
            marker_line[loop_cols[0]] = "["
            marker_line[loop_cols[1]] = "]"

        wave = track.waveform
        n_samples = len(wave)
        text = Text(header + "\n")
        for x in range(width):
            amp = wave[min(n_samples - 1, int(x / width * n_samples))]
            level = round(amp * (len(BAR_CHARS) - 1))
            ch = BAR_CHARS[level]
            in_loop = loop_cols is not None and loop_cols[0] <= x <= loop_cols[1]
            if x == played_col:
                style = f"bold white on {self.accent}"
            elif in_loop:
                style = "bold yellow"
            elif x < played_col:
                style = f"bold {self.accent}"
            else:
                style = "grey50"
            text.append(ch, style=style)
        text.append("\n")
        text.append("".join(marker_line), style="grey62")
        self.update(text)


class DJApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "cli-dj (visual prototype)"

    crossfade = reactive(0.5)

    def __init__(self) -> None:
        super().__init__()
        self.library: list[Track] = DEMO_LIBRARY
        self.decks: dict[str, Deck] = {"A": Deck("A"), "B": Deck("B")}
        self.decks["A"].load(self.library[0])
        self.decks["B"].load(self.library[2])

    def compose(self) -> ComposeResult:
        with Horizontal(id="root"):
            with Vertical(id="library-pane"):
                yield DataTable(id="library-table")
            with Vertical(id="right-pane"):
                with Vertical(id="console-pane"):
                    yield RichLog(id="console-log", markup=True, wrap=True)
                    yield Input(
                        id="command-input",
                        placeholder="a.play()  b.cue(1)  a.loop(2, 4)  help()",
                    )
                with Vertical(id="decks-pane"):
                    yield DeckView(self.decks["A"], accent="cyan", id="deck-a")
                    yield DeckView(self.decks["B"], accent="magenta", id="deck-b")

    def on_mount(self) -> None:
        self.query_one("#library-pane").border_title = "LIBRARY"
        self.query_one("#console-pane").border_title = "CONSOLE"
        self.query_one("#deck-a").border_title = "DECK A"
        self.query_one("#deck-b").border_title = "DECK B"

        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.add_columns("#", "Title", "Artist", "BPM", "Key", "Time")
        for t in self.library:
            m, s = divmod(int(t.duration), 60)
            table.add_row(str(t.id), t.title, t.artist, f"{t.bpm:.0f}", t.key, f"{m}:{s:02d}")

        log = self.query_one("#console-log", RichLog)
        log.write("[bold]cli-dj visual prototype[/] — type help() and press Enter")
        log.write("A: #1 Nightdrive   B: #3 Glass Horizon   (visual only, no audio)")

        self._refresh_all_decks()
        self.set_interval(0.1, self._on_tick)
        self.query_one("#command-input", Input).focus()

    def _on_tick(self) -> None:
        for deck in self.decks.values():
            deck.tick(0.1)
        self._refresh_all_decks()

    def _refresh_all_decks(self) -> None:
        for view in self.query(DeckView):
            view.refresh_view()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return

        log = self.query_one("#console-log", RichLog)
        log.write(f"[bold cyan]>[/] {text}")
        try:
            result = execute(text, self)
            if result:
                log.write(f"[green]{result}[/]")
        except CommandError as exc:
            log.write(f"[red]error:[/] {exc}")
        self._refresh_all_decks()

    def clear_console(self) -> None:
        self.query_one("#console-log", RichLog).clear()


if __name__ == "__main__":
    DJApp().run()
