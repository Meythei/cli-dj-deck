"""Parser for the DJ command line: a.play(), b.cue(1), a.loop(2, 4), xfade(0.7), ..."""
from __future__ import annotations

import re

DECK_CALL = re.compile(r"^([abAB])\.(\w+)\(\s*(.*?)\s*\)$")
GLOBAL_CALL = re.compile(r"^(\w+)\(\s*(.*?)\s*\)$")

HELP_TEXT = """commands:
  a.load(n)       load library track #n into deck A (b.load(n) for deck B)
  a.play()        a.pause()        a.stop()
  a.cue(n)        jump to cue point n (1-4)
  a.loop(a, b)    loop between beat a and beat b
  a.loop()        clear the loop
  a.sync()        match this deck's tempo to the other deck
  a.unsync()      restore this deck's own BPM
  xfade(0..1)     move the crossfader (0 = full A, 1 = full B)
  clear()         clear this log
  help()          show this text""".strip()


class CommandError(Exception):
    pass


def _parse_args(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(",")]


def execute(text: str, app) -> str:
    text = text.strip()
    if not text:
        return ""

    m = DECK_CALL.match(text)
    if m:
        deck_name, method, raw_args = m.group(1).upper(), m.group(2).lower(), m.group(3)
        deck = app.decks[deck_name]
        other = app.decks["B" if deck_name == "A" else "A"]
        return _run_deck_method(deck, other, method, _parse_args(raw_args), app)

    m = GLOBAL_CALL.match(text)
    if m:
        func, raw_args = m.group(1).lower(), m.group(2)
        return _run_global(func, _parse_args(raw_args), app)

    raise CommandError(f"parse error: {text!r} (try help())")


def _run_deck_method(deck, other, method: str, args: list[str], app) -> str:
    if method == "play":
        if not deck.track:
            raise CommandError(f"deck {deck.name}: no track loaded")
        deck.playing = True
        return f"{deck.name} ▶ playing"

    if method == "pause":
        deck.playing = False
        return f"{deck.name} ❚❚ paused"

    if method == "stop":
        deck.playing = False
        deck.position = 0.0
        return f"{deck.name} ■ stopped"

    if method == "cue":
        if not args:
            raise CommandError("cue(n) needs a cue number")
        try:
            n = int(args[0])
        except ValueError:
            raise CommandError(f"cue number must be an integer, got {args[0]!r}") from None
        if not deck.jump_to_cue(n):
            raise CommandError(f"deck {deck.name}: no cue {n}")
        return f"{deck.name} → cue {n}"

    if method == "loop":
        if not args:
            deck.clear_loop()
            return f"{deck.name} loop off"
        if len(args) != 2:
            raise CommandError("loop(a, b) needs two beat numbers")
        try:
            a, b = float(args[0]), float(args[1])
        except ValueError:
            raise CommandError("loop(a, b) beat numbers must be numeric") from None
        if not deck.set_loop(a, b):
            raise CommandError(f"deck {deck.name}: invalid loop range")
        return f"{deck.name} loop {a:g}-{b:g}"

    if method == "sync":
        if not deck.track:
            raise CommandError(f"deck {deck.name}: no track loaded")
        if not other.track:
            raise CommandError(f"deck {other.name}: no track loaded to sync to")
        deck.effective_bpm = other.bpm
        deck.synced_to = other.name
        return f"{deck.name} synced to {other.name} @ {other.bpm:.1f} BPM"

    if method == "unsync":
        deck.effective_bpm = None
        deck.synced_to = None
        return f"{deck.name} unsynced"

    if method == "load":
        if not args:
            raise CommandError("load(n) needs a track id")
        try:
            track_id = int(args[0])
        except ValueError:
            raise CommandError(f"track id must be an integer, got {args[0]!r}") from None
        track = next((t for t in app.library if t.id == track_id), None)
        if track is None:
            raise CommandError(f"no track #{track_id} in library")
        deck.load(track)
        return f"{deck.name} loaded #{track_id} {track.title}"

    raise CommandError(f"unknown deck command: {method}()")


def _run_global(func: str, args: list[str], app) -> str:
    if func == "xfade":
        if not args:
            raise CommandError("xfade(value) needs a number from 0 to 1")
        try:
            value = float(args[0])
        except ValueError:
            raise CommandError(f"xfade value must be numeric, got {args[0]!r}") from None
        app.crossfade = max(0.0, min(1.0, value))
        return f"crossfader → {app.crossfade:.2f}"

    if func == "help":
        return HELP_TEXT

    if func == "clear":
        app.clear_console()
        return ""

    raise CommandError(f"unknown command: {func}()")
