"""Snippets: short, pre-cut pieces of a track (the DJ "tools" of this tool).

Cutting is offline/prep-time work -- it happens before a set runs, from
`snip()` calls typed at the REPL or read from a .djs set file. A Snippet
itself is inert data; a Lane is what actually plays one against the
Transport.
"""
from __future__ import annotations

from dataclasses import dataclass

from .library import Track

# Bar length assumed when cutting from a track's own bars (`bar=` / `bars=`).
# This is about the *source track's* structure, not the live set's Transport,
# so it deliberately doesn't come from Transport.beats_per_bar.
BEATS_PER_BAR = 4


class SnippetError(Exception):
    pass


@dataclass
class Snippet:
    name: str
    track: Track
    start_beat: float
    length_beats: float
    loop: bool = False
    role: str = "other"
    gain: float = 1.0

    @property
    def key(self) -> str:
        return self.track.key

    @property
    def bpm(self) -> float:
        return self.track.bpm


def resolve_track(library: list[Track], track: Track | int | str) -> Track:
    if isinstance(track, Track):
        return track
    if isinstance(track, bool):
        raise SnippetError("track must be a Track, an id, or a title")
    if isinstance(track, int):
        for t in library:
            if t.id == track:
                return t
        raise SnippetError(f"no track #{track} in library")
    if isinstance(track, str):
        for t in library:
            if t.title.lower() == track.lower():
                return t
        raise SnippetError(f"no track titled {track!r} in library")
    raise SnippetError("track must be a Track, an id, or a title")


_auto_name_counter = 0


def snip(
    track: Track | int | str,
    library: list[Track],
    *,
    cue: int | None = None,
    bar: int | None = None,
    bars: float = 8,
    loop: bool = False,
    role: str = "other",
    name: str | None = None,
) -> Snippet:
    """Cut a snippet out of `track`, starting at a cue point or a bar number.

    Exactly one of `cue` or `bar` must be given. Length is `bars` bars (of
    the track's own bar length, BEATS_PER_BAR beats each) and must end up as
    a whole number of beats.
    """
    global _auto_name_counter
    if (cue is None) == (bar is None):
        raise SnippetError("snip() needs exactly one of cue= or bar=")

    resolved = resolve_track(library, track)

    if cue is not None:
        if cue not in resolved.cues:
            raise SnippetError(f"{resolved.title} has no cue {cue}")
        start_beat = resolved.cues[cue]
    else:
        if bar < 1:
            raise SnippetError("bar is 1-based; must be >= 1")
        start_beat = (bar - 1) * BEATS_PER_BAR

    if bars <= 0:
        raise SnippetError("bars must be positive")
    length_beats = float(bars) * BEATS_PER_BAR
    if abs(length_beats - round(length_beats)) > 1e-9:
        raise SnippetError("snippet length must be a whole number of beats")

    if start_beat < 0 or start_beat + length_beats > resolved.duration_beats + 1e-9:
        raise SnippetError(
            f"{resolved.title}: {start_beat:g}+{length_beats:g} beats exceeds "
            f"track length ({resolved.duration_beats:.1f} beats)"
        )

    if name is None:
        _auto_name_counter += 1
        name = f"snip{_auto_name_counter}"

    return Snippet(
        name=name,
        track=resolved,
        start_beat=start_beat,
        length_beats=length_beats,
        loop=loop,
        role=role,
    )
