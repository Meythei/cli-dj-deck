"""Lanes: N playback slots that replace the old fixed-2-deck model.

A Lane holds no time of its own -- its position inside whatever Snippet it's
playing is always computed fresh from the Transport (`started_at_beat` plus
however many beats the Transport has moved since). That's what makes every
lane's beat grid line up automatically: they're all reading the same clock.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from snippets import Snippet
from transport import Transport

LANE_NAMES: tuple[str, ...] = ("L1", "L2", "L3", "L4")

BPM_DRIFT_WARN_RATIO = 0.08  # source bpm vs transport bpm more than this apart -> warn


@dataclass
class Lane:
    name: str
    snippet: Optional[Snippet] = None
    started_at_beat: float = 0.0
    gain: float = 1.0
    muted: bool = False
    lo: float = 1.0
    mid: float = 1.0
    hi: float = 1.0

    def start_snippet(self, snippet: Snippet, transport: Transport) -> None:
        self.snippet = snippet
        self.started_at_beat = transport.position_beats
        self.gain = 1.0

    def stop(self) -> None:
        self.snippet = None

    def local_beat(self, transport: Transport) -> Optional[float]:
        """Position within the snippet's own length, or None if nothing is
        playing (including a finished, non-looping snippet)."""
        if self.snippet is None:
            return None
        local = transport.position_beats - self.started_at_beat
        if local < 0:
            return None
        if self.snippet.loop:
            return local % self.snippet.length_beats
        if local >= self.snippet.length_beats:
            return None
        return local

    def track_beat(self, transport: Transport) -> Optional[float]:
        """Position within the *source track*, for waveform lookups."""
        local = self.local_beat(transport)
        if local is None:
            return None
        return self.snippet.start_beat + local

    def update(self, transport: Transport) -> None:
        """Clear a finished, non-looping snippet. Call once per tick."""
        if self.snippet is not None and self.local_beat(transport) is None:
            local = transport.position_beats - self.started_at_beat
            if local >= 0:
                self.snippet = None


def _parse_camelot(code: str) -> Optional[tuple[int, str]]:
    m = re.match(r"^\s*(\d{1,2})([ABab])\s*$", code)
    if not m:
        return None
    return int(m.group(1)), m.group(2).upper()


def keys_compatible(a: str, b: str) -> bool:
    """Camelot-wheel compatibility: identical, same number (relative major/minor),
    or adjacent number with the same letter. Unparsable codes are never flagged."""
    pa, pb = _parse_camelot(a), _parse_camelot(b)
    if pa is None or pb is None:
        return True
    na, la = pa
    nb, lb = pb
    if na == nb:
        return True
    diff = (na - nb) % 12
    return la == lb and diff in (1, 11)


def check_warnings(lane: Lane, others: list[Lane], transport_bpm: float) -> list[str]:
    """Human-readable warnings for what `lane` is currently playing, given
    what's playing in `others`. Never blocks anything -- purely advisory."""
    warnings: list[str] = []
    snippet = lane.snippet
    if snippet is None:
        return warnings

    if snippet.bpm > 0:
        drift = abs(snippet.bpm - transport_bpm) / snippet.bpm
        if drift > BPM_DRIFT_WARN_RATIO:
            warnings.append(
                f"{lane.name}: source bpm {snippet.bpm:.1f} is {drift * 100:.0f}% "
                f"off the transport ({transport_bpm:.1f})"
            )

    active_others = [o for o in others if o.snippet is not None]

    if snippet.role == "vocal":
        other_vocals = [o for o in active_others if o.snippet.role == "vocal"]
        if other_vocals:
            names = ", ".join(o.name for o in other_vocals)
            warnings.append(f"{lane.name}: another vocal is already playing ({names})")

    for other in active_others:
        if not keys_compatible(snippet.key, other.snippet.key):
            warnings.append(
                f"{lane.name}/{other.name}: keys {snippet.key}/{other.snippet.key} may clash"
            )

    return warnings
