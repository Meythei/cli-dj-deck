"""Deck playback state -- purely a simulation, no audio engine underneath."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from library import Track


@dataclass
class Deck:
    name: str  # "A" or "B"
    track: Optional[Track] = None
    position: float = 0.0  # seconds into the track
    playing: bool = False
    loop: Optional[tuple[float, float]] = None  # (start_sec, end_sec)
    synced_to: Optional[str] = None  # name of the deck this one's tempo follows
    effective_bpm: Optional[float] = None  # overrides track.bpm while synced

    @property
    def bpm(self) -> float:
        if self.effective_bpm is not None:
            return self.effective_bpm
        return self.track.bpm if self.track else 0.0

    def load(self, track: Track) -> None:
        self.track = track
        self.position = 0.0
        self.playing = False
        self.loop = None
        self.synced_to = None
        self.effective_bpm = None

    def tick(self, dt: float) -> None:
        """Advance playback by dt seconds of wall-clock time."""
        if not self.playing or not self.track:
            return
        rate = (self.bpm / self.track.bpm) if self.track.bpm else 1.0
        self.position += dt * rate
        if self.loop:
            start, end = self.loop
            if self.position >= end:
                self.position = start + (self.position - end)
        if self.position >= self.track.duration:
            self.position = self.track.duration
            self.playing = False

    def jump_to_cue(self, n: int) -> bool:
        if not self.track or n not in self.track.cues:
            return False
        self.position = self.track.cues[n]
        self.playing = False
        return True

    def set_loop(self, beat_a: float, beat_b: float) -> bool:
        if not self.track or not self.track.bpm:
            return False
        seconds_per_beat = 60.0 / self.track.bpm
        start, end = sorted((beat_a * seconds_per_beat, beat_b * seconds_per_beat))
        end = min(end, self.track.duration)
        if end <= start:
            return False
        self.loop = (start, end)
        return True

    def clear_loop(self) -> None:
        self.loop = None
