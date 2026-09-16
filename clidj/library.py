"""Fake demo library: tracks with beat-resolution procedural waveforms and cues.

Visual-only prototype -- no audio is decoded or played, so every waveform here
is synthesized from a seed instead of read from a real file. Waveform and cue
data are expressed in beats (not seconds) so a Lane can look up "what does
this track look like right now" purely from a beat position, matching the
Transport's own units.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

SAMPLES_PER_BEAT = 16


@dataclass
class Track:
    id: int
    title: str
    artist: str
    bpm: float
    key: str
    duration: float  # seconds; display only (e.g. the library pane's Time column)
    first_beat: float = 0.0  # seconds from file start to the first beat (future beatgrid use)
    waveform: list[float] = field(default_factory=list)  # amplitude, SAMPLES_PER_BEAT per beat
    cues: dict[int, float] = field(default_factory=dict)  # cue number -> beat position

    def __post_init__(self) -> None:
        if not self.waveform:
            self.waveform = _fake_waveform(seed=self.id, duration=self.duration, bpm=self.bpm)
        if not self.cues:
            phrase = 32.0  # 8 bars of 4/4, a typical DJ phrase length
            total = self.duration_beats
            self.cues = {
                1: 0.0,
                2: min(phrase, total * 0.25),
                3: min(phrase * 2, total * 0.55),
                4: min(phrase * 3, total * 0.8),
            }

    @property
    def duration_beats(self) -> float:
        return self.duration * self.bpm / 60.0

    def amplitude_at_beat(self, beat: float) -> float:
        """Fake waveform amplitude (0..1) at a given beat position into the track."""
        if not self.waveform:
            return 0.0
        index = int(beat * SAMPLES_PER_BEAT)
        index = max(0, min(len(self.waveform) - 1, index))
        return self.waveform[index]


def _fake_waveform(seed: int, duration: float, bpm: float, samples_per_beat: int = SAMPLES_PER_BEAT) -> list[float]:
    """Song-structure envelope (intro/build/drop/breakdown/outro) with a kick
    transient at the start of every beat, so a zoomed-in view shows distinct
    beats rather than smooth noise."""
    rng = random.Random(seed)
    duration_beats = duration * bpm / 60.0
    n = max(samples_per_beat, round(duration_beats * samples_per_beat))
    envelope_points = [0.15, 0.35, 0.9, 0.55, 0.95, 0.4, 0.1]
    values: list[float] = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.0
        seg = t * (len(envelope_points) - 1)
        lo = int(seg)
        hi = min(lo + 1, len(envelope_points) - 1)
        frac = seg - lo
        base = envelope_points[lo] * (1 - frac) + envelope_points[hi] * frac
        phase_in_beat = i % samples_per_beat
        kick = math.exp(-phase_in_beat / 2.2) * 0.6  # sharp decay after each beat's first sample
        wobble = rng.uniform(-0.06, 0.06)
        amp = base * (0.4 + kick) + wobble
        values.append(max(0.03, min(1.0, amp)))
    return values


DEMO_LIBRARY: list[Track] = [
    Track(1, "Nightdrive", "Vektroid Cell", 128.0, "8A", 214),
    Track(2, "Concrete Bloom", "Sable Arc", 126.0, "5A", 231),
    Track(3, "Glass Horizon", "Kite Parade", 130.0, "8A", 198),
    Track(4, "Low Tide Static", "Moriah Deep", 122.0, "3A", 256),
    Track(5, "Afterimage", "Nova Kessler", 128.0, "8B", 207),
    Track(6, "Chrome Petals", "Yui Osprey", 132.0, "10A", 220),
    Track(7, "Faultline", "Dren & Coe", 125.0, "5A", 244),
    Track(8, "Half Light", "Ruin Choir", 128.0, "8A", 233),
]
