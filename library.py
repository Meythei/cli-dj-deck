"""Fake demo library: tracks with procedurally generated waveforms and cue points.

Visual-only prototype -- no audio is decoded or played, so every waveform here
is synthesized from a seed instead of read from a real file.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Track:
    id: int
    title: str
    artist: str
    bpm: float
    key: str
    duration: float  # seconds
    waveform: list[float] = field(default_factory=list)
    cues: dict[int, float] = field(default_factory=dict)  # cue number -> seconds

    def __post_init__(self) -> None:
        if not self.waveform:
            self.waveform = _fake_waveform(seed=self.id, duration=self.duration)
        if not self.cues:
            beat = 60.0 / self.bpm
            phrase = beat * 32  # 8 bars of 4/4, a typical DJ phrase length
            self.cues = {
                1: 0.0,
                2: min(phrase, self.duration * 0.25),
                3: min(phrase * 2, self.duration * 0.55),
                4: min(phrase * 3, self.duration * 0.8),
            }


def _fake_waveform(seed: int, duration: float, resolution: int = 600) -> list[float]:
    """Song-structure-shaped fake amplitude curve: intro/build/drop/breakdown/outro."""
    rng = random.Random(seed)
    n = resolution
    envelope_points = [0.15, 0.35, 0.9, 0.55, 0.95, 0.4, 0.1]
    values: list[float] = []
    for i in range(n):
        t = i / (n - 1)
        seg = t * (len(envelope_points) - 1)
        lo = int(seg)
        hi = min(lo + 1, len(envelope_points) - 1)
        frac = seg - lo
        base = envelope_points[lo] * (1 - frac) + envelope_points[hi] * frac
        wobble = 0.15 * math.sin(t * 90 + seed) + rng.uniform(-0.12, 0.12)
        values.append(max(0.03, min(1.0, base + wobble)))
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
