"""The one clock everything else reads from.

No other module may track its own notion of elapsed time: lanes ask the
transport where "now" is, the scheduler asks it which beat boundaries a tick
crossed. This is deliberate -- when a real audio engine replaces the visual
simulation, its sample clock should be the only thing calling `advance()`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

QUANT_UNITS_BARS = {"beat": None, "bar": 1, "phrase": 8}  # beat is handled separately (1 beat, not 1 bar)

# Repeatedly summing small dt slices drifts a few ulps below the "true" value
# (e.g. 75 steps of 0.025s at 128 BPM lands on 3.9999999999999947, not 4.0).
# Every boundary computation below nudges by this before flooring/modulo so a
# position that is "supposed to be" exactly on a beat doesn't get quantized
# down to the beat before it.
EPSILON = 1e-9


@dataclass
class Transport:
    bpm: float = 128.0
    beats_per_bar: int = 4
    position_beats: float = 0.0
    running: bool = False
    # How far past the position a boundary must be to still be reachable.
    # With a realtime engine a command needs a few ms to get there, so a bar
    # head closer than this counts as already gone (see Session).
    commit_margin_beats: float = 0.0

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def advance(self, dt_seconds: float) -> tuple[float, float]:
        """Advance the clock by dt_seconds of wall time (no-op while stopped).

        Returns (prev_position_beats, new_position_beats) so a caller (the
        scheduler) can tell which beat range this tick crossed.
        """
        prev = self.position_beats
        if self.running and dt_seconds > 0:
            self.position_beats += dt_seconds * self.bpm / 60.0
        return prev, self.position_beats

    @property
    def bar(self) -> int:
        return self.bar_at(self.position_beats)

    @property
    def beat_in_bar(self) -> float:
        return ((self.position_beats + EPSILON) % self.beats_per_bar) + 1

    @property
    def display(self) -> str:
        return self.display_at(self.position_beats)

    def bar_at(self, beats: float) -> int:
        return int((beats + EPSILON) // self.beats_per_bar) + 1

    def display_at(self, beats: float) -> str:
        bar = self.bar_at(beats)
        beat_in_bar = ((beats + EPSILON) % self.beats_per_bar) + 1
        return f"{bar:03d}.{int(beat_in_bar)}"

    def beats_at_bar(self, bar: int) -> float:
        """Absolute beat position (0-based) where the given 1-based bar starts."""
        return (bar - 1) * self.beats_per_bar

    def next_boundary_beats(self, quant: str) -> float:
        """The next beat matching `quant`, strictly after the current position
        plus the commit margin.

        quant: "beat" | "bar" | "phrase" | "none" (returns position_beats as-is).
        """
        if quant == "none":
            return self.position_beats
        if quant == "beat":
            unit = 1.0
        elif quant in ("bar", "phrase"):
            unit = self.beats_per_bar * (QUANT_UNITS_BARS[quant] or 1)
        else:
            raise ValueError(f"unknown quantize mode {quant!r}")
        n = math.floor((self.position_beats + self.commit_margin_beats + EPSILON) / unit) + 1
        return n * unit
