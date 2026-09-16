"""Tempo map: the only conversion between musical time (beats) and audio
time (transport samples).

Beats are the source of truth for *where* something happens musically;
samples are derived through this map. The map is piecewise constant-tempo:
each segment starts at a beat, at the exact (fractional) transport sample
that beat falls on under the previous tempo.

An event scheduled for beat `b` happens at the integer sample
`event_sample(beat_to_sample(b))` = floor(exact + 0.5). Every consumer uses
that same rounding, so "the sample a beat lands on" has one answer.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass


def event_sample(exact: float) -> int:
    return math.floor(exact + 0.5)


@dataclass(frozen=True)
class TempoSegment:
    beat: float
    sample: float  # exact transport sample where `beat` falls
    bpm: float

    def samples_per_beat(self, samplerate: int) -> float:
        return 60.0 * samplerate / self.bpm


class TempoMap:
    def __init__(self, bpm: float, samplerate: int) -> None:
        if bpm <= 0:
            raise ValueError("bpm must be positive")
        self.samplerate = samplerate
        self.segments: list[TempoSegment] = [TempoSegment(0.0, 0.0, float(bpm))]

    # ---- lookups -----------------------------------------------------------------

    def _index_for_beat(self, beat: float) -> int:
        return max(0, bisect.bisect_right([s.beat for s in self.segments], beat) - 1)

    def _index_for_sample(self, sample: float) -> int:
        starts = [event_sample(s.sample) for s in self.segments]
        return max(0, bisect.bisect_right(starts, sample) - 1)

    def segment_for_beat(self, beat: float) -> TempoSegment:
        return self.segments[self._index_for_beat(beat)]

    def segment_for_sample(self, sample: float) -> TempoSegment:
        """The segment governing transport sample `sample` (segments take
        effect from their event sample)."""
        return self.segments[self._index_for_sample(sample)]

    def next_change_sample(self, sample: int) -> float:
        """Event sample of the first tempo change strictly after `sample`,
        or +inf."""
        for segment in self.segments:
            start = event_sample(segment.sample)
            if start > sample:
                return start
        return math.inf

    def beat_to_sample(self, beat: float) -> float:
        seg = self.segment_for_beat(beat)
        return seg.sample + (beat - seg.beat) * seg.samples_per_beat(self.samplerate)

    def sample_to_beat(self, sample: float) -> float:
        seg = self.segment_for_sample(sample)
        return seg.beat + (sample - seg.sample) / seg.samples_per_beat(self.samplerate)

    def bpm_at_beat(self, beat: float) -> float:
        return self.segment_for_beat(beat).bpm

    def bpm_at_sample(self, sample: float) -> float:
        return self.segment_for_sample(sample).bpm

    # ---- edits ----------------------------------------------------------------------

    def set_tempo(self, beat: float, bpm: float) -> TempoSegment:
        """From `beat` on, play at `bpm`. Later segments are discarded (a
        tempo change replaces whatever was planned after it)."""
        if bpm <= 0:
            raise ValueError("bpm must be positive")
        sample = self.beat_to_sample(beat)
        index = self._index_for_beat(beat)
        kept = self.segments[: index + 1]
        if abs(kept[-1].beat - beat) < 1e-12:
            kept = kept[:-1]
        segment = TempoSegment(float(beat), sample, float(bpm))
        self.segments = kept + [segment] if kept else [segment]
        return segment
