"""DSP building blocks for the engine: parameter smoothing, the per-lane
3-band isolator EQ, and the master bus with its limiter.

All of these keep their state across blocks and never allocate more than a
few block-sized temporaries per call.
"""
from __future__ import annotations

import math
from typing import Union

import numpy as np

Curve = Union[float, np.ndarray]  # a scalar when the value is constant over the block


class Smoother:
    """A parameter that glides linearly to a new target over `ramp` samples
    instead of jumping (no zipper noise on immediate changes)."""

    __slots__ = ("value", "target", "step", "remaining", "ramp")

    def __init__(self, value: float, ramp: int) -> None:
        self.value = float(value)
        self.target = float(value)
        self.step = 0.0
        self.remaining = 0
        self.ramp = max(1, int(ramp))

    def set(self, target: float) -> None:
        target = float(target)
        if target == self.target and self.remaining == 0:
            return
        self.target = target
        self.remaining = self.ramp
        self.step = (target - self.value) / self.ramp

    def jump(self, value: float) -> None:
        self.value = self.target = float(value)
        self.remaining = 0
        self.step = 0.0

    def render(self, n: int, arange: np.ndarray) -> Curve:
        """Values for the next n samples (a scalar if constant); advances."""
        if self.remaining == 0:
            return self.value
        k = min(n, self.remaining)
        curve = np.empty(n, dtype=np.float64)
        curve[:k] = self.value + self.step * (arange[1:k + 1])
        self.remaining -= k
        if self.remaining == 0:
            self.value = self.target
            curve[k:] = self.target
        else:
            self.value = float(curve[k - 1])
        return curve

    def advance(self, n: int) -> None:
        k = min(n, self.remaining)
        self.remaining -= k
        self.value = self.target if self.remaining == 0 else self.value + self.step * k


# ---- 3-band isolator ------------------------------------------------------------------------

ISOLATOR_LOW_HZ = 250.0
ISOLATOR_HIGH_HZ = 2500.0


def _butter2_sos(freq: float, btype: str, sr: int) -> np.ndarray:
    from scipy.signal import butter

    return butter(2, freq, btype=btype, fs=sr, output="sos")


def _lr4(freq: float, btype: str, sr: int) -> np.ndarray:
    """Linkwitz-Riley 4th order = two identical 2nd-order Butterworths."""
    sos = _butter2_sos(freq, btype, sr)
    return np.vstack([sos, sos])


def _allpass2(freq: float, sr: int) -> np.ndarray:
    """2nd-order allpass with Q = 1/sqrt(2): exactly LR4 lowpass + highpass at
    `freq`, used to phase-align the low band with the mid/high split."""
    w0 = 2 * math.pi * freq / sr
    alpha = math.sin(w0) / (2 * (1 / math.sqrt(2)))
    cos = math.cos(w0)
    a0 = 1 + alpha
    return np.array([[(1 - alpha) / a0, -2 * cos / a0, (1 + alpha) / a0, 1.0, -2 * cos / a0, (1 - alpha) / a0]])


class Isolator3:
    """DJ-mixer style EQ: the signal is split into low / mid / high bands by
    Linkwitz-Riley crossovers, each band is multiplied by its gain (0..1,
    0 = band removed) and the bands are summed. With all gains at 1 the sum
    is an allpass: flat magnitude, so the EQ can stay in circuit at all
    times. Filter coefficients never change, so moving a gain can't make
    the filters misbehave -- the gains themselves are smoothed."""

    def __init__(self, sr: int, low_hz: float = ISOLATOR_LOW_HZ, high_hz: float = ISOLATOR_HIGH_HZ, channels: int = 2) -> None:
        from scipy.signal import sosfilt

        self._sosfilt = sosfilt
        self.low_lp = _lr4(low_hz, "lowpass", sr)
        self.low_hp = _lr4(low_hz, "highpass", sr)
        self.high_lp = _lr4(high_hz, "lowpass", sr)
        self.high_hp = _lr4(high_hz, "highpass", sr)
        self.high_ap = _allpass2(high_hz, sr)
        self._zi = {
            name: np.zeros((len(sos), 2, channels))
            for name, sos in (("low_lp", self.low_lp), ("low_hp", self.low_hp), ("high_lp", self.high_lp),
                              ("high_hp", self.high_hp), ("high_ap", self.high_ap))
        }

    def _run(self, name: str, sos: np.ndarray, x: np.ndarray) -> np.ndarray:
        y, self._zi[name] = self._sosfilt(sos, x, axis=0, zi=self._zi[name])
        return y

    def process(self, x: np.ndarray, lo: Curve, mid: Curve, hi: Curve) -> np.ndarray:
        low = self._run("high_ap", self.high_ap, self._run("low_lp", self.low_lp, x))
        rest = self._run("low_hp", self.low_hp, x)
        middle = self._run("high_lp", self.high_lp, rest)
        high = self._run("high_hp", self.high_hp, rest)
        return (
            low * _column(lo)
            + middle * _column(mid)
            + high * _column(hi)
        ).astype(np.float32, copy=False)


def _column(curve: Curve):
    return curve if np.isscalar(curve) else curve[:, None]


# ---- master bus ----------------------------------------------------------------------------------


class MasterBus:
    """Master gain -> lookahead brickwall limiter -> hard clip at +-1.0.

    The limiter delays the signal by its lookahead; `latency` is that delay
    in samples (measured, not assumed), and callers that need sample-exact
    output (offline rendering, playhead display) compensate for it."""

    def __init__(self, sr: int, ceiling_db: float = -1.0, gain: float = 1.0, ramp: int = 240,
                 release_ms: float = 100.0, lookahead_ms: float = 5.0) -> None:
        from pedalboard import BrickwallLimiter

        self.sr = sr
        self.gain = Smoother(gain, ramp)
        self._make = lambda: BrickwallLimiter(ceiling_db=ceiling_db, release_ms=release_ms, lookahead_ms=lookahead_ms)
        self.latency = self._measure_latency()
        self.limiter = self._make()
        # The limiter returns fewer samples than it is given until its
        # lookahead is filled; this FIFO turns that into a fixed delay.
        self._fifo = np.zeros((self.latency + 16384, 2), dtype=np.float32)
        self._fifo_len = self.latency
        self.last_peak = (0.0, 0.0)

    def _measure_latency(self) -> int:
        limiter = self._make()
        probe = np.zeros((2, 4096), dtype=np.float32)
        probe[:, 100] = 0.1  # well under any ceiling: passes unchanged, only delayed
        out = limiter.process(probe, self.sr, reset=False)
        consumed = probe.shape[1] - out.shape[1]
        if out.shape[1] and np.max(np.abs(out)) > 0:
            return int(np.argmax(np.abs(out[0])) + consumed - 100)
        return consumed

    def process(self, block: np.ndarray, arange: np.ndarray) -> np.ndarray:
        n = len(block)
        gain = self.gain.render(n, arange)
        mixed = block * _column(gain) if not (np.isscalar(gain) and gain == 1.0) else block
        limited = self.limiter.process(np.ascontiguousarray(mixed.T, dtype=np.float32), self.sr, reset=False).T
        m = len(limited)
        if self._fifo_len + m > len(self._fifo):
            grown = np.zeros((self._fifo_len + m + 16384, 2), dtype=np.float32)
            grown[: self._fifo_len] = self._fifo[: self._fifo_len]
            self._fifo = grown
        self._fifo[self._fifo_len:self._fifo_len + m] = limited
        self._fifo_len += m
        take = min(n, self._fifo_len)
        out = np.zeros((n, 2), dtype=np.float32)
        out[n - take:] = self._fifo[:take]
        self._fifo[: self._fifo_len - take] = self._fifo[take:self._fifo_len]
        self._fifo_len -= take
        np.clip(out, -1.0, 1.0, out=out)
        self.last_peak = (float(np.max(np.abs(out[:, 0]))), float(np.max(np.abs(out[:, 1]))))
        return out
