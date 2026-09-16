"""Deterministic synthetic audio for tests, listening renders and the demo
library. Nothing copyrighted ever enters the repository: every signal the
tests analyse or render is generated here from a seed.

Array convention (used throughout clidj): audio is float32 shaped
(frames, channels). pedalboard wants (channels, frames); convert only at the
pedalboard call site.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def beat_times(bpm: float, first_beat: float, duration: float) -> np.ndarray:
    period = 60.0 / bpm
    count = int(np.floor((duration - first_beat) / period)) + 1
    times = first_beat + period * np.arange(max(count, 0))
    return times[times < duration]


def _mix_at(out: np.ndarray, start: int, grain: np.ndarray) -> None:
    if start >= len(out) or start + len(grain) <= 0:
        return
    src_lo = max(0, -start)
    dst_lo = max(0, start)
    n = min(len(grain) - src_lo, len(out) - dst_lo)
    out[dst_lo:dst_lo + n] += grain[src_lo:src_lo + n]


def click_grain(sr: int, kind: str = "click", seed: int = 0) -> np.ndarray:
    """A short percussive grain whose attack starts at index 0."""
    if kind == "impulse":
        return np.array([1.0], dtype=np.float32)
    if kind == "click":
        n = int(0.010 * sr)
        rng = np.random.default_rng(seed)
        env = np.exp(-np.arange(n) / (0.002 * sr))
        return (rng.standard_normal(n) * env * 0.8).astype(np.float32)
    if kind == "kick":
        n = int(0.15 * sr)
        t = np.arange(n) / sr
        freq = 50.0 + 100.0 * np.exp(-t * 40.0)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        return (np.sin(phase) * np.exp(-t * 12.0) * 0.9).astype(np.float32)
    raise ValueError(f"unknown grain kind {kind!r}")


def click_track(
    bpm: float,
    first_beat: float,
    duration: float,
    sr: int,
    kind: str = "click",
    channels: int = 2,
    noise: float = 0.003,
    seed: int = 0,
) -> np.ndarray:
    """A grain on every beat of a constant-tempo grid, plus a little noise so
    silence isn't digitally exact."""
    mono = np.zeros(int(round(duration * sr)), dtype=np.float32)
    grain = click_grain(sr, kind, seed)
    for t in beat_times(bpm, first_beat, duration):
        _mix_at(mono, int(round(t * sr)), grain)
    if noise:
        mono += (np.random.default_rng(seed + 1).standard_normal(len(mono)) * noise).astype(np.float32)
    return np.repeat(mono[:, None], channels, axis=1)


def sine(freq: float, duration: float, sr: int, amplitude: float = 0.5, channels: int = 2, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(round(duration * sr))) / sr
    mono = (amplitude * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)
    return np.repeat(mono[:, None], channels, axis=1)


def note_freq(pitch_class: int, octave: int) -> float:
    midi = 12 * (octave + 1) + pitch_class
    return 440.0 * 2 ** ((midi - 69) / 12)


def key_track(root: int, minor: bool, duration: float, sr: int, channels: int = 2) -> np.ndarray:
    """Tonic triad pad plus a slow walk up the (natural minor / major) scale,
    enough tonal evidence for a chroma-profile key detector."""
    scale = (0, 2, 3, 5, 7, 8, 10) if minor else (0, 2, 4, 5, 7, 9, 11)
    triad = (0, 3, 7) if minor else (0, 4, 7)
    n = int(round(duration * sr))
    t = np.arange(n) / sr
    mono = np.zeros(n, dtype=np.float64)
    for interval in triad:
        for octave in (3, 4):
            mono += 0.12 * np.sin(2 * np.pi * note_freq((root + interval) % 12, octave) * t)
    step = int(0.5 * sr)
    for i, start in enumerate(range(0, n, step)):
        degree = scale[i % len(scale)]
        seg = slice(start, min(n, start + step))
        tt = t[seg] - t[start]
        mono[seg] += 0.1 * np.sin(2 * np.pi * note_freq((root + degree) % 12, 5) * tt) * np.exp(-tt * 3)
    mono = mono / np.max(np.abs(mono)) * 0.5
    return np.repeat(mono.astype(np.float32)[:, None], channels, axis=1)


def techno_loop_track(
    bpm: float, key_root: int, minor: bool, duration: float, sr: int, first_beat: float = 0.0, seed: int = 0
) -> np.ndarray:
    """A plausible, fully deterministic 4/4 track: kick on every beat, offbeat
    hats, a root-note bassline and a chord stab every other bar, with a
    16-bar intro/build/drop/break arc. Good enough to hear crossfades, loop
    seams and time-stretch artifacts."""
    n = int(round(duration * sr))
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)
    period = 60.0 / bpm
    rng = np.random.default_rng(seed)
    kick = click_grain(sr, "kick")
    hat_n = int(0.04 * sr)
    hat_env = np.exp(-np.arange(hat_n) / (0.008 * sr)).astype(np.float32)
    bass_n = int(period * 0.45 * sr)
    bt = np.arange(bass_n) / sr
    bass_env = np.minimum(1.0, bt / 0.004) * np.exp(-bt * 6.0)
    chord = (0, 3, 7) if minor else (0, 4, 7)
    stab_n = int(period * 1.5 * sr)
    st = np.arange(stab_n) / sr
    stab_env = np.minimum(1.0, st / 0.01) * np.exp(-st * 2.5)

    for beat, t in enumerate(beat_times(bpm, first_beat, duration)):
        bar = beat // 4
        section = (bar // 16) % 4  # 0 intro, 1 build, 2 drop, 3 break
        start = int(round(t * sr))
        if section != 3:
            _mix_at(left, start, kick * 0.9)
            _mix_at(right, start, kick * 0.9)
        if section >= 1:
            hat = (rng.standard_normal(hat_n).astype(np.float32) * hat_env * 0.18)
            hat -= np.convolve(hat, np.ones(4, dtype=np.float32) / 4, mode="same")  # crude high-pass
            off = int(round((t + period / 2) * sr))
            _mix_at(left, off, hat * 0.8)
            _mix_at(right, off, hat)
        if section == 2 or (section == 1 and bar % 2 == 1):
            f = note_freq(key_root, 1) * (2 if beat % 4 == 2 else 1)
            off = int(round((t + period / 2) * sr))
            tone = (np.tanh(np.sin(2 * np.pi * f * bt) * 3) * bass_env * 0.35).astype(np.float32)
            _mix_at(left, off, tone)
            _mix_at(right, off, tone)
        if section in (2, 3) and beat % 8 == 0:
            stab = np.zeros(stab_n)
            for interval in chord:
                stab += np.sin(2 * np.pi * note_freq((key_root + interval) % 12, 4) * st)
            stab = (stab / len(chord) * stab_env * 0.22).astype(np.float32)
            _mix_at(left, start, stab * 1.0)
            _mix_at(right, start, stab * 0.7)
    out = np.stack([left, right], axis=1)
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.8).astype(np.float32)


def write_audio(path: Path, audio: np.ndarray, sr: int) -> Path:
    """Write (frames, channels) float32 audio; format from the extension."""
    from pedalboard.io import AudioFile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).T)
    with AudioFile(str(path), "w", samplerate=sr, num_channels=data.shape[0]) as f:
        f.write(data)
    return path
