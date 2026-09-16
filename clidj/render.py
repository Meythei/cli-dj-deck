"""Snippet rendering: cut a snippet out of its track, time-stretch it to the
set's BPM, and trim it to an exact sample length. Prep-time work only --
nothing here runs during playback (docs/TASK_real-audio.md 3.5, 7).

Output buffers are float32 (frames, 2) `.npy` files named by a content key,
so the engine can `numpy.load(mmap_mode="r")` them without copying between
processes. A key is never rewritten once it exists: a different grid, range,
tempo or renderer version is simply a different file (docs/decisions.md).

Stretching uses Rubber Band's R2 engine through pedalboard. Its R3 engine
("high quality") drifts in time by up to ~0.7 ms per beat depending on the
ratio, which would break sample-accurate beat alignment. R2 has no drift but
a constant, content-independent offset that depends on the ratio; it is
measured once per ratio with a click train and compensated
(`stretch_offset_samples`). The short FFT window keeps short transients
within ~1 ms of the grid (the long window let them wander by up to 5 ms).
docs/decisions.md D7 has the measurements.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

RENDERER_VERSION = 1
MARGIN_BEATS = 1.0  # extra source audio read on both sides, dropped after stretching
EDGE_FADE_SECONDS = 0.003  # fade in/out on non-looping snippets
LOOP_XFADE_SECONDS = 0.010  # loop seam: the audio after the loop end crossfades into its start
# R2 engine with the short FFT window (Rubber Band's percussive setting): see module docstring
STRETCH_OPTIONS = {"high_quality": False, "use_long_fft_window": False}


@dataclass(frozen=True)
class RenderSpec:
    track_id: str
    path: str
    source_bpm: float
    first_beat: float  # seconds
    start_beat: float
    length_beats: float
    loop: bool
    set_bpm: float
    samplerate: int
    version: int = RENDERER_VERSION

    @property
    def length_samples(self) -> int:
        return int(round(self.length_beats * 60.0 / self.set_bpm * self.samplerate))

    @property
    def stretch_factor(self) -> float:
        return self.set_bpm / self.source_bpm

    def cache_key(self) -> str:
        """Everything that changes the rendered samples, and nothing else --
        the file path is left out because the track id already identifies
        the audio content."""
        fields = asdict(self)
        fields.pop("path")
        canonical = json.dumps(
            {k: (round(v, 9) if isinstance(v, float) else v) for k, v in sorted(fields.items())}, sort_keys=True
        )
        return hashlib.blake2b(canonical.encode(), digest_size=12).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RenderSpec":
        return cls(**data)


class RenderError(Exception):
    pass


# ---- source reading ------------------------------------------------------------------


def _read_segment(path: str, start_seconds: float, end_seconds: float) -> tuple[np.ndarray, int, float]:
    """Decode [start, end) seconds at the file's own rate as (frames, 2).
    Returns (audio, native_sr, time of the first returned frame). Regions
    before the file start or past its end are zero-padded."""
    from pedalboard.io import AudioFile

    with AudioFile(path) as f:
        sr = int(round(f.samplerate))
        first = math.floor(start_seconds * sr)
        last = math.ceil(end_seconds * sr)
        pad_front = max(0, -first)
        read_from = max(0, first)
        count = max(0, last - read_from)
        audio = np.zeros((0, f.num_channels), dtype=np.float32)
        if count and read_from < f.frames:
            f.seek(read_from)
            audio = f.read(count).T
    if audio.ndim != 2:
        audio = audio.reshape(-1, 1)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    total = last - first
    out = np.zeros((total, 2), dtype=np.float32)
    out[pad_front:pad_front + len(audio)] = audio[: total - pad_front]
    return out, sr, first / sr


def _resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio
    import soxr

    return soxr.resample(audio, source_sr, target_sr, quality="VHQ").astype(np.float32)


# ---- stretching ----------------------------------------------------------------------


def _stretch(audio: np.ndarray, sr: int, factor: float) -> np.ndarray:
    from pedalboard import time_stretch

    stretched = time_stretch(np.ascontiguousarray(audio.T), sr, stretch_factor=factor, **STRETCH_OPTIONS)
    return np.ascontiguousarray(stretched.T)


@functools.lru_cache(maxsize=128)
def stretch_offset_samples(factor: float, sr: int) -> float:
    """How many samples later than `t / factor` the stretcher places audio
    that was at input time `t`, measured on a click train. Content
    independent for R2, so one measurement per (factor, rate) is enough."""
    from .synth import click_grain

    spacing = int(0.25 * sr)
    count = 24
    mono = np.zeros(spacing * (count + 2), dtype=np.float32)
    grain = click_grain(sr, "click", seed=7)
    positions = [spacing * (i + 1) for i in range(count)]
    for pos in positions:
        mono[pos:pos + len(grain)] += grain
    stretched = _stretch(np.repeat(mono[:, None], 2, axis=1), sr, factor)
    env = np.abs(stretched[:, 0])
    threshold = 0.3 * float(env.max())
    window = int(0.06 * sr)
    offsets = []
    reference = np.abs(mono)
    ref_threshold = 0.3 * float(reference.max())
    for pos in positions[2:-2]:
        ref_idx = np.nonzero(reference[pos - 10:pos + window] > ref_threshold)[0]
        expected = (pos - 10 + ref_idx[0]) / factor
        lo = max(0, int(expected) - window)
        hit = np.nonzero(env[lo:int(expected) + window] > threshold)[0]
        if len(hit):
            offsets.append(lo + hit[0] - expected)
    if not offsets:
        return 0.0
    return float(np.median(offsets))


# ---- rendering ------------------------------------------------------------------------


def _hann_ramp(n: int) -> np.ndarray:
    """0 -> 1 raised-cosine ramp whose first value is already above zero."""
    k = np.arange(1, n + 1, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(np.pi * k / (n + 1))).astype(np.float32)


def render_snippet(spec: RenderSpec) -> np.ndarray:
    """(length_samples, 2) float32 audio of the snippet at `spec.set_bpm`."""
    if spec.source_bpm <= 0 or spec.set_bpm <= 0:
        raise RenderError("render needs a positive source and set BPM")
    if spec.length_beats <= 0:
        raise RenderError("snippet length must be positive")
    sr = spec.samplerate
    beat_seconds = 60.0 / spec.source_bpm
    start_t = spec.first_beat + spec.start_beat * beat_seconds
    end_t = start_t + spec.length_beats * beat_seconds
    margin = MARGIN_BEATS * beat_seconds

    segment, native_sr, segment_t0 = _read_segment(spec.path, start_t - margin, end_t + margin)
    segment = _resample(segment, native_sr, sr)
    body_offset = (start_t - segment_t0) * sr  # fractional sample where the snippet starts

    factor = spec.stretch_factor
    if abs(factor - 1.0) > 1e-9:
        segment = _stretch(segment, sr, factor)
        body_offset = body_offset / factor + stretch_offset_samples(round(factor, 9), sr)

    n = spec.length_samples
    xfade = int(round(LOOP_XFADE_SECONDS * sr)) if spec.loop else 0
    start = int(math.floor(body_offset + 0.5))
    needed = start + n + xfade
    if start < 0 or needed > len(segment):
        segment = np.pad(segment, ((max(0, -start), max(0, needed - len(segment))), (0, 0)))
        start = max(0, start)
    body = np.array(segment[start:start + n], dtype=np.float32)

    if spec.loop:
        # Blend the audio that follows the loop end into the loop start, so
        # the jump from the last sample back to the first is continuous.
        tail = segment[start + n:start + n + xfade]
        ramp = _hann_ramp(xfade)[:, None]
        body[:xfade] = body[:xfade] * ramp + tail * (1.0 - ramp)
    else:
        fade = min(int(round(EDGE_FADE_SECONDS * sr)), n // 2)
        if fade:
            ramp = _hann_ramp(fade)[:, None]
            body[:fade] *= ramp
            body[-fade:] *= ramp[::-1]
    return body


def render_to_cache(spec: RenderSpec, render_dir: Path) -> dict:
    """Render unless `<key>.npy` already exists. Returns metadata only."""
    render_dir = Path(render_dir)
    key = spec.cache_key()
    path = render_dir / f"{key}.npy"
    meta_path = render_dir / f"{key}.json"
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {"key": key, "path": str(path), "frames": meta["frames"], "cached": True}
    render_dir.mkdir(parents=True, exist_ok=True)
    audio = render_snippet(spec)
    tmp = render_dir / f"{key}.{os.getpid()}.tmp.npy"
    np.save(tmp, np.ascontiguousarray(audio, dtype=np.float32))
    os.replace(tmp, path)
    meta = {"frames": int(audio.shape[0]), "spec": spec.to_dict()}
    tmp_meta = render_dir / f"{key}.{os.getpid()}.tmp.json"
    tmp_meta.write_text(json.dumps(meta), encoding="utf-8")
    os.replace(tmp_meta, meta_path)
    return {"key": key, "path": str(path), "frames": int(audio.shape[0]), "cached": False}


def cached_render(spec: RenderSpec, render_dir: Path) -> Optional[dict]:
    key = spec.cache_key()
    path = Path(render_dir) / f"{key}.npy"
    meta_path = Path(render_dir) / f"{key}.json"
    if path.exists() and meta_path.exists():
        return {"key": key, "path": str(path), "frames": json.loads(meta_path.read_text(encoding="utf-8"))["frames"]}
    return None


def loudness_gain(track_lufs: Optional[float], target_lufs: float, max_boost_db: float = 12.0) -> float:
    """Linear gain bringing a track's integrated loudness to the target.
    Applied by the engine, not baked into buffers, so changing the target
    never invalidates the render cache."""
    if track_lufs is None:
        return 1.0
    db = min(max_boost_db, target_lufs - track_lufs)
    return float(10 ** (db / 20))
