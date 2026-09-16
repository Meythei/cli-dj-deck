"""Per-track analysis: beat grid, key, loudness, waveform envelope.

Runs once per track in a background worker process (never in the UI or the
audio engine) and is cached on disk by the library. Every result here is a
*guess* -- the library layers user overrides (regrid/setkey) on top, and
nothing downstream reads the raw analysis without going through that.

Beat grid model: one constant tempo plus the time of the first beat. That is
deliberately all we support (docs/TASK_real-audio.md 14: no variable-tempo
grids).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

ANALYSIS_VERSION = 1
ANALYSIS_SR = 22050
ONSET_HOP = 128  # ~5.8 ms at ANALYSIS_SR
ENVELOPE_HOP = 128  # waveform envelope resolution, same ~5.8 ms
DEFAULT_BPM_RANGE = (88.0, 176.0)  # half-open; detected tempi are folded into it by octaves
AUDIBLE_RELATIVE_DB = -35.0  # "audible" = 10 ms RMS within 35 dB of the loudest 10 ms
KICK_BAND_HZ = 120.0  # low-pass for telling downbeats (kick) from offbeats (hats, rolling bass)
FIRST_BEAT_TOLERANCE = 0.06  # seconds an attack may lead the grid line and still own that beat

# Krumhansl-Kessler key profiles, index 0 = tonic.
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


class AnalysisError(Exception):
    pass


@dataclass
class TrackAnalysis:
    version: int
    duration: float  # seconds
    samplerate: int  # of the source file
    channels: int
    bpm: float
    first_beat: float  # seconds from file start to beat 0
    key: Optional[str]  # Camelot, e.g. "8A"
    lufs: Optional[float]  # integrated loudness (BS.1770-4)
    envelope_rate: float  # envelope values per second
    envelope: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=np.float32))

    def to_json(self) -> dict:
        data = asdict(self)
        data.pop("envelope")
        return data

    @classmethod
    def from_json(cls, data: dict, envelope: np.ndarray) -> "TrackAnalysis":
        known = {k: data[k] for k in cls.__dataclass_fields__ if k != "envelope" and k in data}
        return cls(**known, envelope=envelope)


# ---- audio loading -----------------------------------------------------------


def read_audio(path: Path, samplerate: Optional[float] = None) -> tuple[np.ndarray, int]:
    """Decode a whole file to (frames, channels) float32, optionally
    resampled. Reads in chunks until the decoder runs dry, since MP3 frame
    counts reported up front are only estimates."""
    from pedalboard.io import AudioFile

    with AudioFile(str(path)) as f:
        native_sr = int(round(f.samplerate))
        reader = f.resampled_to(samplerate) if samplerate and samplerate != f.samplerate else f
        chunks = []
        chunk = int(reader.samplerate * 30)
        while True:
            block = reader.read(chunk)
            if block.shape[1] == 0:
                break
            chunks.append(block)
            if block.shape[1] < chunk:
                break
    if not chunks:
        raise AnalysisError(f"{Path(path).name}: no audio frames")
    audio = np.concatenate(chunks, axis=1).T.astype(np.float32, copy=False)
    return np.ascontiguousarray(audio), int(round(samplerate)) if samplerate else native_sr


def to_mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1).astype(np.float32) if audio.ndim == 2 else audio.astype(np.float32)


# ---- loudness (ITU-R BS.1770-4) ------------------------------------------------


def _k_weighting_sos(sr: float) -> np.ndarray:
    """K-weighting pre-filter (high shelf) + RLB high-pass as two biquads,
    designed for any sample rate (libebur128's analogue-prototype formulas)."""
    f0 = 1681.974450955533
    gain_db = 3.999843853973347
    q = 0.7071752369554196
    k = math.tan(math.pi * f0 / sr)
    vh = 10 ** (gain_db / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = [
        (vh + vb * k / q + k * k) / a0,
        2 * (k * k - vh) / a0,
        (vh - vb * k / q + k * k) / a0,
        1.0,
        2 * (k * k - 1) / a0,
        (1 - k / q + k * k) / a0,
    ]
    f0 = 38.13547087602444
    q = 0.5003270373238773
    k = math.tan(math.pi * f0 / sr)
    a0 = 1 + k / q + k * k
    highpass = [1.0, -2.0, 1.0, 1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    return np.array([shelf, highpass])


def integrated_loudness(audio: np.ndarray, sr: int) -> Optional[float]:
    """Gated integrated loudness in LUFS of (frames, channels) audio, with
    unit channel weights (fine for mono/stereo). None if everything is
    gated out (digital silence)."""
    from scipy.signal import sosfilt

    if audio.ndim == 1:
        audio = audio[:, None]
    weighted = sosfilt(_k_weighting_sos(sr), audio.astype(np.float64), axis=0)
    block = int(round(0.4 * sr))
    step = int(round(0.1 * sr))
    if len(weighted) < block:
        return None
    squares = np.cumsum(np.concatenate([np.zeros((1, weighted.shape[1])), weighted**2]), axis=0)
    starts = np.arange(0, len(weighted) - block + 1, step)
    mean_sq = (squares[starts + block] - squares[starts]) / block  # (blocks, channels)
    power = mean_sq.sum(axis=1)
    with np.errstate(divide="ignore"):
        loudness = -0.691 + 10 * np.log10(power)
    gated = power[loudness > -70.0]
    if len(gated) == 0:
        return None
    relative = -0.691 + 10 * math.log10(gated.mean()) - 10.0
    final = power[(loudness > -70.0) & (loudness > relative)]
    if len(final) == 0:
        return None
    return float(-0.691 + 10 * math.log10(final.mean()))


# ---- beat grid ---------------------------------------------------------------


def _fold_bpm(bpm: float, bpm_range: tuple[float, float]) -> float:
    lo, hi = bpm_range
    if bpm <= 0:
        return bpm
    while bpm < lo:
        bpm *= 2
    while bpm >= hi:
        bpm /= 2
    return bpm


def _phase_histogram(weights: np.ndarray, period_frames: float, bins: int) -> np.ndarray:
    frames = np.arange(len(weights), dtype=np.float64)
    idx = np.minimum(((frames % period_frames) / period_frames * bins).astype(np.int64), bins - 1)
    return np.bincount(idx, weights=weights, minlength=bins)


def _best_bpm(weights: np.ndarray, centre: float, span: float, step: float, frame_rate: float) -> float:
    """The tempo whose beat period folds the onset envelope into the sharpest
    peak: with the right period every kick lands in the same phase bin."""
    best_score, best_bpm = -1.0, centre
    for bpm in np.arange(centre - span, centre + span + step / 2, step):
        hist = _phase_histogram(weights, 60.0 / bpm * frame_rate, 64)
        smooth = hist + 0.5 * (np.roll(hist, 1) + np.roll(hist, -1))
        score = float(smooth.max())
        if score > best_score:
            best_score, best_bpm = score, float(bpm)
    return best_bpm


def _onset_phase(weights: np.ndarray, bpm: float, frame_rate: float) -> float:
    """Beat phase (seconds, in [0, period)) from the onset envelope's folded
    peak. Onset envelopes lag the attack by several ms; _refine_phase fixes
    that against the raw signal."""
    period_frames = 60.0 / bpm * frame_rate
    bins = 256
    hist = _phase_histogram(weights, period_frames, bins)
    kernel = np.hanning(9)
    padded = np.concatenate([hist[-4:], hist, hist[:4]])
    smooth = np.convolve(padded, kernel, mode="valid")
    i = int(np.argmax(smooth))
    a, b, c = smooth[(i - 1) % bins], smooth[i], smooth[(i + 1) % bins]
    denom = a - 2 * b + c
    delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
    return ((i + delta + 0.5) / bins) * period_frames / frame_rate


def _refine_phase(mono: np.ndarray, sr: int, bpm: float, coarse_phase: float, search_ms: int = 40) -> float:
    """Move the phase to the steepest rise of the ~1 ms peak envelope folded
    over one beat period, within +-search_ms of the onset-based estimate."""
    hop = max(1, sr // 1000)
    n = len(mono) // hop
    if n == 0:
        return coarse_phase
    env = np.abs(mono[: n * hop]).reshape(n, hop).max(axis=1)
    t = (np.arange(n) * hop + hop / 2) / sr
    period = 60.0 / bpm
    bins = max(8, int(round(period * 1000)))
    idx = np.minimum(((t % period) / period * bins).astype(np.int64), bins - 1)
    folded = np.bincount(idx, weights=env, minlength=bins) / np.maximum(np.bincount(idx, minlength=bins), 1)
    rise = np.roll(folded, -1) - folded
    centre = int(round(coarse_phase / period * bins))
    candidates = [(centre + k) % bins for k in range(-search_ms, search_ms + 1)]
    best = max(candidates, key=lambda i: rise[i])
    return (((best + 1) % bins) / bins) * period


def estimate_grid(
    mono: np.ndarray, sr: int, bpm_range: tuple[float, float] = DEFAULT_BPM_RANGE
) -> tuple[float, float]:
    """(bpm, first_beat_seconds) for a constant-tempo track."""
    import librosa

    if len(mono) < sr * 4:
        raise AnalysisError("track too short to find a beat grid (< 4 s)")
    onset = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=ONSET_HOP, n_fft=1024)
    frame_rate = sr / ONSET_HOP
    weights = np.maximum(onset - np.median(onset), 0.0)
    if not np.any(weights > 0):
        raise AnalysisError("no onsets found")

    coarse = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=onset, sr=sr, hop_length=ONSET_HOP))[0])
    coarse = _fold_bpm(coarse, bpm_range)
    if coarse <= 0:
        raise AnalysisError("no tempo found")
    bpm = _best_bpm(weights, coarse, coarse * 0.04, 0.02, frame_rate)
    bpm = _best_bpm(weights, bpm, 0.03, 0.001, frame_rate)
    bpm = _fold_bpm(bpm, bpm_range)

    coarse_phase = _downbeat_side(mono, sr, bpm, _onset_phase(weights, bpm, frame_rate))
    phase = _refine_phase(mono, sr, bpm, coarse_phase)
    first_beat = _first_audible_beat(mono, sr, bpm, phase)
    return round(bpm, 3), first_beat


def _first_audible_beat(mono: np.ndarray, sr: int, bpm: float, phase: float) -> float:
    """Beat 0 is the first grid beat once the track is audible, so leading
    silence doesn't become bars of nothing -- while a quiet intro (kick only,
    ambience) still counts, however much louder the drop is."""
    period = 60.0 / bpm
    window = max(1, int(0.010 * sr))
    n = len(mono) // window
    if n == 0:
        return phase
    rms = np.sqrt(np.mean(mono[: n * window].astype(np.float64).reshape(n, window) ** 2, axis=1))
    loudest = float(rms.max())
    if loudest <= 0:
        return phase
    audible = np.nonzero(rms > loudest * 10 ** (AUDIBLE_RELATIVE_DB / 20))[0]
    onset_time = audible[0] * window / sr if len(audible) else 0.0
    # Tolerate an attack starting a hair before the estimated grid line.
    k = max(0, math.ceil((onset_time - FIRST_BEAT_TOLERANCE - phase) / period))
    return phase + k * period


def _downbeat_side(mono: np.ndarray, sr: int, bpm: float, phase: float) -> float:
    """Onset envelopes can't tell a beat from its offbeat when hats and an
    offbeat bassline are busier than the kick. Of `phase` and `phase + half
    a beat`, keep the one followed by more low-frequency energy: in dance
    music that's where the kick is."""
    from scipy.signal import butter, sosfilt

    period = 60.0 / bpm
    low = np.abs(sosfilt(butter(4, KICK_BAND_HZ, btype="low", fs=sr, output="sos"), mono))
    hop = max(1, sr // 1000)
    n = len(low) // hop
    if n == 0:
        return phase
    env = low[: n * hop].reshape(n, hop).max(axis=1)
    t = np.arange(n) * hop / sr
    bins = max(8, int(round(period * 1000)))
    idx = np.minimum(((t % period) / period * bins).astype(np.int64), bins - 1)
    folded = np.bincount(idx, weights=env, minlength=bins) / np.maximum(np.bincount(idx, minlength=bins), 1)
    span = max(1, int(round(0.050 / period * bins)))  # ~50 ms of bins

    def energy_after(p: float) -> float:
        start = int(round(p / period * bins))
        return float(np.sum(folded[[(start + i) % bins for i in range(span)]]))

    offbeat = (phase + period / 2) % period
    return offbeat if energy_after(offbeat) > energy_after(phase) else phase


# ---- key ---------------------------------------------------------------------


def camelot(pitch_class: int, minor: bool) -> str:
    """Camelot code for a key: C major = 8B, A minor = 8A, one fifth up = +1."""
    major_pc = (pitch_class + 3) % 12 if minor else pitch_class
    number = ((major_pc * 7) % 12 + 7) % 12 + 1
    return f"{number}{'A' if minor else 'B'}"


def estimate_key(mono: np.ndarray, sr: int) -> Optional[str]:
    import librosa

    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr, hop_length=4096)
    profile = chroma.mean(axis=1)
    if not np.any(profile > 0):
        return None
    best_score, best = -2.0, None
    for tonic in range(12):
        rotated = np.roll(profile, -tonic)
        for minor, template in ((False, _MAJOR_PROFILE), (True, _MINOR_PROFILE)):
            score = float(np.corrcoef(rotated, template)[0, 1])
            if score > best_score:
                best_score, best = score, camelot(tonic, minor)
    return best


# ---- waveform ------------------------------------------------------------------


def peak_envelope(mono: np.ndarray, hop: int = ENVELOPE_HOP) -> np.ndarray:
    n = len(mono) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return np.abs(mono[: n * hop]).reshape(n, hop).max(axis=1).astype(np.float32)


def beat_waveform(
    envelope: np.ndarray, envelope_rate: float, bpm: float, first_beat: float, samples_per_beat: int
) -> list[float]:
    """Resample a time-based peak envelope onto a beat grid (0..1 per slot,
    from beat 0). Kept separate from analysis so a regrid() only redoes this
    cheap step instead of decoding the file again."""
    if len(envelope) == 0 or bpm <= 0:
        return []
    duration = len(envelope) / envelope_rate
    slot_seconds = 60.0 / bpm / samples_per_beat
    slots = int((duration - first_beat) / slot_seconds)
    if slots <= 0:
        return []
    starts = first_beat + slot_seconds * np.arange(slots + 1)
    idx = np.clip((starts * envelope_rate).astype(np.int64), 0, len(envelope) - 1)
    # reduceat takes the single element at idx[i] whenever idx[i+1] <= idx[i],
    # which is exactly the "slot shorter than one envelope step" case.
    peaks = np.maximum.reduceat(envelope, idx)[:slots]
    scale = float(np.percentile(envelope, 99.5)) or 1.0
    return np.clip(peaks / scale, 0.0, 1.0).astype(np.float32).tolist()


# ---- entry point ---------------------------------------------------------------


def analyze_file(path: Path, bpm_range: tuple[float, float] = DEFAULT_BPM_RANGE) -> TrackAnalysis:
    import librosa

    audio, native_sr = read_audio(path)
    duration = len(audio) / native_sr
    lufs = integrated_loudness(audio, native_sr)
    mono = librosa.resample(to_mono(audio), orig_sr=native_sr, target_sr=ANALYSIS_SR, res_type="soxr_hq")
    channels = audio.shape[1]
    del audio

    bpm, first_beat = estimate_grid(mono, ANALYSIS_SR, bpm_range)
    key = estimate_key(mono, ANALYSIS_SR)
    envelope = peak_envelope(mono, ENVELOPE_HOP)
    return TrackAnalysis(
        version=ANALYSIS_VERSION,
        duration=duration,
        samplerate=native_sr,
        channels=channels,
        bpm=bpm,
        first_beat=first_beat,
        key=key,
        lufs=lufs,
        envelope_rate=ANALYSIS_SR / ENVELOPE_HOP,
        envelope=envelope,
    )
