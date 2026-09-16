"""Snippet preparation: rendering jobs, their states, and demo audio.

A snippet is "prepared" at a BPM when its render for that BPM exists in the
cache. States are per (snippet, BPM) because a tempo change needs every
snippet rendered at the new BPM before it may happen (docs/TASK_real-audio.md
7). Render jobs run in the worker pool; `poll()` on the UI thread collects
them.

Demo tracks have no audio files. The first time one is needed, a worker
synthesises a deterministic track for it (clidj.synth) into the cache, and
the renders that were waiting on it are submitted afterwards.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .library import SAMPLES_PER_BEAT, Track
from .render import RenderSpec, cached_render, render_to_cache
from .snippets import Snippet

NONE = "none"
PENDING = "pending"
READY = "ready"
FAILED = "failed"

DEMO_AUDIO_VERSION = 1
DEMO_AUDIO_SR = 44100

LogFn = Callable[[str, str], None]


@dataclass(frozen=True)
class PreparedBuffer:
    key: str
    path: str
    frames: int
    bpm: float


# ---- worker jobs (module level so spawn can pickle them) ---------------------------------


def render_job(spec_data: dict, render_dir: str) -> dict:
    try:
        return {"ok": True, **render_to_cache(RenderSpec.from_dict(spec_data), Path(render_dir))}
    except Exception as exc:  # noqa: BLE001 -- failures come back as data
        return {"ok": False, "key": RenderSpec.from_dict(spec_data).cache_key(), "error": f"{type(exc).__name__}: {exc}"}


def camelot_to_key(code: str) -> tuple[int, bool]:
    """Inverse of analysis.camelot(): (tonic pitch class, minor)."""
    number, letter = int(code[:-1]), code[-1].upper()
    major_pc = ((number - 8) * 7) % 12
    if letter == "B":
        return major_pc, False
    return (major_pc - 3) % 12, True


def demo_audio_job(track_id: str, bpm: float, key: str, duration: float, out_dir: str) -> dict:
    from .analysis import ENVELOPE_HOP, integrated_loudness, peak_envelope, to_mono
    from .synth import techno_loop_track, write_audio

    out = Path(out_dir)
    stem = f"{track_id}-{bpm:g}-{key}-v{DEMO_AUDIO_VERSION}"
    audio_path, meta_path, env_path = out / f"{stem}.flac", out / f"{stem}.json", out / f"{stem}.env.npy"
    try:
        if not (audio_path.exists() and meta_path.exists() and env_path.exists()):
            out.mkdir(parents=True, exist_ok=True)
            root, minor = camelot_to_key(key)
            seed = int(track_id.rsplit("-", 1)[-1]) if track_id.rsplit("-", 1)[-1].isdigit() else 0
            audio = techno_loop_track(bpm, root, minor, duration, DEMO_AUDIO_SR, first_beat=0.0, seed=seed)
            tmp = out / f"{stem}.{os.getpid()}.tmp.flac"
            write_audio(tmp, audio, DEMO_AUDIO_SR)
            os.replace(tmp, audio_path)
            envelope = peak_envelope(to_mono(audio), ENVELOPE_HOP)
            np.save(env_path, envelope)
            meta_path.write_text(json.dumps({
                "lufs": integrated_loudness(audio, DEMO_AUDIO_SR),
                "envelope_rate": DEMO_AUDIO_SR / ENVELOPE_HOP,
            }), encoding="utf-8")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {"ok": True, "path": str(audio_path), "envelope": str(env_path), **meta}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# ---- manager --------------------------------------------------------------------------------


class PrepManager:
    def __init__(self, jobs, render_dir: Path, demo_audio_dir: Path, samplerate: int, log: LogFn) -> None:
        self.jobs = jobs
        self.render_dir = Path(render_dir)
        self.demo_audio_dir = Path(demo_audio_dir)
        self.samplerate = samplerate
        self.log = log
        self._render_jobs: dict[str, Future] = {}
        self._ready: dict[str, PreparedBuffer] = {}
        self._failed: dict[str, str] = {}
        self._source_jobs: dict[str, Future] = {}  # track_id -> demo audio synthesis
        self._source_failed: dict[str, str] = {}
        self._waiting_for_source: dict[str, list[tuple[Snippet, float]]] = {}
        self._render_bpm: dict[str, float] = {}  # key -> BPM of an in-flight render
        self.version = 0  # bumped whenever any state changes

    # ---- keys and states -------------------------------------------------------

    def spec(self, snippet: Snippet, bpm: float) -> Optional[RenderSpec]:
        track = snippet.track
        if track.path is None:
            return None
        return RenderSpec(
            track_id=track.track_id,
            path=str(track.path),
            source_bpm=track.bpm,
            first_beat=track.first_beat,
            start_beat=snippet.start_beat,
            length_beats=snippet.length_beats,
            loop=snippet.loop,
            set_bpm=float(bpm),
            samplerate=self.samplerate,
        )

    def state(self, snippet: Snippet, bpm: float) -> str:
        track = snippet.track
        spec = self.spec(snippet, bpm)
        if spec is None:
            if track.track_id in self._source_failed:
                return FAILED
            return PENDING if track.track_id in self._source_jobs or self._is_waiting(snippet, bpm) else NONE
        key = spec.cache_key()
        if key in self._ready:
            return READY
        if key in self._failed:
            return FAILED
        if key in self._render_jobs:
            return PENDING
        return NONE

    def buffer(self, snippet: Snippet, bpm: float) -> Optional[PreparedBuffer]:
        spec = self.spec(snippet, bpm)
        return None if spec is None else self._ready.get(spec.cache_key())

    def _is_waiting(self, snippet: Snippet, bpm: float) -> bool:
        return any(s is snippet and b == bpm for s, b in self._waiting_for_source.get(snippet.track.track_id, []))

    @property
    def outstanding(self) -> int:
        waiting = sum(len(v) for v in self._waiting_for_source.values())
        return len(self._render_jobs) + waiting

    # ---- requests -----------------------------------------------------------------

    def request(self, snippet: Snippet, bpm: float, retry: bool = False) -> str:
        """Make sure a render of `snippet` at `bpm` exists or is on its way.
        Returns the resulting state. A previous failure stays failed unless
        `retry` is set (so polling code can't resubmit a broken job forever)."""
        track = snippet.track
        spec = self.spec(snippet, bpm)
        if spec is None:
            if track.is_demo:
                if track.track_id in self._source_failed and not retry:
                    return FAILED
                self._source_failed.pop(track.track_id, None)
                if not self._is_waiting(snippet, bpm):
                    self._waiting_for_source.setdefault(track.track_id, []).append((snippet, float(bpm)))
                self._ensure_demo_source(track)
                self.version += 1
                return PENDING
            self._failed_note(f"{snippet.name}: #{track.id} {track.title} has no audio file")
            return FAILED
        key = spec.cache_key()
        if key in self._ready or key in self._render_jobs:
            return READY if key in self._ready else PENDING
        if key in self._failed and not retry:
            return FAILED
        self._failed.pop(key, None)
        hit = cached_render(spec, self.render_dir)
        if hit is not None:
            self._ready[key] = PreparedBuffer(key, hit["path"], hit["frames"], float(bpm))
            self.version += 1
            return READY
        try:
            self._render_jobs[key] = self.jobs.run_in_process(render_job, spec.to_dict(), str(self.render_dir))
        except Exception as exc:  # noqa: BLE001
            self._failed[key] = f"could not start render: {exc}"
            self.version += 1
            return FAILED
        self._render_bpm[key] = float(bpm)
        self.version += 1
        return PENDING

    def _failed_note(self, message: str) -> None:
        self.log(message, "error")

    def _ensure_demo_source(self, track: Track) -> None:
        if track.track_id in self._source_jobs:
            return
        try:
            self._source_jobs[track.track_id] = self.jobs.run_in_process(
                demo_audio_job, track.track_id, track.bpm, track.key, track.duration, str(self.demo_audio_dir)
            )
        except Exception as exc:  # noqa: BLE001
            future: Future = Future()
            future.set_exception(exc)
            self._source_jobs[track.track_id] = future

    # ---- harvesting ----------------------------------------------------------------------

    def poll(self, tracks_by_id: dict[str, Track]) -> list[tuple[str, bool]]:
        """Collect finished jobs. Returns [(key or track id, ok)] for logging."""
        finished: list[tuple[str, bool]] = []
        for track_id, future in list(self._source_jobs.items()):
            if not future.done():
                continue
            del self._source_jobs[track_id]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {"ok": False, "error": f"worker crashed: {exc}"}
            waiting = self._waiting_for_source.pop(track_id, [])
            track = tracks_by_id.get(track_id)
            if not result.get("ok") or track is None:
                self._source_failed[track_id] = result.get("error", "unknown track")
                self.log(f"demo audio for {track_id} failed: {self._source_failed[track_id]}", "error")
                finished.append((track_id, False))
            else:
                self._attach_demo_audio(track, result)
                finished.append((track_id, True))
                for snippet, bpm in waiting:
                    self.request(snippet, bpm)
            self.version += 1

        for key, future in list(self._render_jobs.items()):
            if not future.done():
                continue
            del self._render_jobs[key]
            bpm = self._render_bpm.pop(key, 0.0)
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {"ok": False, "error": f"worker crashed: {exc}"}
            if result.get("ok"):
                self._ready[key] = PreparedBuffer(key, result["path"], result["frames"], bpm)
            else:
                self._failed[key] = result.get("error", "render failed")
            finished.append((key, bool(result.get("ok"))))
            self.version += 1
        return finished

    def error_for(self, snippet: Snippet, bpm: float) -> str:
        spec = self.spec(snippet, bpm)
        if spec is None:
            return self._source_failed.get(snippet.track.track_id, "")
        return self._failed.get(spec.cache_key(), "")

    @staticmethod
    def _attach_demo_audio(track: Track, result: dict) -> None:
        from .analysis import beat_waveform

        track.path = Path(result["path"])
        track.lufs = result.get("lufs")
        envelope = np.load(result["envelope"])
        track.waveform = beat_waveform(envelope, float(result["envelope_rate"]), track.bpm, 0.0, SAMPLES_PER_BEAT)
