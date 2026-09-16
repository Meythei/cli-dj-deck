"""Background jobs: a thread for filesystem scans, a spawn-based process pool
for CPU-heavy work (analysis, snippet rendering).

Worker processes redirect their stdout/stderr file descriptors to a log file
before doing anything else: decoders and numba print straight to fd 1/2, and
on Windows a spawned child shares the parent's console, so one stray warning
would scribble over the Textual screen.

Job functions return small plain dicts and write their bulky results
(envelopes, rendered buffers) to disk themselves, so nothing large is ever
pickled back to the UI process.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import sys
import traceback
import warnings
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Callable, Optional


def quiet_stdio(log_path: Optional[str]) -> None:
    """Point this process's fd 1 and 2 (and sys.stdout/err) at a log file,
    or at the null device when no path is given."""
    target = log_path or os.devnull
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    for std_fd in (1, 2):
        try:
            os.dup2(fd, std_fd)
        except OSError:
            pass
    stream = os.fdopen(os.dup(fd), "w", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = stream
    sys.stderr = stream
    warnings.simplefilter("ignore")


def init_worker(log_path: Optional[str]) -> None:
    quiet_stdio(log_path)


def _atomic_bytes(path: Path, write: Callable[[Path], None]) -> None:
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    write(tmp)
    os.replace(tmp, path)


def analyze_job(path: str, track_id: str, analysis_dir: str, bpm_range: tuple[float, float]) -> dict:
    """Analyse one file and store `<id>.json` + `<id>.env.npy` (or
    `<id>.failed.json`). Never raises: failures come back as data."""
    import numpy as np

    from .analysis import analyze_file

    out = Path(analysis_dir)
    out.mkdir(parents=True, exist_ok=True)
    failed = out / f"{track_id}.failed.json"
    try:
        result = analyze_file(Path(path), tuple(bpm_range))
        _atomic_bytes(out / f"{track_id}.env.npy", lambda p: np.save(p, result.envelope.astype(np.float32)))
        _atomic_bytes(out / f"{track_id}.json", lambda p: p.write_text(json.dumps(result.to_json()), encoding="utf-8"))
        failed.unlink(missing_ok=True)
        return {"track_id": track_id, "ok": True, "bpm": result.bpm, "key": result.key, "lufs": result.lufs}
    except Exception as exc:  # noqa: BLE001 -- a bad file must not take the pool down
        message = f"{type(exc).__name__}: {exc}"
        detail = traceback.format_exc()
        _atomic_bytes(failed, lambda p: p.write_text(json.dumps({"error": message, "detail": detail}), encoding="utf-8"))
        return {"track_id": track_id, "ok": False, "error": message}


class JobRunner:
    def __init__(self, workers: int = 2, log_path: Optional[Path] = None) -> None:
        self.workers = max(1, workers)
        self.log_path = str(log_path) if log_path else None
        self._threads = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clidj-scan")
        self._processes: Optional[ProcessPoolExecutor] = None

    def run_in_thread(self, fn: Callable[..., Any], *args: Any) -> Future:
        return self._threads.submit(fn, *args)

    def run_in_process(self, fn: Callable[..., Any], *args: Any) -> Future:
        if self._processes is None:
            self._processes = self._new_pool()
        try:
            return self._processes.submit(fn, *args)
        except BrokenProcessPool:
            # A worker died (e.g. a decoder crash). Its in-flight futures have
            # already failed; start a fresh pool for everything after.
            self._processes.shutdown(wait=False, cancel_futures=True)
            self._processes = self._new_pool()
            return self._processes.submit(fn, *args)

    def _new_pool(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=init_worker,
            initargs=(self.log_path,),
        )

    def shutdown(self) -> None:
        self._threads.shutdown(wait=False, cancel_futures=True)
        if self._processes is not None:
            self._processes.shutdown(wait=False, cancel_futures=True)
            self._processes = None


class InlineJobRunner:
    """Runs every job synchronously in the calling thread. For tests and the
    offline renderer, where determinism matters more than responsiveness."""

    def run_in_thread(self, fn: Callable[..., Any], *args: Any) -> Future:
        return self._run(fn, *args)

    def run_in_process(self, fn: Callable[..., Any], *args: Any) -> Future:
        return self._run(fn, *args)

    @staticmethod
    def _run(fn: Callable[..., Any], *args: Any) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args))
        except BaseException as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future

    def shutdown(self) -> None:
        pass
