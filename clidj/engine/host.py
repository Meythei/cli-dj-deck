"""The engine process and the UI-side client that talks to it.

Process layout (docs/TASK_real-audio.md 9):

    UI process                              engine process (spawn)
    RealtimeEngineClient.send() --Queue-->  command thread: load/prefault buffers,
                                            deque.append ------> audio callback:
    RealtimeEngineClient.status() <--shared memory (seqlock)---  engine.process(),
                                                                 StatusWriter.write()

The audio callback only pops the engine's deque, renders and writes the
status array. The command thread does everything that may block (queue
reads, mmap + page faults of new buffers). The engine process ignores
Ctrl+C (the UI owns shutdown) and exits by itself if the UI process dies.
"""
from __future__ import annotations

import atexit
import gc
import multiprocessing
import queue
import signal
import sys
import time
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import numpy as np

from . import commands as cmd
from .core import EngineStatus
from .status import H, decode_status, read_status, status_size

STARTUP_TIMEOUT = 30.0
HEARTBEAT_STALE_S = 2.0


@dataclass(frozen=True)
class HostConfig:
    backend: str = "sounddevice"  # "sounddevice" | "null"
    samplerate: int = 48000
    blocksize: int = 512
    lanes: int = 4
    bpm: float = 128.0
    device: object = None
    hostapi: str = "WASAPI"
    limiter_ceiling_db: float = -1.0
    master_gain: float = 1.0
    log_path: Optional[str] = None
    fallback_to_null: bool = True


def _make_backend(config: HostConfig, kind: str):
    from .backends import NullBackend, SoundDeviceBackend

    if kind == "null":
        return NullBackend(config.samplerate, config.blocksize)
    return SoundDeviceBackend(config.samplerate, config.blocksize, config.device, config.hostapi)


def engine_main(config: HostConfig, commands: multiprocessing.Queue, events: multiprocessing.Queue, shm_name: str) -> None:
    """Entry point of the engine process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from ..workers import quiet_stdio

    quiet_stdio(config.log_path)
    # The callback thread needs the GIL right away when a block is due; don't
    # let the command thread hold it for the default 5 ms slice.
    sys.setswitchinterval(0.001)
    _raise_process_priority()

    from .core import Engine, load_buffer
    from .status import StatusWriter

    shm = SharedMemory(name=shm_name, track=False)
    array = np.ndarray((status_size(config.lanes),), dtype=np.float64, buffer=shm.buf)
    engine = Engine(config.samplerate, config.bpm, config.lanes, render_audio=True,
                    limiter_ceiling_db=config.limiter_ceiling_db, master_gain=config.master_gain)
    writer = StatusWriter(array, config.lanes)
    ipc = {"last": 0.0, "max": 0.0}
    backend = None
    # Everything imported and built so far lives for the whole process: take it
    # out of the cyclic collector's view, and never let a collection start on
    # the audio thread. The command thread collects young objects when idle.
    gc.collect()
    gc.freeze()
    gc.disable()
    last_collect = time.monotonic()
    gc_log: list = []
    process_started = time.perf_counter()

    def on_block(out: np.ndarray) -> None:
        block = engine.process(len(out))
        out[:] = block
        writer.write(engine, backend.stats, ipc["last"], ipc["max"])

    kinds = [config.backend] + (["null"] if config.fallback_to_null and config.backend != "null" else [])
    failure = None
    for kind in kinds:
        try:
            backend = _make_backend(config, kind)
            writer.write(engine, backend.stats, 0.0, 0.0)
            backend.start(on_block)
            engine.output_latency_samples = int(round(backend.stats["output_latency_s"] * config.samplerate))
            break
        except Exception as exc:  # noqa: BLE001
            failure = f"{kind}: {exc}"
            backend = None
    if backend is None:
        events.put(("failed", failure or "no backend"))
        shm.close()
        return
    events.put(("started", {
        "backend": backend.name, "device": backend.stats.get("device", ""), "hostapi": backend.stats.get("hostapi", ""),
        "samplerate": config.samplerate, "blocksize": config.blocksize,
        "output_latency_s": backend.stats.get("output_latency_s", 0.0), "limiter_latency": engine.master.latency,
        "fallback_reason": failure,
    }))

    parent = multiprocessing.parent_process()
    try:
        while True:
            try:
                sent_ns, command = commands.get(timeout=0.05)
            except queue.Empty:
                if parent is not None and not parent.is_alive():
                    break
                if time.monotonic() - last_collect > 30.0:
                    t0 = time.perf_counter()
                    gc.collect(1)
                    gc_log.append((t0, "gc-collect", (time.perf_counter() - t0) * 1000.0))
                    last_collect = time.monotonic()
                if getattr(backend, "error", None):
                    events.put(("error", f"audio callback: {backend.error}"))
                    backend.error = None
                continue
            ipc["last"] = (time.perf_counter_ns() - sent_ns) / 1e6
            ipc["max"] = max(ipc["max"], ipc["last"])
            if isinstance(command, cmd.Shutdown):
                break
            if isinstance(command, cmd.RegisterBuffer):
                try:
                    loaded = load_buffer(command.info, prefault=True)
                except Exception as exc:  # noqa: BLE001
                    events.put(("error", f"could not load buffer for snippet {command.info.uid}: {exc}"))
                    continue
                engine.submit(loaded)
            else:
                engine.submit(command)
    finally:
        backend.stop()
        from .backends import SPIKES

        stats = backend.stats
        print(f"engine stopped: callbacks {stats['callbacks']} underruns {stats['underruns']} "
              f"slow {stats['slow_callbacks']} load max {stats['callback_load_max']:.2f} "
              f"wake late max {stats['wake_late_ms_max']:.1f} ms")
        for t, kind, ms in sorted(list(SPIKES) + gc_log):
            print(f"  t={t - process_started:9.3f}s {kind:<14} {ms:7.2f} ms")
        del array
        shm.close()


def _raise_process_priority() -> None:
    """Best effort: HIGH priority class and 1 ms timer resolution for the
    engine process (Windows). Affects only this process."""
    import os

    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00000080)  # HIGH_PRIORITY_CLASS
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:  # noqa: BLE001
        pass


class RealtimeEngineClient:
    """UI-side handle to the engine process. Never raises into the UI: if the
    process dies, `alive` turns False and sends become no-ops."""

    realtime = True
    render_audio = True

    def __init__(self, config: HostConfig) -> None:
        self.config = config
        self.samplerate = config.samplerate
        self.info: dict = {}
        self.error: Optional[str] = None
        self.dropped_commands = 0
        self._last_status = EngineStatus(samplerate=config.samplerate, bpm=config.bpm,
                                         lanes=[_empty_lane() for _ in range(config.lanes)])
        self._last_extra: dict = {}
        self._shm: Optional[SharedMemory] = None
        self._process = None
        self._closed = False

    # ---- lifecycle ------------------------------------------------------------------

    def start(self, timeout: float = STARTUP_TIMEOUT) -> dict:
        ctx = multiprocessing.get_context("spawn")
        size = status_size(self.config.lanes) * 8
        self._shm = SharedMemory(create=True, size=size)
        self._array = np.ndarray((status_size(self.config.lanes),), dtype=np.float64, buffer=self._shm.buf)
        self._array[:] = 0.0
        self._commands = ctx.Queue()
        self._events = ctx.Queue()
        self._process = ctx.Process(target=engine_main, args=(self.config, self._commands, self._events, self._shm.name),
                                    name="clidj-engine", daemon=True)
        self._process.start()
        atexit.register(self.close)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                kind, payload = self._events.get(timeout=0.1)
            except queue.Empty:
                if not self._process.is_alive():
                    self.error = f"engine process exited during startup (code {self._process.exitcode})"
                    break
                continue
            if kind == "started":
                self.info = payload
                return payload
            if kind == "failed":
                self.error = payload
                break
        else:
            self.error = "engine process did not start in time"
        self.close()
        raise RuntimeError(self.error)

    @property
    def alive(self) -> bool:
        if self._closed or self._process is None or not self._process.is_alive():
            return False
        heartbeat = self._array[H["heartbeat_ns"]]
        return heartbeat == 0 or (time.perf_counter_ns() - heartbeat) / 1e9 < HEARTBEAT_STALE_S

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process else None

    def close(self, timeout: float = 3.0) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is not None:
            try:
                if process.is_alive():
                    self._commands.put((time.perf_counter_ns(), cmd.Shutdown()))
                    process.join(timeout)
                if process.is_alive():
                    process.terminate()
                    process.join(1.0)
                if process.is_alive():
                    process.kill()
            except Exception:  # noqa: BLE001 -- shutting down; nothing sensible to report to
                pass
        for q in (getattr(self, "_commands", None), getattr(self, "_events", None)):
            if q is not None:
                q.cancel_join_thread()
                q.close()
        if self._shm is not None:
            try:
                del self._array
                self._shm.close()
                self._shm.unlink()
            except Exception:  # noqa: BLE001
                pass
            self._shm = None

    # ---- the client surface the Session uses -----------------------------------------

    def send(self, command) -> None:
        if self._closed or self._process is None or not self._process.is_alive():
            self.dropped_commands += 1
            return
        self._commands.put((time.perf_counter_ns(), command))

    def flush(self) -> None:
        pass

    def advance(self, frames: int):
        raise RuntimeError("a realtime engine advances by itself")

    def status(self) -> EngineStatus:
        if self._shm is not None and not self._closed:
            row = read_status(self._array, self.config.lanes)
            if row is not None and row[H["samplerate"]] > 0:
                self._last_status, self._last_extra = decode_status(row, self.config.lanes)
        return self._last_status

    @property
    def extra(self) -> dict:
        return self._last_extra

    def events(self) -> list[tuple[str, object]]:
        found = []
        if self._closed:
            return found
        while True:
            try:
                found.append(self._events.get_nowait())
            except (queue.Empty, OSError, ValueError):
                return found

    def beat_at_sample_offset(self, frames: int) -> float:
        status = self._last_status
        return status.position_beats + frames / self.samplerate * status.bpm / 60.0


def _empty_lane():
    from .core import LaneStatus

    return LaneStatus()
