"""Audio backends: what calls the engine's block callback, and how often.

- SoundDeviceBackend: a real output device through PortAudio (sounddevice).
  On Windows the WASAPI host API is preferred.
- NullBackend: no device; a thread calls the callback at the stream's real
  time pace and counts a missed deadline as an underrun. For machines
  without audio, CI, and load tests.
- Offline rendering doesn't need a backend object: clidj.offline pulls
  blocks from an in-process engine as fast as it can.

A backend callback receives a writable (frames, 2) float32 array to fill.
Counters live in `backend.stats` (a plain dict the engine host publishes).
"""
from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

Callback = Callable[[np.ndarray], None]


class BackendError(Exception):
    pass


def _new_stats(samplerate: int, blocksize: int) -> dict:
    return {
        "samplerate": samplerate, "blocksize": blocksize, "output_latency_s": 0.0, "underruns": 0,
        "callbacks": 0, "callback_ms": 0.0, "callback_load": 0.0, "callback_load_max": 0.0,
        "callback_load_avg": 0.0, "slow_callbacks": 0, "wake_late_ms_max": 0.0, "device": "", "hostapi": "",
    }


def raise_thread_priority() -> None:
    """Best effort: make the calling thread time-critical (Windows only)."""
    import os

    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), 15)  # THREAD_PRIORITY_TIME_CRITICAL
    except Exception:  # noqa: BLE001
        pass


# (perf_counter, kind, ms) of the most recent timing outliers, for the engine log
SPIKES: collections.deque = collections.deque(maxlen=256)


def _account(stats: dict, started: float, block_seconds: float) -> None:
    elapsed = time.perf_counter() - started
    load = elapsed / block_seconds
    stats["callbacks"] += 1
    stats["callback_ms"] = elapsed * 1000.0
    stats["callback_load"] = load
    if stats["callbacks"] > 20:  # ignore warm-up blocks (first-touch caches, lazy imports)
        stats["callback_load_max"] = max(stats["callback_load_max"], load)
        if load > 0.8:
            stats["slow_callbacks"] += 1
        if load > 0.5:
            SPIKES.append((started, "slow-callback", elapsed * 1000.0))
    stats["callback_load_avg"] = 0.99 * stats["callback_load_avg"] + 0.01 * load


class NullBackend:
    name = "null"

    def __init__(self, samplerate: int = 48000, blocksize: int = 512) -> None:
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.stats = _new_stats(samplerate, blocksize)
        self.stats["device"] = "null (no audio device)"
        self.stats["output_latency_s"] = blocksize / samplerate
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.error: Optional[str] = None

    def start(self, callback: Callback) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, args=(callback,), name="clidj-null-audio", daemon=True)
        self._thread.start()

    def _run(self, callback: Callback) -> None:
        raise_thread_priority()
        period = self.blocksize / self.samplerate
        out = np.zeros((self.blocksize, 2), dtype=np.float32)
        deadline = time.perf_counter() + period
        wake_at = time.perf_counter()
        while self._running:
            started = time.perf_counter()
            if self.stats["callbacks"] > 20:
                late_ms = (started - wake_at) * 1000.0
                self.stats["wake_late_ms_max"] = max(self.stats["wake_late_ms_max"], late_ms)
                if late_ms > 3.0:
                    SPIKES.append((started, "late-wake", late_ms))
            try:
                callback(out)
            except Exception as exc:  # noqa: BLE001 -- keep pacing; the host reports it
                self.error = f"{type(exc).__name__}: {exc}"
            _account(self.stats, started, period)
            now = time.perf_counter()
            if now > deadline + period:
                # The next block was due before this one was even done: a real
                # device would have played silence here.
                self.stats["underruns"] += 1
                SPIKES.append((now, "UNDERRUN", (now - deadline) * 1000.0))
                deadline = now + period
            else:
                deadline += period
            wake_at = deadline
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                wake_at = time.perf_counter()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)


@dataclass
class OutputDevice:
    index: int
    name: str
    hostapi: str
    channels: int
    default_samplerate: float
    is_default: bool


def list_output_devices() -> list[OutputDevice]:
    import sounddevice as sd

    hostapis = sd.query_hostapis()
    defaults = {api["default_output_device"] for api in hostapis}
    devices = []
    for index, info in enumerate(sd.query_devices()):
        if info["max_output_channels"] < 1:
            continue
        devices.append(OutputDevice(index, info["name"], hostapis[info["hostapi"]]["name"],
                                    info["max_output_channels"], info["default_samplerate"], index in defaults))
    return devices


def find_output_device(spec: Union[str, int, None], preferred_hostapi: str = "WASAPI") -> OutputDevice:
    """Index -> that device; text -> first output device whose name contains
    it (case-insensitive), preferring the host API; None -> the preferred host
    API's default output, else the system default."""
    import sounddevice as sd

    devices = list_output_devices()
    if not devices:
        raise BackendError("no audio output devices found")
    prefer = preferred_hostapi.lower()

    def ranked(candidates):
        return sorted(candidates, key=lambda d: (prefer not in d.hostapi.lower(), d.index))

    if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().isdigit()):
        index = int(spec)
        for device in devices:
            if device.index == index:
                return device
        raise BackendError(f"no output device #{index} (see --list-devices)")
    if isinstance(spec, str) and spec.strip():
        matches = [d for d in devices if spec.lower() in d.name.lower()]
        if not matches:
            raise BackendError(f"no output device matching {spec!r} (see --list-devices)")
        return ranked(matches)[0]
    for api in sd.query_hostapis():
        if prefer in api["name"].lower() and api["default_output_device"] >= 0:
            for device in devices:
                if device.index == api["default_output_device"]:
                    return device
    default = sd.default.device[1]
    for device in devices:
        if device.index == default:
            return device
    return ranked(devices)[0]


class SoundDeviceBackend:
    name = "sounddevice"

    def __init__(self, samplerate: int = 48000, blocksize: int = 512, device: Union[str, int, None] = None,
                 hostapi: str = "WASAPI") -> None:
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device_spec = device
        self.preferred_hostapi = hostapi
        self.stats = _new_stats(samplerate, blocksize)
        self.stream = None
        self.error: Optional[str] = None

    def start(self, callback: Callback) -> None:
        import sounddevice as sd

        device = find_output_device(self.device_spec, self.preferred_hostapi)
        period = self.blocksize / self.samplerate
        stats = self.stats

        def on_audio(outdata, frames, time_info, status) -> None:
            started = time.perf_counter()
            if status.output_underflow:
                stats["underruns"] += 1
            try:
                if frames == len(outdata):
                    callback(outdata)
                else:  # never expected with a fixed blocksize, but never raise here
                    outdata.fill(0.0)
            except Exception as exc:  # noqa: BLE001
                outdata.fill(0.0)
                self.error = f"{type(exc).__name__}: {exc}"
            _account(stats, started, period)

        extra = None
        if "wasapi" in device.hostapi.lower():
            # Shared mode resamples if the device mix format isn't 48 kHz.
            extra = sd.WasapiSettings(auto_convert=True)
        try:
            self.stream = sd.OutputStream(
                samplerate=self.samplerate, blocksize=self.blocksize, device=device.index, channels=2,
                dtype="float32", latency="low", callback=on_audio, extra_settings=extra,
            )
            self.stream.start()
        except Exception as exc:  # noqa: BLE001 -- PortAudio errors come in several types
            raise BackendError(f"could not open {device.name} ({device.hostapi}): {exc}") from None
        stats["device"] = device.name
        stats["hostapi"] = device.hostapi
        stats["output_latency_s"] = float(self.stream.latency)

    def stop(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:  # noqa: BLE001
                pass
            self.stream = None
