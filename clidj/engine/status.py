"""Engine status in a flat float64 array, shared between the engine process
(writer, from the audio callback) and the UI process (reader, every frame).

No locks: a seqlock. The writer bumps `seq` to odd, writes, bumps it back to
even; a reader copies the array and retries if `seq` was odd or changed
meanwhile. The writer never waits on the reader.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .core import EngineStats, EngineStatus, LaneStatus

HEADER = (
    "seq",
    "heartbeat_ns",  # perf_counter_ns() of the last write; staleness = engine stuck or gone
    "stream_sample",
    "transport_sample",
    "position_beats",
    "heard_beats",
    "bpm",
    "running",
    "samplerate",
    "blocksize",
    "latency_samples",
    "output_latency_s",
    "underruns",
    "callbacks",
    "callback_ms",
    "callback_load",
    "callback_load_max",
    "callback_load_avg",
    "late_commands",
    "late_max_ms",
    "late_last_ms",
    "missing_buffers",
    "automation_overrides",
    "errors",
    "commands_applied",
    "master_peak_l",
    "master_peak_r",
    "ipc_ms_last",
    "ipc_ms_max",
    "slow_callbacks",
    "wake_late_ms_max",
)
LANE_FIELDS = (
    "uid", "start_beat", "length_beats", "loop", "render_bpm", "gain", "muted", "lo", "mid", "hi", "peak",
    "automation_id", "automation_start", "automation_end", "releasing",
)
H = {name: i for i, name in enumerate(HEADER)}
L = {name: i for i, name in enumerate(LANE_FIELDS)}


def status_size(lanes: int) -> int:
    return len(HEADER) + lanes * len(LANE_FIELDS)


class StatusWriter:
    """Fills the array from an engine plus backend counters. Called from the
    audio thread after each block, so it only does array assignments."""

    def __init__(self, array: np.ndarray, lanes: int) -> None:
        self.array = array
        self.lanes = lanes
        self._row = np.zeros(status_size(lanes), dtype=np.float64)

    def write(self, engine, backend: dict, ipc_ms_last: float, ipc_ms_max: float) -> None:
        row = self._row
        stats = engine.stats
        position = engine.position_beats
        heard = max(0, engine.transport_sample - (engine.latency_samples if engine.running else 0))
        master = engine.master.last_peak if engine.master is not None else (0.0, 0.0)
        row[: len(HEADER)] = (
            0.0, float(time.perf_counter_ns()), engine.stream_sample, engine.transport_sample, position,
            engine.tempo.sample_to_beat(heard), engine.bpm, 1.0 if engine.running else 0.0, engine.samplerate,
            backend.get("blocksize", 0), engine.latency_samples, backend.get("output_latency_s", 0.0),
            backend.get("underruns", 0), backend.get("callbacks", 0), backend.get("callback_ms", 0.0),
            backend.get("callback_load", 0.0), backend.get("callback_load_max", 0.0),
            backend.get("callback_load_avg", 0.0), stats.late_commands, stats.late_max_ms, stats.late_last_ms,
            stats.missing_buffers, stats.automation_overrides, stats.errors, stats.commands_applied,
            master[0], master[1], ipc_ms_last, ipc_ms_max,
            backend.get("slow_callbacks", 0), backend.get("wake_late_ms_max", 0.0),
        )
        base = len(HEADER)
        width = len(LANE_FIELDS)
        for i, lane in enumerate(engine.lanes):
            voice = lane.voice
            automation = lane.automation
            gain = lane.current_gain
            if automation is not None and automation.param == "gain":
                gain = automation.value_at(position)
            row[base + i * width: base + (i + 1) * width] = (
                voice.info.uid if voice else 0.0,
                voice.start_beat if voice else 0.0,
                voice.info.length_beats if voice else 0.0,
                1.0 if voice and voice.info.loop else 0.0,
                voice.info.bpm if voice else 0.0,
                gain,
                1.0 if lane.params["mute"].target >= 0.5 else 0.0,
                lane.params["lo"].target, lane.params["mid"].target, lane.params["hi"].target,
                lane.peak,
                automation.automation_id if automation else 0.0,
                automation.start_beat if automation else 0.0,
                automation.end_beat if automation else 0.0,
                len(lane.releases),
            )
        array = self.array
        seq = array[0] + 1.0
        array[0] = seq  # odd: writing
        array[1:] = row[1:]
        array[0] = seq + 1.0  # even: consistent


def read_status(array: np.ndarray, lanes: int, retries: int = 50) -> Optional[np.ndarray]:
    """A consistent copy of the array, or None if the writer never let go
    (it can't hold it for long; None means something is badly wrong)."""
    for _ in range(retries):
        before = array[0]
        if int(before) % 2:
            time.sleep(0)
            continue
        copy = np.array(array, copy=True)
        if array[0] == before:
            return copy
    return None


def decode_status(row: np.ndarray, lanes: int) -> tuple[EngineStatus, dict]:
    """(EngineStatus, extra backend numbers) from a status copy."""
    stats = EngineStats(
        commands_applied=int(row[H["commands_applied"]]),
        late_commands=int(row[H["late_commands"]]),
        late_max_ms=float(row[H["late_max_ms"]]),
        late_last_ms=float(row[H["late_last_ms"]]),
        missing_buffers=int(row[H["missing_buffers"]]),
        automation_overrides=int(row[H["automation_overrides"]]),
        errors=int(row[H["errors"]]),
    )
    lane_list = []
    base = len(HEADER)
    width = len(LANE_FIELDS)
    for i in range(lanes):
        v = row[base + i * width: base + (i + 1) * width]
        lane_list.append(LaneStatus(
            uid=int(v[L["uid"]]), start_beat=float(v[L["start_beat"]]), length_beats=float(v[L["length_beats"]]),
            loop=bool(v[L["loop"]]), render_bpm=float(v[L["render_bpm"]]), gain=float(v[L["gain"]]),
            muted=bool(v[L["muted"]]), lo=float(v[L["lo"]]), mid=float(v[L["mid"]]), hi=float(v[L["hi"]]),
            peak=float(v[L["peak"]]), automation_id=int(v[L["automation_id"]]),
            automation_start=float(v[L["automation_start"]]), automation_end=float(v[L["automation_end"]]),
            releasing=int(v[L["releasing"]]),
        ))
    status = EngineStatus(
        stream_sample=int(row[H["stream_sample"]]),
        transport_sample=int(row[H["transport_sample"]]),
        position_beats=float(row[H["position_beats"]]),
        heard_beats=float(row[H["heard_beats"]]),
        bpm=float(row[H["bpm"]]) or 120.0,
        running=bool(row[H["running"]]),
        samplerate=int(row[H["samplerate"]]),
        latency_samples=int(row[H["latency_samples"]]),
        master_peak=(float(row[H["master_peak_l"]]), float(row[H["master_peak_r"]])),
        stats=stats,
        lanes=lane_list,
    )
    extra = {name: float(row[H[name]]) for name in (
        "heartbeat_ns", "blocksize", "output_latency_s", "underruns", "callbacks", "callback_ms", "callback_load",
        "callback_load_max", "callback_load_avg", "ipc_ms_last", "ipc_ms_max", "slow_callbacks", "wake_late_ms_max",
    )}
    return status, extra
