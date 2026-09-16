"""The audio engine: sample-accurate lanes on a tempo map.

Time is the transport sample counter; musical position is derived from it
through the TempoMap. A block is processed in segments split exactly at the
samples where something happens (a command's beat, an automation ending, a
one-shot snippet ending), so every event lands on its sample, not on a block
boundary (docs/TASK_real-audio.md 3.1-3.3).

Playback position inside a snippet is always recomputed from beats
(`(beat - start_beat) mod length`), never accumulated from sample counts, so
rounding can't build up across loop iterations.

`Engine.process()` only touches preallocated state, numpy on block-sized
arrays and the (already loaded, pre-faulted) buffers: no file I/O, no locks,
no logging, no exceptions escaping. Commands arrive through `submit()`, a
deque append that the processing side pops without locking.

`render_audio=False` runs the same state machine without producing sound:
that is the `--no-audio` visual mode, so both modes share one implementation
of timing.
"""
from __future__ import annotations

import bisect
import collections
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..tempo import TempoMap, event_sample
from . import commands as cmd
from .dsp import Isolator3, MasterBus, Smoother

DECLICK_IN = 48  # samples (1 ms @ 48 kHz) of fade-in on every voice start
RELEASE = 240  # samples (5 ms) of fade-out for stopped/replaced voices; also the tempo-swap crossfade
PARAM_RAMP = 240  # samples (5 ms) of smoothing for immediate parameter changes
MAX_BLOCK = 8192
EPSILON_BEATS = 1e-9


@dataclass
class LoadedBuffer:
    info: cmd.BufferInfo
    data: Optional[np.ndarray]  # (frames, 2) float32, usually a read-only memmap; None = silent

    @property
    def samples_per_beat(self) -> float:
        return self.info.frames / self.info.length_beats if self.info.length_beats else 0.0


def load_buffer(info: cmd.BufferInfo, prefault: bool = True) -> LoadedBuffer:
    """Map a rendered buffer and touch every page, so the audio callback
    never takes a page fault (= disk I/O) reading it. Runs outside the
    callback (command thread / in-process client)."""
    if info.path is None:
        return LoadedBuffer(info, None)
    data = np.load(info.path, mmap_mode="r")
    if data.ndim != 2 or data.shape[1] != 2 or data.dtype != np.float32:
        raise ValueError(f"{info.path}: expected float32 (frames, 2), got {data.dtype} {data.shape}")
    if prefault:
        step = 1 << 16
        for start in range(0, len(data), step):
            float(data[start:start + step, 0].sum())
    return LoadedBuffer(info, data)


class Voice:
    """A snippet playing on a lane, positioned by beats."""

    __slots__ = ("buffer", "start_beat", "ramp_in", "rendered", "last_index", "spb")

    def __init__(self, buffer: LoadedBuffer, start_beat: float, samplerate: int, ramp_in: int = DECLICK_IN) -> None:
        self.buffer = buffer
        self.start_beat = start_beat
        self.ramp_in = ramp_in
        self.rendered = 0
        self.last_index: Optional[int] = None
        self.spb = 60.0 * samplerate / buffer.info.bpm  # buffer samples per beat

    @property
    def info(self) -> cmd.BufferInfo:
        return self.buffer.info

    def end_beat(self) -> float:
        return math.inf if self.info.loop else self.start_beat + self.info.length_beats

    def render(self, out: np.ndarray, beats: np.ndarray, arange: np.ndarray) -> None:
        n = len(beats)
        info = self.info
        local = np.maximum(beats - self.start_beat, 0.0)  # the first sample can sit <= half a sample early
        if info.loop:
            positions = np.floor((local % info.length_beats) * self.spb + 0.5).astype(np.int64) % info.frames
            valid = None
        else:
            positions = np.floor(local * self.spb + 0.5).astype(np.int64)
            valid = positions < info.frames
            positions = np.minimum(positions, info.frames - 1)
        self.last_index = int(positions[-1])
        data = self.buffer.data
        if data is not None:
            samples = data[positions]
            weight = np.full(n, info.gain, dtype=np.float64)
            if self.rendered < self.ramp_in:
                weight *= np.minimum(1.0, (self.rendered + arange[1:n + 1]) / (self.ramp_in + 1))
            if valid is not None:
                weight *= valid
            out += samples * weight[:, None]
        self.rendered += n


class Release:
    """A voice fading out after a stop/replace, running freely from where it
    was (the transport may already be stopped)."""

    __slots__ = ("buffer", "next_index", "remaining", "total")

    def __init__(self, voice: Voice, length: int = RELEASE) -> None:
        self.buffer = voice.buffer
        self.next_index = (voice.last_index or 0) + 1
        self.total = length
        self.remaining = length if voice.last_index is not None else 0

    def render(self, out: Optional[np.ndarray], arange: np.ndarray) -> None:
        n = len(out) if out is not None else 0
        k = min(n, self.remaining)
        if k and self.buffer.data is not None:
            info = self.buffer.info
            idx = self.next_index + arange[:k].astype(np.int64)
            if info.loop:
                idx %= info.frames
                valid = None
            else:
                valid = idx < info.frames
                idx = np.minimum(idx, info.frames - 1)
            done = self.total - self.remaining
            fade = (self.total - done - arange[:k]) / (self.total + 1) * info.gain
            if valid is not None:
                fade = fade * valid
            out[:k] += self.buffer.data[idx] * fade[:, None]
        self.next_index += k
        self.remaining -= k if out is not None else self.remaining


@dataclass
class Automation:
    automation_id: int
    param: str
    start_beat: float
    end_beat: float
    start_value: float
    end_value: float
    stop_lane_at_end: bool

    def curve(self, beats: np.ndarray):
        span = self.end_beat - self.start_beat
        if span <= 0:
            return self.end_value
        ratio = np.clip((beats - self.start_beat) / span, 0.0, 1.0)
        return self.start_value + (self.end_value - self.start_value) * ratio

    def value_at(self, beat: float) -> float:
        span = self.end_beat - self.start_beat
        ratio = 1.0 if span <= 0 else min(1.0, max(0.0, (beat - self.start_beat) / span))
        return self.start_value + (self.end_value - self.start_value) * ratio


class LaneState:
    def __init__(self, index: int, samplerate: int, render_audio: bool) -> None:
        self.index = index
        self.voice: Optional[Voice] = None
        self.releases: list[Release] = []
        self.params = {
            "gain": Smoother(1.0, PARAM_RAMP),
            "mute": Smoother(0.0, PARAM_RAMP),  # 1 = muted
            "lo": Smoother(1.0, PARAM_RAMP),
            "mid": Smoother(1.0, PARAM_RAMP),
            "hi": Smoother(1.0, PARAM_RAMP),
        }
        self.automation: Optional[Automation] = None
        self.eq = Isolator3(samplerate) if render_audio else None
        self.scratch = np.zeros((MAX_BLOCK, 2), dtype=np.float32)
        self.peak = 0.0
        self.current_gain = 1.0

    def release_voice(self, length: int = RELEASE) -> None:
        if self.voice is not None:
            release = Release(self.voice, length)
            if release.remaining:
                self.releases.append(release)
        self.voice = None


@dataclass
class EngineStats:
    commands_applied: int = 0
    late_commands: int = 0
    late_max_ms: float = 0.0
    late_last_ms: float = 0.0
    missing_buffers: int = 0
    automation_overrides: int = 0
    errors: int = 0
    last_error: str = ""


@dataclass
class LaneStatus:
    uid: int = 0
    start_beat: float = 0.0
    length_beats: float = 0.0
    loop: bool = False
    render_bpm: float = 0.0
    gain: float = 1.0
    muted: bool = False
    lo: float = 1.0
    mid: float = 1.0
    hi: float = 1.0
    peak: float = 0.0
    automation_id: int = 0
    automation_start: float = 0.0
    automation_end: float = 0.0
    releasing: int = 0


@dataclass
class EngineStatus:
    stream_sample: int = 0
    transport_sample: int = 0
    position_beats: float = 0.0
    heard_beats: float = 0.0
    bpm: float = 120.0
    running: bool = False
    samplerate: int = 48000
    latency_samples: int = 0
    master_peak: tuple[float, float] = (0.0, 0.0)
    stats: EngineStats = field(default_factory=EngineStats)
    lanes: list[LaneStatus] = field(default_factory=list)


class Engine:
    def __init__(
        self,
        samplerate: int = 48000,
        bpm: float = 120.0,
        lanes: int = 4,
        render_audio: bool = True,
        limiter_ceiling_db: float = -1.0,
        master_gain: float = 1.0,
    ) -> None:
        self.samplerate = samplerate
        self.render_audio = render_audio
        self.tempo = TempoMap(bpm, samplerate)
        self.running = False
        self.transport_sample = 0
        self.stream_sample = 0
        self.inbox: collections.deque = collections.deque()
        self._pending: list[tuple[float, int, object]] = []  # (beat, seq, command), sorted
        self._seq = 0
        self.buffers: dict[int, dict[float, LoadedBuffer]] = {}
        self.lanes = [LaneState(i, samplerate, render_audio) for i in range(lanes)]
        self.master = MasterBus(samplerate, limiter_ceiling_db, master_gain, PARAM_RAMP) if render_audio else None
        self.output_latency_samples = 0  # set by a realtime backend (device latency)
        self.stats = EngineStats()
        self.shutdown_requested = False
        self._arange = np.arange(MAX_BLOCK + 1, dtype=np.float64)
        self._out = np.zeros((MAX_BLOCK, 2), dtype=np.float32)

    # ---- command intake (any thread) ---------------------------------------------------

    def submit(self, command) -> None:
        """Queue a command (or a LoadedBuffer registration). Thread-safe:
        deque.append is atomic, and the processing side only pops."""
        self.inbox.append(command)

    # ---- queries ----------------------------------------------------------------------------

    @property
    def position_beats(self) -> float:
        return self.tempo.sample_to_beat(self.transport_sample)

    @property
    def bpm(self) -> float:
        return self.tempo.bpm_at_sample(self.transport_sample)

    @property
    def latency_samples(self) -> int:
        return (self.master.latency if self.master else 0) + self.output_latency_samples

    def status(self) -> EngineStatus:
        lanes = []
        position = self.position_beats
        for lane in self.lanes:
            status = LaneStatus(
                gain=lane.current_gain,
                muted=lane.params["mute"].target >= 0.5,
                lo=lane.params["lo"].target,
                mid=lane.params["mid"].target,
                hi=lane.params["hi"].target,
                peak=lane.peak,
                releasing=len(lane.releases),
            )
            if lane.voice is not None:
                info = lane.voice.info
                status.uid, status.start_beat = info.uid, lane.voice.start_beat
                status.length_beats, status.loop, status.render_bpm = info.length_beats, info.loop, info.bpm
            if lane.automation is not None:
                a = lane.automation
                status.automation_id, status.automation_start, status.automation_end = a.automation_id, a.start_beat, a.end_beat
                if a.param == "gain":
                    status.gain = a.value_at(position)
            lanes.append(status)
        heard_sample = max(0, self.transport_sample - (self.latency_samples if self.running else 0))
        return EngineStatus(
            stream_sample=self.stream_sample,
            transport_sample=self.transport_sample,
            position_beats=position,
            heard_beats=self.tempo.sample_to_beat(heard_sample),
            bpm=self.bpm,
            running=self.running,
            samplerate=self.samplerate,
            latency_samples=self.latency_samples,
            master_peak=self.master.last_peak if self.master else (0.0, 0.0),
            stats=dataclasses.replace(self.stats),
            lanes=lanes,
        )

    # ---- processing ---------------------------------------------------------------------------

    def process(self, frames: int) -> Optional[np.ndarray]:
        """Advance `frames` samples. Returns the (frames, 2) master output, or
        None in visual mode. Never raises: an internal error is counted,
        reported in stats, and the block is silent."""
        try:
            return self._process(frames)
        except Exception as exc:  # noqa: BLE001 -- the audio thread must survive
            self.stats.errors += 1
            self.stats.last_error = f"{type(exc).__name__}: {exc}"
            return np.zeros((frames, 2), dtype=np.float32) if self.render_audio else None

    def flush(self) -> None:
        """Apply queued immediate commands (and timed ones already due)
        without advancing time. For in-process clients between blocks."""
        try:
            self._drain_inbox()
            self._apply_due()
        except Exception as exc:  # noqa: BLE001
            self.stats.errors += 1
            self.stats.last_error = f"{type(exc).__name__}: {exc}"

    def _process(self, frames: int) -> Optional[np.ndarray]:
        self._drain_inbox()
        out = None
        if self.render_audio:
            out = self._out[:frames] if frames <= MAX_BLOCK else np.zeros((frames, 2), dtype=np.float32)
            out.fill(0.0)
        done = 0
        while done < frames:
            self._apply_due()
            span = frames - done
            if self.running:
                boundary = self._next_boundary()
                span = int(min(span, boundary - self.transport_sample))
                if span <= 0:  # defensive: never stall
                    span = 1
            self._render(None if out is None else out[done:done + span], span)
            if self.running:
                self.transport_sample += span
            self.stream_sample += span
            done += span
        self._apply_due()
        if self.master is not None:
            return self.master.process(out, self._arange)
        return None

    def _drain_inbox(self) -> None:
        inbox = self.inbox
        while inbox:
            item = inbox.popleft()
            if isinstance(item, LoadedBuffer):
                self.buffers.setdefault(item.info.uid, {})[_bpm_key(item.info.bpm)] = item
                continue
            beat = cmd.command_beat(item)
            if beat is None:
                self._apply(item, late_ms=0.0)
            else:
                self._seq += 1
                bisect.insort(self._pending, (float(beat), self._seq, item))

    def _apply_due(self) -> None:
        pending = self._pending
        while pending:
            beat, _, command = pending[0]
            if self.running:
                due = event_sample(self.tempo.beat_to_sample(beat))
                if due > self.transport_sample:
                    return
                late_ms = (self.transport_sample - due) * 1000.0 / self.samplerate
            else:
                if beat > self.position_beats + EPSILON_BEATS:
                    return
                late_ms = 0.0
            pending.pop(0)
            self._apply(command, late_ms)

    def _next_boundary(self) -> float:
        now = self.transport_sample
        candidates = []
        if self._pending:
            candidates.append(event_sample(self.tempo.beat_to_sample(self._pending[0][0])))
        for lane in self.lanes:
            if lane.voice is not None and not lane.voice.info.loop:
                candidates.append(event_sample(self.tempo.beat_to_sample(lane.voice.end_beat())))
            if lane.automation is not None:
                candidates.append(event_sample(self.tempo.beat_to_sample(lane.automation.end_beat)))
        future = [c for c in candidates if c > now]
        return min(future) if future else math.inf

    def _render(self, out: Optional[np.ndarray], n: int) -> None:
        arange = self._arange if n <= MAX_BLOCK else np.arange(n + 1, dtype=np.float64)
        beats = None
        if self.running:
            seg = self.tempo.segment_for_sample(self.transport_sample)
            spb = seg.samples_per_beat(self.samplerate)
            beats = seg.beat + (self.transport_sample + arange[:n] - seg.sample) / spb
        end_sample = self.transport_sample + n
        end_beat = self.tempo.sample_to_beat(end_sample) if self.running else self.position_beats

        for lane in self.lanes:
            if out is None:
                for smoother in lane.params.values():
                    smoother.advance(n)
                lane.releases.clear()
                lane.current_gain = lane.automation.value_at(end_beat) if lane.automation else lane.params["gain"].value
            else:
                self._render_lane(lane, out, n, beats, arange)
            if self.running:
                self._finish_lane_events(lane, end_sample)

    def _render_lane(self, lane: LaneState, out: np.ndarray, n: int, beats, arange: np.ndarray) -> None:
        buf = lane.scratch[:n] if n <= MAX_BLOCK else np.zeros((n, 2), dtype=np.float32)
        buf.fill(0.0)
        if lane.voice is not None and beats is not None:
            lane.voice.render(buf, beats, arange)
        if lane.releases:
            for release in lane.releases:
                release.render(buf, arange)
            lane.releases = [r for r in lane.releases if r.remaining > 0]

        values = {}
        automation = lane.automation
        for name, smoother in lane.params.items():
            if automation is not None and automation.param == name and beats is not None:
                values[name] = automation.curve(beats)
                smoother.advance(n)
            else:
                values[name] = smoother.render(n, arange)
        gain = values["gain"]
        shaped = lane.eq.process(buf, values["lo"], values["mid"], values["hi"])
        level = gain * (1.0 - values["mute"])
        if np.isscalar(level):
            shaped *= level
        else:
            shaped *= level[:, None]
        out += shaped
        lane.peak = float(np.max(np.abs(shaped))) if n else 0.0
        lane.current_gain = float(gain if np.isscalar(gain) else gain[-1])

    def _finish_lane_events(self, lane: LaneState, reached: int) -> None:
        """Retire whatever ends at or before the sample just reached. Ends are
        compared as event samples, the same rounding that placed the boundary."""
        tempo = self.tempo
        if lane.automation is not None and reached >= event_sample(tempo.beat_to_sample(lane.automation.end_beat)):
            automation = lane.automation
            lane.automation = None
            lane.params[automation.param].jump(automation.end_value)
            lane.current_gain = lane.params["gain"].value
            if automation.stop_lane_at_end:
                lane.release_voice()
        voice = lane.voice
        if voice is not None and not voice.info.loop and reached >= event_sample(tempo.beat_to_sample(voice.end_beat())):
            lane.voice = None  # one-shot finished; its buffer already faded out

    # ---- command handling ------------------------------------------------------------------------

    def _apply(self, command, late_ms: float) -> None:
        self.stats.commands_applied += 1
        if late_ms > 0.0:
            self.stats.late_commands += 1
            self.stats.late_last_ms = late_ms
            self.stats.late_max_ms = max(self.stats.late_max_ms, late_ms)
        position = self.position_beats
        if isinstance(command, cmd.Play):
            lane = self.lanes[command.lane]
            start = position if command.beat is None else command.beat
            buffer = self._buffer_for(command.uid, self.tempo.bpm_at_beat(start))
            if buffer is None:
                self.stats.missing_buffers += 1
                return
            lane.release_voice()
            lane.voice = Voice(buffer, start, self.samplerate)
        elif isinstance(command, cmd.Stop):
            self.lanes[command.lane].release_voice()
        elif isinstance(command, cmd.SetParam):
            self._set_param(command)
        elif isinstance(command, cmd.Automate):
            lane = self.lanes[command.lane]
            if lane.automation is not None:
                self.stats.automation_overrides += 1
            smoother = lane.params[command.param]
            start_value = smoother.value if command.start_value is None else command.start_value
            if lane.automation is not None and lane.automation.param == command.param and command.start_value is None:
                start_value = lane.automation.value_at(position)
            smoother.jump(start_value)
            lane.automation = Automation(command.automation_id, command.param, command.start_beat, command.end_beat,
                                         start_value, command.end_value, command.stop_lane_at_end)
            lane.current_gain = start_value if command.param == "gain" else lane.current_gain
        elif isinstance(command, cmd.CancelAutomation):
            for lane in self.lanes:
                if lane.automation and command.automation_id in (None, lane.automation.automation_id):
                    value = lane.automation.value_at(position)
                    lane.params[lane.automation.param].jump(value)
                    lane.automation = None
        elif isinstance(command, cmd.SetTempo):
            self._set_tempo(command, late_ms)
        elif isinstance(command, cmd.TransportStart):
            if not self.running:
                self.running = True
                for lane in self.lanes:
                    if lane.voice is not None:
                        lane.voice.rendered = 0  # ramp in again from the paused position
        elif isinstance(command, cmd.TransportStop):
            if self.running:
                self.running = False
                for lane in self.lanes:
                    if lane.voice is not None:
                        release = Release(lane.voice)
                        if release.remaining:
                            lane.releases.append(release)
        elif isinstance(command, cmd.Shutdown):
            self.shutdown_requested = True

    def _set_param(self, command: cmd.SetParam) -> None:
        if command.lane == cmd.MASTER:
            if self.master is not None and command.param == "gain":
                self.master.gain.set(command.value)
            return
        lane = self.lanes[command.lane]
        if command.param not in lane.params:
            self.stats.errors += 1
            self.stats.last_error = f"unknown lane parameter {command.param!r}"
            return
        if lane.automation is not None and lane.automation.param == command.param:
            # A manual move takes over from the automation, from where it is now.
            lane.params[command.param].jump(lane.automation.value_at(self.position_beats))
            lane.automation = None
            self.stats.automation_overrides += 1
        lane.params[command.param].set(command.value)

    def _set_tempo(self, command: cmd.SetTempo, late_ms: float) -> None:
        # On time, the new segment starts exactly at the scheduled beat; late,
        # it can only start where the transport is now.
        beat = self.position_beats if (command.beat is None or late_ms > 0) else command.beat
        self.tempo.set_tempo(beat, command.bpm)
        for lane in self.lanes:
            voice = lane.voice
            if voice is None:
                continue
            replacement = self._buffer_for(voice.info.uid, command.bpm)
            if replacement is None:
                self.stats.missing_buffers += 1
                continue  # keep the old buffer: stays on the beat grid, at the wrong pitch
            lane.release_voice(RELEASE)
            lane.voice = Voice(replacement, voice.start_beat, self.samplerate, ramp_in=RELEASE if self.running else DECLICK_IN)

    def _buffer_for(self, uid: int, bpm: float) -> Optional[LoadedBuffer]:
        return self.buffers.get(uid, {}).get(_bpm_key(bpm))


def _bpm_key(bpm: float) -> float:
    return round(float(bpm), 6)

