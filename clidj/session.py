"""Session: everything behind the interpreter that isn't UI -- library,
background jobs, snippet preparation, the scheduler and the audio engine.

Everything here is driven from one thread (the UI's): background work runs
in JobRunner futures, and `poll()` harvests finished futures and applies
their results, so no library or lane state is ever touched from a worker.

The engine owns time. `transport` and `lanes` here are *views*, refreshed
from engine status (`_sync_view`); lane and tempo changes are sent to the
engine as EngineCommands carrying the beat they must happen on. The
scheduler (Thunks, at/after/every) stays on this side and fires events
shortly before their beat, so the command arrives in time.

When preparation is enabled (audio mode), two rules from
docs/TASK_real-audio.md 7 are enforced here:

- a snippet that isn't rendered at the current tempo never starts silently:
  the play is *held* and re-issued at the first quantize boundary after its
  render is ready;
- bpm(x) first renders every known snippet at x, and only then schedules
  the tempo change for the next bar head.
"""
from __future__ import annotations

import dataclasses
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Callable, Optional, Union

from . import prep as prep_states
from .config import Config, Paths
from .engine import commands as cmd
from .engine.client import LocalEngineClient
from .engine.core import Engine, EngineStatus
from .lanes import LANE_NAMES, Lane
from .library import DEMO_LIBRARY, Library, Track, discover
from .prep import PrepManager
from .render import loudness_gain
from .scheduler import Scheduler, SchedulerError
from .snippets import Snippet
from .transport import Transport
from .workers import JobRunner, analyze_job

LogFn = Callable[[str, str], None]


class SessionError(Exception):
    pass


@dataclass
class Hold:
    snippet: Snippet
    description: str
    retry: Callable[[], None]


@dataclass
class Crossfade:
    automation_id: int
    from_lane: str
    to_lane: str
    start_beat: float
    end_beat: float


class Session:
    def __init__(
        self,
        paths: Paths,
        config: Config,
        log: LogFn,
        *,
        demo: bool = False,
        jobs=None,
        prepare: bool = False,
        engine=None,
        bpm: float = 128.0,
    ) -> None:
        self.paths = paths
        self.config = config
        self.log = log
        self.demo = demo
        self.jobs = jobs if jobs is not None else JobRunner(config.workers, paths.log_dir / "workers.log")
        self.library: Optional[Library] = None
        if demo:
            # Copies: demo tracks gain synthesised audio and waveforms per session.
            self.tracks: list[Track] = [dataclasses.replace(t) for t in DEMO_LIBRARY]
        else:
            self.library = Library.from_paths(paths)
            self.library.load()
            self.tracks = self.library.tracks
        self._scan_future: Optional[Future] = None
        self._retry_failed = False
        self._analysis: dict[str, Future] = {}
        self._analysis_total = 0
        self._analysis_done = 0

        self.engine = engine if engine is not None else LocalEngineClient(
            Engine(config.samplerate, bpm, len(LANE_NAMES), render_audio=False)
        )
        self.samplerate = self.engine.samplerate
        self.transport = Transport(bpm=bpm)
        self.scheduler = Scheduler(self.transport, log)
        self.lanes: dict[str, Lane] = {name: Lane(name) for name in LANE_NAMES}
        self.crossfades: dict[int, Crossfade] = {}
        self.heard_beats = 0.0
        self.last_status: Optional[EngineStatus] = None
        self._registered: set[tuple] = set()
        self._frame_remainder = 0.0
        self._starting = False

        self.prep: Optional[PrepManager] = (
            PrepManager(self.jobs, paths.render_dir, paths.demo_audio_dir, self.samplerate, log) if prepare else None
        )
        self.snippets: dict[int, Snippet] = {}  # uid -> every snippet defined this session
        self._next_uid = 1
        self.holds: list[Hold] = []
        self.pending_bpm: Optional[float] = None
        self._prep_batch: Optional[tuple[float, list[Snippet]]] = None
        self._sync_view()

    # ---- library --------------------------------------------------------------

    @property
    def library_version(self) -> int:
        return self.library.version if self.library else 0

    def resolve_track(self, ref: Union[int, str, Track]) -> Track:
        if isinstance(ref, Track):
            return ref
        if self.library is not None:
            try:
                return self.library.resolve(ref)
            except KeyError as exc:
                raise SessionError(str(exc).strip("'\"")) from None
        for track in self.tracks:
            if (isinstance(ref, int) and track.id == ref) or (isinstance(ref, str) and track.title.lower() == ref.lower()):
                return track
        raise SessionError(f"no track {ref!r} in library")

    def scan(self, retry: bool = False) -> None:
        """Rescan the library folders and analyse new tracks in the
        background. retry=True also re-analyses tracks that failed before."""
        if self.library is None:
            self.log("scan(): the demo library has no folders to scan", "warn")
            return
        folders = self.config.library_folders
        if not folders:
            self.log(f"scan(): no library folders configured -- add them to {self.paths.config_file}", "warn")
            return
        if self._scan_future is not None:
            self.log("scan(): already scanning", "warn")
            return
        missing = [str(f) for f in folders if not f.is_dir()]
        for folder in missing:
            self.log(f"scan(): folder not found: {folder}", "warn")
        self._retry_failed = retry
        self._scan_future = self.jobs.run_in_thread(discover, list(folders), self.library.known_files())
        self.log(f"scanning {len(folders)} folder(s)...", "info")

    def _poll_library(self) -> bool:
        changed = False
        if self._scan_future is not None and self._scan_future.done():
            future, self._scan_future = self._scan_future, None
            changed = True
            try:
                infos = future.result()
            except Exception as exc:  # noqa: BLE001
                self.log(f"scan failed: {exc}", "error")
            else:
                report = self.library.merge_scan(infos)
                self.log(
                    f"scan: {report.total} track(s), {report.new} new, {report.moved} moved, {report.missing} missing",
                    "info",
                )
                self._start_analysis()

        for track_id, future in list(self._analysis.items()):
            if not future.done():
                continue
            del self._analysis[track_id]
            self._analysis_done += 1
            changed = True
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 -- e.g. a worker process died
                result = {"track_id": track_id, "ok": False, "error": f"worker crashed: {exc}"}
                self.library.record_failure(track_id, result["error"])
            self.library.analysis_finished(track_id)
            track = self.library.resolve(track_id)
            if result.get("ok"):
                self.log(f"analyzed #{track.id} {track.title}: {track.bpm:.2f} BPM {track.key or '?'}", "info")
            else:
                self.log(f"analysis failed for #{track.id} {track.title}: {result.get('error')}", "error")
        if not self._analysis and self._analysis_total:
            self.log(f"analysis finished ({self._analysis_done} track(s))", "info")
            self._analysis_total = self._analysis_done = 0
        return changed

    def _start_analysis(self) -> None:
        candidates = self.library.needs_analysis()
        if self._retry_failed:
            for track in self.library.needs_retry():
                self.library.clear_failure(track.track_id)
                candidates.append(track)
        pending = [t for t in candidates if t.track_id not in self._analysis]
        for track in pending:
            self.library.analysis_started(track.track_id)
            try:
                future = self.jobs.run_in_process(
                    analyze_job, str(track.path), track.track_id, str(self.library.analysis_dir), self.config.bpm_range
                )
            except Exception as exc:  # noqa: BLE001 -- e.g. worker processes cannot be started at all
                future = Future()
                future.set_exception(exc)
            self._analysis[track.track_id] = future
            self._analysis_total += 1
        if pending:
            self.log(f"analyzing {len(pending)} track(s) in the background", "info")

    def regrid(self, ref, bpm=None, offset_ms=None, first_beat=None, reset=False) -> Track:
        track = self.resolve_track(ref)
        if self.library is None or track.is_demo:
            raise SessionError("regrid(): demo tracks have a fixed grid")
        try:
            updated = self.library.regrid(track, bpm=bpm, offset_ms=offset_ms, first_beat=first_beat, reset=reset)
        except ValueError as exc:
            raise SessionError(str(exc)) from None
        self._reprepare_track(updated)
        return updated

    def setkey(self, ref, key) -> Track:
        track = self.resolve_track(ref)
        if self.library is None or track.is_demo:
            raise SessionError("setkey(): demo tracks have a fixed key")
        try:
            return self.library.setkey(track, key)
        except ValueError as exc:
            raise SessionError(str(exc)) from None

    def _reprepare_track(self, track: Track) -> None:
        """A new grid means new render keys: render this track's snippets again."""
        if self.prep is None:
            return
        affected = [s for s in self.snippets.values() if s.track is track]
        for snippet in affected:
            for bpm in self._bpms_in_play():
                self.prep.request(snippet, bpm)
        if affected:
            self.log(f"re-rendering {len(affected)} snippet(s) of #{track.id} for the new grid", "info")

    # ---- snippets and preparation --------------------------------------------------

    def register_snippet(self, snippet: Snippet) -> Snippet:
        if snippet.uid == 0:
            snippet.uid = self._next_uid
            self._next_uid += 1
        self.snippets[snippet.uid] = snippet
        if self.prep is not None:
            for bpm in self._bpms_in_play():
                self.prep.request(snippet, bpm)
        return snippet

    def _bpms_in_play(self) -> list[float]:
        bpms = [self.transport.bpm]
        if self.pending_bpm is not None and self.pending_bpm != self.transport.bpm:
            bpms.append(self.pending_bpm)
        return bpms

    @property
    def preparing(self) -> bool:
        return self.prep is not None

    def prep_state(self, snippet: Snippet, bpm: Optional[float] = None) -> str:
        if self.prep is None:
            return prep_states.READY
        return self.prep.state(snippet, self.transport.bpm if bpm is None else bpm)

    def ready_to_play(self, snippet: Snippet) -> bool:
        return all(self.prep_state(snippet, bpm) == prep_states.READY for bpm in self._bpms_in_play())

    def snippet_gain(self, snippet: Snippet) -> float:
        return loudness_gain(snippet.track.lufs, self.config.loudness_target_lufs)

    def hold(self, snippet: Snippet, description: str, retry: Callable[[], None]) -> None:
        """Park a play whose snippet isn't rendered yet. It is re-issued (via
        `retry`, which re-quantizes) once the render is ready."""
        if self.prep is not None:
            for bpm in self._bpms_in_play():
                self.prep.request(snippet, bpm)
        self.holds.append(Hold(snippet, description, retry))
        self.log(f"{description}: {snippet.name} is not prepared yet -- waiting, then playing at the next boundary", "warn")

    def prep_all(self) -> None:
        if self.prep is None:
            self.log("prep(): nothing to prepare without audio (--no-audio)", "info")
            return
        snippets = list(self.snippets.values())
        if not snippets:
            self.log("prep(): no snippets defined", "info")
            return
        for snippet in snippets:
            for bpm in self._bpms_in_play():
                self.prep.request(snippet, bpm, retry=True)
        self._prep_batch = (self.transport.bpm, snippets)
        self.log(f"prep(): preparing {len(snippets)} snippet(s) at {self.transport.bpm:g} BPM", "info")

    def _batch_progress(self, bpm: float, snippets: list[Snippet]) -> tuple[int, int, int]:
        states = [self.prep.state(s, bpm) for s in snippets]
        return states.count(prep_states.READY), states.count(prep_states.FAILED), len(states)

    # ---- engine: buffers ---------------------------------------------------------------------

    def _buffer_info(self, snippet: Snippet, bpm: float) -> Optional[cmd.BufferInfo]:
        if self.prep is None:
            frames = int(round(snippet.length_beats * 60.0 / bpm * self.samplerate))
            return cmd.BufferInfo(snippet.uid, bpm, None, frames, snippet.length_beats, snippet.loop)
        buffer = self.prep.buffer(snippet, bpm)
        if buffer is None:
            return None
        return cmd.BufferInfo(snippet.uid, bpm, buffer.path, buffer.frames, snippet.length_beats, snippet.loop,
                              self.snippet_gain(snippet))

    def _ensure_registered(self, snippet: Snippet, bpm: float) -> bool:
        info = self._buffer_info(snippet, bpm)
        if info is None:
            return False
        key = (info.uid, round(info.bpm, 6), info.path, round(info.gain, 6))
        if key not in self._registered:
            self.engine.send(cmd.RegisterBuffer(info))
            self._registered.add(key)
        return True

    # ---- engine: lane and transport commands ------------------------------------------------------

    def immediate_beat(self) -> float:
        """The beat something "immediate" that needs a beat (an xf) starts on:
        where the engine is now."""
        if not self.engine.realtime:
            self.engine.flush()
            self._sync_view()
        return self.transport.position_beats

    @staticmethod
    def lane_index(lane: Lane) -> int:
        return LANE_NAMES.index(lane.name)

    def _after_immediate(self) -> None:
        if not self.engine.realtime:
            self.engine.flush()
            self._sync_view()

    def play(self, lane: Lane, snippet: Snippet, beat: Optional[float]) -> None:
        for bpm in self._bpms_in_play():
            self._ensure_registered(snippet, bpm)
        self.engine.send(cmd.Play(self.lane_index(lane), snippet.uid, beat))
        if beat is None or not self.transport.running:
            self._after_immediate()

    def stop_lane(self, lane: Lane, beat: Optional[float]) -> None:
        self.engine.send(cmd.Stop(self.lane_index(lane), beat))
        if beat is None or not self.transport.running:
            self._after_immediate()

    def set_param(self, lane: Lane, param: str, value: float) -> None:
        self.engine.send(cmd.SetParam(self.lane_index(lane), param, float(value)))
        # the view changes now; the engine glides there over a few ms
        if param == "mute":
            lane.muted = value >= 0.5
        else:
            setattr(lane, param, float(value))
        self._after_immediate()

    def crossfade(self, from_lane: Lane, to_lane: Lane, start_beat: float, bars: float, description: str) -> int:
        end_beat = start_beat + bars * self.transport.beats_per_bar
        for lane in (from_lane, to_lane):
            for xf in list(self.crossfades.values()):
                if lane.name in (xf.from_lane, xf.to_lane) and xf.end_beat > start_beat:
                    self.log(f"{lane.name}: gain automation overridden by new xf", "warn")
                    del self.crossfades[xf.automation_id]
        automation_id = self.scheduler.new_id()
        self.engine.send(cmd.Automate(automation_id, self.lane_index(from_lane), "gain", start_beat, end_beat,
                                      None, 0.0, stop_lane_at_end=True))
        self.engine.send(cmd.Automate(automation_id, self.lane_index(to_lane), "gain", start_beat, end_beat, 0.0, 1.0))
        self.crossfades[automation_id] = Crossfade(automation_id, from_lane.name, to_lane.name, start_beat, end_beat)
        if not self.transport.running:
            self._after_immediate()
        return automation_id

    def cancel(self, event_id: Optional[int] = None) -> str:
        if event_id is None:
            count = len(self.scheduler.events) + len(self.crossfades)
            self.scheduler.cancel()
            self.crossfades.clear()
            self.engine.send(cmd.CancelAutomation(None))
            self._after_immediate()
            return f"cancelled {count} event(s)"
        if event_id in self.crossfades:
            del self.crossfades[event_id]
            self.engine.send(cmd.CancelAutomation(event_id))
            self._after_immediate()
            return f"cancelled #{event_id}"
        try:
            return self.scheduler.cancel(event_id)
        except SchedulerError as exc:
            raise SessionError(str(exc)) from None

    def start(self) -> None:
        if self.transport.running:
            return
        self.transport.running = True
        # Reservations exactly at the start position (at(1, ...) typed before
        # start()) fire now, so their commands reach the engine before the start.
        self._starting = True
        try:
            self.scheduler.poll(self.transport.position_beats)
        finally:
            self._starting = False
        self.engine.send(cmd.TransportStart())
        self._after_immediate()

    def stop(self) -> None:
        self.engine.send(cmd.TransportStop())
        self.transport.running = False
        self._after_immediate()

    # ---- tempo ---------------------------------------------------------------------------

    def request_bpm(self, bpm: float, in_scheduled_context: bool = False) -> None:
        """Change the tempo. Without preparation: at the next bar head (now
        if stopped, or at the firing beat inside at/after/every). With it,
        every known snippet is rendered at `bpm` first."""
        if bpm <= 0:
            raise SessionError("bpm must be positive")
        if self.prep is not None:
            missing = [s for s in self.snippets.values() if self.prep.state(s, bpm) != prep_states.READY]
            if missing:
                self.pending_bpm = bpm
                for snippet in missing:
                    self.prep.request(snippet, bpm)
                self.log(
                    f"bpm({bpm:g}): rendering {len(missing)} snippet(s) first; "
                    "the tempo changes at the first bar after they are ready",
                    "warn" if in_scheduled_context else "info",
                )
                return
            self.pending_bpm = None
        self._schedule_tempo(bpm, immediate=in_scheduled_context)

    def _send_tempo(self, bpm: float, beat: Optional[float]) -> None:
        for snippet in self.snippets.values():
            self._ensure_registered(snippet, bpm)
        self.engine.send(cmd.SetTempo(bpm, beat))
        self.log(f"bpm -> {bpm:g}", "info")

    def _schedule_tempo(self, bpm: float, immediate: bool) -> None:
        if immediate or not self.transport.running:
            beat = self.scheduler.firing_beat if (immediate and self.transport.running) else None
            self._send_tempo(bpm, beat)
            if beat is None or not self.transport.running:
                self._after_immediate()
            return
        target = self.transport.next_boundary_beats("bar")
        self.scheduler.schedule_at(target, lambda: self._send_tempo(bpm, self.scheduler.firing_beat), f"bpm({bpm:g})")
        self.log(f"bpm({bpm:g}) scheduled for bar {self.transport.bar_at(target)}", "info")

    # ---- time ------------------------------------------------------------------------------------

    def _sync_view(self) -> None:
        status = self.engine.status()
        self.last_status = status
        transport = self.transport
        transport.position_beats = status.position_beats
        transport.bpm = status.bpm
        if not self._starting:
            transport.running = status.running
        self.heard_beats = status.heard_beats
        for lane, lane_status in zip(self.lanes.values(), status.lanes):
            lane.snippet = self.snippets.get(lane_status.uid) if lane_status.uid else None
            lane.started_at_beat = lane_status.start_beat
            lane.gain = lane_status.gain
            lane.muted = lane_status.muted
            lane.lo, lane.mid, lane.hi = lane_status.lo, lane_status.mid, lane_status.hi
        for automation_id, xf in list(self.crossfades.items()):
            if xf.end_beat <= status.position_beats:
                del self.crossfades[automation_id]

    def advance(self, frames: int):
        """Local engines only: fire every reservation due within the next
        `frames` samples, then run the engine over them. Returns the audio
        (or None in visual mode)."""
        self._sync_view()
        if self.transport.running and frames > 0:
            self.scheduler.poll(self.engine.beat_at_sample_offset(frames))
        out = self.engine.advance(frames)
        self._sync_view()
        return out

    def poll(self) -> bool:
        """Apply finished background work, release holds and pending tempo
        changes. Returns True if anything visible changed."""
        changed = self._poll_library()
        if self.prep is None:
            return changed
        before = self.prep.version
        for key, ok in self.prep.poll({t.track_id: t for t in self.tracks}):
            if not ok:
                self.log(f"render failed ({key})", "error")
        changed |= self.prep.version != before

        if self.pending_bpm is not None:
            bpm = self.pending_bpm
            states = [self.prep.state(s, bpm) for s in self.snippets.values()]
            if all(state in (prep_states.READY, prep_states.FAILED) for state in states):
                failed = states.count(prep_states.FAILED)
                if failed:
                    self.log(f"bpm({bpm:g}): {failed} snippet(s) failed to render and cannot play at {bpm:g}", "error")
                self.pending_bpm = None
                self._schedule_tempo(bpm, immediate=False)
                changed = True

        for hold in list(self.holds):
            states = {self.prep.state(hold.snippet, bpm) for bpm in self._bpms_in_play()}
            if prep_states.FAILED in states:
                self.holds.remove(hold)
                reason = self.prep.error_for(hold.snippet, self.transport.bpm) or "render failed"
                self.log(f"{hold.description} dropped: {hold.snippet.name} could not be prepared ({reason})", "error")
                changed = True
            elif states == {prep_states.READY}:
                self.holds.remove(hold)
                self.log(f"{hold.snippet.name} is ready", "info")
                hold.retry()
                changed = True

        if self._prep_batch is not None:
            bpm, snippets = self._prep_batch
            ready, failed, total = self._batch_progress(bpm, snippets)
            if ready + failed == total:
                self._prep_batch = None
                level = "error" if failed else "info"
                summary = f"prep(): {ready}/{total} snippet(s) ready at {bpm:g} BPM"
                self.log(summary + (f", {failed} failed" if failed else ""), level)
                changed = True
        return changed

    def tick(self, dt_seconds: float) -> None:
        """Called by the UI every frame with the wall-clock time since the
        last call. A local (visual) engine is advanced by that much time."""
        self.poll()
        if self.transport.running and dt_seconds > 0:
            exact = dt_seconds * self.samplerate + self._frame_remainder
            frames = int(exact)
            self._frame_remainder = exact - frames
            self.advance(frames)
        else:
            self._after_immediate()

    @property
    def activity(self) -> Optional[str]:
        """One-line description of background work, for the UI."""
        parts = []
        if self._scan_future is not None:
            parts.append("scanning...")
        if self._analysis_total:
            parts.append(f"analyzing {self._analysis_done}/{self._analysis_total}")
        if self.prep is not None:
            if self.pending_bpm is not None:
                ready, failed, total = self._batch_progress(self.pending_bpm, list(self.snippets.values()))
                parts.append(f"bpm {self.pending_bpm:g}: rendering {ready + failed}/{total}")
            elif self._prep_batch is not None:
                ready, failed, total = self._batch_progress(*self._prep_batch)
                parts.append(f"prep {ready + failed}/{total}")
            elif self.prep.outstanding:
                parts.append(f"rendering {self.prep.outstanding}")
        return "  ".join(parts) if parts else None

    def close(self) -> None:
        self.engine.close()
        self.jobs.shutdown()
