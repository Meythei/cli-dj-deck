"""Session: the non-UI services behind the interpreter -- library, background
jobs, and (in later phases) snippet preparation and the audio engine.

Everything here is driven from one thread (the UI's): background work runs
in JobRunner futures, and `poll()` harvests finished futures and applies
their results, so no library state is ever touched from a worker thread.
"""
from __future__ import annotations

from concurrent.futures import Future
from typing import Callable, Optional, Union

from .config import Config, Paths
from .library import DEMO_LIBRARY, Library, Track, discover
from .workers import JobRunner, analyze_job

LogFn = Callable[[str, str], None]


class SessionError(Exception):
    pass


class Session:
    def __init__(
        self,
        paths: Paths,
        config: Config,
        log: LogFn,
        *,
        demo: bool = False,
        jobs=None,
    ) -> None:
        self.paths = paths
        self.config = config
        self.log = log
        self.demo = demo
        self.jobs = jobs if jobs is not None else JobRunner(config.workers, paths.log_dir / "workers.log")
        self.library: Optional[Library] = None
        if demo:
            self.tracks: list[Track] = list(DEMO_LIBRARY)
        else:
            self.library = Library.from_paths(paths)
            self.library.load()
            self.tracks = self.library.tracks
        self._scan_future: Optional[Future] = None
        self._retry_failed = False
        self._analysis: dict[str, Future] = {}
        self._analysis_total = 0
        self._analysis_done = 0

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

    def poll(self) -> bool:
        """Apply finished background work. Returns True if anything changed."""
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

    @property
    def activity(self) -> Optional[str]:
        """One-line description of background work, for the UI."""
        if self._scan_future is not None:
            return "scanning..."
        if self._analysis_total:
            return f"analyzing {self._analysis_done}/{self._analysis_total}"
        return None

    def regrid(self, ref, bpm=None, offset_ms=None, first_beat=None, reset=False) -> Track:
        track = self.resolve_track(ref)
        if self.library is None or track.is_demo:
            raise SessionError("regrid(): demo tracks have a fixed grid")
        try:
            return self.library.regrid(track, bpm=bpm, offset_ms=offset_ms, first_beat=first_beat, reset=reset)
        except ValueError as exc:
            raise SessionError(str(exc)) from None

    def setkey(self, ref, key) -> Track:
        track = self.resolve_track(ref)
        if self.library is None or track.is_demo:
            raise SessionError("setkey(): demo tracks have a fixed key")
        try:
            return self.library.setkey(track, key)
        except ValueError as exc:
            raise SessionError(str(exc)) from None

    def close(self) -> None:
        self.jobs.shutdown()
