"""Tracks: the demo library and the real, on-disk music library.

Waveform and cue data are expressed in beats (not seconds) so a Lane can look
up "what does this track look like right now" purely from a beat position,
matching the Transport's own units.

A real track is identified by a content hash (`track_id_for`), not its path,
so moving or renaming files never loses its analysis or overrides. The short
number shown in the TRACKS table (`Track.id`, used as `snip(3, ...)`) is
handed out in order of first discovery and persisted, so adding files never
renumbers the ones already there.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

SAMPLES_PER_BEAT = 16
AUDIO_EXTENSIONS = frozenset({".wav", ".flac", ".mp3", ".aiff", ".aif", ".ogg"})
ID_CHUNK_BYTES = 64 * 1024
LIBRARY_FILE_VERSION = 1
CAMELOT_RE = re.compile(r"^\s*(1[0-2]|[1-9])([ABab])\s*$")


@dataclass
class Track:
    id: int
    title: str
    artist: str
    bpm: float
    key: str
    duration: float  # seconds
    first_beat: float = 0.0  # seconds from file start to beat 0 of the grid
    waveform: list[float] = field(default_factory=list)  # amplitude, SAMPLES_PER_BEAT per beat from beat 0
    cues: dict[int, float] = field(default_factory=dict)  # cue number -> beat position
    path: Optional[Path] = None  # None for demo tracks
    track_id: str = ""  # stable content id ("demo-N" for demo tracks)
    status: str = "demo"  # demo | new | analyzing | ready | failed | missing
    lufs: Optional[float] = None
    error: str = ""
    waveform_source: Optional[Callable[[], list[float]]] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.track_id and self.path is None:
            self.track_id = f"demo-{self.id}"
        if not self.waveform and self.path is None and self.bpm > 0:
            self.waveform = _fake_waveform(seed=self.id, duration=self.duration, bpm=self.bpm)
        if not self.cues:
            self.cues = phrase_cues(self.duration_beats)

    @property
    def duration_beats(self) -> float:
        if self.bpm <= 0:
            return 0.0
        return max(0.0, (self.duration - self.first_beat) * self.bpm / 60.0) if self.path else self.duration * self.bpm / 60.0

    @property
    def is_demo(self) -> bool:
        return self.path is None

    @property
    def playable(self) -> bool:
        """Has a beat grid, so snippets can be cut from it."""
        return self.bpm > 0 and self.status in ("demo", "ready")

    def amplitude_at_beat(self, beat: float) -> float:
        """Waveform amplitude (0..1) at a given beat position into the track."""
        if not self.waveform and self.waveform_source is not None:
            self.waveform = self.waveform_source()
        if not self.waveform:
            return 0.0
        index = int(beat * SAMPLES_PER_BEAT)
        index = max(0, min(len(self.waveform) - 1, index))
        return self.waveform[index]


def phrase_cues(duration_beats: float, beats_per_bar: int = 4) -> dict[int, float]:
    """Cue points 1-4: track start, then 8/16/24 bars in, clipped to the track
    and snapped down to a bar head on the grid."""
    phrase = 8 * beats_per_bar
    raw = {1: 0.0, 2: min(phrase, duration_beats * 0.25), 3: min(phrase * 2, duration_beats * 0.55),
           4: min(phrase * 3, duration_beats * 0.8)}
    return {n: max(0.0, math.floor(beat / beats_per_bar) * beats_per_bar) for n, beat in raw.items()}


def _fake_waveform(seed: int, duration: float, bpm: float, samples_per_beat: int = SAMPLES_PER_BEAT) -> list[float]:
    """Song-structure envelope (intro/build/drop/breakdown/outro) with a kick
    transient at the start of every beat, so a zoomed-in view shows distinct
    beats rather than smooth noise."""
    rng = random.Random(seed)
    duration_beats = duration * bpm / 60.0
    n = max(samples_per_beat, round(duration_beats * samples_per_beat))
    envelope_points = [0.15, 0.35, 0.9, 0.55, 0.95, 0.4, 0.1]
    values: list[float] = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.0
        seg = t * (len(envelope_points) - 1)
        lo = int(seg)
        hi = min(lo + 1, len(envelope_points) - 1)
        frac = seg - lo
        base = envelope_points[lo] * (1 - frac) + envelope_points[hi] * frac
        phase_in_beat = i % samples_per_beat
        kick = math.exp(-phase_in_beat / 2.2) * 0.6  # sharp decay after each beat's first sample
        wobble = rng.uniform(-0.06, 0.06)
        amp = base * (0.4 + kick) + wobble
        values.append(max(0.03, min(1.0, amp)))
    return values


DEMO_LIBRARY: list[Track] = [
    Track(1, "Nightdrive", "Vektroid Cell", 128.0, "8A", 214),
    Track(2, "Concrete Bloom", "Sable Arc", 126.0, "5A", 231),
    Track(3, "Glass Horizon", "Kite Parade", 130.0, "8A", 198),
    Track(4, "Low Tide Static", "Moriah Deep", 122.0, "3A", 256),
    Track(5, "Afterimage", "Nova Kessler", 128.0, "8B", 207),
    Track(6, "Chrome Petals", "Yui Osprey", 132.0, "10A", 220),
    Track(7, "Faultline", "Dren & Coe", 125.0, "5A", 244),
    Track(8, "Half Light", "Ruin Choir", 128.0, "8A", 233),
]


# ---- real library ----------------------------------------------------------------


def normalize_camelot(key: str) -> str:
    m = CAMELOT_RE.match(str(key))
    if not m:
        raise ValueError(f"{key!r} is not a Camelot key like 8A or 11B")
    return f"{int(m.group(1))}{m.group(2).upper()}"


def track_id_for(path: Path) -> str:
    """Content id from the file size plus its first and last 64 KiB: stable
    across moves/renames, cheap for large libraries, and changed by any
    re-encode or edit of the audio."""
    size = path.stat().st_size
    digest = hashlib.blake2b(digest_size=10)
    digest.update(str(size).encode())
    with path.open("rb") as f:
        digest.update(f.read(ID_CHUNK_BYTES))
        if size > ID_CHUNK_BYTES:
            f.seek(max(ID_CHUNK_BYTES, size - ID_CHUNK_BYTES))
            digest.update(f.read(ID_CHUNK_BYTES))
    return digest.hexdigest()


@dataclass
class FileInfo:
    path: str
    size: int
    mtime_ns: int
    track_id: str
    title: str
    artist: str
    duration: float


def _read_tags(path: Path) -> tuple[str, str, float]:
    title, artist, duration = "", "", 0.0
    try:
        import mutagen

        audio = mutagen.File(str(path), easy=True)
        if audio is not None:
            title = (audio.get("title") or [""])[0]
            artist = (audio.get("artist") or [""])[0]
            duration = float(getattr(audio.info, "length", 0.0) or 0.0)
    except Exception:  # noqa: BLE001 -- unreadable tags just fall back to the file name
        pass
    if not title:
        stem = path.stem
        if " - " in stem and not artist:
            artist, title = (part.strip() for part in stem.split(" - ", 1))
        else:
            title = stem
    return title, artist, duration


def discover(folders: Iterable[Path], known: dict[str, tuple[int, int, str]]) -> list[FileInfo]:
    """Walk the library folders. `known` maps path -> (size, mtime_ns,
    track_id) from the last scan so unchanged files aren't re-hashed. Runs
    off the UI thread."""
    found: list[FileInfo] = []
    for folder in folders:
        folder = Path(folder)
        if not folder.is_dir():
            continue
        for root, _dirs, files in os.walk(folder):
            for name in sorted(files):
                path = Path(root) / name
                if path.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue
                try:
                    stat = path.stat()
                    cached = known.get(str(path))
                    if cached and cached[0] == stat.st_size and cached[1] == stat.st_mtime_ns:
                        track_id = cached[2]
                    else:
                        track_id = track_id_for(path)
                    title, artist, duration = _read_tags(path)
                except OSError:
                    continue
                found.append(FileInfo(str(path), stat.st_size, stat.st_mtime_ns, track_id, title, artist, duration))
    return found


@dataclass
class ScanReport:
    new: int = 0
    moved: int = 0
    missing: int = 0
    total: int = 0


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class Library:
    """The user's music library: persisted track list, analysis results and
    manual overrides, merged into effective `Track` objects.

    `tracks` is mutated in place (never rebound) so anything holding the
    list -- the interpreter's snip() resolution, the UI table -- sees rescans.
    """

    def __init__(self, library_file: Path, analysis_dir: Path, overrides_file: Path) -> None:
        self.library_file = library_file
        self.analysis_dir = analysis_dir
        self.overrides_file = overrides_file
        self.tracks: list[Track] = []
        self.version = 0  # bumped on every change, so views know when to redraw
        self._entries: dict[str, dict] = {}
        self._next_num = 1
        self._analysis: dict[str, dict] = {}
        self._in_flight: set[str] = set()  # track ids a worker is analysing right now
        self.overrides: dict[str, dict] = {}

    @classmethod
    def from_paths(cls, paths) -> "Library":
        return cls(paths.library_file, paths.analysis_dir, paths.overrides_file)

    # ---- persistence -------------------------------------------------------

    def load(self) -> None:
        if self.library_file.exists():
            data = json.loads(self.library_file.read_text(encoding="utf-8"))
            self._entries = data.get("tracks", {})
            self._next_num = int(data.get("next_num", 1))
        if self.overrides_file.exists():
            self.overrides = json.loads(self.overrides_file.read_text(encoding="utf-8"))
        self._analysis = {}
        for track_id in self._entries:
            self._load_analysis_json(track_id)
        self._rebuild()

    def save(self) -> None:
        payload = {"version": LIBRARY_FILE_VERSION, "next_num": self._next_num, "tracks": self._entries}
        _atomic_write_text(self.library_file, json.dumps(payload, indent=1, ensure_ascii=False))

    def _save_overrides(self) -> None:
        _atomic_write_text(self.overrides_file, json.dumps(self.overrides, indent=1, ensure_ascii=False))

    def analysis_json_path(self, track_id: str) -> Path:
        return self.analysis_dir / f"{track_id}.json"

    def envelope_path(self, track_id: str) -> Path:
        return self.analysis_dir / f"{track_id}.env.npy"

    def failure_path(self, track_id: str) -> Path:
        return self.analysis_dir / f"{track_id}.failed.json"

    def _load_analysis_json(self, track_id: str) -> None:
        from .analysis import ANALYSIS_VERSION

        path = self.analysis_json_path(track_id)
        if path.exists() and self.envelope_path(track_id).exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("version") == ANALYSIS_VERSION:
                self._analysis[track_id] = data
                return
        self._analysis.pop(track_id, None)

    # ---- scanning -----------------------------------------------------------

    def known_files(self) -> dict[str, tuple[int, int, str]]:
        return {e["path"]: (e.get("size", -1), e.get("mtime_ns", -1), tid) for tid, e in self._entries.items()}

    def merge_scan(self, infos: list[FileInfo]) -> ScanReport:
        report = ScanReport()
        seen: set[str] = set()
        for info in infos:
            if info.track_id in seen:
                continue  # the same audio twice: keep the first path
            seen.add(info.track_id)
            entry = self._entries.get(info.track_id)
            if entry is None:
                entry = {"num": self._next_num}
                self._next_num += 1
                self._entries[info.track_id] = entry
                report.new += 1
            elif entry.get("path") != info.path:
                report.moved += 1
            entry.update(path=info.path, size=info.size, mtime_ns=info.mtime_ns, title=info.title,
                         artist=info.artist, duration=info.duration or entry.get("duration", 0.0), missing=False)
        for track_id, entry in self._entries.items():
            if track_id not in seen and not entry.get("missing"):
                entry["missing"] = True
                report.missing += 1
        report.total = len(self._entries)
        self.save()
        self._rebuild()
        return report

    def needs_analysis(self) -> list[Track]:
        return [t for t in self.tracks if t.status in ("new",)]

    # ---- analysis results -----------------------------------------------------

    def analysis_started(self, track_id: str) -> None:
        self._in_flight.add(track_id)
        self._rebuild()

    def analysis_finished(self, track_id: str) -> None:
        """A worker wrote (or failed to write) this track's analysis."""
        self._in_flight.discard(track_id)
        self._load_analysis_json(track_id)
        self._rebuild()

    def record_failure(self, track_id: str, error: str) -> None:
        """Persist a failure the worker could not write itself (it crashed)."""
        _atomic_write_text(self.failure_path(track_id), json.dumps({"error": error}))
        self._rebuild()

    def needs_retry(self) -> list[Track]:
        return [t for t in self.tracks if t.status == "failed"]

    def clear_failure(self, track_id: str) -> None:
        self.failure_path(track_id).unlink(missing_ok=True)

    # ---- overrides -------------------------------------------------------------

    def regrid(self, track: Track, bpm: Optional[float] = None, offset_ms: Optional[float] = None,
               first_beat: Optional[float] = None, reset: bool = False) -> Track:
        if track.is_demo:
            raise ValueError("demo tracks have a fixed grid")
        override = self.overrides.setdefault(track.track_id, {})
        override["title"] = track.title
        if reset:
            override.pop("bpm", None)
            override.pop("first_beat", None)
        if bpm is not None:
            if bpm <= 0:
                raise ValueError("bpm must be positive")
            override["bpm"] = float(bpm)
        if first_beat is not None:
            override["first_beat"] = float(first_beat)
        if offset_ms is not None:
            base = override.get("first_beat", track.first_beat)
            override["first_beat"] = round(float(base) + float(offset_ms) / 1000.0, 6)
        self._save_overrides()
        self._rebuild()
        return self.resolve(track.track_id)

    def setkey(self, track: Track, key: Optional[str]) -> Track:
        if track.is_demo:
            raise ValueError("demo tracks have a fixed key")
        override = self.overrides.setdefault(track.track_id, {})
        override["title"] = track.title
        if key is None:
            override.pop("key", None)
        else:
            override["key"] = normalize_camelot(key)
        self._save_overrides()
        self._rebuild()
        return self.resolve(track.track_id)

    # ---- effective tracks ---------------------------------------------------------

    def _rebuild(self) -> None:
        """Recompute every Track from entry + analysis + overrides, reusing
        existing Track objects so snippets keep pointing at the same ones."""
        existing = {t.track_id: t for t in self.tracks}
        rebuilt: list[Track] = []
        for track_id, entry in sorted(self._entries.items(), key=lambda kv: kv[1]["num"]):
            track = existing.get(track_id)
            if track is None:
                track = Track(entry["num"], entry.get("title", ""), entry.get("artist", ""), 0.0, "",
                              entry.get("duration", 0.0), path=Path(entry["path"]), track_id=track_id, status="new")
            self._apply(track, entry)
            rebuilt.append(track)
        self.tracks[:] = rebuilt
        self.version += 1

    def _apply(self, track: Track, entry: dict) -> None:
        from .analysis import beat_waveform

        analysis = self._analysis.get(track.track_id)
        override = self.overrides.get(track.track_id, {})
        track.id = entry["num"]
        track.path = Path(entry["path"])
        track.title = entry.get("title") or track.path.stem
        track.artist = entry.get("artist", "")
        previous_grid = (track.bpm, track.first_beat)
        if analysis is None:
            failure = self.failure_path(track.track_id)
            if track.track_id in self._in_flight:
                track.status = "analyzing"
            else:
                track.status = "failed" if failure.exists() else "new"
            track.error = json.loads(failure.read_text(encoding="utf-8")).get("error", "") if failure.exists() else ""
            track.bpm = float(override.get("bpm", 0.0))
            track.first_beat = float(override.get("first_beat", 0.0))
            track.key = override.get("key", "")
            track.duration = entry.get("duration", 0.0)
        else:
            track.status = "ready"
            track.error = ""
            track.bpm = float(override.get("bpm", analysis["bpm"]))
            track.first_beat = float(override.get("first_beat", analysis["first_beat"]))
            track.key = override.get("key", analysis.get("key") or "")
            track.lufs = analysis.get("lufs")
            track.duration = float(analysis["duration"])
        if entry.get("missing"):
            track.status = "missing"
        track.cues = phrase_cues(track.duration_beats)
        if analysis is not None and (previous_grid != (track.bpm, track.first_beat) or not track.waveform):
            envelope_path = self.envelope_path(track.track_id)
            rate = float(analysis["envelope_rate"])
            bpm, first_beat = track.bpm, track.first_beat

            def load_waveform(path=envelope_path, rate=rate, bpm=bpm, first_beat=first_beat) -> list[float]:
                import numpy as np

                try:
                    envelope = np.load(path)
                except OSError:
                    return []
                return beat_waveform(envelope, rate, bpm, first_beat, SAMPLES_PER_BEAT)

            track.waveform = []
            track.waveform_source = load_waveform

    def resolve(self, ref: Union[int, str, Track]) -> Track:
        if isinstance(ref, Track):
            return ref
        for track in self.tracks:
            if (isinstance(ref, int) and not isinstance(ref, bool) and track.id == ref) or (
                isinstance(ref, str) and (track.track_id == ref or track.title.lower() == ref.lower())
            ):
                return track
        raise KeyError(f"no track {ref!r} in library")
