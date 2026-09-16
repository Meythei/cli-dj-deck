"""Real library: stable ids, persistent numbering, scanning, analysis jobs and
overrides (docs/TASK_real-audio.md 6)."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from clidj import synth
from clidj.config import Config, Paths
from clidj.interpreter import Interpreter
from clidj.lanes import LANE_NAMES, Lane
from clidj.library import Library, discover, normalize_camelot, track_id_for
from clidj.scheduler import Scheduler
from clidj.session import Session, SessionError
from clidj.snippets import Snippet
from clidj.transport import Transport
from clidj.workers import InlineJobRunner, JobRunner, analyze_job

SR = 44100


@pytest.fixture(scope="module")
def audio_files(tmp_path_factory):
    """Three short synthetic tracks, generated once per module."""
    folder = tmp_path_factory.mktemp("music")
    specs = [("Alpha - One", 124.0, 9, True, 0.37), ("Beta - Two", 128.0, 2, False, 0.1), ("Gamma", 126.0, 0, True, 0.25)]
    files = []
    for name, bpm, root, minor, first in specs:
        audio = synth.techno_loop_track(bpm, root, minor, duration=24.0, sr=SR, first_beat=first, seed=len(files))
        files.append(synth.write_audio(folder / f"{name}.flac", audio, SR))
    return folder, files, specs


def make_session(tmp_path: Path, folders, logs=None) -> Session:
    paths = Paths(tmp_path / "config", tmp_path / "data", tmp_path / "cache")
    config = Config(library_folders=[Path(f) for f in folders])
    log = (lambda m, level="info": logs.append((level, m))) if logs is not None else (lambda m, level="info": None)
    return Session(paths, config, log, jobs=InlineJobRunner())


def copy_folder(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


# ---- ids and numbering ---------------------------------------------------------


def test_track_id_survives_move_and_rename_but_not_edits(tmp_path, audio_files):
    _, files, _ = audio_files
    original = track_id_for(files[0])
    moved = tmp_path / "elsewhere" / "renamed.flac"
    moved.parent.mkdir()
    shutil.copy(files[0], moved)
    assert track_id_for(moved) == original
    data = bytearray(moved.read_bytes())
    data[-10] ^= 0xFF
    moved.write_bytes(bytes(data))
    assert track_id_for(moved) != original


def test_numbers_are_assigned_once_and_persist(tmp_path, audio_files):
    src, files, _ = audio_files
    music = copy_folder(src, tmp_path / "music")
    session = make_session(tmp_path, [music])
    session.scan()
    session.poll()
    numbers = {t.title: t.id for t in session.tracks}
    assert sorted(numbers.values()) == [1, 2, 3]

    # a new file gets the next number; existing ones keep theirs
    extra = synth.click_track(120.0, 0.0, 6.0, SR)
    synth.write_audio(music / "Aaa - First Alphabetically.wav", extra, SR)
    session.scan()
    session.poll()
    after = {t.title: t.id for t in session.tracks}
    assert {k: after[k] for k in numbers} == numbers
    assert after["First Alphabetically"] == 4

    reloaded = make_session(tmp_path, [music])
    assert {t.title: t.id for t in reloaded.tracks} == after


def test_moved_file_keeps_number_and_analysis_and_missing_files_are_flagged(tmp_path, audio_files):
    src, _, _ = audio_files
    music = copy_folder(src, tmp_path / "music")
    session = make_session(tmp_path, [music])
    session.scan()
    session.poll()
    track = session.resolve_track("One")
    number, bpm, track_id = track.id, track.bpm, track.track_id
    assert track.status == "ready"

    (music / "sub").mkdir()
    (music / "Alpha - One.flac").rename(music / "sub" / "moved.flac")
    (music / "Gamma.flac").unlink()
    session.scan()
    session.poll()
    moved = session.resolve_track(track_id)  # untagged file: the title now comes from the new name
    assert moved.id == number and moved.bpm == bpm and moved.status == "ready"
    assert moved.path.name == "moved.flac"
    assert session.resolve_track("Gamma").status == "missing"


def test_tags_fall_back_to_artist_dash_title_file_names(tmp_path, audio_files):
    src, _, _ = audio_files
    infos = {Path(i.path).stem: i for i in discover([src], {})}
    assert (infos["Alpha - One"].artist, infos["Alpha - One"].title) == ("Alpha", "One")
    assert (infos["Gamma"].artist, infos["Gamma"].title) == ("", "Gamma")


# ---- analysis through the session -------------------------------------------------


def test_scan_analyses_new_tracks_and_snip_works_afterwards(tmp_path, audio_files):
    src, _, specs = audio_files
    logs = []
    session = make_session(tmp_path, [src], logs)
    assert session.tracks == []
    session.scan()
    session.poll()
    assert [t.status for t in session.tracks] == ["ready"] * 3
    for (name, bpm, _root, _minor, first), track in zip(sorted(specs), sorted(session.tracks, key=lambda t: t.path.name)):
        assert track.bpm == pytest.approx(bpm, abs=0.1), name
        assert track.first_beat == pytest.approx(first, abs=0.010), name
        assert track.lufs is not None
    assert any("analysis finished" in m for _, m in logs)

    interp = Interpreter(session, Path("."), log=lambda m, level="info": logs.append((level, m)))
    interp.run("a = snip(1, bar=2, bars=4, loop=True)")
    assert isinstance(interp.env["a"], Snippet)
    assert interp.env["a"].start_beat == 4.0
    # the analysed envelope feeds the beat-resolution waveform lazily
    assert 0.0 < max(interp.env["a"].track.amplitude_at_beat(b / 4) for b in range(16)) <= 1.0


def test_snip_on_an_unanalysed_track_is_a_clear_error(tmp_path, audio_files):
    src, _, _ = audio_files
    session = make_session(tmp_path, [src])
    lib = session.library
    lib.merge_scan(discover([src], {}))  # scanned, never analysed
    logs = []
    interp = Interpreter(session, Path("."), log=lambda m, level="info": logs.append((level, m)))
    interp.run("a = snip(1, bar=1, bars=4)")
    errors = [m for level, m in logs if level == "error"]
    assert errors and "scan()" in errors[0]


def test_corrupt_file_fails_analysis_without_breaking_the_rest(tmp_path, audio_files):
    src, _, _ = audio_files
    music = copy_folder(src, tmp_path / "music")
    (music / "Broken - File.mp3").write_bytes(b"ID3" + b"\x00" * 5000)
    logs = []
    session = make_session(tmp_path, [music], logs)
    session.scan()
    session.poll()
    statuses = {t.title: t.status for t in session.tracks}
    assert statuses.pop("File") == "failed"
    assert set(statuses.values()) == {"ready"}
    assert any(level == "error" and "File" in m for level, m in logs)
    assert session.resolve_track("File").error


def test_real_worker_process_runs_analysis_job(tmp_path, audio_files):
    """Spawn-based pool on the real platform (Windows spawn in CI/dev)."""
    _, files, _ = audio_files
    runner = JobRunner(workers=1, log_path=tmp_path / "workers.log")
    try:
        result = runner.run_in_process(analyze_job, str(files[1]), "abc123", str(tmp_path / "analysis"), (88.0, 176.0))
        payload = result.result(timeout=180)
    finally:
        runner.shutdown()
    assert payload["ok"], payload
    assert payload["bpm"] == pytest.approx(128.0, abs=0.1)
    assert (tmp_path / "analysis" / "abc123.json").exists()
    assert np.load(tmp_path / "analysis" / "abc123.env.npy").size > 0


# ---- overrides ---------------------------------------------------------------------


def test_regrid_overrides_analysis_persists_and_can_be_reset(tmp_path, audio_files):
    src, _, _ = audio_files
    session = make_session(tmp_path, [src])
    session.scan()
    session.poll()
    track = session.resolve_track("One")
    auto_bpm, auto_first = track.bpm, track.first_beat
    waveform_before = [track.amplitude_at_beat(b / 16) for b in range(64)]

    session.regrid(1 if track.id == 1 else track.id, bpm=127.98, offset_ms=12)
    assert track.bpm == 127.98
    assert track.first_beat == pytest.approx(auto_first + 0.012)
    # the waveform is rebuilt from the envelope on the new grid
    assert [track.amplitude_at_beat(b / 16) for b in range(64)] != waveform_before

    stored = json.loads(session.paths.overrides_file.read_text(encoding="utf-8"))
    assert stored[track.track_id]["bpm"] == 127.98

    reloaded = make_session(tmp_path, [src])
    again = reloaded.resolve_track("One")
    assert again.bpm == 127.98 and again.first_beat == pytest.approx(auto_first + 0.012)

    reloaded.regrid(again.id, reset=True)
    assert again.bpm == auto_bpm and again.first_beat == pytest.approx(auto_first)


def test_setkey_normalises_and_validates(tmp_path, audio_files):
    src, _, _ = audio_files
    session = make_session(tmp_path, [src])
    session.scan()
    session.poll()
    track = session.resolve_track("Two")
    session.setkey(track.id, "11b")
    assert track.key == "11B"
    with pytest.raises(SessionError):
        session.setkey(track.id, "13A")
    assert normalize_camelot(" 8a ") == "8A"


def test_demo_tracks_cannot_be_regridded(tmp_path):
    paths = Paths(tmp_path / "c", tmp_path / "d", tmp_path / "k")
    session = Session(paths, Config(), lambda *a: None, demo=True, jobs=InlineJobRunner())
    with pytest.raises(SessionError):
        session.regrid(1, bpm=120)


# ---- config ---------------------------------------------------------------------------


def test_config_template_is_written_and_parsed(tmp_path):
    paths = Paths(tmp_path / "config", tmp_path / "data", tmp_path / "cache")
    default = Config.load(paths)
    assert paths.config_file.exists()
    assert default.samplerate == 48000 and default.blocksize == 512
    text = paths.config_file.read_text(encoding="utf-8")
    text = text.replace("folders = []", 'folders = ["D:/Music/DJ"]').replace('device = ""', 'device = "3"')
    paths.config_file.write_text(text, encoding="utf-8")
    loaded = Config.load(paths)
    assert loaded.library_folders == [Path("D:/Music/DJ")]
    assert loaded.device == 3


def test_clidj_home_redirects_every_path(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIDJ_HOME", str(tmp_path))
    paths = Paths.default()
    for p in (paths.config_file, paths.library_file, paths.render_dir, paths.overrides_file):
        assert tmp_path in p.parents


class _CrashingRunner(InlineJobRunner):
    """Simulates a worker process dying mid-job (no failure file written)."""

    def run_in_process(self, fn, *args):
        from concurrent.futures import Future
        from concurrent.futures.process import BrokenProcessPool

        future = Future()
        future.set_exception(BrokenProcessPool("worker died"))
        return future


def test_crashed_worker_marks_tracks_failed_and_scan_retry_recovers(tmp_path, audio_files):
    src, _, _ = audio_files
    logs = []
    paths = Paths(tmp_path / "config", tmp_path / "data", tmp_path / "cache")
    config = Config(library_folders=[src])
    crashing = Session(paths, config, lambda m, level="info": logs.append((level, m)), jobs=_CrashingRunner())
    crashing.scan()
    crashing.poll()
    assert [t.status for t in crashing.tracks] == ["failed"] * 3
    assert all("worker crashed" in t.error for t in crashing.tracks)

    healthy = make_session(tmp_path, [src])
    assert [t.status for t in healthy.tracks] == ["failed"] * 3  # persisted
    healthy.scan()
    healthy.poll()
    assert [t.status for t in healthy.tracks] == ["failed"] * 3  # a plain scan does not retry
    healthy.scan(retry=True)
    healthy.poll()
    assert [t.status for t in healthy.tracks] == ["ready"] * 3
