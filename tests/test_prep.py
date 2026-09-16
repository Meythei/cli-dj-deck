"""Snippet preparation in the session: states, holding plays until renders are
ready, and tempo changes that wait for renders (docs/TASK_real-audio.md 7)."""
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest

from clidj import prep as prep_states
from clidj import synth
from clidj.config import Config, Paths
from clidj.interpreter import Interpreter
from clidj.library import Track
from clidj.prep import camelot_to_key, render_job
from clidj.analysis import camelot
from clidj.session import Session
from clidj.workers import InlineJobRunner, JobRunner

SR_SOURCE = 48000


class ManualJobRunner(InlineJobRunner):
    """Process jobs wait until the test calls run_all(), like a slow worker."""

    def __init__(self):
        self.queue: list[tuple] = []

    def run_in_process(self, fn, *args):
        future = Future()
        self.queue.append((fn, args, future))
        return future

    def run_all(self):
        while self.queue:
            fn, args, future = self.queue.pop(0)
            try:
                future.set_result(fn(*args))
            except BaseException as exc:  # noqa: BLE001
                future.set_exception(exc)


@pytest.fixture(scope="module")
def click_source(tmp_path_factory):
    folder = tmp_path_factory.mktemp("src")
    audio = synth.click_track(124.0, 0.37, 60.0, SR_SOURCE, kind="kick")
    return synth.write_audio(folder / "clicks.wav", audio, SR_SOURCE)


def make(tmp_path, click_source, jobs, bpm=124.0):
    logs: list[tuple[str, str]] = []
    log = lambda message, level="info": logs.append((level, message))  # noqa: E731
    paths = Paths(tmp_path / "config", tmp_path / "data", tmp_path / "cache")
    session = Session(paths, Config(), log, demo=True, jobs=jobs, prepare=True, bpm=bpm)
    session.tracks[:] = [
        Track(1, "Clicks", "Test", 124.0, "8A", 60.0, first_beat=0.37, path=click_source, track_id="clicks", status="ready")
    ]
    interp = Interpreter(session, Path("."), log=log)
    return session, interp, logs


def seconds_for_beats(beats, bpm):
    return beats * 60.0 / bpm


def errors(logs):
    return [m for level, m in logs if level == "error"]


# ---- states ----------------------------------------------------------------------------


def test_snip_starts_a_render_that_becomes_ready(tmp_path, click_source):
    jobs = ManualJobRunner()
    session, interp, logs = make(tmp_path, click_source, jobs)
    interp.run("kick = snip(1, bar=2, bars=2, loop=True)")
    kick = interp.env["kick"]
    assert kick.uid > 0
    assert session.prep_state(kick) == prep_states.PENDING
    assert "rendering" in session.activity
    jobs.run_all()
    session.poll()
    assert session.prep_state(kick) == prep_states.READY
    buffer = session.prep.buffer(kick, 124.0)
    data = np.load(buffer.path, mmap_mode="r")
    assert data.shape == (round(8 * 60 / 124 * 48000), 2)


def test_prep_renders_everything_and_reports_completion(tmp_path, click_source):
    jobs = ManualJobRunner()
    session, interp, logs = make(tmp_path, click_source, jobs)
    interp.run("a = snip(1, bar=1, bars=1)")
    interp.run("b = snip(1, bar=3, bars=1, loop=True)")
    jobs.run_all()  # the renders started by snip()
    interp.run("bpm(126)")  # stopped transport, but renders at 126 are still missing
    assert session.pending_bpm == 126.0
    assert session.transport.bpm == 124.0
    interp.run("prep()")
    assert "rendering" in session.activity
    jobs.run_all()
    session.poll()
    assert session.transport.bpm == 126.0  # stopped: applied as soon as everything is ready
    assert session.prep_state(interp.env["a"]) == prep_states.READY
    assert any("prep(): 2/2" in m for _, m in logs)


# ---- holds ------------------------------------------------------------------------------


def test_unprepared_play_is_held_and_starts_on_the_first_boundary_after_ready(tmp_path, click_source):
    jobs = ManualJobRunner()
    session, interp, logs = make(tmp_path, click_source, jobs)
    interp.run("kick = snip(1, bar=2, bars=2, loop=True)")
    interp.run("start()")
    session.tick(seconds_for_beats(1.5, 124.0))
    interp.run("L1 << kick")  # queued for beat 4

    session.tick(seconds_for_beats(3.0, 124.0))  # past beat 4: the play fired but must hold
    assert session.lanes["L1"].snippet is None
    assert len(session.holds) == 1
    assert any(level == "warn" and "not prepared" in m for level, m in logs)

    session.tick(seconds_for_beats(3.0, 124.0))  # still rendering through beat 8
    assert session.lanes["L1"].snippet is None

    jobs.run_all()  # render lands at beat 7.5
    session.tick(1 / 30)
    assert session.holds == []
    assert session.lanes["L1"].snippet is None  # not started mid-bar...
    session.tick(seconds_for_beats(1.0, 124.0))
    assert session.lanes["L1"].snippet is interp.env["kick"]  # ...but on the next bar head
    assert session.lanes["L1"].started_at_beat == pytest.approx(8.0)


def test_prepared_reservation_plays_exactly_on_its_bar(tmp_path, click_source):
    jobs = ManualJobRunner()
    session, interp, logs = make(tmp_path, click_source, jobs)
    interp.run("kick = snip(1, bar=2, bars=2, loop=True)")
    jobs.run_all()
    session.poll()
    interp.run("at(3, L1 << kick)")
    interp.run("start()")
    for _ in range(200):
        session.tick(1 / 30)
    assert session.holds == []
    assert session.lanes["L1"].started_at_beat == pytest.approx(8.0)


def test_held_play_with_stopped_transport_starts_immediately_once_ready(tmp_path, click_source):
    jobs = ManualJobRunner()
    session, interp, logs = make(tmp_path, click_source, jobs)
    interp.run("kick = snip(1, bar=2, bars=2, loop=True)")
    interp.run("L1 << kick")
    assert session.lanes["L1"].snippet is None
    jobs.run_all()
    session.poll()
    assert session.lanes["L1"].snippet is interp.env["kick"]


def test_failed_render_drops_the_held_play_with_an_error(tmp_path, click_source):
    jobs = ManualJobRunner()
    session, interp, logs = make(tmp_path, click_source, jobs)
    session.tracks[0].path = tmp_path / "missing.wav"
    interp.run("kick = snip(1, bar=2, bars=2, loop=True)")
    interp.run("L1 << kick")
    jobs.run_all()
    session.poll()
    session.poll()  # a failure must not be resubmitted every poll
    assert jobs.queue == []
    assert session.holds == []
    assert session.lanes["L1"].snippet is None
    assert any("dropped" in m for m in errors(logs))
    assert session.prep_state(interp.env["kick"]) == prep_states.FAILED


# ---- tempo changes -------------------------------------------------------------------------


def test_bpm_change_waits_for_renders_then_switches_on_a_bar_head(tmp_path, click_source):
    jobs = ManualJobRunner()
    session, interp, logs = make(tmp_path, click_source, jobs)
    interp.run("kick = snip(1, bar=2, bars=2, loop=True)")
    jobs.run_all()
    session.poll()
    interp.run("L1 << kick")  # stopped: immediate, already prepared
    interp.run("start()")
    session.tick(seconds_for_beats(1.0, 124.0))
    interp.run("bpm(128)")
    assert session.pending_bpm == 128.0
    for _ in range(3):
        session.tick(seconds_for_beats(2.0, session.transport.bpm))
    assert session.transport.bpm == 124.0  # beat 7: renders at 128 are not done yet
    assert "bpm 128" in session.activity

    jobs.run_all()
    session.tick(1 / 60)  # ready inside bar 2 -> scheduled for bar 3 (beat 8)
    assert session.pending_bpm is None
    assert any("scheduled for bar 3" in m for _, m in logs)
    assert session.transport.bpm == 124.0
    session.tick(seconds_for_beats(1.0, 124.0))  # crosses beat 8
    assert session.transport.bpm == 128.0
    # the lane keeps its beat position across the switch
    assert session.lanes["L1"].started_at_beat == 0.0


def test_bpm_change_with_everything_prepared_goes_to_the_next_bar(tmp_path, click_source):
    jobs = ManualJobRunner()
    session, interp, logs = make(tmp_path, click_source, jobs)
    interp.run("kick = snip(1, bar=2, bars=2, loop=True)")
    session.prep.request(interp.env["kick"], 128.0)
    jobs.run_all()
    session.poll()
    interp.run("start()")
    session.tick(seconds_for_beats(1.0, 124.0))
    interp.run("bpm(128)")
    assert session.pending_bpm is None
    assert any("scheduled for bar 2" in m for _, m in logs)


def test_snippet_defined_while_a_bpm_change_is_pending_renders_at_both_tempi(tmp_path, click_source):
    jobs = ManualJobRunner()
    session, interp, logs = make(tmp_path, click_source, jobs)
    interp.run("a = snip(1, bar=1, bars=1)")
    interp.run("bpm(130)")
    interp.run("b = snip(1, bar=2, bars=1)")
    jobs.run_all()
    session.poll()
    for name in ("a", "b"):
        assert session.prep_state(interp.env[name], 124.0) == prep_states.READY
        assert session.prep_state(interp.env[name], 130.0) == prep_states.READY


# ---- demo audio and real workers ---------------------------------------------------------


def test_camelot_round_trip():
    for pc in range(12):
        for minor in (False, True):
            assert camelot_to_key(camelot(pc, minor)) == (pc, minor)


def test_demo_snippet_synthesises_its_track_then_renders(tmp_path):
    logs = []
    paths = Paths(tmp_path / "config", tmp_path / "data", tmp_path / "cache")
    session = Session(paths, Config(), lambda m, level="info": logs.append((level, m)), demo=True,
                      jobs=InlineJobRunner(), prepare=True)
    track = session.tracks[0]
    track.duration = 40.0  # keep the synthesis short for the test
    interp = Interpreter(session, Path("."), log=lambda m, level="info": logs.append((level, m)))
    fake_waveform = list(track.waveform)
    interp.run("kick = snip(1, bar=1, bars=2, loop=True)")
    assert session.prep_state(interp.env["kick"]) == prep_states.PENDING
    session.poll()  # source job done -> render submitted (inline) -> ready on this poll's request
    session.poll()
    assert track.path is not None and track.path.exists()
    assert track.lufs is not None
    assert track.waveform != fake_waveform
    assert session.prep_state(interp.env["kick"]) == prep_states.READY
    assert errors(logs) == []


def test_render_job_runs_in_a_real_worker_process(tmp_path, click_source):
    from clidj.render import RenderSpec

    spec = RenderSpec("clicks", str(click_source), 124.0, 0.37, 4.0, 8.0, True, 128.0, 48000)
    runner = JobRunner(workers=1, log_path=tmp_path / "workers.log")
    try:
        result = runner.run_in_process(render_job, spec.to_dict(), str(tmp_path / "renders")).result(timeout=120)
    finally:
        runner.shutdown()
    assert result["ok"], result
    assert np.load(result["path"], mmap_mode="r").shape == (spec.length_samples, 2)
