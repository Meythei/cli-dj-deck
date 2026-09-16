"""Session <-> engine integration on the visual-only engine: crossfades,
lane views synced from engine status, cancel(). (The crossfade tests came
from test_scheduler.py when gain automation moved into the engine.)"""
from pathlib import Path

import pytest

from clidj.config import Config, Paths
from clidj.interpreter import Interpreter
from clidj.library import Track
from clidj.session import Session
from clidj.snippets import Snippet
from clidj.workers import InlineJobRunner


def make_session(tmp_path, bpm=60.0):
    logs: list[tuple[str, str]] = []
    session = Session(Paths(tmp_path / "c", tmp_path / "d", tmp_path / "k"), Config(),
                      lambda m, level="info": logs.append((level, m)), demo=True, jobs=InlineJobRunner(), bpm=bpm)
    return session, logs


def long_loop(session) -> Snippet:
    track = Track(1, "T", "A", 60.0, "8A", duration=600.0)
    return session.register_snippet(Snippet(name="s", track=track, start_beat=0.0, length_beats=64.0, loop=True))


def test_xf_gain_interpolates_start_mid_end_and_stops_from_lane(tmp_path):
    session, _ = make_session(tmp_path, bpm=60.0)  # 1 beat per second
    lane_a, lane_b = session.lanes["L1"], session.lanes["L2"]
    session.play(lane_a, long_loop(session), None)
    session.start()
    session.crossfade(lane_a, lane_b, start_beat=0.0, bars=4, description="xf")  # 16 beats = 16 s
    session.tick(0.001)
    assert lane_b.gain == pytest.approx(0.0, abs=1e-3)  # to_lane starts at 0 immediately

    session.tick(7.999)  # halfway
    assert lane_a.gain == pytest.approx(0.5, abs=1e-6)
    assert lane_b.gain == pytest.approx(0.5, abs=1e-6)
    assert lane_a.snippet is not None  # not stopped yet

    session.tick(8.0)  # end
    assert lane_a.gain == pytest.approx(0.0, abs=1e-6)
    assert lane_b.gain == pytest.approx(1.0, abs=1e-6)
    assert lane_a.snippet is None  # from_lane stops at the end
    assert session.crossfades == {}


def test_xf_overlap_on_same_lane_warns_and_replaces(tmp_path):
    session, logs = make_session(tmp_path)
    lane_a, lane_b, lane_c = session.lanes["L1"], session.lanes["L2"], session.lanes["L3"]
    session.start()
    session.crossfade(lane_a, lane_b, 0.0, bars=4, description="xf1")
    second = session.crossfade(lane_a, lane_c, 0.0, bars=4, description="xf2")
    assert any(level == "warn" for level, _ in logs)
    assert list(session.crossfades) == [second]
    session.tick(0.01)
    assert session.last_status.lanes[0].automation_id == second


def test_lane_views_follow_the_engine(tmp_path):
    session, _ = make_session(tmp_path, bpm=120.0)
    interp = Interpreter(session, Path("."))
    snippet = long_loop(session)
    interp.env["s"] = snippet
    interp.run("L3 << s")  # stopped: immediate
    assert session.lanes["L3"].snippet is snippet
    interp.run("L3.eq(lo=0.25)")
    interp.run("L3.mute()")
    status = session.last_status.lanes[2]
    assert status.lo == 0.25 and status.muted
    assert session.lanes["L3"].lo == 0.25 and session.lanes["L3"].muted
    interp.run("start()")
    interp.run("L3.stop()")  # queued for bar 2
    session.tick(1.9)
    assert session.lanes["L3"].snippet is snippet
    session.tick(0.2)
    assert session.lanes["L3"].snippet is None


def test_cancel_removes_a_pending_crossfade(tmp_path):
    session, _ = make_session(tmp_path)
    session.start()
    xf = session.crossfade(session.lanes["L1"], session.lanes["L2"], 4.0, bars=1, description="xf")
    assert session.cancel(xf) == f"cancelled #{xf}"
    session.tick(5.0)
    assert session.lanes["L2"].gain == pytest.approx(1.0)  # the fade never ran


def test_a_huge_stall_does_not_allocate_per_sample(tmp_path):
    """A visual-mode tick covering days of audio must be cheap (no per-sample
    arrays): the regression test for every() limits advances 100000 s."""
    session, _ = make_session(tmp_path, bpm=120.0)
    session.start()
    session.tick(10_000_000.0)
    assert session.transport.position_beats == pytest.approx(20_000_000.0)
