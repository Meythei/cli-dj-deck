"""Realtime engine process and lookahead scheduling (docs/TASK_real-audio.md
9, 11 "リアルタイム"). Everything runs on the NullBackend: no audio device."""
import time
from pathlib import Path

import numpy as np
import pytest

from clidj.config import Config, Paths
from clidj.engine import commands as c
from clidj.engine.core import EngineStatus, LaneStatus
from clidj.engine.host import HostConfig, RealtimeEngineClient
from clidj.interpreter import Interpreter
from clidj.library import Track
from clidj.session import Session
from clidj.snippets import Snippet
from clidj.workers import InlineJobRunner


@pytest.fixture
def engine_process():
    client = RealtimeEngineClient(HostConfig(backend="null", bpm=120.0))
    client.start()
    yield client
    client.close()


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_engine_process_starts_runs_reports_and_shuts_down(engine_process, tmp_path):
    client = engine_process
    assert client.info["backend"] == "null"
    assert client.alive
    buffer = tmp_path / "b.npy"
    np.save(buffer, np.full((96000, 2), 0.1, dtype=np.float32))
    client.send(c.RegisterBuffer(c.BufferInfo(7, 120.0, str(buffer), 96000, 4.0, True)))
    client.send(c.Play(2, 7, beat=None))
    client.send(c.TransportStart())
    assert wait_for(lambda: client.status().position_beats > 1.0)
    status = client.status()
    assert status.running and status.samplerate == 48000
    assert status.lanes[2].uid == 7
    assert client.extra["callbacks"] > 10
    assert client.extra["underruns"] == 0
    pid = client.pid
    client.close()
    assert not client.alive
    assert not client._process.is_alive(), pid


def test_late_command_is_detected_and_counted(engine_process):
    client = engine_process
    client.send(c.RegisterBuffer(c.BufferInfo(1, 120.0, None, 48000, 2.0, True)))
    client.send(c.TransportStart())
    assert wait_for(lambda: client.status().position_beats > 1.0)
    client.send(c.Play(0, 1, beat=0.5))  # already in the past
    assert wait_for(lambda: client.status().stats.late_commands >= 1)
    assert client.status().stats.late_max_ms > 100


def test_killed_engine_does_not_take_the_ui_down(engine_process, tmp_path):
    client = engine_process
    logs = []
    session = Session(Paths(tmp_path / "c", tmp_path / "d", tmp_path / "k"), Config(),
                      lambda m, level="info": logs.append((level, m)), demo=True, jobs=InlineJobRunner(),
                      engine=client, bpm=120.0)
    interp = Interpreter(session, Path("."))
    interp.run("start()")
    client._process.kill()
    client._process.join(5)
    assert not client.alive
    for _ in range(3):
        session.tick(1 / 30)  # must not raise
    interp.run("L1.gain(0.5)")  # sends are dropped, not raised
    interp.run("stop()")
    assert client.dropped_commands >= 2
    assert sum("stopped responding" in m for level, m in logs if level == "error") == 1


class FakeRealtimeClient:
    """A realtime client whose engine position the test moves by hand."""

    realtime = True
    render_audio = True
    samplerate = 48000
    alive = True

    def __init__(self, bpm):
        self.sent = []
        self.position = 0.0
        self.bpm = bpm
        self.extra = {}

    def send(self, command):
        self.sent.append((self.position, command))

    def flush(self):
        pass

    def status(self):
        return EngineStatus(position_beats=self.position, heard_beats=self.position, bpm=self.bpm, running=True,
                            lanes=[LaneStatus() for _ in range(4)])

    def events(self):
        return []

    def beat_at_sample_offset(self, frames):
        return self.position + frames / self.samplerate * self.bpm / 60

    def close(self):
        pass


def test_reservations_are_evaluated_only_inside_the_lookahead_window(tmp_path):
    fake = FakeRealtimeClient(bpm=120.0)  # lookahead 200 ms = 0.4 beats
    session = Session(Paths(tmp_path / "c", tmp_path / "d", tmp_path / "k"), Config(lookahead_ms=200.0),
                      lambda *a: None, demo=True, jobs=InlineJobRunner(), engine=fake, bpm=120.0)
    interp = Interpreter(session, Path("."))
    track = Track(1, "T", "A", 120.0, "8A", 600.0)
    interp.env["s"] = session.register_snippet(Snippet("s", track, 0.0, 16.0, loop=True))
    evaluated = []
    interp.env["probe"] = None
    interp.run("start()")
    interp.run("at(2, L1 << s)")  # beat 4
    original_play = session.play
    session.play = lambda lane, snippet, beat: (evaluated.append((fake.position, beat)), original_play(lane, snippet, beat))

    for position in (0.5, 2.0, 3.55, 3.59):
        fake.position = position
        session.tick(1 / 30)
    assert evaluated == []  # 3.59 + 0.4 < 4.0

    fake.position = 3.61
    session.tick(1 / 30)
    assert evaluated == [(3.61, 4.0)]  # evaluated early, but for beat 4 exactly
    plays = [cmd for _, cmd in fake.sent if isinstance(cmd, c.Play)]
    assert plays == [c.Play(0, interp.env["s"].uid, 4.0)]

    fake.position = 4.2
    session.tick(1 / 30)
    assert len(evaluated) == 1


def test_session_on_a_real_engine_process_lands_reservations_on_their_beat(engine_process, tmp_path):
    client = engine_process
    session = Session(Paths(tmp_path / "c", tmp_path / "d", tmp_path / "k"), Config(lookahead_ms=200.0),
                      lambda *a: None, demo=True, jobs=InlineJobRunner(), engine=client, bpm=120.0)
    interp = Interpreter(session, Path("."))
    track = Track(1, "T", "A", 120.0, "8A", 600.0)
    interp.env["s"] = session.register_snippet(Snippet("s", track, 0.0, 16.0, loop=True))
    interp.run("at(2, L3 << s)")
    interp.run("start()")
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline and session.transport.position_beats < 5.0:
        session.tick(1 / 30)
        time.sleep(1 / 30)
    status = client.status()
    assert status.lanes[2].uid == interp.env["s"].uid
    assert status.lanes[2].start_beat == 4.0
    assert status.stats.late_commands == 0
