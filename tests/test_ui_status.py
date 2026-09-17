"""Phase F UI integration (docs/TASK_real-audio.md 10): the audio status row,
a playhead that follows what is heard, lane notes for holds / queued /
late commands, and ruler alignment with analysed waveforms."""
import time
from pathlib import Path

import pytest

from clidj import synth
from clidj.config import Config, Paths
from clidj.engine.core import EngineStats, EngineStatus, LaneStatus
from clidj.engine.host import HostConfig, RealtimeEngineClient
from clidj.interpreter import Interpreter
from clidj.library import Track
from clidj.session import Hold, Session
from clidj.snippets import Snippet
from clidj.ui.app import LanesView
from clidj.workers import InlineJobRunner


class FakeRealtime:
    realtime = True
    render_audio = True
    samplerate = 48000
    alive = True

    def __init__(self, position=0.0, heard=None, bpm=120.0, late=0):
        self.position, self.heard, self.bpm, self.late = position, heard, bpm, late
        self.sent = []
        self.info = {"backend": "sounddevice", "device": "Speakers (Test)", "hostapi": "Windows WASAPI",
                     "blocksize": 512}
        self.extra = {"blocksize": 512, "callback_load_avg": 0.12, "callback_load_max": 0.61, "underruns": 2}

    def send(self, command):
        self.sent.append(command)

    def flush(self):
        pass

    def status(self):
        return EngineStatus(position_beats=self.position, bpm=self.bpm, running=True, samplerate=48000,
                            heard_beats=self.position if self.heard is None else self.heard, latency_samples=1440,
                            stats=EngineStats(late_commands=self.late, late_max_ms=12.0),
                            lanes=[LaneStatus() for _ in range(4)])

    def events(self):
        return []

    def beat_at_sample_offset(self, frames):
        return self.position

    def close(self):
        pass


def make(tmp_path, engine=None, prepare=False, jobs=None):
    logs = []
    session = Session(Paths(tmp_path / "c", tmp_path / "d", tmp_path / "k"), Config(),
                      lambda m, level="info": logs.append((level, m)), demo=True, jobs=jobs or InlineJobRunner(),
                      engine=engine, bpm=120.0, prepare=prepare)
    interp = Interpreter(session, Path("."))
    view = LanesView(session.transport, session.lanes, lambda: interp.quant_mode, session=session)
    return session, interp, view, logs


def lines(view, width=150):
    return view.build_text(width).plain.split("\n")


def audio_row(view, width=160):
    """(text, styles used on it) of the AUDIO row only."""
    text = view.build_text(width)
    start = text.plain.index("\nAUDIO") + 1
    end = text.plain.index("\n", start)
    styles = {str(span.style) for span in text.spans if span.start < end and span.end > start}
    return text.plain[start:end], styles


def test_audio_row_says_off_in_visual_mode(tmp_path):
    session, interp, view, _ = make(tmp_path)
    assert lines(view)[1].startswith("AUDIO")
    assert "off" in lines(view)[1]


def test_audio_row_shows_device_format_latency_load_and_problems(tmp_path):
    fake = FakeRealtime(position=4.0, late=3)
    session, interp, view, _ = make(tmp_path, engine=fake)
    session.tick(1 / 30)
    audio, styles = audio_row(view)
    for piece in ("Speakers (Test)", "WASAPI", "48k/512", "lat 30ms", "cpu 12% (max 61%)", "xrun 2", "late 3"):
        assert piece in audio, (piece, audio)
    assert "bold yellow" not in styles  # 12% now, one 61% peak: not a sustained overload
    assert any("red" in style for style in styles)  # underruns / late
    fake.extra["callback_load_avg"] = 0.55
    audio, styles = audio_row(view)
    assert "cpu 55%" in audio and "bold yellow" in styles


def test_audio_row_reports_a_dead_engine(tmp_path):
    fake = FakeRealtime()
    session, interp, view, _ = make(tmp_path, engine=fake)
    fake.alive = False
    assert "ENGINE DOWN" in lines(view)[1]


def test_real_null_engine_fills_the_audio_row(tmp_path):
    client = RealtimeEngineClient(HostConfig(backend="null", bpm=120.0))
    client.start()
    try:
        session, interp, view, _ = make(tmp_path, engine=client)
        time.sleep(0.3)
        session.tick(1 / 30)
        audio = lines(view, 160)[1]
        assert "null backend" in audio and "48k/512" in audio and "xrun 0" in audio
    finally:
        client.close()


def test_playhead_and_ruler_follow_the_heard_position_not_the_engine_head(tmp_path):
    fake = FakeRealtime(position=10.0, heard=9.5)
    session, interp, view, _ = make(tmp_path, engine=fake)
    session.tick(1 / 30)
    text = lines(view)
    assert text[0].startswith("TRANSPORT  003.2")  # beat 9.5 = bar 3, beat 2 (not 003.3 at beat 10)
    ruler = text[2]
    centre = ruler.index("▼")
    # half a beat before a beat head: the next "·"/"|" is 2 columns right of the playhead
    assert ruler[centre + 2] in "·|"
    assert ruler[centre + 1] == " "


def test_lane_notes_show_holds_queued_commands_late_arrivals_and_crossfades(tmp_path):
    fake = FakeRealtime(position=5.0)
    session, interp, view, _ = make(tmp_path, engine=fake)
    track = Track(1, "T", "A", 120.0, "8A", 600.0)
    s = session.register_snippet(Snippet("kick", track, 0.0, 4.0, loop=True))
    interp.env["kick"] = s
    interp.run("start()")
    session.tick(1 / 30)

    interp.run("at(9, L2 << kick)")
    assert any("next @009.1: L2 << kick" in text for text, _ in session.lane_notes("L2"))

    session.play(session.lanes["L3"], s, 4.0)  # engine already at beat 5
    [(late_text, late_style)] = [n for n in session.lane_notes("L3") if n[0].startswith("LATE")]
    assert late_text == "LATE +500ms" and "red" in late_style

    session.holds.append(Hold(s, "L4 << kick", lambda: None))
    assert ("waiting for render: kick", "bold yellow") in session.lane_notes("L4")

    session.crossfade(session.lanes["L1"], session.lanes["L2"], 8.0, 2, "xf")
    assert any(text.startswith("xf L1->L2 @003.1") for text, _ in session.lane_notes("L1"))

    rows = "\n".join(lines(view, 200))
    for fragment in ("next @009.1: L2 << kick", "LATE +500ms", "waiting for render: kick", "xf L1->L2"):
        assert fragment in rows


def test_a_bar_head_closer_than_the_commit_margin_rolls_to_the_next_bar(tmp_path):
    """A command typed 20 ms before a bar head can't reach a realtime engine
    in time: it must go to the following bar, not arrive late."""
    fake = FakeRealtime(position=8.0 - 0.04)  # 120 BPM: 0.04 beats = 20 ms before bar 3
    session, interp, view, _ = make(tmp_path, engine=fake)
    track = Track(1, "T", "A", 120.0, "8A", 600.0)
    interp.env["kick"] = session.register_snippet(Snippet("kick", track, 0.0, 4.0, loop=True))
    interp.run("start()")
    session.tick(1 / 30)
    interp.run("L1 << kick")
    [(_, beat, _)] = session.scheduler.pending()
    assert beat == 12.0  # bar 4, not bar 3
    fake.position = 8.0 - 0.2  # 100 ms before: still reachable
    session.tick(1 / 30)
    interp.run("L2 << kick")
    assert sorted(b for _, b, _ in session.scheduler.pending()) == [8.0, 12.0]


def test_lane_row_shows_eq_and_mute(tmp_path):
    session, interp, view, _ = make(tmp_path)
    interp.run("L2.eq(lo=0)")
    interp.run("L2.mute()")
    row = next(line for line in lines(view, 160) if line.startswith("L2"))
    assert "eq ██" in row and "MUTE" in row


def test_pending_tempo_change_is_shown_in_the_transport_header(tmp_path):
    session, interp, view, _ = make(tmp_path)
    interp.run("start()")
    session.tick(0.3)
    interp.run("bpm(126)")
    assert "-> bpm(126) @ bar 2" in lines(view)[0]


def test_ruler_lines_up_with_an_analysed_click_track_waveform(tmp_path):
    """The waveform here comes from real analysis (envelope -> beat grid),
    not the fake demo generator."""
    music = tmp_path / "music"
    synth.write_audio(music / "Clicks - Grid.wav", synth.click_track(120.0, 0.3, 40.0, 44100, noise=0.0), 44100)
    logs = []
    session = Session(Paths(tmp_path / "c", tmp_path / "d", tmp_path / "k"), Config(library_folders=[music]),
                      lambda m, level="info": logs.append((level, m)), jobs=InlineJobRunner(), bpm=120.0)
    session.scan()
    session.poll()
    [track] = session.tracks
    assert track.status == "ready" and track.bpm == pytest.approx(120.0, abs=0.05)
    session.library.regrid(track, bpm=120.0, first_beat=0.3)  # exact grid: every click lands in its column
    interp = Interpreter(session, Path("."))
    interp.run("c = snip(1, bar=2, bars=4, loop=True)")
    interp.run("L1 << c")
    view = LanesView(session.transport, session.lanes, lambda: "bar", session=session)
    for lane in session.lanes.values():
        lane.gain = 0.0
    for width in (80, 121):
        text = lines(view, width)
        ruler = text[2]
        playhead = ruler.index("▼")
        beat_columns = {i for i, ch in enumerate(ruler) if ch in "|·"} | {playhead}
        spikes = set()
        for row in text[4:6]:  # L1's two waveform rows
            spikes |= {i for i, ch in enumerate(row) if ch == "█"}
        assert spikes == beat_columns
