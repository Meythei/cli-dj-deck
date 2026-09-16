"""End-to-end offline rendering: interpreter -> session -> prep -> engine ->
WAV (docs/TASK_real-audio.md 8 and 11)."""
from pathlib import Path

import numpy as np
import pytest

from clidj import synth
from clidj.analysis import read_audio
from clidj.config import Config, Paths
from clidj.interpreter import Interpreter
from clidj.library import Track
from clidj.offline import OfflineRenderer, ScriptError, build_session, main, parse_script
from clidj.tempo import TempoMap, event_sample

SR = 48000


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sources")
    a = synth.write_audio(folder / "a.wav", synth.click_track(128.0, 0.25, 60.0, SR, noise=0.0, seed=1), SR)
    b = synth.write_audio(folder / "b.wav", synth.click_track(128.0, 0.5, 60.0, SR, kind="kick", noise=0.0), SR)
    return a, b


def renderer_for(tmp_path, sources, set_text):
    logs = []
    paths = Paths(tmp_path / "config", tmp_path / "data", tmp_path / "cache")
    session = build_session(paths, Config(), lambda m, level="info": logs.append((level, m)), demo=True)
    a, b = sources
    session.tracks[:] = [
        Track(1, "A", "T", 128.0, "8A", 60.0, first_beat=0.25, path=a, track_id="src-a", status="ready"),
        Track(2, "B", "T", 128.0, "8A", 60.0, first_beat=0.5, path=b, track_id="src-b", status="ready"),
    ]
    interp = Interpreter(session, tmp_path)
    interp.run(set_text)
    return OfflineRenderer(session, interp), session, logs


def first_nonzero(audio):
    idx = np.nonzero(np.any(audio != 0.0, axis=1))[0]
    return int(idx[0]) if len(idx) else None


SNIPS = "a = snip(1, bar=1, bars=8, loop=True)\nb = snip(2, bar=1, bars=8, loop=True)\n"


@pytest.mark.parametrize("plays", [["at(3, L1 << a)"], ["at(3, L2 << b)"], ["at(3, L1 << a)", "at(3, L2 << b)"]])
def test_reservations_for_bar_3_start_at_sample_180000(tmp_path, sources, plays):
    renderer, session, logs = renderer_for(tmp_path, sources, SNIPS + "\n".join(plays))
    audio, report = renderer.render(bars=4)
    assert report.frames == 4 * 90000
    assert first_nonzero(audio) == 180000
    assert report.stats.late_commands == 0
    assert [m for level, m in logs if level == "error"] == []


def test_script_command_typed_mid_bar_plays_on_the_next_bar_head(tmp_path, sources):
    renderer, session, logs = renderer_for(tmp_path, sources, SNIPS)
    audio, _ = renderer.render(bars=4, script=parse_script("2.3: L1 << a"))
    assert first_nonzero(audio) == 180000


def test_bpm_change_waits_for_renders_and_then_follows_the_tempo_map(tmp_path, sources):
    renderer, session, logs = renderer_for(tmp_path, sources, SNIPS + "L1 << a\nat(5, bpm(140))\n")
    audio, report = renderer.render(bars=10)
    # bpm(140) fired at bar 5 with nothing rendered at 140 yet: it waits for
    # the renders, then switches at the next bar head (bar 6 = beat 20).
    assert any("rendering" in m for _, m in logs)
    tempo = TempoMap(128.0, SR)
    tempo.set_tempo(20.0, 140.0)
    assert report.frames == event_sample(tempo.beat_to_sample(40.0))
    env = np.abs(audio[:, 0])
    for beat in (18, 19, 21, 25, 30, 39):
        expected = event_sample(tempo.beat_to_sample(beat))
        window = env[expected - 200:expected + 200]
        onset = int(np.nonzero(window > 0.3 * window.max())[0][0]) - 200
        assert abs(onset) <= 0.003 * SR, (beat, onset)
    assert report.stats.missing_buffers == 0


def test_parse_script():
    commands = parse_script("# comment\n1: start()\n\n2.3: L1 << a  # typed late\n9.1.5: xf(L1, L2, 8)\n")
    assert [(c.beat, c.text) for c in commands] == [(0.0, "start()"), (6.0, "L1 << a"), (32.5, "xf(L1, L2, 8)")]
    for bad in ("L1 << a", "0: start()", "3.5: stop()"):
        with pytest.raises(ScriptError):
            parse_script(bad)


def test_render_cli_writes_the_demo_set(tmp_path, capsys):
    out = tmp_path / "demo.wav"
    sets = Path(__file__).resolve().parents[1] / "sets" / "demo.djs"
    code = main([str(sets), "--bars", "2", "-o", str(out), "--demo"])
    assert code == 0, capsys.readouterr()
    audio, sr = read_audio(out)
    assert sr == SR and audio.shape == (180000, 2)
    assert np.max(np.abs(audio)) > 0.1
    assert "late commands 0" in capsys.readouterr().out
