"""Analysis accuracy on synthetic audio (docs/TASK_real-audio.md 11: BPM
within +-0.1, first beat within +-10 ms)."""
import numpy as np
import pytest

from clidj import analysis, synth
from clidj.library import phrase_cues

SR = analysis.ANALYSIS_SR


def mono(audio):
    return analysis.to_mono(audio)


@pytest.mark.parametrize("kind", ["click", "kick"])
def test_grid_of_a_124_bpm_click_track_starting_at_370_ms(kind):
    audio = synth.click_track(124.0, 0.37, 60.0, SR, kind=kind)
    bpm, first_beat = analysis.estimate_grid(mono(audio), SR)
    assert bpm == pytest.approx(124.0, abs=0.1)
    assert first_beat == pytest.approx(0.37, abs=0.010)


@pytest.mark.parametrize("bpm, first_beat", [(128.0, 0.05), (126.3, 0.25), (140.0, 1.2)])
def test_grid_other_tempi(bpm, first_beat):
    audio = synth.click_track(bpm, first_beat, 60.0, SR, kind="kick", seed=3)
    est_bpm, est_first = analysis.estimate_grid(mono(audio), SR)
    assert est_bpm == pytest.approx(bpm, abs=0.1)
    period = 60.0 / bpm
    # a grid that is correct but starts a whole beat later is still wrong
    assert est_first == pytest.approx(first_beat, abs=0.010), (est_first, first_beat, period)


def test_leading_silence_does_not_put_beat_zero_in_the_silence():
    audio = synth.click_track(125.0, 3.0, 40.0, SR, kind="kick")
    _, first_beat = analysis.estimate_grid(mono(audio), SR)
    assert first_beat == pytest.approx(3.0, abs=0.010)


def test_quiet_kick_only_intro_is_still_the_start_of_the_grid():
    """Regression: a first-beat heuristic based on onset strength skipped the
    16-bar kick-only intro whenever the later sections were busier."""
    audio = synth.techno_loop_track(128.0, key_root=2, minor=True, duration=70.0, sr=SR, first_beat=0.2)
    bpm, first_beat = analysis.estimate_grid(mono(audio), SR)
    assert bpm == pytest.approx(128.0, abs=0.1)
    assert first_beat == pytest.approx(0.2, abs=0.010)


def test_tempo_is_folded_into_the_configured_range():
    audio = synth.click_track(87.0, 0.2, 60.0, SR, kind="click")
    bpm, _ = analysis.estimate_grid(mono(audio), SR, bpm_range=(88.0, 176.0))
    assert 88.0 <= bpm < 176.0
    assert bpm == pytest.approx(174.0, abs=0.2)


@pytest.mark.parametrize("pitch_class, minor, expected", [(9, True, "8A"), (0, False, "8B"), (7, False, "9B"),
                                                          (4, True, "9A"), (5, True, "4A")])
def test_camelot_mapping(pitch_class, minor, expected):
    assert analysis.camelot(pitch_class, minor) == expected


@pytest.mark.parametrize("root, minor, expected", [(9, True, "8A"), (2, False, "10B"), (0, True, "5A")])
def test_key_of_a_synthetic_key_track(root, minor, expected):
    audio = synth.key_track(root, minor, 20.0, SR)
    assert analysis.estimate_key(mono(audio), SR) == expected


def test_integrated_loudness_of_a_stereo_minus_23_dbfs_sine_is_minus_23_lufs():
    amplitude = 10 ** (-23 / 20)
    audio = synth.sine(997.0, 20.0, 48000, amplitude=amplitude, channels=2)
    assert analysis.integrated_loudness(audio, 48000) == pytest.approx(-23.0, abs=0.1)


def test_integrated_loudness_full_scale_sine_in_one_channel():
    audio = synth.sine(997.0, 10.0, 44100, amplitude=1.0, channels=2)
    audio[:, 1] = 0.0
    assert analysis.integrated_loudness(audio, 44100) == pytest.approx(-3.01, abs=0.1)


def test_silence_has_no_loudness():
    assert analysis.integrated_loudness(np.zeros((48000 * 2, 2), dtype=np.float32), 48000) is None


def test_beat_waveform_puts_each_click_at_the_start_of_its_beat():
    audio = synth.click_track(120.0, 0.5, 20.0, SR, kind="click", noise=0.0)
    env = analysis.peak_envelope(mono(audio))
    wave = analysis.beat_waveform(env, SR / analysis.ENVELOPE_HOP, 120.0, 0.5, samples_per_beat=16)
    wave = np.array(wave)
    # the envelope drops a partial last hop, so the grid may lose its last slot
    assert abs(len(wave) - int((20.0 - 0.5) * 2 * 16)) <= 1
    beat_heads = wave[::16]
    off_beat = wave[8::16]
    assert beat_heads.min() > 0.5
    assert off_beat.max() < 0.05


def test_phrase_cues_snap_to_bars_and_fit_the_track():
    cues = phrase_cues(duration_beats=50.0)
    assert cues[1] == 0.0
    assert all(beat % 4 == 0 for beat in cues.values())
    assert all(beat <= 50.0 for beat in cues.values())
    assert phrase_cues(400.0) == {1: 0.0, 2: 32.0, 3: 64.0, 4: 96.0}


def test_analyze_file_end_to_end(tmp_path):
    audio = synth.techno_loop_track(126.0, key_root=9, minor=True, duration=50.0, sr=44100, first_beat=0.3)
    path = synth.write_audio(tmp_path / "loop.flac", audio, 44100)
    result = analysis.analyze_file(path)
    assert result.samplerate == 44100
    assert result.channels == 2
    assert result.duration == pytest.approx(50.0, abs=0.01)
    assert result.bpm == pytest.approx(126.0, abs=0.1)
    assert result.first_beat == pytest.approx(0.3, abs=0.010)
    assert result.lufs is not None and -30 < result.lufs < 0
    assert len(result.envelope) == pytest.approx(50.0 * result.envelope_rate, rel=0.01)
    restored = analysis.TrackAnalysis.from_json(result.to_json(), result.envelope)
    assert restored.to_json() == result.to_json()
    assert np.array_equal(restored.envelope, result.envelope)
