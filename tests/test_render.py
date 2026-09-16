"""Snippet rendering accuracy (docs/TASK_real-audio.md 11: exact length,
clicks within +-3 ms after stretching, sample-exact without stretching, no
clicks at loop seams or snippet edges)."""
import json

import numpy as np
import pytest

from clidj import render, synth
from clidj.render import RenderSpec, render_snippet, render_to_cache

SR = 48000


def click_file(tmp_path, bpm, first_beat, seconds, sr=SR, name="clicks.wav", kind="click"):
    audio = synth.click_track(bpm, first_beat, seconds, sr, kind=kind, noise=0.0)
    return synth.write_audio(tmp_path / name, audio, sr), audio


def spec_for(path, source_bpm, first_beat, start_beat, length_beats, loop, set_bpm, sr=SR, track_id="t"):
    return RenderSpec(track_id, str(path), source_bpm, first_beat, start_beat, length_beats, loop, set_bpm, sr)


def onset_errors_ms(buffer, set_bpm, beats, sr=SR, window_ms=25):
    """For each beat k, how far (ms) the first sample above 30% of the peak
    is from k * 60/set_bpm seconds into the buffer."""
    env = np.abs(buffer).max(axis=1)
    threshold = 0.3 * env.max()
    errors = []
    for k in beats:
        expected = int(round(k * 60.0 / set_bpm * sr))
        lo = max(0, expected - int(window_ms * sr / 1000))
        hi = min(len(env), expected + int(window_ms * sr / 1000))
        hits = np.nonzero(env[lo:hi] > threshold)[0]
        errors.append(None if len(hits) == 0 else (lo + hits[0] - expected) / sr * 1000)
    return errors


# ---- length and timing -------------------------------------------------------------


def test_eight_bars_of_124_bpm_rendered_at_128_have_the_exact_length(tmp_path):
    path, _ = click_file(tmp_path, 124.0, 0.37, 30.0)
    spec = spec_for(path, 124.0, 0.37, start_beat=8, length_beats=32, loop=False, set_bpm=128.0)
    buffer = render_snippet(spec)
    assert buffer.shape == (round(32 * 60 / 128 * SR), 2) == (720000, 2)
    assert buffer.dtype == np.float32


@pytest.mark.parametrize("source_bpm, set_bpm", [(124.0, 128.0), (128.0, 124.0), (120.0, 132.0), (140.0, 128.0)])
@pytest.mark.parametrize("kind", ["click", "kick"])
def test_clicks_land_on_the_set_bpm_grid_within_3_ms(tmp_path, source_bpm, set_bpm, kind):
    path, _ = click_file(tmp_path, source_bpm, 0.37, 40.0, kind=kind)
    spec = spec_for(path, source_bpm, 0.37, start_beat=4, length_beats=32, loop=False, set_bpm=set_bpm)
    buffer = render_snippet(spec)
    errors = onset_errors_ms(buffer, set_bpm, beats=range(1, 32))
    assert None not in errors
    assert max(abs(e) for e in errors) <= 3.0, errors


def test_resampled_44k1_source_keeps_length_and_timing(tmp_path):
    path, _ = click_file(tmp_path, 126.0, 0.25, 30.0, sr=44100, name="clicks44.flac")
    spec = spec_for(path, 126.0, 0.25, start_beat=0, length_beats=16, loop=False, set_bpm=128.0)
    buffer = render_snippet(spec)
    assert buffer.shape[0] == round(16 * 60 / 128 * SR)
    errors = onset_errors_ms(buffer, 128.0, beats=range(1, 16))
    assert max(abs(e) for e in errors) <= 3.0, errors


def test_same_bpm_render_is_sample_identical_to_the_source(tmp_path):
    rng = np.random.default_rng(5)
    source = (rng.standard_normal((SR * 20, 2)) * 0.2).astype(np.float32)
    path = synth.write_audio(tmp_path / "noise.wav", source, SR)
    from clidj.analysis import read_audio

    source, _ = read_audio(path)  # compare against what the file holds (16-bit PCM), not the float input
    # 120 BPM -> 24000 samples per beat; beat 0 at 0.5 s -> snippet starts at sample 12000 + 8 * 24000
    spec = spec_for(path, 120.0, 0.5, start_beat=8, length_beats=16, loop=False, set_bpm=120.0)
    buffer = render_snippet(spec)
    start = int(0.5 * SR) + 8 * 24000
    fade = int(round(render.EDGE_FADE_SECONDS * SR))
    assert buffer.shape[0] == 16 * 24000
    np.testing.assert_array_equal(buffer[fade:-fade], source[start + fade:start + 16 * 24000 - fade])


def test_snippet_starting_before_the_file_start_is_padded_not_shifted(tmp_path):
    path, _ = click_file(tmp_path, 120.0, 0.0, 10.0)
    spec = spec_for(path, 120.0, 0.0, start_beat=0, length_beats=8, loop=True, set_bpm=120.0)
    buffer = render_snippet(spec)
    assert buffer.shape[0] == 8 * 24000
    errors = onset_errors_ms(buffer, 120.0, beats=range(1, 8))
    assert max(abs(e) for e in errors) < 0.1


# ---- seams and edges -----------------------------------------------------------------------


def max_natural_step(freq, amplitude, sr=SR):
    return 2 * np.pi * freq / sr * amplitude


def seam_sine(tmp_path, amplitude=0.5):
    """A sine that lands exactly half a cycle out of phase at the loop end
    of a 16-beat 127 BPM snippet starting at beat 4, peaking at the start:
    the worst case for a hard loop."""
    n = round(16 * 60 / 127 * SR)
    freq = (3326 + 0.5) * SR / n  # ~440 Hz, a half-integer number of cycles per loop
    start = round((0.1 + 4 * 60 / 127) * SR)
    phase = np.pi / 2 - 2 * np.pi * freq * start / SR
    source = synth.sine(freq, 30.0, SR, amplitude=amplitude, phase=phase)
    return synth.write_audio(tmp_path / "sine.wav", source, SR), freq


def seam_steps(buffer):
    looped = np.concatenate([buffer, buffer, buffer])[:, 0]
    return [abs(float(looped[len(buffer) * i]) - float(looped[len(buffer) * i - 1])) for i in (1, 2)]


@pytest.mark.parametrize("set_bpm", [127.0, 131.0])
def test_loop_seam_has_no_click(tmp_path, set_bpm):
    amplitude = 0.5
    path, freq = seam_sine(tmp_path, amplitude)
    spec = spec_for(path, 127.0, 0.1, start_beat=4, length_beats=16, loop=True, set_bpm=set_bpm)
    limit = 2.0 * max_natural_step(freq, amplitude)
    assert max(seam_steps(render_snippet(spec))) <= limit


def test_loop_seam_check_would_catch_a_hard_loop(tmp_path, monkeypatch):
    """The seam test above is only meaningful if a loop without the
    crossfade actually fails it."""
    amplitude = 0.5
    path, freq = seam_sine(tmp_path, amplitude)
    monkeypatch.setattr(render, "LOOP_XFADE_SECONDS", 0.0)
    spec = spec_for(path, 127.0, 0.1, start_beat=4, length_beats=16, loop=True, set_bpm=127.0)
    assert max(seam_steps(render_snippet(spec))) > 10 * max_natural_step(freq, amplitude)


def test_non_looping_snippet_edges_fade_to_near_silence(tmp_path):
    amplitude = 0.5
    source = synth.sine(441.0, 30.0, SR, amplitude=amplitude)
    path = synth.write_audio(tmp_path / "sine.wav", source, SR)
    buffer = render_snippet(spec_for(path, 127.0, 0.1, 4, 16, loop=False, set_bpm=129.0))
    limit = 2.0 * max_natural_step(441.0, amplitude)
    assert abs(buffer[0, 0]) <= limit  # the step up from silence
    assert abs(buffer[-1, 0]) <= limit  # the step down to silence
    assert buffer[0, 0] != 0.0  # but the snippet does start on its first sample
    steps = np.abs(np.diff(buffer[:, 0]))
    assert steps.max() <= limit


# ---- cache ----------------------------------------------------------------------------------


def test_cache_key_covers_what_changes_the_samples_and_nothing_else(tmp_path):
    base = spec_for("a.wav", 124.0, 0.37, 8, 32, False, 128.0)
    variants = [
        spec_for("a.wav", 124.1, 0.37, 8, 32, False, 128.0),
        spec_for("a.wav", 124.0, 0.38, 8, 32, False, 128.0),
        spec_for("a.wav", 124.0, 0.37, 12, 32, False, 128.0),
        spec_for("a.wav", 124.0, 0.37, 8, 16, False, 128.0),
        spec_for("a.wav", 124.0, 0.37, 8, 32, True, 128.0),
        spec_for("a.wav", 124.0, 0.37, 8, 32, False, 130.0),
        spec_for("a.wav", 124.0, 0.37, 8, 32, False, 128.0, sr=44100),
        spec_for("a.wav", 124.0, 0.37, 8, 32, False, 128.0, track_id="other"),
    ]
    keys = {base.cache_key()} | {v.cache_key() for v in variants}
    assert len(keys) == len(variants) + 1
    moved = spec_for("elsewhere/a.wav", 124.0, 0.37, 8, 32, False, 128.0)
    assert moved.cache_key() == base.cache_key()
    from dataclasses import replace

    assert replace(base, version=base.version + 1).cache_key() != base.cache_key()


def test_render_to_cache_writes_an_mmappable_npy_once(tmp_path):
    path, _ = click_file(tmp_path, 124.0, 0.37, 20.0)
    spec = spec_for(path, 124.0, 0.37, 0, 8, True, 128.0)
    first = render_to_cache(spec, tmp_path / "renders")
    assert not first["cached"]
    mapped = np.load(first["path"], mmap_mode="r")
    assert mapped.dtype == np.float32 and mapped.shape == (spec.length_samples, 2)
    assert json.loads((tmp_path / "renders" / f"{first['key']}.json").read_text())["frames"] == spec.length_samples
    del mapped
    second = render_to_cache(spec, tmp_path / "renders")
    assert second["cached"] and second["path"] == first["path"]


def test_loudness_gain():
    assert render.loudness_gain(-14.0, -14.0) == pytest.approx(1.0)
    assert render.loudness_gain(-8.0, -14.0) == pytest.approx(10 ** (-6 / 20))
    assert render.loudness_gain(-60.0, -14.0) == pytest.approx(10 ** (12 / 20))  # boost is capped
    assert render.loudness_gain(None, -14.0) == 1.0
