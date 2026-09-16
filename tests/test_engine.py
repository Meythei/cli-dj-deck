"""Offline engine accuracy (docs/TASK_real-audio.md 11, "エンジン(オフライン)").

Buffers are built directly as arrays here (no renderer) so every expected
sample position is known exactly."""
import math

import numpy as np
import pytest

from clidj.engine import commands as c
from clidj.engine.core import DECLICK_IN, Engine, LoadedBuffer, load_buffer
from clidj.tempo import TempoMap, event_sample

SR = 48000
BLOCK = 512


# ---- helpers ------------------------------------------------------------------------------


def add_buffer(engine, tmp_path, uid, bpm, audio, length_beats, loop, gain=1.0):
    path = tmp_path / f"buf-{uid}-{bpm:g}.npy"
    np.save(path, np.ascontiguousarray(audio, dtype=np.float32))
    info = c.BufferInfo(uid, bpm, str(path), len(audio), length_beats, loop, gain)
    engine.submit(load_buffer(info))
    return info


def sine(freq, frames, amplitude=0.5, phase=np.pi / 2):
    """Starts at its peak by default, so "first non-zero sample" is the start."""
    t = np.arange(frames) / SR
    mono = amplitude * np.sin(2 * np.pi * freq * t + phase)
    return np.repeat(mono[:, None], 2, axis=1).astype(np.float32)


def clicks(bpm, beats, width=16):
    """A short decaying burst on every beat of a buffer rendered at `bpm`.
    (Not a single sample: when a loop is a fractional number of samples long,
    one index per loop is skipped or repeated at the wrap.)"""
    frames = round(beats * 60 / bpm * SR)
    audio = np.zeros((frames, 2), dtype=np.float32)
    burst = 0.9 * np.exp(-np.arange(width) / 4.0)
    for k in range(beats):
        start = round(k * 60 / bpm * SR)
        audio[start:start + width] = burst[: frames - start, None]
    return audio


class Capture:
    """Pulls blocks from an engine and returns output aligned to the transport
    (the limiter's lookahead delay removed once, at the start), like the
    offline renderer does."""

    def __init__(self, engine, block=BLOCK):
        self.engine = engine
        self.block = block
        self.skip = engine.master.latency if engine.master else 0
        self.buffered = np.zeros((0, 2), dtype=np.float32)

    def run(self, frames):
        chunks, have = [self.buffered], len(self.buffered)
        while have < frames + self.skip:
            block = self.engine.process(self.block)
            chunks.append(block)
            have += len(block)
        self.buffered = np.concatenate(chunks)
        if self.skip:
            self.buffered = self.buffered[self.skip:]
            self.skip = 0
        out, self.buffered = self.buffered[:frames], self.buffered[frames:]
        return out


_captures = {}


def run(engine, frames, block=BLOCK):
    capture = _captures.get(id(engine))
    if capture is None or capture.engine is not engine:
        capture = _captures[id(engine)] = Capture(engine, block)
    return capture.run(frames)


def first_nonzero(audio):
    idx = np.nonzero(np.any(audio != 0.0, axis=1))[0]
    return int(idx[0]) if len(idx) else None


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


# ---- tempo map ------------------------------------------------------------------------------


def test_tempo_map_beats_and_samples_round_trip_across_changes():
    tempo = TempoMap(128.0, SR)
    assert tempo.beat_to_sample(8.0) == 180000.0
    tempo.set_tempo(16.0, 140.0)
    assert tempo.beat_to_sample(16.0) == 360000.0
    assert tempo.beat_to_sample(24.0) == pytest.approx(360000 + 8 * 60 / 140 * SR)
    for beat in (0.0, 3.3, 16.0, 16.01, 99.5):
        assert tempo.sample_to_beat(tempo.beat_to_sample(beat)) == pytest.approx(beat)
    assert tempo.bpm_at_beat(15.99) == 128.0 and tempo.bpm_at_beat(16.0) == 140.0
    tempo.set_tempo(12.0, 120.0)  # replaces the later change
    assert [s.bpm for s in tempo.segments] == [128.0, 120.0]


# ---- sample-accurate starts -----------------------------------------------------------------------


@pytest.mark.parametrize("lanes", [(0,), (1,), (0, 1)])
def test_snippets_scheduled_for_bar_3_start_exactly_at_sample_180000(tmp_path, lanes):
    engine = Engine(SR, 128.0)
    add_buffer(engine, tmp_path, 1, 128.0, clicks(128.0, 8), 8, loop=True)
    add_buffer(engine, tmp_path, 2, 128.0, sine(440.0, 180000), 8, loop=True)
    for lane in lanes:
        engine.submit(c.Play(lane, lane + 1, beat=8.0))  # bar 3 head
    engine.submit(c.TransportStart())
    out = run(engine, 200000)
    assert first_nonzero(out) == 180000
    assert np.all(out[:180000] == 0.0)


def test_start_lands_on_its_sample_whatever_the_block_size(tmp_path):
    starts = set()
    for block in (64, 500, 512, 1024):
        engine = Engine(SR, 124.0)  # 124 BPM: a beat is a fractional number of samples
        add_buffer(engine, tmp_path, 1, 124.0, sine(300.0, 100000), 4, loop=True)
        engine.submit(c.Play(0, 1, beat=5.0))
        engine.submit(c.TransportStart())
        starts.add(first_nonzero(Capture(engine, block).run(150000)))
    assert starts == {event_sample(5 * 60 / 124 * SR)}


def test_bpm_change_mid_play_keeps_beats_on_the_tempo_map(tmp_path):
    engine = Engine(SR, 128.0)
    add_buffer(engine, tmp_path, 1, 128.0, clicks(128.0, 4), 4, loop=True)
    add_buffer(engine, tmp_path, 1, 140.0, clicks(140.0, 4), 4, loop=True)
    engine.submit(c.Play(0, 1, beat=0.0))
    engine.submit(c.SetTempo(140.0, beat=16.0))
    engine.submit(c.TransportStart())
    out = run(engine, 900000)
    tempo = TempoMap(128.0, SR)
    tempo.set_tempo(16.0, 140.0)
    env = np.abs(out[:, 0])
    for beat in list(range(1, 16)) + list(range(17, 38)):
        expected = event_sample(tempo.beat_to_sample(beat))
        window = env[expected - 30:expected + 30]
        # the click (smeared a little by the EQ's allpass) peaks right at the grid
        assert abs(int(np.argmax(window)) - 30) <= 2, (beat, int(np.argmax(window)) - 30)
    assert engine.bpm == 140.0


def test_loops_do_not_drift_over_many_repeats_at_a_fractional_beat_length(tmp_path):
    engine = Engine(SR, 124.0)
    add_buffer(engine, tmp_path, 1, 124.0, clicks(124.0, 1), 1, loop=True)
    engine.submit(c.Play(0, 1, beat=0.0))
    engine.submit(c.TransportStart())
    out = Capture(engine, block=4096).run(round(130 * 60 / 124 * SR))
    env = np.abs(out[:, 0])
    for beat in (1, 40, 80, 120, 129):
        expected = event_sample(beat * 60 / 124 * SR)
        window = env[expected - 30:expected + 30]
        assert abs(int(np.argmax(window)) - 30) <= 2, beat


# ---- gains and automation ----------------------------------------------------------------------------


def test_crossfade_midpoint_gain_is_linear_within_0_01(tmp_path):
    engine = Engine(SR, 120.0)  # 24000 samples per beat
    add_buffer(engine, tmp_path, 1, 120.0, sine(1000.0, 96000), 4, loop=True)
    engine.submit(c.Play(0, 1, beat=0.0))
    engine.submit(c.Automate(1, 0, "gain", start_beat=8.0, end_beat=24.0, start_value=None, end_value=0.0,
                             stop_lane_at_end=True))
    engine.submit(c.TransportStart())
    out = run(engine, 26 * 24000)[:, 0]
    reference = rms(out[4 * 24000:8 * 24000])
    for beat, expected in ((12.0, 0.75), (16.0, 0.5), (20.0, 0.25)):
        centre = int(beat * 24000)
        window = out[centre - 480:centre + 480]  # 20 ms: 20 cycles, gain changes <0.2% across it
        assert rms(window) / reference == pytest.approx(expected, abs=0.01)
    assert rms(out[25 * 24000:]) == 0.0  # stopped at the end of the fade
    assert engine.status().lanes[0].uid == 0


def test_immediate_gain_change_has_no_zipper_noise(tmp_path):
    amplitude = 0.5
    engine = Engine(SR, 120.0)
    add_buffer(engine, tmp_path, 1, 120.0, sine(440.0, 96000, amplitude), 4, loop=True)
    engine.submit(c.Play(0, 1, beat=0.0))
    engine.submit(c.TransportStart())
    before = run(engine, 20000)
    engine.submit(c.SetParam(0, "gain", 0.0))
    engine.submit(c.SetParam(0, "mute", 1.0))
    after = run(engine, 20000)
    out = np.concatenate([before, after])[:, 0]
    natural = 2 * np.pi * 440.0 / SR * amplitude
    assert np.max(np.abs(np.diff(out))) <= natural * 1.2
    assert np.max(np.abs(after[2000:])) == 0.0  # and it does reach silence


def test_manual_gain_takes_over_from_a_running_automation(tmp_path):
    engine = Engine(SR, 120.0)
    add_buffer(engine, tmp_path, 1, 120.0, sine(440.0, 96000), 4, loop=True)
    engine.submit(c.Play(0, 1, beat=0.0))
    engine.submit(c.Automate(9, 0, "gain", 0.0, 32.0, None, 0.0))
    engine.submit(c.TransportStart())
    run(engine, 24000 * 8)
    assert engine.status().lanes[0].automation_id == 9
    engine.submit(c.SetParam(0, "gain", 1.0))
    run(engine, BLOCK)
    status = engine.status()
    assert status.lanes[0].automation_id == 0
    assert status.stats.automation_overrides == 1


def test_four_full_scale_lanes_never_exceed_full_scale(tmp_path):
    engine = Engine(SR, 120.0, limiter_ceiling_db=-1.0)
    for lane in range(4):
        add_buffer(engine, tmp_path, lane + 1, 120.0, sine(110.0, 96000, amplitude=0.99), 4, loop=True)
        engine.submit(c.Play(lane, lane + 1, beat=0.0))
    engine.submit(c.TransportStart())
    out = run(engine, SR * 3)
    assert np.max(np.abs(out)) <= 1.0
    assert np.max(np.abs(out)) <= 10 ** (-1 / 20) + 1e-3


@pytest.mark.parametrize("band, freq", [("lo", 60.0), ("mid", 1000.0), ("hi", 8000.0)])
def test_isolator_band_at_zero_removes_its_band(tmp_path, band, freq):
    engine = Engine(SR, 120.0)
    add_buffer(engine, tmp_path, 1, 120.0, sine(freq, 96000, 0.5), 4, loop=True)
    engine.submit(c.Play(0, 1, beat=0.0))
    engine.submit(c.SetParam(0, band, 0.0))
    engine.submit(c.TransportStart())
    out = run(engine, SR * 2)[SR:, 0]  # after the filters and the ramp settle
    ratio_db = 20 * math.log10(rms(out) / (0.5 / math.sqrt(2)))
    assert ratio_db <= -30.0, ratio_db


@pytest.mark.parametrize("freq", [40.0, 250.0, 1000.0, 2500.0, 10000.0])
def test_isolator_is_flat_with_all_bands_open(tmp_path, freq):
    engine = Engine(SR, 120.0)
    add_buffer(engine, tmp_path, 1, 120.0, sine(freq, 96000, 0.5), 4, loop=True)
    engine.submit(c.Play(0, 1, beat=0.0))
    engine.submit(c.TransportStart())
    out = run(engine, SR * 2)[SR:, 0]
    assert 20 * math.log10(rms(out) / (0.5 / math.sqrt(2))) == pytest.approx(0.0, abs=0.1)


# ---- voices ---------------------------------------------------------------------------------------------


def test_replacing_a_voice_on_a_lane_does_not_click(tmp_path):
    amplitude = 0.5
    engine = Engine(SR, 120.0)
    add_buffer(engine, tmp_path, 1, 120.0, sine(220.0, 96000, amplitude), 4, loop=True)
    add_buffer(engine, tmp_path, 2, 120.0, sine(330.0, 96000, amplitude, phase=np.pi / 2), 4, loop=True)
    engine.submit(c.Play(0, 1, beat=0.0))
    engine.submit(c.Play(0, 2, beat=4.3))  # mid-cycle of both
    engine.submit(c.TransportStart())
    out = run(engine, 24000 * 6)[:, 0]
    natural = 2 * np.pi * 330.0 / SR * amplitude
    seam = out[int(4.3 * 24000) - 500:int(4.3 * 24000) + 1000]
    assert np.max(np.abs(np.diff(seam))) <= natural * 1.5


def test_one_shot_ends_on_its_last_beat_and_frees_the_lane(tmp_path):
    engine = Engine(SR, 120.0)
    add_buffer(engine, tmp_path, 1, 120.0, sine(440.0, 48000), 2, loop=False)
    engine.submit(c.Play(0, 1, beat=1.0))
    engine.submit(c.TransportStart())
    engine.process(24000 * 3 - 1)  # transport time, not latency-aligned output
    assert engine.status().lanes[0].uid == 1
    engine.process(1)
    assert engine.status().lanes[0].uid == 0


def test_stop_and_resume_fade_instead_of_stepping(tmp_path):
    amplitude = 0.5
    engine = Engine(SR, 120.0)
    add_buffer(engine, tmp_path, 1, 120.0, sine(440.0, 96000, amplitude), 4, loop=True)
    engine.submit(c.Play(0, 1, beat=0.0))
    engine.submit(c.TransportStart())
    a = run(engine, 10000)
    position = engine.position_beats
    engine.submit(c.TransportStop())
    b = run(engine, 5000)
    assert engine.position_beats == position
    engine.submit(c.TransportStart())
    d = run(engine, 5000)
    out = np.concatenate([a, b, d])[:, 0]
    natural = 2 * np.pi * 440.0 / SR * amplitude
    assert np.max(np.abs(np.diff(out))) <= natural * 1.5
    assert np.max(np.abs(b[1000:4000])) < 1e-5  # silent apart from the EQ filters' decaying tail


def test_voice_start_ramp_begins_on_the_scheduled_sample(tmp_path):
    engine = Engine(SR, 120.0, render_audio=True)
    add_buffer(engine, tmp_path, 1, 120.0, np.full((96000, 2), 0.5, dtype=np.float32), 4, loop=True)
    engine.submit(c.Play(0, 1, beat=2.0))
    engine.submit(c.TransportStart())
    out = run(engine, 60000)[:, 0]
    assert out[47999] == 0.0 and out[48000] != 0.0
    assert out[48000 + DECLICK_IN + 2000] == pytest.approx(0.5, abs=1e-3)  # DC through the allpass EQ settles


def test_missing_buffer_does_not_start_silently(tmp_path):
    engine = Engine(SR, 120.0)
    engine.submit(c.Play(0, 42, beat=0.0))
    engine.submit(c.TransportStart())
    run(engine, 2048)
    status = engine.status()
    assert status.lanes[0].uid == 0
    assert status.stats.missing_buffers == 1


def test_late_command_applies_at_the_next_block_on_the_grid_and_is_counted(tmp_path):
    engine = Engine(SR, 120.0)
    add_buffer(engine, tmp_path, 1, 120.0, clicks(120.0, 4), 4, loop=True)
    engine.submit(c.TransportStart())
    run(engine, 30000)  # past beat 1 (24000)
    engine.submit(c.Play(0, 1, beat=1.0))
    out = run(engine, 30000)
    status = engine.status()
    assert status.stats.late_commands == 1
    assert status.stats.late_max_ms > 100
    env = np.abs(out[:, 0])
    # it stays on the grid: the next click is at beat 2 (48000), not 24000 samples after it started
    peak = int(np.argmax(env[48000 - 30000 - 50:48000 - 30000 + 50]))
    assert abs(peak - 50) <= 2


def test_limiter_latency_is_exactly_what_the_master_bus_reports(tmp_path):
    engine = Engine(SR, 120.0)
    impulse = np.zeros((24000, 2), dtype=np.float32)
    impulse[0] = 0.25
    add_buffer(engine, tmp_path, 1, 120.0, impulse, 1, loop=False)
    engine.submit(c.Play(0, 1, beat=1.0))
    engine.submit(c.TransportStart())
    raw = np.concatenate([engine.process(BLOCK) for _ in range(120)])
    assert first_nonzero(raw) == 24000 + engine.master.latency


# ---- visual mode ------------------------------------------------------------------------------------------


def test_visual_mode_runs_the_same_timeline_without_audio():
    engine = Engine(SR, 128.0, render_audio=False)
    info = c.BufferInfo(1, 128.0, None, 180000, 8.0, True)
    engine.submit(LoadedBuffer(info, None))
    engine.submit(c.Play(2, 1, beat=8.0))
    engine.submit(c.Automate(1, 2, "gain", 8.0, 12.0, 0.0, 1.0))
    engine.submit(c.TransportStart())
    assert engine.process(179999) is None
    assert engine.status().lanes[2].uid == 0
    engine.process(1 + 45000)  # to beat 10: halfway through the automation
    status = engine.status()
    assert status.lanes[2].uid == 1 and status.lanes[2].start_beat == 8.0
    assert status.lanes[2].gain == pytest.approx(0.5)
