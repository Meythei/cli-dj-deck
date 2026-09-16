import pytest

from lanes import Lane
from library import Track
from scheduler import Scheduler, SchedulerError
from snippets import Snippet
from transport import Transport


def make_log():
    lines: list[tuple[str, str]] = []
    return lines, (lambda message, level="info": lines.append((level, message)))


def test_schedule_default_runs_immediately_when_stopped():
    transport = Transport(bpm=120.0)  # not started
    _, log = make_log()
    scheduler = Scheduler(transport, log)
    calls = []
    event_id = scheduler.schedule_default("bar", lambda: calls.append(1), "test")
    assert event_id == -1
    assert calls == [1]


def test_schedule_default_fires_only_at_next_bar():
    transport = Transport(bpm=120.0, beats_per_bar=4)
    transport.start()
    _, log = make_log()
    scheduler = Scheduler(transport, log)
    calls = []
    scheduler.schedule_default("bar", lambda: calls.append(1), "test")

    scheduler.tick(1.9)  # 3.8 beats: not yet at bar 2 (beat 4)
    assert calls == []

    scheduler.tick(0.1)  # 4.0 beats: crosses the bar head
    assert calls == [1]

    scheduler.tick(5.0)  # long past; must not fire again
    assert calls == [1]


def test_large_dt_fires_all_due_events_once_in_order():
    transport = Transport(bpm=60.0, beats_per_bar=4)  # 1 beat per second
    transport.start()
    _, log = make_log()
    scheduler = Scheduler(transport, log)
    order = []
    scheduler.schedule_at(2, lambda: order.append("a"), "a")
    scheduler.schedule_at(2, lambda: order.append("b"), "b")  # same beat, registered after "a"
    scheduler.schedule_at(5, lambda: order.append("c"), "c")
    scheduler.schedule_at(10, lambda: order.append("d"), "d")

    scheduler.tick(100.0)  # one huge jump past every boundary

    assert order == ["a", "b", "c", "d"]
    scheduler.tick(10.0)
    assert order == ["a", "b", "c", "d"]  # none fire twice


def test_past_target_is_rescheduled_to_next_boundary_with_warning():
    transport = Transport(bpm=60.0, beats_per_bar=4)
    transport.start()
    transport.position_beats = 10.0
    logs, log = make_log()
    scheduler = Scheduler(transport, log)
    calls = []
    scheduler.schedule_at(5, lambda: calls.append(1), "old")

    assert calls == []  # not run immediately, not an error
    assert any(level == "warn" for level, _ in logs)
    [(event_id, fire_at, desc)] = scheduler.pending()
    assert fire_at == pytest.approx(12.0)  # next bar head after beat 10


def test_cancel_removes_a_pending_event():
    transport = Transport(bpm=60.0, beats_per_bar=4)
    transport.start()
    _, log = make_log()
    scheduler = Scheduler(transport, log)
    calls = []
    event_id = scheduler.schedule_at(4, lambda: calls.append(1), "test")

    scheduler.cancel(event_id)
    scheduler.tick(10.0)
    assert calls == []
    assert scheduler.pending() == []


def test_cancel_unknown_id_raises():
    transport = Transport(bpm=60.0)
    _, log = make_log()
    scheduler = Scheduler(transport, log)
    with pytest.raises(SchedulerError):
        scheduler.cancel(999)


def test_cancel_all_clears_everything():
    transport = Transport(bpm=60.0, beats_per_bar=4)
    transport.start()
    _, log = make_log()
    scheduler = Scheduler(transport, log)
    scheduler.schedule_at(4, lambda: None, "a")
    scheduler.schedule_at(8, lambda: None, "b")
    scheduler.cancel()
    assert scheduler.pending() == []


def test_every_bars_repeats_until_cancelled():
    transport = Transport(bpm=60.0, beats_per_bar=4)  # 1 beat/sec, bar = 4 sec
    transport.start()
    _, log = make_log()
    scheduler = Scheduler(transport, log)
    calls = []
    event_id = scheduler.schedule_every_bars(1, lambda: calls.append(1), "every-1-bar")

    scheduler.tick(4.0)  # first bar head
    assert calls == [1]
    scheduler.tick(4.0)  # second
    assert calls == [1, 1]

    scheduler.cancel(event_id)
    scheduler.tick(4.0)
    assert calls == [1, 1]  # no more firings after cancel


def make_snippet_for_lane() -> Snippet:
    track = Track(1, "T", "A", 120.0, "8A", duration=600.0)
    return Snippet(name="s", track=track, start_beat=0.0, length_beats=64.0, loop=True)


def test_xf_gain_interpolates_start_mid_end_and_stops_from_lane():
    transport = Transport(bpm=60.0, beats_per_bar=4)  # 1 beat/sec
    transport.start()
    _, log = make_log()
    scheduler = Scheduler(transport, log)
    lane_a, lane_b = Lane("L1"), Lane("L2")
    lane_a.start_snippet(make_snippet_for_lane(), transport.position_beats)
    lane_a.gain = 1.0

    scheduler.start_automation(lane_a, lane_b, bars=4, description="xf")  # 16 beats = 16s
    assert lane_b.gain == pytest.approx(0.0)  # to_lane starts at 0 immediately

    scheduler.tick(8.0)  # halfway
    assert lane_a.gain == pytest.approx(0.5, abs=1e-6)
    assert lane_b.gain == pytest.approx(0.5, abs=1e-6)
    assert lane_a.snippet is not None  # not stopped yet

    scheduler.tick(8.0)  # end
    assert lane_a.gain == pytest.approx(0.0, abs=1e-6)
    assert lane_b.gain == pytest.approx(1.0, abs=1e-6)
    assert lane_a.snippet is None  # from_lane stops at the end


def test_xf_overlap_on_same_lane_warns_and_replaces():
    transport = Transport(bpm=60.0, beats_per_bar=4)
    transport.start()
    logs, log = make_log()
    scheduler = Scheduler(transport, log)
    lane_a, lane_b, lane_c = Lane("L1"), Lane("L2"), Lane("L3")

    scheduler.start_automation(lane_a, lane_b, bars=4, description="xf1")
    scheduler.start_automation(lane_a, lane_c, bars=4, description="xf2")

    assert any(level == "warn" for level, _ in logs)
    assert len(scheduler.automations) == 1
    assert scheduler.automations[0].to_lane is lane_c
