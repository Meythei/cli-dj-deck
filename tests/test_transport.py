import pytest

from clidj.transport import Transport


def test_advance_reaches_next_bar_head():
    t = Transport(bpm=128.0, beats_per_bar=4)
    t.start()
    t.advance(1.875)
    assert t.position_beats == pytest.approx(4.0)
    assert t.bar == 2
    assert t.beat_in_bar == pytest.approx(1.0)


def test_advance_result_independent_of_step_size():
    a = Transport(bpm=128.0, beats_per_bar=4)
    a.start()
    a.advance(1.875)

    b = Transport(bpm=128.0, beats_per_bar=4)
    b.start()
    for _ in range(75):
        b.advance(0.025)

    assert a.position_beats == pytest.approx(b.position_beats)
    assert b.bar == 2
    assert b.beat_in_bar == pytest.approx(1.0)


def test_stopped_transport_does_not_advance():
    t = Transport(bpm=128.0)
    t.advance(10.0)
    assert t.position_beats == 0.0


def test_advance_returns_prev_and_new_position():
    t = Transport(bpm=120.0)
    t.start()
    prev, new = t.advance(1.0)
    assert prev == pytest.approx(0.0)
    assert new == pytest.approx(2.0)


def test_next_boundary_strictly_after_current_position():
    t = Transport(bpm=120.0, beats_per_bar=4)
    t.position_beats = 8.0  # exactly on a bar head already
    assert t.next_boundary_beats("bar") == pytest.approx(12.0)
    assert t.next_boundary_beats("beat") == pytest.approx(9.0)
    assert t.next_boundary_beats("phrase") == pytest.approx(32.0)


def test_next_boundary_mid_bar():
    t = Transport(bpm=120.0, beats_per_bar=4)
    t.position_beats = 5.5
    assert t.next_boundary_beats("beat") == pytest.approx(6.0)
    assert t.next_boundary_beats("bar") == pytest.approx(8.0)


def test_next_boundary_none_is_now():
    t = Transport(bpm=120.0)
    t.position_beats = 5.5
    assert t.next_boundary_beats("none") == pytest.approx(5.5)


def test_bar_at_and_display_at_are_pure():
    t = Transport(bpm=120.0, beats_per_bar=4)
    assert t.bar_at(0.0) == 1
    assert t.bar_at(3.99) == 1
    assert t.bar_at(4.0) == 2
    assert t.display_at(4.0) == "002.1"
    assert t.display_at(6.0) == "002.3"


def test_beats_at_bar_round_trips_bar_at():
    t = Transport(beats_per_bar=4)
    for bar in (1, 2, 5, 33):
        assert t.bar_at(t.beats_at_bar(bar)) == bar
