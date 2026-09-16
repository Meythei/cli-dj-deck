import pytest

from lanes import Lane, check_warnings, keys_compatible
from library import Track
from snippets import Snippet
from transport import Transport


def make_snippet(loop=False, length_beats=16.0, role="drums", key="8A", bpm=128.0, start_beat=0.0):
    track = Track(1, "Test Track", "Nobody", bpm, key, duration=600.0)
    return Snippet(name="s", track=track, start_beat=start_beat, length_beats=length_beats, loop=loop, role=role)


def test_looping_lane_local_beat_always_in_range():
    transport = Transport(bpm=128.0)
    lane = Lane("L1")
    snippet = make_snippet(loop=True, length_beats=16.0)
    lane.start_snippet(snippet, transport.position_beats)

    for beats in [0, 1, 15.9, 16, 16.1, 100, 1000.5, 12345.25]:
        transport.position_beats = beats
        local = lane.local_beat(transport)
        assert local is not None
        assert 0 <= local < snippet.length_beats


def test_non_looping_lane_stops_after_length():
    transport = Transport(bpm=128.0)
    lane = Lane("L1")
    snippet = make_snippet(loop=False, length_beats=16.0)
    lane.start_snippet(snippet, transport.position_beats)

    transport.position_beats = 15.9
    assert lane.local_beat(transport) == pytest.approx(15.9)

    transport.position_beats = 16.0
    assert lane.local_beat(transport) is None

    transport.position_beats = 500.0
    assert lane.local_beat(transport) is None


def test_non_looping_lane_update_clears_snippet():
    transport = Transport(bpm=128.0)
    lane = Lane("L1")
    snippet = make_snippet(loop=False, length_beats=16.0)
    lane.start_snippet(snippet, transport.position_beats)

    transport.position_beats = 20.0
    assert lane.snippet is not None
    lane.update(transport)
    assert lane.snippet is None


def test_loop_position_recomputed_directly_not_iteratively():
    """Regression test for the old Deck.tick bug: its loop wrap used
    `start + (position - end)`, which only unwound one loop-length per
    tick, so a position far past the loop end crept backwards over many
    ticks instead of wrapping immediately. Lane carries no such state --
    local_beat is a plain `local % length_beats`, correct on a single call
    no matter how far past the loop end the transport has jumped.
    """
    transport = Transport(bpm=128.0)
    lane = Lane("L1")
    snippet = make_snippet(loop=True, length_beats=8.0)
    lane.start_snippet(snippet, transport.position_beats)

    transport.position_beats = 100.35  # far past many loop cycles, in one jump
    local = lane.local_beat(transport)
    assert local == pytest.approx(100.35 % 8.0)
    assert 0.0 <= local < 8.0


def test_track_beat_adds_snippet_start_offset():
    transport = Transport(bpm=128.0)
    lane = Lane("L1")
    snippet = make_snippet(loop=True, length_beats=16.0, start_beat=64.0)
    lane.start_snippet(snippet, transport.position_beats)
    transport.position_beats = 3.0
    assert lane.track_beat(transport) == pytest.approx(67.0)


def test_keys_compatible():
    assert keys_compatible("8A", "8A")
    assert keys_compatible("8A", "8B")
    assert keys_compatible("8A", "9A")
    assert keys_compatible("8A", "7A")
    assert not keys_compatible("8A", "3A")
    assert not keys_compatible("8A", "9B")


def test_check_warnings_bpm_drift():
    transport_bpm = 140.0
    lane = Lane("L1")
    transport = Transport(bpm=transport_bpm)
    lane.start_snippet(make_snippet(bpm=128.0), transport.position_beats)
    warnings = check_warnings(lane, [], transport_bpm)
    assert any("bpm" in w.lower() for w in warnings)


def test_check_warnings_no_drift_below_threshold():
    transport_bpm = 130.0
    lane = Lane("L1")
    transport = Transport(bpm=transport_bpm)
    lane.start_snippet(make_snippet(bpm=128.0), transport.position_beats)
    warnings = check_warnings(lane, [], transport_bpm)
    assert not any("bpm" in w.lower() for w in warnings)


def test_check_warnings_vocal_conflict():
    transport = Transport(bpm=128.0)
    lane_a, lane_b = Lane("L1"), Lane("L2")
    lane_a.start_snippet(make_snippet(role="vocal"), transport.position_beats)
    lane_b.start_snippet(make_snippet(role="vocal"), transport.position_beats)
    warnings = check_warnings(lane_a, [lane_b], 128.0)
    assert any("vocal" in w.lower() for w in warnings)


def test_check_warnings_key_clash():
    transport = Transport(bpm=128.0)
    lane_a, lane_b = Lane("L1"), Lane("L2")
    lane_a.start_snippet(make_snippet(key="8A"), transport.position_beats)
    lane_b.start_snippet(make_snippet(key="3A"), transport.position_beats)
    warnings = check_warnings(lane_a, [lane_b], 128.0)
    assert any("key" in w.lower() for w in warnings)


def test_check_warnings_empty_lane_has_no_warnings():
    assert check_warnings(Lane("L1"), [], 128.0) == []
