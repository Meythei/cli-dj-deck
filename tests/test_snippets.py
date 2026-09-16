import pytest

from library import DEMO_LIBRARY, Track
from snippets import SnippetError, snip


def test_snip_from_cue_uses_cue_beat_position():
    track = next(t for t in DEMO_LIBRARY if t.id == 1)
    s = snip(1, DEMO_LIBRARY, cue=2, bars=8)
    assert s.start_beat == pytest.approx(track.cues[2])
    assert s.length_beats == pytest.approx(32.0)
    assert s.track is track


def test_snip_from_bar_is_zero_indexed_internally():
    s = snip(1, DEMO_LIBRARY, bar=17, bars=4)
    assert s.start_beat == pytest.approx((17 - 1) * 4)
    assert s.length_beats == pytest.approx(16.0)


def test_snip_resolves_track_by_title():
    s = snip("Glass Horizon", DEMO_LIBRARY, bar=1, bars=1)
    assert s.track.title == "Glass Horizon"


def test_snip_requires_exactly_one_of_cue_or_bar():
    with pytest.raises(SnippetError):
        snip(1, DEMO_LIBRARY, bars=8)
    with pytest.raises(SnippetError):
        snip(1, DEMO_LIBRARY, cue=1, bar=1, bars=8)


def test_snip_beyond_track_length_is_an_error():
    track = Track(99, "Tiny", "Nobody", 120.0, "1A", duration=4.0)  # 8 beats total
    with pytest.raises(SnippetError):
        snip(track, [track], bar=1, bars=8)  # 32 beats, way past the 8-beat track


def test_snip_unknown_track_id_or_title_errors():
    with pytest.raises(SnippetError):
        snip(999, DEMO_LIBRARY, bar=1, bars=1)
    with pytest.raises(SnippetError):
        snip("Nonexistent Song", DEMO_LIBRARY, bar=1, bars=1)


def test_snip_key_and_bpm_come_from_track():
    s = snip(1, DEMO_LIBRARY, bar=1, bars=1)
    assert s.key == s.track.key
    assert s.bpm == s.track.bpm


def test_snip_length_must_be_whole_beats():
    with pytest.raises(SnippetError):
        snip(1, DEMO_LIBRARY, bar=1, bars=0.3)
