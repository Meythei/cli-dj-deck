"""Layout regressions for the TRANSPORT / LANES pane and the library pane
(docs/TASK_real-audio.md 5.1, 5.8, 5.9)."""
import pytest
from textual.widgets import DataTable

from clidj.ui.app import DJApp, LanesView
from clidj.lanes import LANE_NAMES, Lane
from clidj.library import SAMPLES_PER_BEAT, Track
from clidj.snippets import Snippet
from clidj.transport import Transport


def spiky_track() -> Track:
    """Silent-ish waveform with a full-height spike on every bar head, so the
    spikes' columns can be compared against the ruler's bar lines."""
    beats = 256
    waveform = [0.03] * (beats * SAMPLES_PER_BEAT)
    for beat in range(0, beats, 4):
        waveform[beat * SAMPLES_PER_BEAT] = 1.0
    return Track(1, "Spikes", "Nobody", 120.0, "8A", duration=beats * 60 / 120, waveform=waveform)


def make_view(position: float):
    transport = Transport(bpm=120.0)
    transport.position_beats = position
    lanes = {name: Lane(name) for name in LANE_NAMES}
    lanes["L1"].start_snippet(Snippet("s", spiky_track(), 0.0, 128.0, loop=True), 0.0)
    for lane in lanes.values():
        lane.gain = 0.0  # empty gain meters: the only full blocks left are waveform spikes
    return LanesView(transport, lanes, lambda: "bar"), lanes


def ruler_and_wave_columns(text, width):
    lines = text.plain.split("\n")
    ruler = lines[1]
    playhead = ruler.index("▼")
    bar_lines = {i for i, ch in enumerate(ruler) if ch == "|"}
    spikes = set()
    for line in lines[2:]:
        if line[:4].strip():  # lane info rows (gain/eq meters) start with the lane name
            continue
        spikes |= {i for i, ch in enumerate(line) if ch == "█"}
    return playhead, bar_lines, spikes


@pytest.mark.parametrize("width", [60, 61, 97, 120])
def test_ruler_bar_lines_and_playhead_line_up_with_the_waveform(width):
    view, _ = make_view(position=40.0)  # exactly on a bar head
    text = view.build_text(width)
    playhead, bar_lines, spikes = ruler_and_wave_columns(text, width)
    assert spikes, "the spiky waveform should be visible"
    assert spikes == bar_lines | {playhead}


@pytest.mark.parametrize("width", [60, 61])
def test_waveform_playhead_column_matches_ruler_marker(width):
    """Past columns use the lane accent, the playhead column and everything
    right of it are drawn in grey: the first grey column is the playhead."""
    view, _ = make_view(position=41.5)
    text = view.build_text(width)
    lines = text.plain.split("\n")
    playhead = lines[1].index("▼")

    offset = 0
    first_grey_cols = []
    for line in lines:
        line_start, line_end = offset, offset + len(line)
        greys = sorted(
            span.start - line_start
            for span in text.spans
            if line_start <= span.start < line_end and "grey50" in str(span.style)
        )
        if greys:
            first_grey_cols.append(greys[0])
        offset = line_end + 1
    assert first_grey_cols, "expected a waveform row"
    assert all(col == playhead for col in first_grey_cols)


def test_each_lane_gets_an_info_row_and_waveform_rows():
    view, _ = make_view(position=40.0)
    lines = view.build_text(100).plain.rstrip("\n").split("\n")
    # header + ruler + (1 info + 2 waveform rows) per lane
    assert len(lines) == 2 + 3 * len(LANE_NAMES)


async def test_lanes_pane_is_sized_to_its_content_not_half_the_screen():
    app = DJApp(demo=True)
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        view = app.query_one("#lanes-view", LanesView)
        content_lines = len(view.build_text(view.content_size.width).plain.rstrip("\n").split("\n"))
        assert view.size.height <= content_lines + 2  # + border


async def test_tracks_table_fits_the_library_pane_without_horizontal_scroll():
    app = DJApp(demo=True)
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        table = app.query_one("#tracks-table", DataTable)
        assert table.virtual_size.width <= table.scrollable_content_region.width
