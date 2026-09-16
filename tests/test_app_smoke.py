import asyncio
from pathlib import Path

from clidj.ui.app import DJApp

DEMO_SET = Path(__file__).resolve().parents[1] / "sets" / "demo.djs"


async def test_app_boots_runs_and_transport_advances():
    app = DJApp(set_path=DEMO_SET, demo=True)
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()

        input_widget = app.query_one("#command-input")
        input_widget.value = "start()"
        await pilot.press("enter")
        await pilot.pause()

        assert app.transport.running

        # Let the real interval timer fire a few times so a genuine dt
        # accumulates (the app reads time.monotonic(), not a fake clock).
        await asyncio.sleep(0.5)
        await pilot.pause()

        assert app.transport.position_beats > 0
        assert app.lanes["L1"].snippet is not None  # demo set's `L1 << kick`


async def test_app_history_and_error_path_do_not_crash():
    app = DJApp(demo=True)
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()

        input_widget = app.query_one("#command-input")
        input_widget.value = "bpm(140)"
        await pilot.press("enter")
        await pilot.pause()

        input_widget.value = ""
        await pilot.press("up")
        assert input_widget.value == "bpm(140)"

        # a rejected command must be logged, not crash the app
        input_widget.value = "__import__('os')"
        await pilot.press("enter")
        await pilot.pause()

        input_widget.value = "help()"
        await pilot.press("enter")
        await pilot.pause()

    # exiting run_test() cleanly (no exception) is itself the assertion that
    # nothing above crashed the app


async def test_app_without_set_path_starts_with_empty_lanes():
    app = DJApp(demo=True)
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        assert all(lane.snippet is None for lane in app.lanes.values())
        assert not app.transport.running


async def test_scan_in_the_app_fills_the_tracks_table_with_grid_and_status(tmp_path):
    from clidj import synth
    from clidj.config import Config, Paths
    from clidj.workers import InlineJobRunner

    music = tmp_path / "music"
    synth.write_audio(music / "Someone - Thing.flac", synth.techno_loop_track(125.0, 9, True, 30.0, 44100), 44100)
    paths = Paths(tmp_path / "config", tmp_path / "data", tmp_path / "cache")
    app = DJApp(paths=paths, config=Config(library_folders=[music]), jobs=InlineJobRunner())
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        input_widget = app.query_one("#command-input")
        input_widget.value = "scan()"
        await pilot.press("enter")
        await pilot.pause(0.3)
        table = app.query_one("#tracks-table")
        assert table.row_count == 1
        number, status, title, artist, bpm, key, length = table.get_row_at(0)
        assert (number, status, title, artist) == ("1", "✓", "Thing", "Someone")
        assert float(bpm) == 125.0


async def test_snips_table_shows_preparation_state(tmp_path):
    from clidj import synth
    from clidj.config import Config, Paths
    from clidj.library import Track
    from clidj.workers import InlineJobRunner

    source = synth.write_audio(tmp_path / "k.wav", synth.click_track(124.0, 0.2, 30.0, 48000, kind="kick"), 48000)
    paths = Paths(tmp_path / "config", tmp_path / "data", tmp_path / "cache")
    app = DJApp(demo=True, audio=True, paths=paths, config=Config(), jobs=InlineJobRunner())
    app.session.tracks[:] = [Track(1, "Kicks", "T", 124.0, "8A", 30.0, first_beat=0.2, path=source,
                                   track_id="kicks", status="ready")]
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        input_widget = app.query_one("#command-input")
        input_widget.value = "kick = snip(1, bar=1, bars=2, loop=True)"
        await pilot.press("enter")
        await pilot.pause(0.2)
        table = app.query_one("#snips-table")
        assert table.row_count == 1
        glyph, name, *_ = table.get_row_at(0)
        assert (glyph, name) == ("✓", "kick")
