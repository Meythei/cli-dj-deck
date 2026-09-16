import asyncio
from pathlib import Path

from app import DJApp

DEMO_SET = Path(__file__).resolve().parents[1] / "sets" / "demo.djs"


async def test_app_boots_runs_and_transport_advances():
    app = DJApp(set_path=DEMO_SET)
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
    app = DJApp()
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
    app = DJApp()
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        assert all(lane.snippet is None for lane in app.lanes.values())
        assert not app.transport.running
