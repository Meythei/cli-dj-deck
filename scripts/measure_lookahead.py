"""Measure what the scheduler's lookahead has to absorb, and how many
commands arrive late at a given lookahead (docs/TASK_real-audio.md 9).

    .venv\\Scripts\\python scripts\\measure_lookahead.py

1. IPC: time from RealtimeEngineClient.send() to the engine process reading
   the command.
2. UI ticks: intervals between DJApp ticks while a set plays (headless
   Textual app, so terminal output time is not included).
3. Late commands: the headless app plays a reservation on every beat with
   each lookahead value; the engine counts commands that arrive after their
   sample has already been rendered.
"""
import argparse
import asyncio
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clidj.config import Config, Paths  # noqa: E402
from clidj.engine import commands as c  # noqa: E402
from clidj.engine.host import HostConfig, RealtimeEngineClient  # noqa: E402
from clidj.library import Track  # noqa: E402
from clidj.snippets import Snippet  # noqa: E402
from clidj.ui.app import DJApp  # noqa: E402
from clidj.workers import InlineJobRunner  # noqa: E402


def percentiles(values):
    values = sorted(values)
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
    return f"p50 {pick(0.5):.2f}  p99 {pick(0.99):.2f}  max {values[-1]:.2f}  (n={len(values)})"


def measure_ipc(samples: int) -> list[float]:
    client = RealtimeEngineClient(HostConfig(backend="null"))
    client.start()
    latencies = []
    try:
        for _ in range(samples):
            client.send(c.SetParam(c.MASTER, "gain", 1.0))
            time.sleep(0.02)
            latencies.append(client.extra.get("ipc_ms_last", 0.0) if client.status() else 0.0)
    finally:
        client.close()
    return latencies[5:]


async def run_app(lookahead_ms: float, seconds: float) -> tuple[list[float], dict]:
    home = Path(tempfile.mkdtemp(prefix="clidj-measure-"))
    paths = Paths(home / "config", home / "data", home / "cache")
    config = Config(lookahead_ms=lookahead_ms)
    app = DJApp(demo=True, audio=True, backend="null", paths=paths, config=config, jobs=InlineJobRunner())
    session = app.session
    # silent snippets: this measures timing, not audio (no rendering needed)
    session.prep = None
    track = Track(1, "T", "A", 128.0, "8A", 600.0)
    ticks: list[float] = []
    original_tick = app._on_tick

    def timed_tick() -> None:
        ticks.append(time.perf_counter())
        original_tick()

    app._on_tick = timed_tick
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        for lane in range(4):
            app.interp.env[f"s{lane}"] = session.register_snippet(Snippet(f"s{lane}", track, 0.0, 4.0, loop=True))
        for lane in range(4):
            app.interp.run(f"every(0.25, L{lane + 1} << s{lane})")
        app.interp.run("start()")
        started = time.monotonic()
        while time.monotonic() - started < seconds:
            await asyncio.sleep(0.25)
            app.query_one("#command-input").value = "queue()"  # some UI work, like typing
            await pilot.press("enter")
        status = session.engine.status()
        extra = dict(session.engine.extra)
    intervals = [(b - a) * 1000 for a, b in zip(ticks, ticks[1:])]
    return intervals, {"late": status.stats.late_commands, "late_max_ms": status.stats.late_max_ms,
                       "applied": status.stats.commands_applied, "underruns": extra.get("underruns")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--lookaheads", default="0,25,50,100,150,200,300")
    args = parser.parse_args()

    ipc = measure_ipc(300)
    print(f"IPC send -> engine read (ms): {percentiles(ipc)}")
    for lookahead in [float(x) for x in args.lookaheads.split(",")]:
        intervals, result = asyncio.run(run_app(lookahead, args.seconds))
        print(f"lookahead {lookahead:5.0f} ms: late {result['late']:4d} / {result['applied']} commands "
              f"(max {result['late_max_ms']:.1f} ms late), underruns {result['underruns']:.0f}; "
              f"UI tick interval (ms): {percentiles(intervals)}; mean {statistics.fmean(intervals):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
