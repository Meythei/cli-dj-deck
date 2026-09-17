"""`python -m clidj selftest` (or `cli-dj.exe selftest`): check that the
parts which only break in a packaged or unusual environment actually work --
spawned worker processes (analysis, rendering), the realtime engine process,
audio device enumeration and the Textual UI -- without playing any sound.

Everything runs in a throwaway CLIDJ_HOME, so the user's library and cache
are never touched.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable


def _step(name: str, fn: Callable[[], str]) -> bool:
    started = time.perf_counter()
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 -- report every failure, keep going
        print(f"  FAIL  {name} ({time.perf_counter() - started:.1f}s): {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False
    print(f"  ok    {name} ({time.perf_counter() - started:.1f}s){': ' + detail if detail else ''}")
    return True


def main(argv: list[str]) -> int:
    from . import __version__

    home = Path(tempfile.mkdtemp(prefix="clidj-selftest-"))
    os.environ["CLIDJ_HOME"] = str(home)
    print(f"cli-dj {__version__} selftest (scratch dir {home})")
    state: dict = {}

    def imports() -> str:
        import librosa
        import numpy
        import pedalboard
        import scipy
        import sounddevice
        import textual

        return (f"numpy {numpy.__version__}, scipy {scipy.__version__}, librosa {librosa.__version__}, "
                f"pedalboard {pedalboard.__version__}, sounddevice {sounddevice.__version__}, textual {textual.__version__}")

    def synthesise() -> str:
        from . import synth

        audio = synth.techno_loop_track(124.0, 9, True, 30.0, 44100, first_beat=0.25)
        state["track"] = synth.write_audio(home / "music" / "Selftest - Loop.flac", audio, 44100)
        return str(state["track"].name)

    def analysis_in_worker() -> str:
        from .workers import JobRunner, analyze_job

        runner = JobRunner(workers=1, log_path=home / "workers.log")
        try:
            result = runner.run_in_process(analyze_job, str(state["track"]), "selftest", str(home / "analysis"),
                                           (88.0, 176.0)).result(timeout=300)
        finally:
            runner.shutdown()
        if not result.get("ok"):
            raise RuntimeError(result.get("error"))
        if abs(result["bpm"] - 124.0) > 0.1:
            raise RuntimeError(f"expected 124 BPM, got {result['bpm']}")
        return f"{result['bpm']:.2f} BPM, key {result['key']}"

    def render_in_worker() -> str:
        from .prep import render_job
        from .render import RenderSpec
        from .workers import JobRunner

        spec = RenderSpec("selftest", str(state["track"]), 124.0, 0.25, 8.0, 8.0, True, 128.0, 48000)
        runner = JobRunner(workers=1, log_path=home / "workers.log")
        try:
            result = runner.run_in_process(render_job, spec.to_dict(), str(home / "renders")).result(timeout=300)
        finally:
            runner.shutdown()
        if not result.get("ok"):
            raise RuntimeError(result.get("error"))
        if result["frames"] != spec.length_samples:
            raise RuntimeError(f"expected {spec.length_samples} frames, got {result['frames']}")
        state["render"] = result
        return f"{result['frames']} frames at 128 BPM"

    def realtime_engine() -> str:
        from .engine import commands as cmd
        from .engine.host import HostConfig, RealtimeEngineClient

        client = RealtimeEngineClient(HostConfig(backend="null", bpm=128.0, log_path=str(home / "engine.log")))
        client.start()
        try:
            render = state["render"]
            client.send(cmd.RegisterBuffer(cmd.BufferInfo(1, 128.0, render["path"], render["frames"], 8.0, True)))
            client.send(cmd.Play(0, 1, beat=None))
            client.send(cmd.TransportStart())
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and client.status().position_beats < 2.0:
                time.sleep(0.05)
            status = client.status()
            if status.position_beats < 2.0 or status.lanes[0].uid != 1:
                raise RuntimeError(f"engine did not play (position {status.position_beats:.2f})")
            return f"null backend, {client.extra.get('underruns', 0):.0f} underruns, peak {status.master_peak[0]:.2f}"
        finally:
            client.close()

    def audio_devices() -> str:
        from .engine.backends import find_output_device, list_output_devices

        devices = list_output_devices()
        if not devices:
            return "no output devices (fine for --no-audio / render)"
        default = find_output_device(None)
        return f"{len(devices)} output device(s), default: {default.name} [{default.hostapi}]"

    def user_interface() -> str:
        from .config import Config, Paths
        from .ui.app import DJApp
        from .workers import InlineJobRunner

        async def run() -> str:
            app = DJApp(set_path=None, demo=True, paths=Paths.default(), config=Config(), jobs=InlineJobRunner())
            async with app.run_test(size=(140, 44)) as pilot:
                await pilot.pause()
                app.interp.run("kick = snip(1, cue=1, bars=4, loop=True)")
                app.interp.run("L1 << kick")
                app.interp.run("start()")
                await asyncio.sleep(0.5)
                await pilot.pause()
                if not app.transport.running or app.lanes["L1"].snippet is None:
                    raise RuntimeError("UI did not start the demo lane")
                return f"TUI mounted, transport at {app.transport.display}"

        return asyncio.run(run())

    steps = [
        ("imports", imports),
        ("synthesise a test track", synthesise),
        ("analysis in a worker process", analysis_in_worker),
        ("render in a worker process", render_in_worker),
        ("realtime engine process", realtime_engine),
        ("audio devices", audio_devices),
        ("user interface (headless)", user_interface),
    ]
    results = []
    for name, fn in steps:
        results.append(_step(name, fn))
        if not results[-1] and name in ("synthesise a test track", "render in a worker process"):
            break  # later steps need this one's output
    failed = len(results) - sum(results) + (len(steps) - len(results))
    print("selftest passed" if failed == 0 else f"selftest FAILED ({failed} step(s))")
    return 0 if failed == 0 else 1
