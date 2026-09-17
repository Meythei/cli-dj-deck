"""Load test: four lanes with EQ, automation and the limiter on the realtime
engine process, paced by the NullBackend (docs/TASK_real-audio.md 12).

Runs for CLIDJ_LOAD_SECONDS (default 5 s). The ten-minute check from the
spec is `CLIDJ_LOAD_SECONDS=600 pytest tests/test_load.py` or
`python scripts/load_test.py --seconds 600`."""
import os
import time

import numpy as np

from clidj.engine import commands as c
from clidj.engine.host import HostConfig, RealtimeEngineClient

SECONDS = float(os.environ.get("CLIDJ_LOAD_SECONDS", "5"))


def run_load(seconds: float, tmp_dir, blocksize: int = 512, log_path=None, backend: str = "null") -> dict:
    """With backend="sounddevice" the master gain is 0: the full DSP load runs
    on the real output device, but what it plays is silence."""
    silent = backend != "null"
    client = RealtimeEngineClient(HostConfig(backend=backend, bpm=124.0, blocksize=blocksize, log_path=log_path,
                                             master_gain=0.0 if silent else 1.0, fallback_to_null=False))
    client.start()
    try:
        rng = np.random.default_rng(0)
        frames = round(16 * 60 / 124 * 48000)
        for lane in range(4):
            path = os.path.join(tmp_dir, f"load{lane}.npy")
            np.save(path, (rng.standard_normal((frames, 2)) * 0.25).astype(np.float32))
            client.send(c.RegisterBuffer(c.BufferInfo(lane + 1, 124.0, path, frames, 16.0, True)))
            client.send(c.Play(lane, lane + 1, beat=None))
            client.send(c.SetParam(lane, "lo", 0.4))
            client.send(c.SetParam(lane, "hi", 0.7))
        client.send(c.TransportStart())
        started = time.monotonic()
        beat = 4.0
        while time.monotonic() - started < seconds:
            # keep automations running the whole time, like continuous mixing
            status = client.status()
            if status.position_beats + 8 > beat:
                client.send(c.Automate(int(beat), int(beat) % 4, "gain", beat, beat + 16, None,
                                       0.3 if int(beat) % 8 else 1.0))
                beat += 4.0
            time.sleep(0.05)
        status = client.status()
        elapsed = time.monotonic() - started
        expected = elapsed * 48000 / blocksize
        return {"backend": client.info.get("device"), "seconds": round(elapsed, 1), "callback_ratio": round(client.extra["callbacks"] / expected, 3),
                "underruns": client.extra["underruns"], "callbacks": client.extra["callbacks"],
                "load_max": client.extra["callback_load_max"], "load_avg": client.extra["callback_load_avg"],
                "slow_callbacks": client.extra["slow_callbacks"], "wake_late_ms_max": client.extra["wake_late_ms_max"],
                "errors": status.stats.errors, "alive": client.alive}
    finally:
        client.close()


def test_four_lanes_with_eq_and_limiter_have_no_underruns(tmp_path):
    result = run_load(SECONDS, str(tmp_path))
    assert result["alive"] and result["errors"] == 0
    # A ratio well below 1 means the machine was paused (e.g. slept): the run proves nothing.
    assert result["callback_ratio"] > 0.95, result
    assert result["underruns"] == 0, result
    assert result["load_avg"] < 0.5, result
