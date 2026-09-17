r"""Ten-minute realtime load test (docs/TASK_real-audio.md 12): four lanes,
EQ, automations and the limiter on the engine process, paced in real time
by the NullBackend. Prints underruns and callback load.

    .venv\Scripts\python scripts\load_test.py --seconds 600
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from test_load import run_load  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--blocksize", type=int, default=512)
    parser.add_argument("--log", default=None, help="engine log file (timing outliers are listed at the end)")
    parser.add_argument("--device", action="store_true",
                        help="run on the default output device (master gain 0: silent) instead of the NullBackend")
    args = parser.parse_args()
    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        result = run_load(args.seconds, tmp, args.blocksize, log_path=args.log,
                          backend="sounddevice" if args.device else "null")
    print(f"ran {time.time() - started:.0f} s, blocksize {args.blocksize}: {result}")
    if result["callback_ratio"] < 0.95:
        print("INVALID: far fewer callbacks than the elapsed time allows -- the machine was paused or asleep")
        return 2
    return 0 if result["underruns"] == 0 and result["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
