"""Entry point of the packaged cli-dj.exe (PyInstaller)."""
import multiprocessing
import os
import sys


def run() -> int:
    from clidj.config import Paths

    # librosa's numba functions cache compiled code next to their source by
    # default, which is read-only inside a packaged build.
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Paths.default().cache_dir / "numba"))
    from clidj.cli import main

    return main()


if __name__ == "__main__":
    # Must run first: the engine and worker processes are this same exe,
    # started by multiprocessing's spawn method.
    multiprocessing.freeze_support()
    sys.exit(run())
