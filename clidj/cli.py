"""Command-line entry point: `python -m clidj [render ... | selftest | --version]`."""
from __future__ import annotations

import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "render":
        from .offline import main as render_main

        return render_main(argv[1:])
    if argv and argv[0] == "selftest":
        from .selftest import main as selftest_main

        return selftest_main(argv[1:])
    if argv and argv[0] in ("--version", "-V"):
        from . import __version__

        print(f"cli-dj {__version__}")
        return 0
    from .ui.app import main as app_main

    return app_main(argv) or 0
