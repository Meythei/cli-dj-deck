"""Command-line entry point: `python -m clidj [render ...]`."""
from __future__ import annotations

import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "render":
        from .offline import main as render_main

        return render_main(argv[1:])
    from .ui.app import main as app_main

    app_main(argv)
    return 0
