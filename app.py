"""Compatibility launcher: `python app.py` still works; see clidj.cli."""
import sys

from clidj.cli import main

if __name__ == "__main__":
    sys.exit(main())
