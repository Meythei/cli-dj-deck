"""Compatibility launcher: `python app.py` still works; the app lives in clidj.ui.app."""
from clidj.ui.app import main

if __name__ == "__main__":
    main()
