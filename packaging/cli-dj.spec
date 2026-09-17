# PyInstaller spec for cli-dj.exe -- build with: python scripts/build_exe.py
# -*- mode: python -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent

hiddenimports = (
    collect_submodules("clidj")
    + collect_submodules("textual")  # widgets are imported lazily by name
    + collect_submodules("rich")  # unicode width tables are imported by version
    + collect_submodules("mutagen")  # mutagen.File imports every format module
    + collect_submodules("soxr")
    + ["sounddevice", "_sounddevice_data", "pedalboard", "pedalboard.io"]
)
datas = (
    [(str(ROOT / "clidj" / "ui" / "app.tcss"), "clidj/ui")]
    + collect_data_files("librosa")  # lazy_loader stubs (__init__.pyi) and data registries
    + collect_data_files("textual")
)

a = Analysis(
    [str(ROOT / "packaging" / "cli_dj.py")],
    pathex=[str(ROOT)],
    hiddenimports=hiddenimports,
    datas=datas,
    excludes=["matplotlib", "tkinter", "IPython", "jupyter", "notebook", "pytest", "PyQt5", "PySide6", "pandas"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="cli-dj",
    console=True,  # a terminal UI needs its console
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="cli-dj", upx=False)
