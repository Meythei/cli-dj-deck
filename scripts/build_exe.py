r"""Build cli-dj.exe with PyInstaller and put it under release\.

    .venv\Scripts\pip install -r requirements-build.txt
    .venv\Scripts\python scripts\build_exe.py

Produces release\cli-dj-<version>-windows-x64\ (cli-dj.exe, _internal\, sets\,
README.md) and a zip of that folder. The build is a folder, not a single-file
exe: cli-dj starts itself again for the audio engine and the worker
processes, and a one-file exe would unpack hundreds of MB on every launch.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clidj import __version__  # noqa: E402


def main() -> int:
    build, dist = ROOT / "build", ROOT / "dist"
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
         "--workpath", str(build), "--distpath", str(dist), str(ROOT / "packaging" / "cli-dj.spec")],
        check=True,
    )
    app_dir = dist / "cli-dj"
    shutil.copytree(ROOT / "sets", app_dir / "sets", dirs_exist_ok=True)
    shutil.copy2(ROOT / "README.md", app_dir / "README.md")

    name = f"cli-dj-{__version__}-windows-x64"
    release = ROOT / "release"
    target = release / name
    if target.exists():
        shutil.rmtree(target)
    release.mkdir(exist_ok=True)
    shutil.copytree(app_dir, target)
    archive = shutil.make_archive(str(release / name), "zip", root_dir=release, base_dir=name)

    version = subprocess.run([str(target / "cli-dj.exe"), "--version"], capture_output=True, text=True, timeout=120)
    print(version.stdout.strip() or version.stderr.strip())
    size_mb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e6
    print(f"release folder: {target} ({size_mb:.0f} MB)")
    print(f"zip: {archive} ({Path(archive).stat().st_size / 1e6:.0f} MB)")
    return version.returncode


if __name__ == "__main__":
    sys.exit(main())
