"""User configuration and where cli-dj keeps its files.

Everything cli-dj writes lives outside the repository, in per-user
directories from platformdirs (docs/decisions.md D4). Setting CLIDJ_HOME
redirects all of them under one folder, which is how tests stay hermetic.
"""
from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

APP_NAME = "cli-dj"

CONFIG_TEMPLATE = """\
# cli-dj configuration.
#
# Windows paths: write them with forward slashes in double quotes,
# "D:/Music/DJ", or keep the backslashes inside single quotes, 'D:\\Music\\DJ'.
# ("D:\\Music" does not work: inside double quotes a backslash starts an escape.)

[library]
# Folders scanned by scan(); subfolders are included. Example:
# folders = ["D:/Music/DJ", 'E:\\Tracks']
folders = []
# Detected tempi are folded by octaves into [bpm_min, bpm_max).
bpm_min = 88.0
bpm_max = 176.0
# Background worker processes for analysis and snippet rendering.
workers = 2

[audio]
samplerate = 48000
blocksize = 512
# Output device: name substring or index from `python -m clidj --list-devices`; empty = default.
device = ""
# Prefer this host API when picking the default device (Windows: WASAPI).
hostapi = "WASAPI"
# How far ahead of the audio the scheduler evaluates reservations.
lookahead_ms = 200.0

[mix]
loudness_target_lufs = -14.0
limiter_ceiling_db = -1.0
master_gain = 1.0
"""


def install_dir() -> Path:
    """Where the files shipped next to the program live (sets/): the folder of
    cli-dj.exe in a packaged build, the repository root otherwise."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


class ConfigError(Exception):
    """config.toml exists but can't be used; the message says how to fix it."""


@dataclass(frozen=True)
class Paths:
    config_dir: Path
    data_dir: Path
    cache_dir: Path

    @classmethod
    def default(cls) -> "Paths":
        home = os.environ.get("CLIDJ_HOME")
        if home:
            root = Path(home)
            return cls(root / "config", root / "data", root / "cache")
        import platformdirs

        return cls(
            Path(platformdirs.user_config_dir(APP_NAME, appauthor=False)),
            Path(platformdirs.user_data_dir(APP_NAME, appauthor=False)),
            Path(platformdirs.user_cache_dir(APP_NAME, appauthor=False)),
        )

    @property
    def config_file(self) -> Path:
        return self.config_dir / "config.toml"

    @property
    def overrides_file(self) -> Path:
        return self.config_dir / "overrides.json"

    @property
    def library_file(self) -> Path:
        return self.data_dir / "library.json"

    @property
    def analysis_dir(self) -> Path:
        return self.data_dir / "analysis"

    @property
    def render_dir(self) -> Path:
        return self.cache_dir / "renders"

    @property
    def demo_audio_dir(self) -> Path:
        return self.cache_dir / "demo-audio"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"


@dataclass
class Config:
    library_folders: list[Path] = field(default_factory=list)
    bpm_min: float = 88.0
    bpm_max: float = 176.0
    workers: int = 2
    samplerate: int = 48000
    blocksize: int = 512
    device: Union[str, int, None] = None
    hostapi: str = "WASAPI"
    lookahead_ms: float = 200.0
    loudness_target_lufs: float = -14.0
    limiter_ceiling_db: float = -1.0
    master_gain: float = 1.0

    @property
    def bpm_range(self) -> tuple[float, float]:
        return (self.bpm_min, self.bpm_max)

    @classmethod
    def load(cls, paths: Paths, create: bool = True) -> "Config":
        """Read config.toml, writing a commented template on first run."""
        path = paths.config_file
        if not path.exists():
            if create:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
            return cls()
        try:
            with path.open("rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            message = f"{path} could not be read: {exc}"
            if "\\" in str(exc):
                message += (
                    "\n  Windows paths in double quotes need forward slashes: \"D:/Music/DJ\"."
                    "\n  Or keep the backslashes and use single quotes: 'D:\\Music\\DJ'."
                )
            raise ConfigError(message) from None
        try:
            return cls.from_dict(data)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{path} has an invalid value: {exc}") from None

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        library = data.get("library", {})
        audio = data.get("audio", {})
        mix = data.get("mix", {})
        device: Optional[Union[str, int]] = audio.get("device") or None
        if isinstance(device, str) and device.strip().isdigit():
            device = int(device)
        defaults = cls()
        return cls(
            library_folders=[Path(os.path.expandvars(os.path.expanduser(p))) for p in library.get("folders", [])],
            bpm_min=float(library.get("bpm_min", defaults.bpm_min)),
            bpm_max=float(library.get("bpm_max", defaults.bpm_max)),
            workers=max(1, int(library.get("workers", defaults.workers))),
            samplerate=int(audio.get("samplerate", defaults.samplerate)),
            blocksize=int(audio.get("blocksize", defaults.blocksize)),
            device=device,
            hostapi=str(audio.get("hostapi", defaults.hostapi)),
            lookahead_ms=float(audio.get("lookahead_ms", defaults.lookahead_ms)),
            loudness_target_lufs=float(mix.get("loudness_target_lufs", defaults.loudness_target_lufs)),
            limiter_ceiling_db=float(mix.get("limiter_ceiling_db", defaults.limiter_ceiling_db)),
            master_gain=float(mix.get("master_gain", defaults.master_gain)),
        )
