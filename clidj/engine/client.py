"""Engine clients: how the UI-side Session talks to an engine.

`LocalEngineClient` runs the engine in the calling process and advances it
explicitly. It backs `--no-audio` (visual mode, no sound) and offline
rendering. The realtime client (engine in its own process, driven by an
audio backend) lives in clidj.engine.host and has the same surface.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import commands as cmd
from .core import Engine, EngineStatus, load_buffer


class LocalEngineClient:
    realtime = False

    def __init__(self, engine: Engine, prefault: bool = False) -> None:
        self.engine = engine
        self.prefault = prefault
        self.error: Optional[str] = None

    @property
    def samplerate(self) -> int:
        return self.engine.samplerate

    @property
    def render_audio(self) -> bool:
        return self.engine.render_audio

    @property
    def alive(self) -> bool:
        return True

    def send(self, command) -> None:
        if isinstance(command, cmd.RegisterBuffer):
            self.engine.submit(load_buffer(command.info, prefault=self.prefault))
        else:
            self.engine.submit(command)

    def flush(self) -> None:
        """Apply queued immediate commands now (and timed ones already due)
        without advancing time."""
        self.engine.flush()

    def advance(self, frames: int) -> Optional[np.ndarray]:
        return self.engine.process(frames)

    def status(self) -> EngineStatus:
        return self.engine.status()

    def beat_at_sample_offset(self, frames: int) -> float:
        engine = self.engine
        if not engine.running:
            return engine.position_beats
        return engine.tempo.sample_to_beat(engine.transport_sample + frames)

    def close(self) -> None:
        pass
