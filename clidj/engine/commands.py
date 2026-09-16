"""Engine commands: plain, picklable data. Nothing here may reference
interpreter objects, Thunks or snippets -- only numbers, strings and paths --
because commands cross a process boundary to the audio engine
(docs/TASK_real-audio.md 8).

`beat=None` means "immediately": at the start of the next block the engine
processes after receiving the command. A beat means "at the sample that beat
lands on", exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

LANE_PARAMS = ("gain", "mute", "lo", "mid", "hi")
MASTER = -1  # lane index for master-bus parameters ("gain" only)


@dataclass(frozen=True)
class BufferInfo:
    uid: int  # snippet uid
    bpm: float  # tempo the buffer was rendered for
    path: Optional[str]  # float32 (frames, 2) .npy; None = silent (visual-only engine)
    frames: int
    length_beats: float
    loop: bool
    gain: float = 1.0  # loudness normalisation, applied by the engine


@dataclass(frozen=True)
class RegisterBuffer:
    info: BufferInfo


@dataclass(frozen=True)
class Play:
    lane: int
    uid: int
    beat: Optional[float]


@dataclass(frozen=True)
class Stop:
    lane: int
    beat: Optional[float]


@dataclass(frozen=True)
class SetParam:
    lane: int  # MASTER for the master bus
    param: str
    value: float
    beat: Optional[float] = None


@dataclass(frozen=True)
class Automate:
    """Ramp a lane parameter from `start_value` (None = its value when the
    ramp starts) to `end_value` over [start_beat, end_beat]."""

    automation_id: int
    lane: int
    param: str
    start_beat: float
    end_beat: float
    start_value: Optional[float]
    end_value: float
    curve: str = "linear"
    stop_lane_at_end: bool = False

    @property
    def beat(self) -> float:
        return self.start_beat


@dataclass(frozen=True)
class CancelAutomation:
    automation_id: Optional[int] = None  # None = every automation
    beat: Optional[float] = None


@dataclass(frozen=True)
class SetTempo:
    bpm: float
    beat: Optional[float]


@dataclass(frozen=True)
class TransportStart:
    beat: Optional[float] = None


@dataclass(frozen=True)
class TransportStop:
    beat: Optional[float] = None


@dataclass(frozen=True)
class Shutdown:
    beat: Optional[float] = None


EngineCommand = Union[
    RegisterBuffer, Play, Stop, SetParam, Automate, CancelAutomation, SetTempo, TransportStart, TransportStop, Shutdown
]


def command_beat(command) -> Optional[float]:
    return None if isinstance(command, RegisterBuffer) else command.beat
