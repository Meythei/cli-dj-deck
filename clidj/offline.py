"""Offline rendering: run a set through the real engine as fast as the CPU
allows and write the result to a WAV file (docs/TASK_real-audio.md 8).

    python -m clidj render sets/demo.djs --bars 48 -o renders/demo.wav --demo

The render is deterministic: snippets are prepared synchronously, the
engine is advanced block by block, and the limiter's lookahead delay is
removed so sample N of the file is transport sample N.

A *script* simulates typing at the REPL during the set. One command per
line, prefixed with the moment it is "typed" as BAR or BAR.BEAT (1-based,
like the transport display):

    # 2.3 = bar 2, beat 3
    2.3: L2 << bass
    9:   xf(L1, L2, bars=8)
    17.1: bpm(130)

A command runs at the first engine block boundary at or after that moment,
then goes through the usual quantizing -- exactly as if typed live.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import Config, ConfigError, Paths
from .engine.client import LocalEngineClient
from .engine.core import Engine, EngineStats
from .interpreter import Interpreter
from .lanes import LANE_NAMES
from .session import Session
from .synth import write_audio
from .tempo import event_sample
from .workers import InlineJobRunner

LogFn = Callable[[str, str], None]
SCRIPT_LINE = re.compile(r"^\s*(\d+)(?:\.(\d+(?:\.\d+)?))?\s*:\s*(.+?)\s*$")
SETTLE_POLLS = 50


class ScriptError(ValueError):
    pass


@dataclass(frozen=True)
class ScriptCommand:
    beat: float  # 0-based transport beat
    text: str
    line: int


def parse_script(text: str, beats_per_bar: int = 4) -> list[ScriptCommand]:
    commands = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0] if not raw.lstrip().startswith("#") else ""
        if not line.strip():
            continue
        match = SCRIPT_LINE.match(line)
        if not match:
            raise ScriptError(f"script line {number}: expected 'BAR[.BEAT]: command', got {raw.strip()!r}")
        bar = int(match.group(1))
        beat_in_bar = float(match.group(2)) if match.group(2) else 1.0
        if bar < 1 or not 1.0 <= beat_in_bar < beats_per_bar + 1:
            raise ScriptError(f"script line {number}: bar must be >= 1 and beat within 1..{beats_per_bar}")
        commands.append(ScriptCommand((bar - 1) * beats_per_bar + beat_in_bar - 1.0, match.group(3), number))
    return sorted(commands, key=lambda c: (c.beat, c.line))


@dataclass
class RenderReport:
    frames: int = 0
    samplerate: int = 48000
    peak: float = 0.0
    stats: EngineStats = field(default_factory=EngineStats)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return self.frames / self.samplerate


class OfflineRenderer:
    """Wraps a Session whose engine is an in-process, audio-producing
    engine, and pulls audio out of it block by block."""

    def __init__(self, session: Session, interpreter: Interpreter, blocksize: int = 512) -> None:
        if session.engine.realtime or not session.engine.render_audio:
            raise ValueError("offline rendering needs an in-process engine that renders audio")
        self.session = session
        self.interp = interpreter
        self.blocksize = blocksize

    @property
    def engine(self) -> Engine:
        return self.session.engine.engine

    def settle(self) -> None:
        """Let preparation finish and held plays go out (jobs run inline, but
        a demo track's audio must exist before its renders can start)."""
        session = self.session
        for _ in range(SETTLE_POLLS):
            session.poll()
            busy = session.holds or session.pending_bpm is not None or (session.prep and session.prep.outstanding)
            if not busy:
                break
        session.engine.flush()

    def render(self, bars: float, script: list[ScriptCommand] = (), autostart: bool = True) -> tuple[np.ndarray, RenderReport]:
        session, engine = self.session, self.engine
        self.settle()
        if autostart and not session.transport.running:
            self.interp.run("start()")
        target_beat = bars * session.transport.beats_per_bar
        pending = list(script)
        chunks: list[np.ndarray] = []
        stalled_blocks = 0
        while engine.position_beats < target_beat:
            position = engine.position_beats
            while pending and pending[0].beat <= position + 1e-9:
                self.interp.run(pending.pop(0).text)
            self.settle()
            if not session.transport.running:
                stalled_blocks += 1
                if stalled_blocks > 1 and not pending:
                    break  # the set stopped the transport and nothing will restart it
                if pending:
                    # fast-forward the script: nothing plays while stopped
                    self.interp.run(pending.pop(0).text)
                    continue
            else:
                stalled_blocks = 0
            chunks.append(session.advance(self.blocksize))
        # The limiter delays output by its lookahead: render that much more,
        # then drop it from the front so file sample N is transport sample N.
        latency = engine.latency_samples
        extra = 0
        while extra < latency:
            chunks.append(session.advance(self.blocksize))
            extra += self.blocksize
        audio = np.concatenate(chunks) if chunks else np.zeros((0, 2), dtype=np.float32)
        frames = event_sample(engine.tempo.beat_to_sample(target_beat))
        audio = audio[latency:latency + frames]
        report = RenderReport(frames=len(audio), samplerate=engine.samplerate,
                              peak=float(np.max(np.abs(audio))) if len(audio) else 0.0,
                              stats=session.last_status.stats if session.last_status else EngineStats())
        return audio, report


def build_session(paths: Paths, config: Config, log: LogFn, demo: bool) -> Session:
    engine = Engine(config.samplerate, 128.0, len(LANE_NAMES), render_audio=True,
                    limiter_ceiling_db=config.limiter_ceiling_db, master_gain=config.master_gain)
    return Session(paths, config, log, demo=demo, jobs=InlineJobRunner(), prepare=True,
                   engine=LocalEngineClient(engine), bpm=128.0)


def render_set_file(
    set_path: Path,
    bars: float,
    output: Path,
    *,
    demo: bool = False,
    script_text: str = "",
    paths: Optional[Paths] = None,
    config: Optional[Config] = None,
    log: Optional[LogFn] = None,
    autostart: bool = True,
) -> RenderReport:
    paths = paths or Paths.default()
    config = config or Config.load(paths)
    messages: list[tuple[str, str]] = []

    def collect(message: str, level: str = "info") -> None:
        messages.append((level, message))
        if log is not None:
            log(message, level)

    script = parse_script(script_text)
    session = build_session(paths, config, collect, demo)
    try:
        interp = Interpreter(session, Path(set_path).parent, log=collect)
        interp.run(Path(set_path).read_text(encoding="utf-8"))
        audio, report = OfflineRenderer(session, interp, config.blocksize).render(bars, script, autostart)
    finally:
        session.close()
    write_audio(output, audio, config.samplerate, bit_depth=24)
    report.errors = [m for level, m in messages if level == "error"]
    report.warnings = [m for level, m in messages if level == "warn"]
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m clidj render",
                                     description="Render a set file offline to a WAV file.")
    parser.add_argument("set_path", type=Path, help="the .djs set file to run")
    parser.add_argument("--bars", type=float, default=32, help="how many bars to render (default 32)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="output WAV (default renders/<set>.wav)")
    parser.add_argument("--script", type=Path, default=None, help="REPL input to replay: 'BAR[.BEAT]: command' lines")
    parser.add_argument("--demo", action="store_true", help="use the demo library")
    parser.add_argument("--no-start", action="store_true", help="don't call start() after loading the set")
    parser.add_argument("-v", "--verbose", action="store_true", help="print the console log while rendering")
    args = parser.parse_args(argv)

    output = args.output or Path("renders") / f"{args.set_path.stem}.wav"
    script_text = args.script.read_text(encoding="utf-8") if args.script else ""

    def echo(message: str, level: str = "info") -> None:
        if args.verbose or level != "info":
            print(f"[{level}] {message}", file=sys.stderr if level != "info" else sys.stdout)

    try:
        report = render_set_file(args.set_path, args.bars, output, demo=args.demo, script_text=script_text,
                                 log=echo, autostart=not args.no_start)
    except (OSError, ScriptError, ConfigError) as exc:
        print(f"render failed: {exc}", file=sys.stderr)
        return 1
    stats = report.stats
    print(f"wrote {output} ({report.seconds:.1f} s, peak {report.peak:.3f}); "
          f"late commands {stats.late_commands}, missing buffers {stats.missing_buffers}, engine errors {stats.errors}")
    return 1 if report.errors or stats.errors else 0
