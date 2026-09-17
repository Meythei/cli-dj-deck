r"""Render the WAV files for the manual listening check
(docs/listening-checklist.md). Everything uses the demo library, whose audio
is synthesised deterministically, so the files are the same on every machine.

    .venv\Scripts\python scripts\render_listening.py            # all, into renders\listening\
    .venv\Scripts\python scripts\render_listening.py 03 05      # only some
"""
import argparse
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clidj.config import Paths  # noqa: E402
from clidj.offline import render_set_file  # noqa: E402


@dataclass(frozen=True)
class Scenario:
    key: str
    name: str
    bars: int
    set_text: str
    script: str = ""


# Demo tracks are 16-bar sections: 1-16 intro (kick), 17-32 build (+hats,
# bass), 33-48 drop (+stabs), 49-64 break (no kick). Track 1 is 128 BPM 8A,
# 8 is 128 BPM 8A, 3 is 130 BPM 8A, 4 is 122 BPM 3A, 6 is 132 BPM 10A.
SCENARIOS = [
    Scenario("01", "crossfade", 20, """
bpm(128)
a = snip(1, bar=33, bars=8, loop=True, role="drums")
b = snip(5, bar=33, bars=8, loop=True, role="drums")
L1 << a
at(5, L2 << b)
at(5, xf(L1, L2, bars=8))
"""),
    Scenario("02", "loop_seams", 16, """
bpm(128)
one_bar = snip(1, bar=35, bars=1, loop=True)
half_bar = snip(3, bar=41, bars=0.5, loop=True)
L1 << one_bar
at(9, L1.stop())
at(9, L2 << half_bar)
"""),
    Scenario("03", "stretch_quality", 16, """
bpm(130)
drop = snip(3, bar=33, bars=4, loop=True)
L1 << drop
""", script="""
4.3: bpm(122)
8.3: bpm(140)
12.3: bpm(150)
"""),
    Scenario("04", "eq_isolator", 16, """
bpm(128)
full = snip(8, bar=33, bars=16, loop=True)
L1 << full
at(3, L1.eq(lo=0))
at(5, L1.eq(lo=1, mid=0))
at(7, L1.eq(mid=1, hi=0))
at(9, L1.eq(hi=1))
at(11, L1.eq(lo=0.3, mid=0.3, hi=0.3))
at(13, L1.eq(lo=1, mid=1, hi=1))
""", script="""
14.2: L1.gain(0.2)
14.4: L1.gain(1.0)
15.2: L1.mute()
15.4: L1.unmute()
"""),
    Scenario("05", "tempo_change", 16, """
bpm(128)
drums = snip(1, bar=33, bars=4, loop=True)
bass = snip(8, bar=33, bars=4, loop=True)
L1 << drums
L2 << bass
prep()
at(5, bpm(132))
at(9, bpm(124))
at(13, bpm(128))
"""),
    Scenario("06", "replace_and_stop", 12, """
bpm(128)
a = snip(1, bar=33, bars=2, loop=True)
b = snip(6, bar=33, bars=2, loop=True)
riser = snip(4, bar=21, bars=2)
L1 << a
at(3, L1 << b)
at(5, L1 << a)
at(5, L2 << riser)
at(7, L1.stop())
at(9, L1 << b)
""", script="""
10.3: stop()
10.4: start()
"""),
    Scenario("07", "demo_set", 48, "load_set(\"demo\")\n"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("only", nargs="*", help="scenario numbers to render (default: all)")
    parser.add_argument("--out", type=Path, default=ROOT / "renders" / "listening")
    args = parser.parse_args()
    home = Path(tempfile.mkdtemp(prefix="clidj-listening-"))
    paths = Paths(home / "config", home / "data", home / "cache")  # demo audio cached once for all scenarios
    failed = 0
    for scenario in SCENARIOS:
        if args.only and scenario.key not in args.only:
            continue
        set_dir = home / "sets"
        set_dir.mkdir(exist_ok=True)
        (set_dir / "demo.djs").write_text((ROOT / "sets" / "demo.djs").read_text(encoding="utf-8"), encoding="utf-8")
        set_path = set_dir / f"{scenario.key}_{scenario.name}.djs"
        set_path.write_text(scenario.set_text.lstrip(), encoding="utf-8")
        output = args.out / f"{scenario.key}_{scenario.name}.wav"
        started = time.time()
        report = render_set_file(set_path, scenario.bars, output, demo=True, script_text=scenario.script,
                                 paths=paths, config=None)
        stats = report.stats
        problems = report.errors + ([f"{stats.late_commands} late commands"] if stats.late_commands else []) + \
            ([f"{stats.missing_buffers} missing buffers"] if stats.missing_buffers else [])
        failed += bool(problems)
        print(f"{output.name}: {report.seconds:.1f} s, peak {report.peak:.3f}, rendered in {time.time() - started:.1f} s"
              + (f"  PROBLEMS: {problems}" if problems else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
