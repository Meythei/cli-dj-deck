from pathlib import Path

import pytest

from interpreter import Interpreter
from lanes import LANE_NAMES, Lane
from library import DEMO_LIBRARY
from scheduler import Scheduler
from snippets import Snippet
from transport import Transport


def make_interp(sets_dir: Path | None = None):
    logs: list[tuple[str, str]] = []

    def log(message: str, level: str = "info") -> None:
        logs.append((level, message))

    transport = Transport(bpm=120.0, beats_per_bar=4)
    scheduler = Scheduler(transport, log)
    lanes = {name: Lane(name) for name in LANE_NAMES}
    interp = Interpreter(transport, scheduler, lanes, DEMO_LIBRARY, log, sets_dir or Path("."))
    return interp, transport, scheduler, lanes, logs


def has_error(logs) -> bool:
    return any(level == "error" for level, _ in logs)


# ---- normal-path syntax ---------------------------------------------------


def test_assignment_defines_a_snippet():
    interp, *_, logs = make_interp()
    interp.run("kick = snip(1, cue=1, bars=8, loop=True, role='drums')")
    assert not has_error(logs)
    assert isinstance(interp.env["kick"], Snippet)
    assert interp.env["kick"].role == "drums"


def test_lshift_and_dot_play_schedule_the_same_way():
    interp_a, transport_a, scheduler_a, *_ = make_interp()
    transport_a.start()
    interp_a.run("kick = snip(1, cue=1, bars=8, loop=True)")
    interp_a.run("L1 << kick")
    assert len(scheduler_a.pending()) == 1

    interp_b, transport_b, scheduler_b, *_ = make_interp()
    transport_b.start()
    interp_b.run("kick = snip(1, cue=1, bars=8, loop=True)")
    interp_b.run("L1.play(kick)")
    assert len(scheduler_b.pending()) == 1


def test_keyword_arguments():
    interp, *_ = make_interp()
    interp.run("hook = snip(3, bar=33, bars=4, role='vocal', loop=True)")
    hook = interp.env["hook"]
    assert hook.role == "vocal"
    assert hook.loop is True
    assert hook.length_beats == pytest.approx(16.0)


def test_semicolon_separated_statements():
    interp, transport, scheduler, lanes, logs = make_interp()
    interp.run("bpm(140); quant('phrase')")
    assert not has_error(logs)
    assert transport.bpm == pytest.approx(140.0)  # stopped transport -> immediate
    assert interp.quant_mode == "phrase"


def test_multiline_statements():
    interp, transport, scheduler, lanes, logs = make_interp()
    interp.run("kick = snip(1, cue=1, bars=8, loop=True)\nbass = snip(2, cue=1, bars=8, loop=True)")
    assert not has_error(logs)
    assert "kick" in interp.env and "bass" in interp.env


# ---- deferred evaluation ---------------------------------------------------


def test_at_does_not_run_its_body_until_the_scheduled_bar_arrives():
    interp, transport, scheduler, lanes, logs = make_interp()
    transport.start()
    interp.run("kick = snip(1, cue=1, bars=32, loop=True)")
    interp.run("hook = snip(3, bar=1, bars=32, loop=True)")
    interp.run("L1 << kick")
    interp.run("at(2, xf(L1, L2, 4))")

    assert scheduler.automations == []  # not run yet, just queued

    seconds_to_bar_2 = transport.beats_at_bar(2) * 60.0 / transport.bpm
    scheduler.tick(seconds_to_bar_2 + 0.01)

    assert len(scheduler.automations) == 1
    assert scheduler.automations[0].to_lane is lanes["L2"]


def test_now_bypasses_quantization_entirely():
    interp, transport, scheduler, lanes, logs = make_interp()
    transport.start()
    interp.run("kick = snip(1, cue=1, bars=8, loop=True)")
    interp.run("now(L1 << kick)")
    assert lanes["L1"].snippet is not None
    assert scheduler.pending() == []  # ran immediately, never queued


def test_command_inside_at_does_not_requantize_a_second_time():
    """L1 << kick nested in at(2, ...) must start exactly at bar 2, not bar 3
    from an inner ambient-quantize re-scheduling itself another bar out."""
    interp, transport, scheduler, lanes, logs = make_interp()
    transport.start()
    interp.run("kick = snip(1, cue=1, bars=32, loop=True)")
    interp.run("at(2, L1 << kick)")

    seconds_to_bar_2 = transport.beats_at_bar(2) * 60.0 / transport.bpm
    scheduler.tick(seconds_to_bar_2 + 0.01)

    assert lanes["L1"].snippet is not None
    # Started within this tick's overshoot of bar 2's head, not pushed out to bar 3.
    assert transport.beats_at_bar(2) <= lanes["L1"].started_at_beat < transport.beats_at_bar(3)


# ---- rejections -------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        '__import__("os")',
        "L1.__class__",
        "[x for x in y]",
        "lambda: 1",
        "import os",
        'open("x")',
        "1 + 1",  # only << is an allowed binop
        "-L1",  # unary minus on a non-number is nonsensical but should error, not crash
    ],
)
def test_dangerous_or_unsupported_syntax_is_rejected(code):
    interp, *_, logs = make_interp()
    interp.run(code)
    assert has_error(logs), f"expected an error for: {code}"


def test_assignment_to_builtin_lane_name_is_rejected():
    interp, transport, scheduler, lanes, logs = make_interp()
    interp.run("L1 = 5")
    assert has_error(logs)
    assert isinstance(lanes["L1"], Lane)


def test_assignment_to_builtin_function_name_is_rejected():
    interp, *_, logs = make_interp()
    interp.run("snip = 5")
    assert has_error(logs)
    assert "snip" in interp.functions


def test_undefined_name_is_rejected():
    interp, *_, logs = make_interp()
    interp.run("L1 << nonexistent")
    assert has_error(logs)


def test_unknown_function_is_rejected():
    interp, *_, logs = make_interp()
    interp.run("frobnicate()")
    assert has_error(logs)


# ---- set files -------------------------------------------------------


def test_load_set_runs_a_djs_file(tmp_path: Path):
    (tmp_path / "mini.djs").write_text(
        "kick = snip(1, cue=1, bars=8, loop=True, role='drums')\n"
        "L1 << kick\n",
        encoding="utf-8",
    )
    interp, transport, scheduler, lanes, logs = make_interp(sets_dir=tmp_path)
    transport.start()
    interp.run('load_set("mini")')
    assert not has_error(logs)
    assert "kick" in interp.env
    assert len(scheduler.pending()) == 1


def test_load_set_missing_file_is_an_error():
    interp, *_, logs = make_interp(sets_dir=Path("."))
    interp.run('load_set("does_not_exist")')
    assert has_error(logs)


def test_demo_set_loads_with_expected_snippets_and_events():
    sets_dir = Path(__file__).resolve().parents[1] / "sets"
    interp, transport, scheduler, lanes, logs = make_interp(sets_dir=sets_dir)
    interp.run('load_set("demo")')

    assert not has_error(logs), [msg for level, msg in logs if level == "error"]
    for name in ("kick", "bass", "hook", "riser", "drop"):
        assert name in interp.env, f"missing snippet '{name}'"

    assert transport.bpm == pytest.approx(128.0)
    assert lanes["L1"].snippet is interp.env["kick"]  # bare command ran immediately (stopped)
    assert len(scheduler.pending()) == 5  # the five at(...) reservations
