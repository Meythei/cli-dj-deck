"""Regression tests for the issues found in the review before the real-audio
work (docs/TASK_real-audio.md, section 5). Each test was written to fail on
the code as it was, before the corresponding fix."""
from pathlib import Path

import pytest

from clidj.interpreter import Interpreter
from clidj.lanes import LANE_NAMES, Lane
from clidj.library import DEMO_LIBRARY
from clidj.scheduler import Scheduler
from clidj.transport import Transport

SETS_DIR = Path(__file__).resolve().parents[1] / "sets"


def make_interp(sets_dir: Path | None = None, bpm: float = 120.0):
    logs: list[tuple[str, str]] = []

    def log(message: str, level: str = "info") -> None:
        logs.append((level, message))

    transport = Transport(bpm=bpm, beats_per_bar=4)
    scheduler = Scheduler(transport, log)
    lanes = {name: Lane(name) for name in LANE_NAMES}
    interp = Interpreter(transport, scheduler, lanes, DEMO_LIBRARY, log, sets_dir or Path("."))
    return interp, transport, scheduler, lanes, logs


def errors(logs):
    return [message for level, message in logs if level == "error"]


# ---- 2. scheduled start beats land exactly on the boundary -----------------


def test_demo_set_lanes_start_exactly_on_their_scheduled_bar_heads():
    interp, transport, scheduler, lanes, logs = make_interp(sets_dir=SETS_DIR)
    interp.run('load_set("demo")')
    interp.run("start()")
    # Tick like the UI does (30 fps) so every firing overshoots its boundary.
    for _ in range(int(40 * 60 / 128 * 4 * 30)):
        scheduler.tick(1 / 30)
    assert lanes["L3"].started_at_beat == pytest.approx(64.0, abs=1e-9)
    assert lanes["L4"].started_at_beat == pytest.approx(128.0, abs=1e-9)
    assert lanes["L2"].started_at_beat == pytest.approx(32.0, abs=1e-9)


def test_quantized_play_starts_on_the_boundary_not_the_tick_that_crossed_it():
    interp, transport, scheduler, lanes, logs = make_interp()
    interp.run("kick = snip(1, cue=1, bars=8, loop=True)")
    interp.run("start()")
    scheduler.tick(0.3)  # 0.6 beats in
    interp.run("L1 << kick")
    scheduler.tick(1.77)  # crosses beat 4 by 0.14 beats
    assert lanes["L1"].started_at_beat == pytest.approx(4.0, abs=1e-9)


def test_xf_start_and_end_beats_come_from_the_scheduled_boundary():
    interp, transport, scheduler, lanes, logs = make_interp()
    interp.run("kick = snip(1, cue=1, bars=32, loop=True)")
    interp.run("L1 << kick")
    interp.run("start()")
    interp.run("at(3, xf(L1, L2, 2))")
    scheduler.tick(8.0 * 60 / 120 + 0.021)
    [automation] = scheduler.automations
    assert automation.start_beat == pytest.approx(8.0, abs=1e-9)
    assert automation.end_beat == pytest.approx(16.0, abs=1e-9)


# ---- 3. assigned snippets take the variable name --------------------------


def test_assigned_snippet_is_named_after_the_variable():
    interp, *_, logs = make_interp()
    interp.run("kick = snip(1, cue=1, bars=8, loop=True)")
    assert interp.env["kick"].name == "kick"
    assert any("kick" in message for level, message in logs if level == "info")


def test_explicit_name_wins_over_the_variable_name():
    interp, *_ = make_interp()
    interp.run("kick = snip(1, cue=1, bars=8, name='big_kick')")
    assert interp.env["kick"].name == "big_kick"


# ---- 4. after() into the past ---------------------------------------------


def test_negative_after_is_an_error_and_never_queued():
    interp, transport, scheduler, lanes, logs = make_interp()
    interp.run("kick = snip(1, cue=1, bars=8, loop=True)")
    interp.run("start()")
    interp.run("after(-4, L1 << kick)")
    assert errors(logs)
    assert scheduler.pending() == []


def test_at_bar_below_one_is_an_error():
    interp, transport, scheduler, lanes, logs = make_interp()
    interp.run("kick = snip(1, cue=1, bars=8, loop=True)")
    interp.run("at(0, L1 << kick)")
    assert errors(logs)
    assert scheduler.pending() == []


# ---- 5. at(1, ...) written before start() ---------------------------------


def test_at_bar_one_before_start_fires_the_moment_transport_starts():
    interp, transport, scheduler, lanes, logs = make_interp()
    interp.run("kick = snip(1, cue=1, bars=8, loop=True)")
    interp.run("at(1, L1 << kick)")
    assert not any(level == "warn" for level, _ in logs)
    interp.run("start()")
    scheduler.tick(1 / 30)
    assert lanes["L1"].snippet is interp.env["kick"]
    assert lanes["L1"].started_at_beat == pytest.approx(0.0, abs=1e-9)


# ---- 6. every() with a tiny interval ---------------------------------------


def test_every_below_one_beat_is_rejected():
    interp, transport, scheduler, lanes, logs = make_interp()
    interp.run("every(0.001, queue())")
    assert errors(logs)
    assert scheduler.pending() == []


def test_one_tick_fires_a_bounded_number_of_events():
    interp, transport, scheduler, lanes, logs = make_interp()
    interp.run("start()")
    interp.run("every(0.25, queue())")  # one firing per beat
    scheduler.tick(100_000.0)  # a pathological stall: 200k beats in one tick
    fired = sum(1 for level, message in logs if message.startswith("queue:") or message.startswith("#"))
    assert 0 < fired <= Scheduler.MAX_FIRES_PER_POLL
    assert any("fire" in message for message in errors(logs))


# ---- 7. load_set recursion ---------------------------------------------------


def test_self_loading_set_stops_at_the_nesting_limit(tmp_path: Path):
    (tmp_path / "loop.djs").write_text('load_set("loop")\n', encoding="utf-8")
    interp, transport, scheduler, lanes, logs = make_interp(sets_dir=tmp_path)
    interp.run('load_set("loop")')
    errs = errors(logs)
    assert len(errs) == 1
    assert "nest" in errs[0]
    assert "recursion" not in errs[0].lower()
    assert len(logs) <= Interpreter.MAX_SET_NESTING + 2


# ---- 10. log wording and order -----------------------------------------------


def test_play_log_comes_before_now_playing_when_run_immediately():
    interp, transport, scheduler, lanes, logs = make_interp()
    interp.run("kick = snip(1, cue=1, bars=8, loop=True)")
    logs.clear()
    interp.run("L1 << kick")
    messages = [message for _, message in logs]
    play = next(i for i, m in enumerate(messages) if "L1 << kick" in m)
    now_playing = next(i for i, m in enumerate(messages) if "now playing" in m)
    assert play < now_playing
    assert "(now)" not in messages[play]


def test_fired_reservation_is_not_labelled_as_now():
    interp, transport, scheduler, lanes, logs = make_interp()
    interp.run("kick = snip(1, cue=1, bars=8, loop=True)")
    interp.run("start()")
    interp.run("at(2, L1 << kick)")
    logs.clear()
    scheduler.tick(4 * 60 / 120 + 0.01)
    fired = [message for _, message in logs if "L1 << kick" in message]
    assert fired, logs
    assert all("(now)" not in message for message in fired)
    assert any("002.1" in message for message in fired)


@pytest.mark.parametrize(
    "command, bar",
    [("at(9, L1 << kick)", 9), ("after(4, L1 << kick)", 6), ("every(8, L1 << kick)", 2)],
)
def test_reservations_log_the_bar_they_were_queued_for(command, bar):
    interp, transport, scheduler, lanes, logs = make_interp()
    interp.run("kick = snip(1, cue=1, bars=8, loop=True)")
    interp.run("start()")
    scheduler.tick(0.5)  # inside bar 1
    logs.clear()
    interp.run(command)
    assert any(f"bar {bar}" in message for level, message in logs if level == "info"), logs
