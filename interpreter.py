"""AST-based safe interpreter for the DJ command language.

Never eval()/exec()s user input -- the parsed AST is walked and evaluated by
hand against a small whitelist of node types, names and callables.

`at`, `after`, `every` and `now` are special forms: the "what to run"
argument is captured as an unevaluated AST node (a Thunk) instead of being
evaluated immediately, so `at(33, xf(L1, L2, 8))` doesn't crossfade the
moment it's typed -- only when bar 33 actually arrives.

While a Thunk is firing (including a `now(...)` firing synchronously right
away), `_in_scheduled_context` is set so that nested lane/xf/bpm commands run
immediately instead of re-quantizing to *another* future boundary -- once
something has been scheduled for a precise beat, what's inside it should
happen exactly then, not be deferred a second time.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable, Optional

import snippets
from lanes import Lane, check_warnings
from library import Track
from scheduler import Scheduler, SchedulerError
from transport import Transport

LogFn = Callable[[str, str], None]

SPECIAL_FORMS = {"at", "after", "every", "now"}

ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.Call, ast.Name, ast.Constant,
    ast.keyword, ast.Attribute, ast.BinOp, ast.LShift, ast.UnaryOp, ast.USub,
    ast.List, ast.Tuple, ast.Load, ast.Store,
)

HELP_TEXT = """commands:
  start() / stop()              run / stop the transport
  bpm(128)                      change tempo (takes effect at the next bar)
  quant("bar")                  default quantize: beat / bar / phrase / none
  kick = snip(1, cue=2, bars=8, loop=True, role="drums")
  hook = snip("Glass Horizon", bar=33, bars=4, role="vocal")
  L1 << kick   /   L1.play(kick)      play a snippet on a lane (next boundary)
  L1.stop()                          stop a lane (next boundary)
  L1.gain(0.5)  L1.mute()  L1.unmute()   (immediate)
  L1.eq(lo=0, mid=1, hi=0.8)             (immediate, visual only)
  xf(L1, L2, bars=8)             crossfade from L1 to L2 over 8 bars
  at(33, expr)                   run expr at the head of bar 33
  after(4, expr)                 run expr 4 bars after the next bar head
  every(8, expr)                 run expr every 8 bars until cancel()led
  now(expr)                      run expr immediately, no quantizing
  queue()                        list pending events
  cancel(3) / cancel()           cancel one event, or everything
  snips()                        list defined snippets
  load_set("demo")               run sets/demo.djs
  clear() / help()               clear the log / show this text""".strip()


class CommandError(Exception):
    pass


class Thunk:
    """An unevaluated expression bound to the interpreter that will
    eventually evaluate it -- deliberately live (re-reads `env` at call
    time), not a snapshot of values taken when it was written."""

    __slots__ = ("_interp", "_node", "_source")

    def __init__(self, interp: "Interpreter", node: ast.AST, source: str) -> None:
        self._interp = interp
        self._node = node
        self._source = source

    def __call__(self):
        prev = self._interp._in_scheduled_context
        self._interp._in_scheduled_context = True
        try:
            return self._interp._eval(self._node)
        finally:
            self._interp._in_scheduled_context = prev

    def __repr__(self) -> str:  # used as the scheduler's queue description
        return self._source


def _validate(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise CommandError(f"unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise CommandError(f"cannot access name '{node.id}'")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise CommandError(f"cannot access attribute '{node.attr}'")
        if isinstance(node, ast.Assign) and (
            len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name)
        ):
            raise CommandError("assignment target must be a single plain name")
        if isinstance(node, ast.BinOp) and not isinstance(node.op, ast.LShift):
            raise CommandError("only the '<<' operator is supported")
        if isinstance(node, ast.UnaryOp) and not isinstance(node.op, ast.USub):
            raise CommandError("only unary '-' is supported")


class Interpreter:
    def __init__(
        self,
        transport: Transport,
        scheduler: Scheduler,
        lanes: dict[str, Lane],
        library: list[Track],
        log: LogFn,
        sets_dir: Path,
        on_clear: Callable[[], None] = lambda: None,
    ) -> None:
        self.transport = transport
        self.scheduler = scheduler
        self.lanes = lanes
        self.library = library
        self.log = log
        self.sets_dir = sets_dir
        self.on_clear = on_clear
        self.quant_mode = "bar"
        self._in_scheduled_context = False

        self.env: dict[str, object] = dict(lanes)
        self.functions: dict[str, Callable] = self._build_functions()
        self.reserved_names = set(self.env) | set(self.functions) | SPECIAL_FORMS
        self.method_whitelist: dict[type, dict[str, Callable]] = {
            Lane: {
                "play": lambda lane, snippet: self._lane_play(lane, snippet),
                "stop": lambda lane: self._lane_stop(lane),
                "gain": lambda lane, value: self._lane_gain(lane, value),
                "mute": lambda lane: self._lane_mute(lane),
                "unmute": lambda lane: self._lane_unmute(lane),
                "eq": lambda lane, lo=None, mid=None, hi=None: self._lane_eq(lane, lo=lo, mid=mid, hi=hi),
            }
        }

    # ---- public entry point -------------------------------------------------

    def run(self, text: str) -> None:
        try:
            tree = ast.parse(text, mode="exec")
        except SyntaxError as exc:
            self.log(f"syntax error: {exc.msg}", "error")
            return
        try:
            _validate(tree)
            for stmt in tree.body:
                self._exec_stmt(stmt)
        except CommandError as exc:
            self.log(str(exc), "error")
        except Exception as exc:  # noqa: BLE001 -- malformed input must not crash the REPL
            self.log(f"error: {exc}", "error")

    # ---- statement / expression evaluation -----------------------------------

    def _exec_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            name = stmt.targets[0].id
            if name.startswith("_"):
                raise CommandError(f"cannot assign to '{name}'")
            if name in self.reserved_names:
                raise CommandError(f"cannot reassign built-in name '{name}'")
            self.env[name] = self._eval(stmt.value)
            return
        if isinstance(stmt, ast.Expr):
            self._eval(stmt.value)
            return
        raise CommandError(f"unsupported statement: {type(stmt).__name__}")

    def _eval(self, node: ast.AST):
        if isinstance(node, ast.Expr):
            return self._eval(node.value)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in self.env:
                raise CommandError(f"undefined name '{node.id}'")
            return self.env[node.id]
        if isinstance(node, ast.List):
            return [self._eval(e) for e in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._eval(e) for e in node.elts)
        if isinstance(node, ast.UnaryOp):
            value = self._eval(node.operand)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise CommandError(f"unary '-' needs a number, got {type(value).__name__}")
            return -value
        if isinstance(node, ast.BinOp):
            return self._lshift(self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.Call):
            return self._eval_call(node)
        raise CommandError(f"unsupported syntax: {type(node).__name__}")

    def _lshift(self, left, right):
        if not isinstance(left, Lane):
            raise CommandError("'<<' needs a lane on the left, e.g. L1 << kick")
        return self._lane_play(left, right)

    def _eval_call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname in SPECIAL_FORMS:
                return self._eval_special_form(fname, node)
            if fname not in self.functions:
                raise CommandError(f"unknown function: {fname}()")
            args = [self._eval(a) for a in node.args]
            kwargs = {kw.arg: self._eval(kw.value) for kw in node.keywords}
            return self.functions[fname](*args, **kwargs)

        if isinstance(node.func, ast.Attribute):
            obj = self._eval(node.func.value)
            method = node.func.attr
            handlers = self.method_whitelist.get(type(obj))
            if not handlers or method not in handlers:
                raise CommandError(f"no such method: {type(obj).__name__}.{method}()")
            args = [self._eval(a) for a in node.args]
            kwargs = {kw.arg: self._eval(kw.value) for kw in node.keywords}
            return handlers[method](obj, *args, **kwargs)

        raise CommandError("unsupported call target")

    def _eval_special_form(self, fname: str, node: ast.Call):
        if node.keywords:
            raise CommandError(f"{fname}() does not take keyword arguments")

        if fname == "now":
            if len(node.args) != 1:
                raise CommandError("now(expr) takes exactly one argument")
            return Thunk(self, node.args[0], ast.unparse(node.args[0]))()

        if len(node.args) != 2:
            raise CommandError(f"{fname}(n, expr) takes exactly two arguments")
        n = self._eval(node.args[0])
        if not isinstance(n, (int, float)) or isinstance(n, bool):
            raise CommandError(f"{fname}()'s first argument must be a number")
        thunk = Thunk(self, node.args[1], ast.unparse(node.args[1]))

        try:
            if fname == "at":
                return self.scheduler.schedule_at(self.transport.beats_at_bar(n), thunk, str(thunk))
            if fname == "after":
                return self.scheduler.schedule_after_bars(n, thunk, str(thunk))
            return self.scheduler.schedule_every_bars(n, thunk, str(thunk))
        except SchedulerError as exc:
            raise CommandError(str(exc)) from None

    # ---- quantized dispatch, shared by lane play/stop and xf -----------------

    def _quantized(self, description: str, action: Callable[[], None]) -> int:
        """Ambient-quantized scheduling, except once we're already running
        inside a scheduled Thunk (an at/after/every/now firing): then the
        beat has already been chosen by the outer schedule, so just run."""
        if self._in_scheduled_context:
            action()
            return -1
        return self.scheduler.schedule_default(self.quant_mode, action, description)

    def _log_scheduled(self, description: str, event_id: int) -> None:
        if event_id == -1:
            self.log(f"{description} (now)", "info")
            return
        target = self.transport.next_boundary_beats(self.quant_mode)
        self.log(f"{description} @ bar {self.transport.bar_at(target)}", "info")

    # ---- lane commands ---------------------------------------------------

    def _lane_play(self, lane: Lane, snippet) -> None:
        if not isinstance(snippet, snippets.Snippet):
            raise CommandError(f"{lane.name} can only play a snippet (got {type(snippet).__name__})")
        description = f"{lane.name} << {snippet.name}"

        def action() -> None:
            lane.start_snippet(snippet, self.transport)
            others = [other for name, other in self.lanes.items() if name != lane.name]
            for message in check_warnings(lane, others, self.transport.bpm):
                self.log(message, "warn")
            self.log(f"{lane.name} now playing {snippet.name}", "info")

        self._log_scheduled(description, self._quantized(description, action))

    def _lane_stop(self, lane: Lane) -> None:
        description = f"{lane.name}.stop()"
        self._log_scheduled(description, self._quantized(description, lane.stop))

    def _lane_gain(self, lane: Lane, value) -> None:
        lane.gain = max(0.0, min(1.0, float(value)))
        self.log(f"{lane.name}.gain -> {lane.gain:.2f}", "info")

    def _lane_mute(self, lane: Lane) -> None:
        lane.muted = True
        self.log(f"{lane.name} muted", "info")

    def _lane_unmute(self, lane: Lane) -> None:
        lane.muted = False
        self.log(f"{lane.name} unmuted", "info")

    def _lane_eq(self, lane: Lane, lo=None, mid=None, hi=None) -> None:
        if lo is not None:
            lane.lo = float(lo)
        if mid is not None:
            lane.mid = float(mid)
        if hi is not None:
            lane.hi = float(hi)
        self.log(f"{lane.name}.eq(lo={lane.lo:.2f}, mid={lane.mid:.2f}, hi={lane.hi:.2f})", "info")

    # ---- global commands ---------------------------------------------------

    def _build_functions(self) -> dict[str, Callable]:
        return {
            "start": self._cmd_start,
            "stop": self._cmd_stop,
            "bpm": self._cmd_bpm,
            "quant": self._cmd_quant,
            "snip": self._cmd_snip,
            "xf": self._cmd_xf,
            "queue": self._cmd_queue,
            "cancel": self._cmd_cancel,
            "snips": self._cmd_snips,
            "load_set": self._cmd_load_set,
            "clear": self._cmd_clear,
            "help": self._cmd_help,
        }

    def _cmd_start(self) -> None:
        self.transport.start()
        self.log("transport running", "info")

    def _cmd_stop(self) -> None:
        self.transport.stop()
        self.log("transport stopped", "info")

    def _cmd_bpm(self, value) -> None:
        value = float(value)
        if value <= 0:
            raise CommandError("bpm must be positive")
        if self._in_scheduled_context or not self.transport.running:
            self.transport.bpm = value
            self.log(f"bpm -> {value:.1f}", "info")
            return

        target = self.transport.next_boundary_beats("bar")

        def action() -> None:
            self.transport.bpm = value
            self.log(f"bpm -> {value:.1f}", "info")

        self.scheduler.schedule_at(target, action, f"bpm({value:g})")
        self.log(f"bpm({value:g}) scheduled for bar {self.transport.bar_at(target)}", "info")

    def _cmd_quant(self, mode) -> None:
        if mode not in ("beat", "bar", "phrase", "none"):
            raise CommandError(f"unknown quantize mode {mode!r} (use beat/bar/phrase/none)")
        self.quant_mode = mode
        self.log(f"quant -> {mode}", "info")

    def _cmd_snip(self, track, *, cue=None, bar=None, bars=8, loop=False, role="other", name=None):
        if name is not None and name in self.reserved_names:
            raise CommandError(f"cannot name a snippet '{name}', it's a built-in")
        try:
            snippet = snippets.snip(
                track, self.library, cue=cue, bar=bar, bars=bars, loop=loop, role=role, name=name
            )
        except snippets.SnippetError as exc:
            raise CommandError(str(exc)) from None
        self.log(
            f"{snippet.name} = {snippet.track.title} "
            f"[{snippet.start_beat:g}+{snippet.length_beats:g} beats] role={snippet.role}",
            "info",
        )
        return snippet

    def _cmd_xf(self, from_lane, to_lane, bars=8) -> None:
        if not isinstance(from_lane, Lane) or not isinstance(to_lane, Lane):
            raise CommandError("xf(from_lane, to_lane, bars=...) needs two lanes")
        description = f"xf({from_lane.name}, {to_lane.name}, {bars:g})"

        def action() -> None:
            self.scheduler.start_automation(from_lane, to_lane, float(bars), description)
            self.log(f"{description} started", "info")

        self._log_scheduled(description, self._quantized(description, action))

    def _cmd_queue(self) -> None:
        items = self.scheduler.pending(limit=5)
        if not items:
            self.log("queue: empty", "info")
            return
        for event_id, beat, desc in items:
            self.log(f"#{event_id} @{self.transport.display_at(beat)} {desc}", "info")

    def _cmd_cancel(self, event_id=None) -> None:
        try:
            message = self.scheduler.cancel(int(event_id) if event_id is not None else None)
        except SchedulerError as exc:
            raise CommandError(str(exc)) from None
        self.log(message, "info")

    def _cmd_snips(self) -> None:
        entries = [(name, value) for name, value in self.env.items() if isinstance(value, snippets.Snippet)]
        if not entries:
            self.log("no snippets defined", "info")
            return
        for name, snippet in entries:
            bars = snippet.length_beats / snippets.BEATS_PER_BAR
            self.log(
                f"{name}: {snippet.track.title} role={snippet.role} bars={bars:g} "
                f"key={snippet.key} loop={snippet.loop}",
                "info",
            )

    def _cmd_load_set(self, name: str) -> None:
        path = self.sets_dir / f"{name}.djs"
        if not path.is_file():
            raise CommandError(f"set file not found: {path.name}")
        self.log(f"loading set '{name}'", "info")
        self.run(path.read_text(encoding="utf-8"))

    def _cmd_clear(self) -> None:
        self.on_clear()

    def _cmd_help(self) -> None:
        for line in HELP_TEXT.splitlines():
            self.log(line, "info")
