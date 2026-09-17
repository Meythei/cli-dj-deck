"""Quantized scheduler: turns "do X at the next bar" into an actual queue.

Everything here reads time from a Transport and nothing else. `poll(limit)`
fires every event whose time is at or before `limit`, in time order --
including, after one big tick, several boundaries at once (a paused/slow UI
shouldn't cause events to be skipped, only to fire late but in the right
order).

While an event's action runs, `firing_beat` holds the beat the event was
*scheduled* for. Actions must use that as their start beat rather than the
transport's current position: the tick that crosses a boundary almost always
overshoots it, and a lane started from the overshoot position would be a few
dozen milliseconds late for good.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .transport import EPSILON, Transport

LogFn = Callable[[str, str], None]  # (message, level) where level in info/warn/error

# Smallest every() interval, in beats. Anything smaller is never musically
# useful and, with a long enough stall, turns one poll into a busy loop.
MIN_EVERY_BEATS = 1.0


class SchedulerError(Exception):
    pass


@dataclass
class ScheduledEvent:
    id: int
    fire_at_beats: float
    action: Callable[[], None]
    description: str
    recurring_bars: Optional[float] = None  # set only for every()'s repeating events


class Scheduler:
    # Safety valve: one poll never fires more than this many events. The rest
    # stay queued for the next poll, so the UI keeps drawing and can be used
    # to cancel() whatever is flooding the queue.
    MAX_FIRES_PER_POLL = 256

    def __init__(self, transport: Transport, log: LogFn) -> None:
        self.transport = transport
        self.log = log
        self.events: list[ScheduledEvent] = []
        self.firing_beat: Optional[float] = None
        self._next_id = 1
        self._recurring_active: dict[int, bool] = {}

    def new_id(self) -> int:
        """Ids for events and for things the session tracks alongside them
        (crossfades), from one sequence so cancel(n) is unambiguous."""
        event_id = self._next_id
        self._next_id += 1
        return event_id

    # ---- scheduling entry points -------------------------------------------------

    def schedule_default(self, quant_mode: str, action: Callable[[], None], description: str) -> int:
        """Ambient-quantized scheduling: the next boundary of `quant_mode`, or
        immediately if the transport is stopped (there is no future boundary
        arriving in real time)."""
        if not self.transport.running:
            self._run(action, description, self.transport.position_beats)
            return -1
        target = self.transport.next_boundary_beats(quant_mode)
        return self._enqueue(target, action, description)

    def schedule_at(self, target_beats: float, action: Callable[[], None], description: str) -> int:
        """Schedule for an absolute beat position (used by at() and bpm()).

        A target behind the transport is moved to the next bar with a
        warning. A target exactly *at* a stopped transport's position is not
        in the past -- it fires the moment the transport starts."""
        position = self.transport.position_beats
        margin = self.transport.commit_margin_beats if self.transport.running else 0.0
        passed = target_beats < position - EPSILON or (
            self.transport.running and target_beats <= position + margin + EPSILON
        )
        if passed:
            fallback = self.transport.next_boundary_beats("bar")
            bar = self.transport.bar_at(fallback)
            self.log(f"{description}: target already passed, rescheduled to bar {bar}", "warn")
            target_beats = fallback
        return self._enqueue(target_beats, action, description)

    def schedule_after_bars(self, bars: float, action: Callable[[], None], description: str) -> int:
        if bars < 0:
            raise SchedulerError("after() needs a bar count >= 0 (it cannot schedule into the past)")
        return self._enqueue(self.after_bars_target(bars), action, description)

    def after_bars_target(self, bars: float) -> float:
        return self.transport.next_boundary_beats("bar") + bars * self.transport.beats_per_bar

    def schedule_every_bars(self, bars: float, action: Callable[[], None], description: str) -> int:
        if bars * self.transport.beats_per_bar < MIN_EVERY_BEATS - EPSILON:
            minimum = MIN_EVERY_BEATS / self.transport.beats_per_bar
            raise SchedulerError(f"every() needs an interval of at least {minimum:g} bars (1 beat)")
        event_id = self.new_id()
        self._recurring_active[event_id] = True
        target = self.transport.next_boundary_beats("bar")
        self.events.append(ScheduledEvent(event_id, target, action, description, recurring_bars=bars))
        return event_id

    # ---- per-frame update ----------------------------------------------------

    def tick(self, dt_seconds: float) -> None:
        """Advance a free-running Transport and fire what became due (used
        when no engine drives the clock, e.g. in unit tests)."""
        _, now = self.transport.advance(dt_seconds)
        self.poll(now)

    def poll(self, limit_beats: float) -> int:
        """Fire every queued event scheduled at or before `limit_beats`, in
        (beat, id) order. A stopped transport fires nothing: events sitting
        exactly at its position wait for start(), which polls again. Returns
        the number of events fired."""
        if not self.transport.running:
            return 0
        fired = 0
        # Loop rather than a single pass: a recurring event's freshly
        # rescheduled next occurrence can itself be due in the same poll.
        while True:
            due = [e for e in self.events if e.fire_at_beats <= limit_beats + EPSILON]
            if not due:
                return fired
            due.sort(key=lambda e: (e.fire_at_beats, e.id))
            for event in due:
                if fired >= self.MAX_FIRES_PER_POLL:
                    self.log(
                        f"scheduler: fired {fired} events in one tick; deferring the rest "
                        "(cancel() a runaway every() if this repeats)",
                        "error",
                    )
                    return fired
                self.events.remove(event)
                self._fire(event)
                fired += 1

    def _fire(self, event: ScheduledEvent) -> None:
        self._run(event.action, event.description, event.fire_at_beats)
        if event.recurring_bars is not None and self._recurring_active.get(event.id, False):
            next_target = event.fire_at_beats + event.recurring_bars * self.transport.beats_per_bar
            self.events.append(
                ScheduledEvent(event.id, next_target, event.action, event.description, event.recurring_bars)
            )

    def _run(self, action: Callable[[], None], description: str, beat: float) -> None:
        previous = self.firing_beat
        self.firing_beat = beat
        try:
            action()
        except Exception as exc:  # noqa: BLE001 -- one bad command must not kill the set
            self.log(f"error running '{description}': {exc}", "error")
        finally:
            self.firing_beat = previous

    # ---- introspection ---------------------------------------------------

    def _enqueue(self, target_beats: float, action: Callable[[], None], description: str) -> int:
        event_id = self.new_id()
        self.events.append(ScheduledEvent(event_id, target_beats, action, description))
        return event_id

    def pending(self, limit: Optional[int] = None) -> list[tuple[int, float, str]]:
        items = sorted(self.events, key=lambda e: (e.fire_at_beats, e.id))
        if limit is not None:
            items = items[:limit]
        return [(e.id, e.fire_at_beats, e.description) for e in items]

    def cancel(self, event_id: Optional[int] = None) -> str:
        if event_id is None:
            count = len(self.events)
            self.events.clear()
            self._recurring_active.clear()
            return f"cancelled {count} event(s)"

        found = any(e.id == event_id for e in self.events) or event_id in self._recurring_active
        if not found:
            raise SchedulerError(f"no such event #{event_id}")

        self.events = [e for e in self.events if e.id != event_id]
        self._recurring_active.pop(event_id, None)
        return f"cancelled #{event_id}"
