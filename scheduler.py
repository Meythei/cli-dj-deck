"""Quantized scheduler: turns "do X at the next bar" into an actual queue.

Everything here reads time from a Transport and nothing else. A tick
processes whatever beat range the Transport just advanced through, firing
every event whose time falls in `(prev, now]` -- including, in a single big
tick, several boundaries at once (a paused/slow UI shouldn't cause events to
be skipped, only to fire late but in the right order).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from lanes import Lane
from transport import Transport

LogFn = Callable[[str, str], None]  # (message, level) where level in info/warn/error


class SchedulerError(Exception):
    pass


@dataclass
class ScheduledEvent:
    id: int
    fire_at_beats: float
    action: Callable[[], None]
    description: str
    recurring_bars: Optional[float] = None  # set only for every()'s repeating events


@dataclass
class Automation:
    id: int
    from_lane: Lane
    to_lane: Lane
    start_beat: float
    end_beat: float
    from_start_gain: float
    description: str


class Scheduler:
    def __init__(self, transport: Transport, log: LogFn) -> None:
        self.transport = transport
        self.log = log
        self.events: list[ScheduledEvent] = []
        self.automations: list[Automation] = []
        self._next_id = 1
        self._recurring_active: dict[int, bool] = {}
        self._lane_automation: dict[str, int] = {}  # lane name -> automation id targeting its gain

    def _new_id(self) -> int:
        event_id = self._next_id
        self._next_id += 1
        return event_id

    # ---- scheduling entry points -------------------------------------------------

    def schedule_default(self, quant_mode: str, action: Callable[[], None], description: str) -> int:
        """Ambient-quantized scheduling used by lane play/stop and xf: the
        next boundary of `quant_mode`, or immediately if the transport is
        stopped (there is no future boundary arriving in real time)."""
        if not self.transport.running:
            self._run(action, description)
            return -1
        target = self.transport.next_boundary_beats(quant_mode)
        return self._enqueue(target, action, description)

    def schedule_at(self, target_beats: float, action: Callable[[], None], description: str) -> int:
        """Schedule for an absolute beat position (used by at() and bpm())."""
        if target_beats <= self.transport.position_beats:
            fallback = self.transport.next_boundary_beats("bar")
            bar = self.transport.bar_at(fallback)
            self.log(f"{description}: target already passed, rescheduled to bar {bar}", "warn")
            target_beats = fallback
        return self._enqueue(target_beats, action, description)

    def schedule_after_bars(self, bars: float, action: Callable[[], None], description: str) -> int:
        target = self.transport.next_boundary_beats("bar") + bars * self.transport.beats_per_bar
        return self._enqueue(target, action, description)

    def schedule_every_bars(self, bars: float, action: Callable[[], None], description: str) -> int:
        if bars <= 0:
            raise SchedulerError("every() needs a positive number of bars")
        event_id = self._new_id()
        self._recurring_active[event_id] = True
        target = self.transport.next_boundary_beats("bar")
        self.events.append(ScheduledEvent(event_id, target, action, description, recurring_bars=bars))
        return event_id

    def start_automation(self, from_lane: Lane, to_lane: Lane, bars: float, description: str) -> int:
        """Linear crossfade: from_lane.gain current->0, to_lane.gain 0->1,
        over `bars` bars starting now. Overriding an in-flight automation on
        either lane logs a warning and replaces it."""
        for lane in (from_lane, to_lane):
            existing = self._lane_automation.get(lane.name)
            if existing is not None:
                self.log(f"{lane.name}: gain automation overridden by new xf", "warn")
                self.automations = [a for a in self.automations if a.id != existing]
                self._lane_automation.pop(lane.name, None)

        automation_id = self._new_id()
        start_beat = self.transport.position_beats
        automation = Automation(
            id=automation_id,
            from_lane=from_lane,
            to_lane=to_lane,
            start_beat=start_beat,
            end_beat=start_beat + bars * self.transport.beats_per_bar,
            from_start_gain=from_lane.gain,
            description=description,
        )
        to_lane.gain = 0.0
        self.automations.append(automation)
        self._lane_automation[from_lane.name] = automation_id
        self._lane_automation[to_lane.name] = automation_id
        return automation_id

    # ---- per-frame update ----------------------------------------------------

    def tick(self, dt_seconds: float) -> None:
        prev, now = self.transport.advance(dt_seconds)
        self._advance_automations(prev, now)
        # Loop rather than a single pass: a recurring event's freshly
        # rescheduled next occurrence can itself fall inside a large (prev, now]
        # jump, and must fire in the same tick rather than waiting a frame.
        while True:
            due = [e for e in self.events if prev < e.fire_at_beats <= now]
            if not due:
                break
            due.sort(key=lambda e: (e.fire_at_beats, e.id))
            for event in due:
                self.events.remove(event)
                self._fire(event)

    def _fire(self, event: ScheduledEvent) -> None:
        self._run(event.action, event.description)
        if event.recurring_bars is not None and self._recurring_active.get(event.id, False):
            next_target = event.fire_at_beats + event.recurring_bars * self.transport.beats_per_bar
            self.events.append(
                ScheduledEvent(event.id, next_target, event.action, event.description, event.recurring_bars)
            )

    def _run(self, action: Callable[[], None], description: str) -> None:
        try:
            action()
        except Exception as exc:  # noqa: BLE001 -- one bad command must not kill the set
            self.log(f"error running '{description}': {exc}", "error")

    def _advance_automations(self, prev: float, now: float) -> None:
        finished = []
        for automation in self.automations:
            span = automation.end_beat - automation.start_beat
            ratio = 1.0 if span <= 0 else max(0.0, min(1.0, (now - automation.start_beat) / span))
            automation.from_lane.gain = automation.from_start_gain * (1 - ratio)
            automation.to_lane.gain = ratio
            if now >= automation.end_beat:
                finished.append(automation)
        for automation in finished:
            automation.from_lane.stop()
            self.automations.remove(automation)
            self._lane_automation.pop(automation.from_lane.name, None)
            self._lane_automation.pop(automation.to_lane.name, None)

    # ---- introspection ---------------------------------------------------

    def _enqueue(self, target_beats: float, action: Callable[[], None], description: str) -> int:
        event_id = self._new_id()
        self.events.append(ScheduledEvent(event_id, target_beats, action, description))
        return event_id

    def pending(self, limit: Optional[int] = None) -> list[tuple[int, float, str]]:
        items = sorted(self.events, key=lambda e: (e.fire_at_beats, e.id))
        if limit is not None:
            items = items[:limit]
        return [(e.id, e.fire_at_beats, e.description) for e in items]

    def cancel(self, event_id: Optional[int] = None) -> str:
        if event_id is None:
            count = len(self.events) + len(self.automations)
            self.events.clear()
            self.automations.clear()
            self._recurring_active.clear()
            self._lane_automation.clear()
            return f"cancelled {count} event(s)"

        found = (
            any(e.id == event_id for e in self.events)
            or event_id in self._recurring_active
            or any(a.id == event_id for a in self.automations)
        )
        if not found:
            raise SchedulerError(f"no such event #{event_id}")

        self.events = [e for e in self.events if e.id != event_id]
        self.automations = [a for a in self.automations if a.id != event_id]
        self._recurring_active.pop(event_id, None)
        for lane_name, aid in list(self._lane_automation.items()):
            if aid == event_id:
                del self._lane_automation[lane_name]
        return f"cancelled #{event_id}"
