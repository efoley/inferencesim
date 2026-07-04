"""Event-driven scheduler core.

A self-contained list/event scheduler over a task graph.  It knows nothing
about chips, ops or engines -- `Task`/`Resource`/`ScheduleResult` are the whole
vocabulary -- so both the stage-level `DESEngine` and the planned
expanded-graph DES can drive it.

Two resource disciplines:

    FIFO (shared=False)     a k-slot server pool: up to `servers` tasks run
                            concurrently, the rest queue.  Dispatch is greedy
                            in (ready_time, key) order, each task taking the
                            earliest-free slot.  servers=1 is a plain FIFO
                            queue and reproduces list scheduling exactly.

    shared (shared=True)    processor sharing: `duration` is the work in
                            seconds of exclusive use; while N flows are active
                            each advances at rate 1/N.  A single flow is
                            identical to FIFO.  `servers` is ignored.

Undeclared resource names default to a 1-server FIFO, so a graph that passes
no `resources` map schedules exactly as plain list scheduling would.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

_EPS = 1e-12


@dataclass
class Task:
    """One unit of work.  `key` MUST equal the task's index in the list handed
    to `schedule` -- dependencies, readiness and results are all indexed by
    key, and builders enforce `key = len(tasks)` at append time."""

    key: int
    resource: str
    duration: float
    deps: list[int]
    label: str = ""


@dataclass
class Resource:
    """A named server.

    FIFO (shared=False): `servers` slots run concurrently; extra tasks queue.
    shared=True: processor-sharing; `servers` is ignored."""

    name: str
    servers: int = 1
    shared: bool = False


@dataclass
class ScheduleResult:
    finish: list[float]
    start: list[float]
    busy: dict[str, float]  # per-resource work done; shared: time with >=1 flow
    makespan: float

    def utilization(self) -> dict[str, float]:
        """busy / makespan per resource (empty if makespan == 0)."""
        if self.makespan <= 0:
            return {}
        return {r: b / self.makespan for r, b in self.busy.items()}


def schedule(
    tasks: list[Task], resources: dict[str, Resource] | None = None
) -> ScheduleResult:
    """Simulate the task graph.  A task starts once its deps are done and its
    resource admits it (a free FIFO slot, or immediately for shared); its
    finish depends on the discipline.  Deterministic: FIFO dispatch keeps
    (ready_time, key) order, so with all-FIFO resources this is exactly list
    scheduling.  Raises ValueError on a dependency cycle."""
    resources = resources or {}
    n = len(tasks)
    if n == 0:
        return ScheduleResult([], [], {}, 0.0)

    children: list[list[int]] = [[] for _ in range(n)]
    missing = [0] * n
    for t in tasks:
        missing[t.key] = len(t.deps)
        for d in t.deps:
            children[d].append(t.key)

    def res(name: str) -> Resource:
        return resources.get(name) or Resource(name)

    start = [0.0] * n
    finish = [0.0] * n
    ready_at = [0.0] * n
    busy: dict[str, float] = {}
    completed = 0

    slots: dict[str, list[float]] = {}  # FIFO: min-heap of slot free-times
    active: dict[str, dict[int, float]] = {}  # shared: key -> remaining work
    last: dict[str, float] = {}  # shared: last time the resource advanced
    epoch: dict[str, int] = {}  # shared: invalidates stale departure events

    ready_heap: list[tuple[float, int]] = [
        (0.0, t.key) for t in tasks if missing[t.key] == 0
    ]
    heapq.heapify(ready_heap)
    dep_heap: list[tuple[float, str, int]] = []  # (time, resource, epoch)

    def add_busy(name: str, x: float) -> None:
        busy[name] = busy.get(name, 0.0) + x

    def release(k: int, when: float) -> None:
        nonlocal completed
        finish[k] = when
        completed += 1
        for c in children[k]:
            if ready_at[c] < when:
                ready_at[c] = when
            missing[c] -= 1
            if missing[c] == 0:
                heapq.heappush(ready_heap, (ready_at[c], c))

    def advance(name: str, now: float) -> None:
        act = active[name]
        dt = now - last[name]
        if dt > 0 and act:
            per = dt / len(act)
            for k in act:
                act[k] -= per
            add_busy(name, dt)
        last[name] = now

    def next_departure(name: str) -> None:
        act = active[name]
        if not act:
            return
        rem = min(act.values())
        heapq.heappush(dep_heap, (last[name] + rem * len(act), name, epoch[name]))

    while completed < n:
        # skip stale / emptied departure events
        dep_time: float | None = None
        while dep_heap:
            dt, name, ep = dep_heap[0]
            if ep != epoch.get(name, -1) or not active.get(name):
                heapq.heappop(dep_heap)
                continue
            dep_time = dt
            break
        ready_time = ready_heap[0][0] if ready_heap else None

        if dep_time is not None and (ready_time is None or dep_time <= ready_time):
            _, name, _ = heapq.heappop(dep_heap)
            advance(name, dep_time)
            act = active[name]
            leaving = sorted(k for k, r in act.items() if r <= _EPS)
            if not leaving:  # float drift: retire the nearest to guarantee progress
                leaving = [min(act, key=lambda k: (act[k], k))]
            for k in leaving:
                del act[k]
            epoch[name] += 1
            for k in leaving:
                release(k, dep_time)
            next_departure(name)
            continue

        if ready_time is None:
            break  # nothing runnable but tasks remain -> cycle
        rt, k = heapq.heappop(ready_heap)
        t = tasks[k]
        r = res(t.resource)
        if r.shared:
            name = t.resource
            if name not in active:
                active[name], last[name], epoch[name] = {}, rt, 0
            advance(name, rt)
            active[name][k] = t.duration
            epoch[name] += 1
            start[k] = rt
            next_departure(name)
        else:
            name = t.resource
            if name not in slots:
                slots[name] = [0.0] * max(1, r.servers)
            slot = heapq.heappop(slots[name])
            st = rt if rt > slot else slot
            heapq.heappush(slots[name], st + t.duration)
            start[k] = st
            add_busy(name, t.duration)
            release(k, st + t.duration)

    if completed != n:
        raise ValueError("task graph has a dependency cycle")
    return ScheduleResult(finish, start, busy, max(finish))


def chrome_trace(
    tasks: list[Task],
    result: ScheduleResult,
    pid_base: int = 0,
    prefix: str = "",
) -> dict:
    """Chrome/Perfetto trace-event JSON: one process per resource, complete
    ("X") events with ts/dur in microseconds and `label` as the name.  Tasks
    on one resource are packed into non-overlapping tid lanes (a k-server slot
    or a shared flow), assigned greedily by start time."""
    by_res: dict[str, list[Task]] = {}
    for t in tasks:
        by_res.setdefault(t.resource, []).append(t)

    events: list[dict] = []
    for i, name in enumerate(sorted(by_res)):
        pid = pid_base + i
        events.append({
            "name": "process_name", "ph": "M", "pid": pid, "tid": 0,
            "args": {"name": prefix + name},
        })
        lane_free: list[float] = []
        for t in sorted(by_res[name], key=lambda x: (result.start[x.key], x.key)):
            s, f = result.start[t.key], result.finish[t.key]
            lane = next((li for li, lf in enumerate(lane_free) if lf <= s + _EPS), None)
            if lane is None:
                lane = len(lane_free)
                lane_free.append(f)
            else:
                lane_free[lane] = f
            events.append({
                "name": t.label or f"task{t.key}", "ph": "X",
                "ts": s * 1e6, "dur": (f - s) * 1e6, "pid": pid, "tid": lane,
            })
    return {"traceEvents": events, "displayTimeUnit": "ms"}
