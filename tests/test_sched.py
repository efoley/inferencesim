"""The event-driven scheduler core (inferencesim.sched)."""

import json

import pytest

from inferencesim.sched import Resource, Task, chrome_trace, schedule


# ---- FIFO list scheduling (behaviour-preserving) ----------------------------


def test_serialises_on_resources_and_respects_deps():
    tasks = [
        Task(0, "u", 2.0, []),
        Task(1, "u", 3.0, []),        # same resource: queues behind 0
        Task(2, "link", 1.0, [0]),    # different resource: overlaps with 1
        Task(3, "u", 1.0, [2]),
    ]
    r = schedule(tasks)
    assert r.finish[0] == 2.0
    assert r.finish[1] == 5.0
    assert r.finish[2] == 3.0   # started at 2 on the free link
    assert r.finish[3] == 6.0   # waited for the unit (busy till 5)


def test_explicit_single_server_matches_default():
    """Declaring the resources as 1-server FIFO reproduces the exact scenario
    that default (undeclared) resources produce."""
    tasks = [
        Task(0, "u", 2.0, []),
        Task(1, "u", 3.0, []),
        Task(2, "link", 1.0, [0]),
        Task(3, "u", 1.0, [2]),
    ]
    res = {"u": Resource("u", servers=1), "link": Resource("link", servers=1)}
    assert schedule(tasks, res).finish == [2.0, 5.0, 3.0, 6.0]


def test_detects_cycles():
    with pytest.raises(ValueError, match="cycle"):
        schedule([Task(0, "u", 1.0, [1]), Task(1, "u", 1.0, [0])])


# ---- k-server FIFO pools ----------------------------------------------------


def test_k_server_runs_tasks_concurrently():
    tasks = [Task(i, "R", 1.0, []) for i in range(3)]
    two = schedule(tasks, {"R": Resource("R", servers=2)})
    assert two.finish == [1.0, 1.0, 2.0]   # two run at once, third queues
    one = schedule(tasks, {"R": Resource("R", servers=1)})
    assert one.finish == [1.0, 2.0, 3.0]   # old single-server behaviour


# ---- shared (processor-sharing) resources -----------------------------------


def test_shared_two_flows_share_bandwidth():
    """Two equal flows starting together each run at half rate -> both 2.0."""
    tasks = [Task(0, "R", 1.0, []), Task(1, "R", 1.0, [])]
    r = schedule(tasks, {"R": Resource("R", shared=True)})
    assert r.finish[0] == pytest.approx(2.0, rel=1e-12)
    assert r.finish[1] == pytest.approx(2.0, rel=1e-12)


def test_shared_staggered_flows():
    """Flow A ready at 0, flow B ready at 0.5 (behind a 0.5s task on another
    resource): A finishes 1.5, B finishes 2.0."""
    tasks = [
        Task(0, "R", 1.0, []),        # flow A
        Task(1, "gate", 0.5, []),     # delays B onto the shared resource
        Task(2, "R", 1.0, [1]),       # flow B, ready at 0.5
    ]
    r = schedule(tasks, {"R": Resource("R", shared=True)})
    assert r.finish[0] == pytest.approx(1.5, rel=1e-12)
    assert r.finish[2] == pytest.approx(2.0, rel=1e-12)


def test_shared_single_flow_equals_fifo():
    tasks = [Task(0, "R", 1.3, [])]
    shared = schedule(tasks, {"R": Resource("R", shared=True)})
    fifo = schedule(tasks, {"R": Resource("R")})
    assert shared.finish == fifo.finish
    assert shared.start == fifo.start
    assert shared.busy["R"] == pytest.approx(fifo.busy["R"], rel=1e-12)


# ---- accounting -------------------------------------------------------------


def test_busy_sums_durations_and_utilisation_bounded():
    tasks = [
        Task(0, "u", 2.0, []),
        Task(1, "u", 3.0, []),
        Task(2, "link", 1.0, [0]),
        Task(3, "u", 1.0, [2]),
    ]
    r = schedule(tasks)
    assert sum(r.busy.values()) == pytest.approx(sum(t.duration for t in tasks))
    assert all(0.0 <= u <= 1.0 for u in r.utilization().values())
    assert r.utilization()["u"] == pytest.approx(6.0 / 6.0)


def test_makespan_and_empty():
    tasks = [Task(0, "u", 2.0, []), Task(1, "link", 1.0, [0])]
    r = schedule(tasks)
    assert r.makespan == max(r.finish)
    empty = schedule([])
    assert empty.makespan == 0.0
    assert empty.finish == [] and empty.utilization() == {}


# ---- Chrome trace export ----------------------------------------------------


def test_chrome_trace_is_serialisable_and_complete():
    tasks = [
        Task(0, "u", 2.0, [], "embed"),
        Task(1, "u", 3.0, [], "ffn"),
        Task(2, "link", 1.0, [0], "hop"),
        Task(3, "u", 1.0, [2], "head"),
    ]
    r = schedule(tasks)
    trace = chrome_trace(tasks, r)
    json.loads(json.dumps(trace))  # round-trips
    x_events = [e for e in trace["traceEvents"] if e["ph"] == "X"]
    assert len(x_events) == len(tasks)
    for e in x_events:
        assert {"name", "ts", "dur", "ph"} <= e.keys()


def test_chrome_trace_lanes_k_server_do_not_overlap():
    tasks = [Task(i, "R", 1.0, []) for i in range(3)]
    r = schedule(tasks, {"R": Resource("R", servers=2)})
    events = [e for e in chrome_trace(tasks, r)["traceEvents"] if e["ph"] == "X"]
    # the two concurrent tasks land on different tids
    concurrent = [e for e in events if e["ts"] == 0.0]
    assert len({e["tid"] for e in concurrent}) == 2
