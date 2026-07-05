"""Collective expansion (collectives.py) vs the closed-form oracles.

Link resources carry only bandwidth occupancy (bytes/bw); propagation latency
rides the dependency chain (barrier / propagation tasks), never a link.  So in
isolation every expansion reproduces its closed form exactly, link busy-time
is pure occupancy, and the engines diverge only under real bandwidth
contention -- which the last two tests exercise directly.
"""

import pytest

from inferencesim import collectives
from inferencesim.engine import ring_allreduce_time
from inferencesim.hardware import Link, Topology
from inferencesim.sched import Task, schedule


# ---- ring allreduce: exact oracle, pure-occupancy links ---------------------


@pytest.mark.parametrize("g", [2, 4, 8])
@pytest.mark.parametrize("lat", [0.0, 1e-6])
def test_ring_allreduce_matches_closed_form(g, lat):
    """The expanded ring allreduce, scheduled in isolation, has makespan
    exactly equal to ring_allreduce_time -- the validation oracle -- while
    each link's busy time is pure bandwidth occupancy 2(g-1)*payload/(g*bw)
    with no phantom latency."""
    payload, bw = 4e6, 100e9
    tasks: list[Task] = []
    exit_key = collectives.ring_allreduce(tasks, None, g, payload, bw, lat,
                                          "s0", "ar")
    r = schedule(tasks)
    assert r.makespan == pytest.approx(
        ring_allreduce_time(payload, g, Link("l", bw, lat)), rel=1e-9
    )
    assert exit_key is not None
    assert r.busy["s0.l0.cw"] == pytest.approx(2 * (g - 1) * payload / (g * bw),
                                               rel=1e-9)  # occupancy only


def test_ring_allreduce_group_one_is_noop():
    """g == 1 emits nothing and passes the entry dependency straight through
    (as the roofline charges 0 for a tp=1 allreduce)."""
    tasks: list[Task] = []
    assert collectives.ring_allreduce(tasks, 7, 1, 1e6, 100e9, 1e-6,
                                      "s0", "ar") == 7
    assert tasks == []


def test_allreduce_dispatch_ring_and_switched_expand_equally():
    """The ring algorithm is bandwidth-optimal on both RING and ALL_TO_ALL
    fabrics, so allreduce() expands identically on each (only MESH_2D differs)."""
    for topo in (Topology.RING, Topology.ALL_TO_ALL):
        tasks: list[Task] = []
        collectives.allreduce(tasks, None, 4, 4e6, 100e9, 1e-6, topo, "s0", "ar")
        assert schedule(tasks).makespan == pytest.approx(
            ring_allreduce_time(4e6, 4, Link("l", 100e9, 1e-6)), rel=1e-9
        )


# ---- switched all-to-all: exact closed form ---------------------------------


@pytest.mark.parametrize("g", [2, 4, 32])
@pytest.mark.parametrize("lat", [0.0, 2e-6])
def test_a2a_switched_matches_closed_form(g, lat):
    """On an ALL_TO_ALL fabric every member serialises the g-1 messages'
    bandwidth occupancies on its one outbound link (total comm_bytes/bw), and
    a single exit barrier carries one propagation latency (flight time
    overlaps across the injected messages).  So the isolation makespan is
    exactly the closed form comm_bytes/bw + lat -- for every g, including the
    32-way MoE case -- and the link busy-time is pure occupancy."""
    comm_bytes, bw = 3.1e6, 100e9
    tasks: list[Task] = []
    collectives.all_to_all(tasks, None, g, comm_bytes, bw, lat,
                           Topology.ALL_TO_ALL, "s0", "a2a")
    r = schedule(tasks)
    assert r.makespan == pytest.approx(comm_bytes / bw + lat, rel=1e-9)
    assert r.busy["s0.l0.cw"] == pytest.approx(comm_bytes / bw, rel=1e-9)


def test_a2a_group_one_is_noop():
    tasks: list[Task] = []
    assert collectives.all_to_all(tasks, 3, 1, 1e6, 100e9, 1e-6,
                                  Topology.ALL_TO_ALL, "s0", "a2a") == 3
    assert tasks == []


# ---- routed ring all-to-all: a hand-computed oracle -------------------------


def test_ring_a2a_g4_hand_computed():
    """Ring all-to-all, g = 4, uniform payload P per member.  Each member
    sends 3 messages of P/3 (one per peer), routed the short way (a message
    exactly halfway round -- distance 2 -- goes clockwise by the tie rule).
    A hop is a link task of occupancy occ = (P/3)/bw; each hop is followed by
    a flight time `lat` on a per-message resource:

        i -> i+1 : cw,  1 hop  on l{i}.cw
        i -> i+2 : cw,  2 hops on l{i}.cw, l{i+1}.cw
        i -> i-1 : ccw, 1 hop  on l{i}.ccw

    Each clockwise link l{m}.cw carries exactly 3 occupancies -- (m,m+1),
    (m,m+2)'s first hop and (m-1,m+1)'s second hop (= P/bw of bandwidth work);
    each ccw link exactly 1.  The last message to arrive is a 2-hop message:
    it queues one occupancy behind a 1-hop message on its first link (2 occ),
    flies once (lat), takes one occupancy on its second link (occ), then flies
    once more (lat) -- so

        makespan = 3*occ + 2*lat = P/bw + 2*lat,

    and every link's busy time is pure occupancy (no latency)."""
    g, P, bw, lat = 4, 3.0e6, 100e9, 1e-6
    tasks: list[Task] = []
    collectives.all_to_all(tasks, None, g, P, bw, lat, Topology.RING,
                           "s0", "a2a")
    r = schedule(tasks)
    occ = (P / (g - 1)) / bw
    assert r.makespan == pytest.approx(3 * occ + 2 * lat, rel=1e-9)
    assert r.makespan == pytest.approx(P / bw + 2 * lat, rel=1e-9)
    for m in range(g):
        assert r.busy[f"s0.l{m}.cw"] == pytest.approx(3 * occ, rel=1e-9)
        assert r.busy[f"s0.l{m}.ccw"] == pytest.approx(occ, rel=1e-9)


# ---- contention: hop occupancy shares member 0's link with the allreduce ----


def test_hop_bandwidth_contends_with_allreduce_on_member0_link():
    """A pipeline hop rides member 0's outbound link.  Its *bandwidth
    occupancy* competes with that member's ring-allreduce sends; its flight
    time does not (it sits on a separate propagation resource).

    Build a g=2 allreduce (2 steps -> member 0 sends twice on s0.l0.cw, each of
    occupancy occ=1, barrier latency lat=0.2) plus a concurrent hop (occupancy
    0.5, flight 0.3).  Allreduce alone: 2*(occ+lat) = 2.4.

    Independent link (hop on its own resource): the allreduce (2.4) and the hop
    (0.8) overlap -> wall = 2.4.

    Shared link: s0.l0.cw runs step-0 send [0,1]; at t=1 the hop (ready at 0,
    blocked) takes the freed link [1,1.5], delaying member 0's step-1 send to
    [1.5,2.5] and the final barrier to [2.5,2.7] -- so the hop's 0.5 of
    occupancy pushes the wall to 2.7 > 2.4."""
    occ, lat, bw = 1.0, 0.2, 1.0
    payload = 2 * occ * bw  # send occupancy = payload/(2*bw) = occ

    shared: list[Task] = []
    collectives.ring_allreduce(shared, None, 2, payload, bw, lat, "s0", "ar")
    k = collectives._add(shared, "s0.l0.cw", 0.5, [], "hop")
    collectives._add(shared, "s0.prop_h", 0.3, [k], "hop")
    assert schedule(shared).makespan == pytest.approx(2.7, rel=1e-12)

    indep: list[Task] = []
    collectives.ring_allreduce(indep, None, 2, payload, bw, lat, "s0", "ar")
    k = collectives._add(indep, "hop_only", 0.5, [], "hop")
    collectives._add(indep, "hop_prop", 0.3, [k], "hop")
    assert schedule(indep).makespan == pytest.approx(2.4, rel=1e-12)


def test_overlapping_collectives_serialize_only_on_shared_links():
    """Two allreduce instances in flight on the *same* stage (e.g. two
    microbatches) share the physical member links but must not serialise their
    latency: each instance gets its own barrier resource.

    Two g=2 allreduces (occ=1, lat=0.2), both ready at t=0 on prefix s0.  The
    shared link s0.l0.cw packs all 4 member-0 sends (2 steps x 2 instances)
    back-to-back with no idle -- the barrier latencies hide behind the other
    instance's link work -- then one final flight time:

        makespan = 4*occ + lat = 4.2,

    NOT 2*(2*occ + 2*lat) = 4.8 (which is what falsely serialised barriers
    would give).  The shared link's busy time is exactly 4 occupancies."""
    occ, lat, bw = 1.0, 0.2, 1.0
    payload = 2 * occ * bw
    tasks: list[Task] = []
    collectives.ring_allreduce(tasks, None, 2, payload, bw, lat, "s0", "arA")
    collectives.ring_allreduce(tasks, None, 2, payload, bw, lat, "s0", "arB")
    r = schedule(tasks)
    single = 2 * (occ + lat)  # one instance in isolation = 2.4
    assert r.makespan == pytest.approx(4 * occ + lat, rel=1e-12)  # 4.2
    assert r.makespan < 2 * single  # latency overlapped, not serialised
    assert r.busy["s0.l0.cw"] == pytest.approx(4 * occ, rel=1e-12)  # occupancy only
