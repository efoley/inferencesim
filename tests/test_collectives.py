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
    with no phantom latency.  (Named on a RING fabric, so sends use `.cw`.)"""
    payload, bw = 4e6, 100e9
    tasks: list[Task] = []
    exit_key = collectives.ring_allreduce(tasks, None, g, payload, bw, lat,
                                          Topology.RING, "s0", "ar")
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
                                      Topology.RING, "s0", "ar") == 7
    assert tasks == []


def test_allreduce_topology_aware_link_naming():
    """The ring algorithm is bandwidth-optimal on both RING and ALL_TO_ALL
    fabrics, so allreduce() expands to the same makespan on each (only MESH_2D
    differs).  The link *names* follow the fabric: a switched fabric egresses a
    member via its single `.out` port; a ring uses the `.cw` cable."""
    oracle = ring_allreduce_time(4e6, 4, Link("l", 100e9, 1e-6))
    ring: list[Task] = []
    collectives.allreduce(ring, None, 4, 4e6, 100e9, 1e-6, Topology.RING,
                          "s0", "ar")
    r_ring = schedule(ring)
    assert r_ring.makespan == pytest.approx(oracle, rel=1e-9)
    assert "s0.l0.cw" in r_ring.busy and not any(".out" in k for k in r_ring.busy)

    sw: list[Task] = []
    collectives.allreduce(sw, None, 4, 4e6, 100e9, 1e-6, Topology.ALL_TO_ALL,
                          "s0", "ar")
    r_sw = schedule(sw)
    assert r_sw.makespan == pytest.approx(oracle, rel=1e-9)  # identical timing
    assert "s0.l0.out" in r_sw.busy and not any(".cw" in k for k in r_sw.busy)


# ---- switched all-to-all: exact closed form ---------------------------------


@pytest.mark.parametrize("g", [2, 4, 32])
@pytest.mark.parametrize("lat", [0.0, 2e-6])
def test_a2a_switched_uniform_oracle_with_fill(g, lat):
    """Store-and-forward switched all-to-all, uniform payloads: each message is
    an egress-occupancy task on the sender's `.out` port THEN an ingress task on
    the receiver's `.in` port.  The staggered rotation is a perfect permutation
    each step, so no ingress ever queues, and the isolation makespan is EXACTLY
    the closed form plus one message of store-and-forward fill:

        g*occ + lat == comm_bytes/bw + (comm_bytes/(g-1))/bw + lat,

    with occ = comm_bytes/((g-1)*bw).  The one-message gap over the switched
    closed form (comm_bytes/bw + lat) is deliberate and bounded -- the analytic
    engine does not charge it (mirroring graphdes' fill/drain).  Egress and
    ingress link busy-times are each pure occupancy comm_bytes/bw."""
    comm_bytes, bw = 3.1e6, 100e9
    tasks: list[Task] = []
    collectives.all_to_all(tasks, None, g, comm_bytes, bw, lat,
                           Topology.ALL_TO_ALL, "s0", "a2a")
    r = schedule(tasks)
    occ = (comm_bytes / (g - 1)) / bw
    closed_form = comm_bytes / bw + lat
    assert r.makespan == pytest.approx(g * occ + lat, rel=1e-9)  # the new exact oracle
    assert r.makespan == pytest.approx(comm_bytes / bw + occ + lat, rel=1e-9)
    # within exactly one message of store-and-forward fill of the closed form
    assert closed_form < r.makespan <= closed_form + occ * (1 + 1e-9)
    assert r.busy["s0.l0.out"] == pytest.approx(comm_bytes / bw, rel=1e-9)  # egress occupancy
    assert r.busy["s0.l0.in"] == pytest.approx(comm_bytes / bw, rel=1e-9)  # ingress occupancy


def test_a2a_switched_skewed_incast_g4():
    """Skewed switched all-to-all, g = 4, member 0 twice as popular as the rest
    (popularity vector w = [2/5, 1/5, 1/5, 1/5]).  Dispatch (weight_on='dest')
    sizes s->r by w[r], so every sender's message to the hot member 0 is
    size 4/3*P*2/5 = 8P/15 while its cold messages are 4/3*P*1/5 = 4P/15
    (occ = size/bw).  With rotation order s->s+1,s+2,s+3, the three hot messages
    aimed at member 0 finish egress at 8P/15, 12P/15, 16P/15 (from senders 3, 2,
    1 respectively), and member 0's INGRESS port serialises them:

        [8/15,16/15], [16/15,24/15], [24/15,32/15]  (units of P/bw)

    -- an incast.  So the makespan is 32/15*P/bw + lat, strictly above the
    uniform g*occ+lat = 4/3*P/bw+lat, and member 0's ingress carries 3*8P/15 of
    occupancy versus a cold member's 3*4P/15."""
    g, P, bw, lat = 4, 3.0e6, 100e9, 1e-6
    weights = [2 / 5, 1 / 5, 1 / 5, 1 / 5]
    tasks: list[Task] = []
    collectives.all_to_all(tasks, None, g, P, bw, lat, Topology.ALL_TO_ALL,
                           "s0", "a2a", weights=weights, weight_on="dest")
    r = schedule(tasks)
    uniform = g * (P / (g - 1) / bw) + lat  # = 4/3 P/bw + lat
    assert r.makespan == pytest.approx(32 / 15 * P / bw + lat, rel=1e-9)
    assert r.makespan > uniform  # incast onto the hot owner
    assert r.busy["s0.l0.in"] == pytest.approx(3 * (8 / 15) * P / bw, rel=1e-9)
    assert r.busy["s0.l1.in"] == pytest.approx(3 * (4 / 15) * P / bw, rel=1e-9)
    assert r.busy["s0.l1.in"] < r.busy["s0.l0.in"]


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


def test_ring_a2a_skewed_payloads_route_and_conserve():
    """Skewed ring all-to-all: message sizes follow the popularity vector (a
    dispatch sizes s->r by w[r]) while the routing -- short way round, cw on
    ties -- is UNCHANGED, and ring forwarding needs no separate ingress port.
    Payload is conserved, just concentrated onto the hot owner's messages: the
    total cw+ccw link occupancy equals the injected bytes times each message's
    hop count.  (The independent oracle re-derives that sum from the same
    routing rule.)"""
    g, P, bw, lat = 4, 3.0e6, 100e9, 0.0
    weights = [0.4, 0.2, 0.2, 0.2]  # member 0 hottest
    tasks: list[Task] = []
    collectives.all_to_all(tasks, None, g, P, bw, lat, Topology.RING,
                           "s0", "a2a", weights=weights, weight_on="dest")
    r = schedule(tasks)
    assert r.makespan > 0
    scale = g / (g - 1) * P
    expected = 0.0
    for i in range(g):
        for j in range(g):
            if i == j:
                continue
            size = scale * weights[j]  # weight_on='dest'
            cw_dist, ccw_dist = (j - i) % g, (i - j) % g
            dist = cw_dist if cw_dist <= ccw_dist else ccw_dist
            expected += size * dist / bw
    link_busy = sum(b for k, b in r.busy.items()
                    if k.endswith(".cw") or k.endswith(".ccw"))
    assert link_busy == pytest.approx(expected, rel=1e-9)


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
    collectives.ring_allreduce(shared, None, 2, payload, bw, lat,
                               Topology.RING, "s0", "ar")
    k = collectives._add(shared, "s0.l0.cw", 0.5, [], "hop")
    collectives._add(shared, "s0.prop_h", 0.3, [k], "hop")
    assert schedule(shared).makespan == pytest.approx(2.7, rel=1e-12)

    indep: list[Task] = []
    collectives.ring_allreduce(indep, None, 2, payload, bw, lat,
                               Topology.RING, "s0", "ar")
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
    collectives.ring_allreduce(tasks, None, 2, payload, bw, lat,
                               Topology.RING, "s0", "arA")
    collectives.ring_allreduce(tasks, None, 2, payload, bw, lat,
                               Topology.RING, "s0", "arB")
    r = schedule(tasks)
    single = 2 * (occ + lat)  # one instance in isolation = 2.4
    assert r.makespan == pytest.approx(4 * occ + lat, rel=1e-12)  # 4.2
    assert r.makespan < 2 * single  # latency overlapped, not serialised
    assert r.busy["s0.l0.cw"] == pytest.approx(4 * occ, rel=1e-12)  # occupancy only
