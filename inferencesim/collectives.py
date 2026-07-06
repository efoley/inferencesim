"""Collective communication expanded to per-step link transfers.

The stage-level DES (des.py) used to charge a collective as a single task
with a closed-form service time (`engine.ring_allreduce_time`, or
`bytes/bw + latency` for an all-to-all).  This module refines that: a
collective becomes its actual per-step message transfers over the group's
fabric topology, on *per-member directional outbound link* resources.  Those
same link resources also carry the pipeline hops, so collective/hop and
collective/collective contention emerges from the schedule instead of being
averaged away.

**Latency vs occupancy.**  A link resource carries only *bandwidth
occupancy* -- the time bytes physically hold the wire, `bytes/bw`.
Propagation latency is flight time, not link occupancy: it overlaps across
back-to-back messages, so it rides the *dependency chain* (on barrier /
propagation tasks) and never sits on a link resource.  Consequently the
isolated makespan of every expansion reproduces its closed form exactly
(ring allreduce == `ring_allreduce_time`; switched all-to-all ==
`comm_bytes/bw + lat`), and the engines diverge only under genuine
bandwidth contention -- which is the physical effect this refinement exists
to capture.

Resources (owned by des.py, named `s{stage}`) -- a member's outbound link is
named for the fabric it egresses onto:
  * switched / mesh (ALL_TO_ALL, MESH_2D fallbacks): `{prefix}.l{i}.out` --
    member i's single egress port (a switched fabric has one port per chip,
    not a directional cable pair).
  * ring (RING): `{prefix}.l{i}.cw` / `{prefix}.l{i}.ccw` -- the two
    physically distinct cables, each full duplex (matching a Link's bandwidth
    being per-direction).  The ring *algorithm* over a switched fabric is a
    logical ring whose sends still leave via the one egress port, so it names
    `.out` on ALL_TO_ALL and `.cw` on RING.
  * `{prefix}.l{i}.in` -- member i's *ingress* port on a switched fabric.  The
    all-to-all is store-and-forward: each message occupies the sender's `.out`
    then the receiver's `.in`, so simultaneous arrivals at one member serialise
    on its ingress port -- the incast a hot-expert MoE all-to-all creates onto
    the busy owners.  A RING has no separate ingress port (arrivals ride the
    same directional cables they are forwarded on), so it reuses `.cw`/`.ccw`.
  Either way the link resource carries bandwidth occupancy only.
  * `{prefix}.bar{inst}` -- the sync/barrier resource of one collective
    *instance* (`inst` uniquifies it so concurrent instances on a stage do
    not falsely serialise their latency segments).  Carries latency.
  * `{prefix}.prop{inst}_{i}_{j}` -- a routed ring message's propagation
    delay, unique per message so in-flight messages never contend.

Sync resources are named with `.bar` / `.prop` markers so the engine can
drop them from the utilisation report (they are timeline sync, not physical
occupancy) while keeping them in the trace.

Each builder appends `sched.Task`s to the caller's list (`key == len(tasks)`
at append), gated by a single `entry` dependency (int | None: the compute
that produced the payload), and returns a single `exit` key to chain the rest
of the stage onto (int | None if the collective was a no-op).
"""

from __future__ import annotations

from .hardware import Topology
from .sched import Task

# resource-name markers for dependency-chain sync tasks (barriers /
# propagation): they carry latency, not bandwidth occupancy, so des.py drops
# them from the utilisation report.
SYNC_MARKERS = (".bar", ".prop")


def is_sync_resource(name: str) -> bool:
    return any(m in name for m in SYNC_MARKERS)


def _add(tasks: list[Task], resource: str, duration: float,
         deps: list[int], label: str) -> int:
    key = len(tasks)
    tasks.append(Task(key, resource, duration, deps, label))
    return key


def _cw(prefix: str, i: int) -> str:
    return f"{prefix}.l{i}.cw"


def _ccw(prefix: str, i: int) -> str:
    return f"{prefix}.l{i}.ccw"


def _out(prefix: str, i: int) -> str:
    return f"{prefix}.l{i}.out"


def _in(prefix: str, i: int) -> str:
    return f"{prefix}.l{i}.in"


def egress(prefix: str, i: int, topology: Topology) -> str:
    """Member i's outbound link resource, named for the fabric it egresses
    onto: a RING has two distinct cables (this is the clockwise one), a
    switched / mesh fabric a single egress port.  Used for ring-algorithm
    sends and (in des.py) for the pipeline hop off the boundary chip."""
    return _cw(prefix, i) if topology is Topology.RING else _out(prefix, i)


def ingress(prefix: str, i: int, topology: Topology) -> str:
    """Member i's inbound link resource.  A switched / mesh fabric has a
    distinct ingress port (`.in`) so simultaneous arrivals at a member serialise
    -- the incast a skewed all-to-all creates onto hot owners.  A RING has no
    separate ingress port; a forwarded message arrives on the same directional
    cable it travelled, so ring ingress reuses the clockwise cable (arrivals are
    already accounted on the cw/ccw link occupancy)."""
    return _cw(prefix, i) if topology is Topology.RING else _in(prefix, i)


def _msg_sizes(group: int, comm_bytes: float, weights: list[float] | None,
               weight_on: str):
    """Return a `size(s, r) -> bytes` for the message from sender s to receiver
    r of an all-to-all.

    Uniform (`weights is None`): every message is `comm_bytes/(group-1)` -- the
    per-member payload split evenly over the g-1 peers, the historical sizing.

    Skewed (`weights` = the per-member routing-popularity vector, `sum == 1`):
    `size(s, r) = group/(group-1) * comm_bytes * w[k]`, where k is the receiver
    r for a dispatch (`weight_on='dest'`: the message carries the fraction of
    s's tokens routed to r's experts, so a hot receiver INCASTS) or the sender s
    for a combine (`weight_on='src'`: the hot expert-owner combines and egresses
    the most).  The `group/(group-1)` factor makes it reduce to the uniform
    `comm_bytes/(group-1)` when `w == 1/group`, and preserves the per-member
    average payload (`comm_bytes`): mean sender egress and mean receiver ingress
    both stay `comm_bytes`, while the hot member's ingress (dispatch) / egress
    (combine) load is `hot_member_factor * comm_bytes`."""
    if weights is None:
        uniform = comm_bytes / (group - 1)
        return lambda s, r: uniform
    scale = group / (group - 1) * comm_bytes
    if weight_on == "src":
        return lambda s, r: scale * weights[s]
    return lambda s, r: scale * weights[r]


def _ring_pass(
    tasks: list[Task], entry: int | None, group: int, payload: float,
    bw: float, lat: float, topology: Topology, prefix: str, label: str,
    steps: int,
) -> int:
    """`steps` barrier-separated ring steps: in every step each member `i` sends
    `payload/g` bytes to `(i+1) % g` via its egress link (the logical ring is
    bandwidth-optimal on a switched fabric too; the send leaves the chip's one
    egress port, so the resource is named `.out` on ALL_TO_ALL and `.cw` on
    RING) -- one task per member per step carrying only the bandwidth occupancy
    `payload/(g*bw)`.  A barrier task carrying the propagation latency `lat`
    joins each step to the next (the next step's sends wait on *all* of this
    step's sends, then one flight time).  Returns the final barrier key.  In
    isolation the makespan is exactly `steps * (payload/(g*bw) + lat)` while each
    link's busy time is pure occupancy `steps*payload/(g*bw)` with no phantom
    latency.  A full ring allreduce is 2(g-1) steps; an allgather / reduce-
    scatter is g-1 (half)."""
    inst = len(tasks)
    bar = f"{prefix}.bar{inst}"
    occ = payload / (group * bw)
    send = [egress(prefix, i, topology) for i in range(group)]  # resolved once
    prev = entry
    for _ in range(steps):
        deps = [prev] if prev is not None else []
        sends = [
            _add(tasks, send[i], occ, list(deps), label)
            for i in range(group)
        ]
        prev = _add(tasks, bar, lat, sends, label)
    assert prev is not None
    return prev


def ring_allreduce(
    tasks: list[Task], entry: int | None, group: int, payload: float,
    bw: float, lat: float, topology: Topology, prefix: str, label: str,
) -> int | None:
    """Bandwidth-optimal ring allreduce as 2(g-1) barrier-separated steps.  In
    isolation the makespan is exactly

        2(g-1) * (payload/(g*bw) + lat) == ring_allreduce_time(payload,g,link),

    while each link's busy time is pure occupancy 2(g-1)*payload/(g*bw) with no
    phantom latency (see `_ring_pass`)."""
    if group <= 1 or payload <= 0.0:
        return entry
    return _ring_pass(tasks, entry, group, payload, bw, lat, topology, prefix,
                      label, steps=2 * (group - 1))


def half_ring(
    tasks: list[Task], entry: int | None, group: int, payload: float,
    bw: float, lat: float, topology: Topology, prefix: str, label: str,
) -> int | None:
    """Expand a one-pass ring collective over `group = tp*adp` chips -- the dense
    attention-DP FFN's allgather (assemble the full-batch hidden state) or
    reduce-scatter (reduce its output back to sequence shards).  It is exactly
    HALF a ring allreduce: g-1 barrier-separated steps, so in isolation the
    makespan is

        (g-1) * (payload/(g*bw) + lat) == ring_gather_time(payload,g,link),

    and each link's busy time is pure occupancy (g-1)*payload/(g*bw).  The ring
    algorithm is bandwidth-optimal on RING and ALL_TO_ALL fabrics alike; MESH_2D
    falls back to the same closed form (occupancy on the port, latency on a
    barrier).  Reduce-scatter is the arithmetic dual of allgather with identical
    communication volume, so both FFN-boundary collectives share this expansion."""
    if group <= 1 or payload <= 0.0:
        return entry
    if topology is Topology.MESH_2D:
        # TODO: expand a 2-D mesh allgather per-step; no preset uses MESH_2D yet.
        steps = group - 1
        deps = [entry] if entry is not None else []
        occ = _add(tasks, _out(prefix, 0), steps / group * payload / bw, deps, label)
        return _add(tasks, f"{prefix}.bar{len(tasks)}", steps * lat, [occ], label)
    return _ring_pass(tasks, entry, group, payload, bw, lat, topology, prefix,
                      label, steps=group - 1)


def _a2a_switched(
    tasks: list[Task], entry: int | None, group: int, comm_bytes: float,
    bw: float, lat: float, prefix: str, label: str,
    weights: list[float] | None = None, weight_on: str = "dest",
) -> int:
    """All-to-all on a switched / full-mesh fabric, store-and-forward: every
    member sends g-1 messages, one to each peer, in a STAGGERED rotation (sender
    s sends to s+1, s+2, ... s+(g-1) mod g in that order).  Each message is an
    egress-occupancy task on the sender's `.out` port THEN an ingress-occupancy
    task on the receiver's `.in` port (both carry the message's `bytes/bw`); a
    single exit barrier carries one propagation latency (flight time overlaps).

    **Uniform oracle (exact).**  With uniform payloads (`weights is None`,
    message occ = `comm_bytes/((g-1)*bw)`) the rotation is a perfect permutation
    each step: in step t every sender pushes to a distinct receiver, and receiver
    r takes from sender s exactly the message whose egress finished at
    `t*occ` where `t = (r-s) mod g` -- so as s ranges over the g-1 senders, r's
    ingress port receives exactly one message in each slot [t*occ, (t+1)*occ],
    t = 1..g-1, and NEVER queues.  The last message (t = g-1) egresses over
    [(g-2)occ, (g-1)occ] then ingresses over [(g-1)occ, g*occ], so the isolation
    makespan is

        g*occ + lat == comm_bytes/bw + (comm_bytes/(g-1))/bw + lat,

    the switched closed form (`comm_bytes/bw + lat`) plus EXACTLY ONE message of
    store-and-forward fill -- a deliberate, bounded gap the analytic engine does
    not charge (mirroring graphdes' fill/drain).  Egress and ingress busy times
    are each pure occupancy `comm_bytes/bw`.

    **Skewed case (incast).**  With a per-member popularity vector the message
    sizes are non-uniform (see `_msg_sizes`): for a dispatch the hot receiver
    takes a large message from every sender, so its ingress port serialises the
    extra arrivals and the makespan exceeds the uniform one -- the incast made
    visible.  A small g=4 case is hand-derived in the tests."""
    inst = len(tasks)
    size = _msg_sizes(group, comm_bytes, weights, weight_on)
    base = [entry] if entry is not None else []
    finals: list[int] = []
    for s in range(group):
        out = _out(prefix, s)  # the sender's single egress port
        prev: int | None = None
        for step in range(1, group):  # staggered rotation: s -> s+step
            r = (s + step) % group
            occ = size(s, r) / bw
            deps = [prev] if prev is not None else list(base)
            # the sender's egress port serialises its sends in rotation order;
            # the message then occupies the receiver's ingress port
            prev = _add(tasks, out, occ, deps, label)
            finals.append(_add(tasks, _in(prefix, r), occ, [prev], label))
    return _add(tasks, f"{prefix}.bar{inst}", lat, finals, label)


def _a2a_ring(
    tasks: list[Task], entry: int | None, group: int, comm_bytes: float,
    bw: float, lat: float, prefix: str, label: str,
    weights: list[float] | None = None, weight_on: str = "dest",
) -> int:
    """All-to-all on a ring: each of the g(g-1) messages routes the short way
    round and is forwarded hop by hop (store-and-forward).  Each hop is a link
    task carrying bandwidth occupancy `msg/bw`, followed by a propagation task
    carrying `lat` on a per-message resource (so a message's own flight time
    never contends, and different messages contend only on the shared links).
    Ties (a message exactly halfway round) go clockwise.  Message sizes may be
    non-uniform under expert-load skew (`weights`); the routing is unchanged, and
    ingress is inherent in ring forwarding (arrivals ride the cw/ccw cables, no
    separate ingress resource).  No closed form exists; small cases (uniform and
    skewed) are hand-derived in the tests."""
    inst = len(tasks)
    size = _msg_sizes(group, comm_bytes, weights, weight_on)
    base = [entry] if entry is not None else []
    finals: list[int] = []
    for i in range(group):
        for j in range(group):
            if i == j:
                continue
            occ = size(i, j) / bw
            cw_dist = (j - i) % group
            ccw_dist = (i - j) % group
            cw = cw_dist <= ccw_dist  # tie -> clockwise
            dist = cw_dist if cw else ccw_dist
            prop = f"{prefix}.prop{inst}_{i}_{j}"  # unique per message
            prev: int | None = None
            for h in range(dist):
                m = (i + h) % group if cw else (i - h) % group
                res = _cw(prefix, m) if cw else _ccw(prefix, m)
                deps = [prev] if prev is not None else list(base)
                prev = _add(tasks, res, occ, deps, label)  # link occupancy
                prev = _add(tasks, prop, lat, [prev], label)  # flight time
            assert prev is not None
            finals.append(prev)
    return _add(tasks, f"{prefix}.bar{inst}", 0.0, finals, label)


def all_to_all(
    tasks: list[Task], entry: int | None, group: int, comm_bytes: float,
    bw: float, lat: float, topology: Topology, prefix: str, label: str,
    weights: list[float] | None = None, weight_on: str = "dest",
) -> int | None:
    """Expand a MoE dispatch/combine all-to-all over `group = tp*ep` chips,
    dispatching on the group's fabric topology.

    `weights` (the per-member routing-popularity vector, `sum == 1`) sizes the
    per-member messages under expert-load skew: `weight_on='dest'` for a dispatch
    (incast onto the hot receiver's ingress), `'src'` for a combine (the hot
    owner egresses the most).  `None` (uniform, `skew=0`) recovers even
    `comm_bytes/(g-1)` messages -- bit-identical routing to before."""
    if group <= 1 or comm_bytes <= 0.0:
        return entry
    if topology is Topology.RING:
        return _a2a_ring(tasks, entry, group, comm_bytes, bw, lat, prefix, label,
                         weights, weight_on)
    if topology is Topology.MESH_2D:
        # TODO: expand a 2-D mesh all-to-all per-step; no preset uses MESH_2D
        # yet, so fall back to the closed form (occupancy on the egress port,
        # latency on a barrier -- same latency/occupancy split as the real
        # expansions).  Skew weights are ignored here (documented; no preset).
        deps = [entry] if entry is not None else []
        occ = _add(tasks, _out(prefix, 0), comm_bytes / bw, deps, label)
        return _add(tasks, f"{prefix}.bar{len(tasks)}", lat, [occ], label)
    # ALL_TO_ALL: switched (NVSwitch) or full mesh.
    return _a2a_switched(tasks, entry, group, comm_bytes, bw, lat, prefix, label,
                         weights, weight_on)


def allreduce(
    tasks: list[Task], entry: int | None, group: int, payload: float,
    bw: float, lat: float, topology: Topology, prefix: str, label: str,
) -> int | None:
    """Expand an allreduce over `group = tp` chips.  The ring algorithm is
    bandwidth-optimal on both RING and ALL_TO_ALL fabrics, so both expand the
    same way; MESH_2D falls back to the ring closed form (occupancy on the
    link, latency on a barrier)."""
    if group <= 1 or payload <= 0.0:
        return entry
    if topology is Topology.MESH_2D:
        # TODO: expand a 2-D mesh allreduce per-step; no preset uses MESH_2D.
        steps = 2 * (group - 1)
        deps = [entry] if entry is not None else []
        occ = _add(tasks, _out(prefix, 0), steps / group * payload / bw, deps, label)
        return _add(tasks, f"{prefix}.bar{len(tasks)}", steps * lat, [occ], label)
    return ring_allreduce(tasks, entry, group, payload, bw, lat, topology,
                          prefix, label)
