"""Simulation engines.

The engine abstraction is deliberately thin: an engine consumes lowered ops
(resource demands) and a hardware description, and produces timed phases.

RooflineEngine is the "speed of light" model: each op runs at the peak rate
of its bottleneck resource, ops execute back-to-back with no overheads, and
compute/memory within an op perfectly overlap (time = max, not sum).  It is
an optimistic lower bound on latency and upper bound on throughput.

A discrete-event engine (roadmap) will consume the same Op stream with
explicit dependencies, queues and contention on each Chip stage (DRAM
channels, NoC hops, SRAM banks, links).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace

from .efficiency import Efficiency
from .hardware import Chip, Link, System, Topology
from .ops import Op, OpKind
from .workload import Deployment


def ring_allreduce_time(payload_bytes: float, group_size: int, link: Link) -> float:
    """Bandwidth-optimal ring allreduce of `payload_bytes` per rank."""
    if group_size <= 1 or payload_bytes <= 0:
        return 0.0
    steps = 2 * (group_size - 1)
    bw_term = (steps / group_size) * payload_bytes / link.bandwidth
    lat_term = steps * link.latency_s
    return bw_term + lat_term


def ring_gather_time(payload_bytes: float, group_size: int, link: Link) -> float:
    """Bandwidth-optimal one-pass ring allgather / reduce-scatter of a
    `payload_bytes`-per-rank result over `group_size` ranks.

    Exactly HALF a ring allreduce: g-1 barrier-separated steps (not 2(g-1)),
    each rank forwarding a payload_bytes/g shard to its neighbour, so per-rank
    egress is (g-1)/g * payload_bytes and the makespan is

        (g-1)/g * payload_bytes/bw + (g-1)*lat.

    Reduce-scatter has the identical closed form (the arithmetic dual of the
    allgather with the same communication volume)."""
    if group_size <= 1 or payload_bytes <= 0:
        return 0.0
    steps = group_size - 1
    bw_term = (steps / group_size) * payload_bytes / link.bandwidth
    lat_term = steps * link.latency_s
    return bw_term + lat_term


@dataclass(frozen=True)
class OpTiming:
    op: Op
    time: float  # total for all `count` instances
    compute_time: float  # time the math units are busy (all instances)
    mem_time: float  # time the DRAM path is busy (all instances)
    comm_time: float  # time spent in collectives (all instances)

    @property
    def bound(self) -> str:
        best = max(
            ("compute", self.compute_time),
            ("memory", self.mem_time),
            ("comm", self.comm_time),
            key=lambda kv: kv[1],
        )
        return best[0]


@dataclass
class Phase:
    """A timed group of ops (one prefill, or one decode step).

    wall_time, when set, is a measured wall-clock duration (e.g. from the
    discrete-event engine, where overlap is emergent); otherwise the phase
    is the analytic sum of its op timings."""

    name: str
    timings: list[OpTiming] = field(default_factory=list)
    wall_time: float | None = None
    # per-resource busy time and the schedule span they are measured over
    # (populated by the discrete-event engine; None for the roofline engine).
    resource_busy: dict[str, float] | None = None
    resource_span: float | None = None

    @property
    def total_time(self) -> float:
        if self.wall_time is not None:
            return self.wall_time
        return sum(t.time for t in self.timings)

    def duration(self, overlap_comm: bool = False) -> float:
        """Wall-clock time for the phase.  With overlap_comm, collectives run
        concurrently with on-chip work: duration = max(comm, everything else).
        A measured wall_time already includes whatever overlap the schedule
        achieved, so the flag is ignored."""
        if self.wall_time is not None:
            return self.wall_time
        if not overlap_comm:
            return self.total_time
        comm = sum(t.time for t in self.timings if t.op.is_comm)
        return max(comm, self.total_time - comm)

    @property
    def compute_busy(self) -> float:
        return sum(t.compute_time for t in self.timings)

    @property
    def mem_busy(self) -> float:
        return sum(t.mem_time for t in self.timings)

    @property
    def comm_busy(self) -> float:
        return sum(t.comm_time for t in self.timings)

    def category_times(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for t in self.timings:
            out[t.op.category] = out.get(t.op.category, 0.0) + t.time
        return out

    def category_bounds(self) -> dict[str, str]:
        """Dominant bottleneck per category, weighted by time."""
        acc: dict[str, dict[str, float]] = {}
        for t in self.timings:
            acc.setdefault(t.op.category, {"compute": 0.0, "memory": 0.0, "comm": 0.0})
            acc[t.op.category][t.bound] += t.time
        return {c: max(v, key=v.get) for c, v in acc.items()}

    def chip_avg_power_w(
        self, chip: Chip, link: Link | None, duration: float | None = None
    ) -> float:
        """Average per-chip power: idle + per-component dynamic power scaled
        by that component's busy fraction."""
        total = duration if duration is not None else self.total_time
        if total <= 0:
            return chip.idle_power_w
        path_power = sum(s.power_w for s in chip.on_chip_path)
        p = chip.idle_power_w
        p += chip.compute.power_w * min(1.0, self.compute_busy / total)
        p += (chip.dram.power_w + path_power) * min(1.0, self.mem_busy / total)
        if link is not None:
            p += link.power_w * min(1.0, self.comm_busy / total)
        return p


@dataclass(frozen=True)
class CommContext:
    """The link each kind of communication travels over, given how the
    replica's chip groups map onto the machine (tp innermost, then ep/adp,
    then pp).

    The `a2a` group is the FFN array -- tp*ep*adp -- shared by the MoE
    dispatch/combine all-to-alls (ep) and the dense attention-DP FFN
    gather/reduce-scatter (adp).  ep and adp are mutually exclusive (MoE vs
    dense), so tp*ep*adp is tp*ep for MoE and tp*adp for dense."""

    tp: int
    tp_link: Link | None  # allreduce: spans the tp group
    a2a: int  # FFN-array group size: tp*ep (MoE) or tp*adp (dense)
    a2a_link: Link | None  # MoE all-to-all / dense FFN halfring: spans the a2a array
    p2p_link: Link | None  # pipeline hop: crosses stage boundaries
    # context-parallel prefill K/V ring width (= adp): the ring is over the adp
    # dimension (cp-1 steps) but its members span the tp*adp array with stride
    # tp, so it rides the a2a fabric (a2a_link / a2a_topology).  1 == no CP.
    cp: int = 1
    # topology of each group's fabric (the discrete-event engine expands
    # collectives per-step over it; the roofline engine ignores it).
    tp_topology: Topology = Topology.ALL_TO_ALL
    a2a_topology: Topology = Topology.ALL_TO_ALL

    @classmethod
    def for_deployment(cls, system: System, dep: Deployment) -> "CommContext":
        a2a_group = dep.tp * dep.ep * dep.adp
        return cls(
            tp=dep.tp,
            tp_link=system.link_for_group(dep.tp),
            a2a=a2a_group,
            a2a_link=system.link_for_group(a2a_group),
            p2p_link=system.link_for_group(dep.replica_chips),
            cp=dep.adp,
            tp_topology=system.topology_for_group(dep.tp),
            a2a_topology=system.topology_for_group(a2a_group),
        )


class Engine(ABC):
    @abstractmethod
    def run_phase(
        self, name: str, ops: list[Op], system: System, dep: Deployment
    ) -> Phase: ...


def _scaled_link(link: Link, factor: float) -> Link:
    """A copy of `link` with its bandwidth derated by `factor` (the collective
    efficiency).  Only the bandwidth is touched -- `latency_s` is physical
    flight time and must not be scaled -- so it feeds directly into the ring /
    a2a closed forms, which read `link.bandwidth` for the occupancy term and
    `link.latency_s` for the (unscaled) latency term.  At factor 1.0 the copy
    carries an identical bandwidth value, so the result is bit-identical."""
    return replace(link, bandwidth=link.bandwidth * factor)


class RooflineEngine(Engine):
    """Speed-of-light timing: t(op) = max(flops/peak, bytes/eff_bw).

    An `Efficiency` derates each peak (compute FLOP/s, memory bandwidth,
    collective link bandwidth) and adds a fixed per-op-instance overhead.  The
    default `Efficiency()` is the identity, so a bare `RooflineEngine()` is
    byte-for-byte the historical speed-of-light engine.
    """

    def __init__(self, efficiency: Efficiency = Efficiency()) -> None:
        self.efficiency = efficiency

    def time_op(self, op: Op, chip: Chip, comm: CommContext) -> OpTiming:
        eff = self.efficiency
        if op.kind is OpKind.ALLREDUCE:
            one = (
                ring_allreduce_time(
                    op.comm_bytes, comm.tp, _scaled_link(comm.tp_link, eff.collective)
                )
                if comm.tp > 1
                else 0.0
            )
            one += eff.op_overhead_s
            return OpTiming(op, op.count * one, 0.0, 0.0, op.count * one)
        if op.kind is OpKind.HALFRING:
            # dense attention-DP FFN allgather / reduce-scatter over the tp*adp
            # array: one-pass ring (half an allreduce).  Collective efficiency
            # scales the bandwidth (occupancy) term, never the flight latency.
            if comm.a2a_link is None:
                raise ValueError(f"{op.name}: no link available for {op.kind.value}")
            one = (
                ring_gather_time(
                    op.comm_bytes, comm.a2a, _scaled_link(comm.a2a_link, eff.collective)
                )
                if comm.a2a > 1
                else 0.0
            )
            one += eff.op_overhead_s
            return OpTiming(op, op.count * one, 0.0, 0.0, op.count * one)
        if op.kind is OpKind.CP_RING:
            # context-parallel prefill K/V ring over the cp = adp groups: cp-1
            # ring steps circulating each group's K/V block, so the closed form
            # is ring_gather_time with group = cp (NOT the tp*adp a2a group) and
            # payload = the full-prompt per-chip K/V.  It rides the a2a fabric
            # (its members span the tp*adp array).  Collective efficiency scales
            # the bandwidth (occupancy) term, never the flight latency.
            if comm.a2a_link is None:
                raise ValueError(f"{op.name}: no link available for {op.kind.value}")
            one = (
                ring_gather_time(
                    op.comm_bytes, comm.cp, _scaled_link(comm.a2a_link, eff.collective)
                )
                if comm.cp > 1
                else 0.0
            )
            one += eff.op_overhead_s
            return OpTiming(op, op.count * one, 0.0, 0.0, op.count * one)
        if op.kind in (OpKind.ALLTOALL, OpKind.P2P):
            link = comm.a2a_link if op.kind is OpKind.ALLTOALL else comm.p2p_link
            if link is None:
                raise ValueError(f"{op.name}: no link available for {op.kind.value}")
            # speed of light: payload streams at (derated) line rate after one
            # latency; the collective efficiency scales bandwidth, not latency.
            one = op.comm_bytes / (link.bandwidth * eff.collective) + link.latency_s
            one += eff.op_overhead_s
            return OpTiming(op, op.count * one, 0.0, 0.0, op.count * one)

        compute_t = (
            op.flops / (chip.compute.flops(op.dtype) * eff.compute) if op.flops else 0.0
        )
        bytes_moved = op.dram_read + op.dram_write
        mem_t = (
            bytes_moved / (chip.effective_dram_bandwidth * eff.memory)
            if bytes_moved
            else 0.0
        )
        one = max(compute_t, mem_t) + eff.op_overhead_s
        return OpTiming(
            op,
            op.count * one,
            op.count * compute_t,
            op.count * mem_t,
            0.0,
        )

    def run_phase(self, name: str, ops: list[Op], system: System, dep: Deployment) -> Phase:
        chip = system.node.chip
        comm = CommContext.for_deployment(system, dep)
        if dep.tp > 1 and comm.tp_link is None:
            raise ValueError("TP > 1 requires an interconnect for collectives")
        return Phase(name, [self.time_op(op, chip, comm) for op in ops])
