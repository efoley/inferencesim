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
from dataclasses import dataclass, field

from .hardware import Chip, Link, System
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
    """A timed group of ops (one prefill, or one decode step)."""

    name: str
    timings: list[OpTiming] = field(default_factory=list)

    @property
    def total_time(self) -> float:
        return sum(t.time for t in self.timings)

    def duration(self, overlap_comm: bool = False) -> float:
        """Wall-clock time for the phase.  With overlap_comm, collectives run
        concurrently with on-chip work: duration = max(comm, everything else)."""
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
    replica's chip groups map onto the machine (tp innermost, then ep,
    then pp)."""

    tp: int
    tp_link: Link | None  # allreduce: spans the tp group
    a2a_link: Link | None  # MoE all-to-all: spans the tp*ep array
    p2p_link: Link | None  # pipeline hop: crosses stage boundaries

    @classmethod
    def for_deployment(cls, system: System, dep: Deployment) -> "CommContext":
        return cls(
            tp=dep.tp,
            tp_link=system.link_for_group(dep.tp),
            a2a_link=system.link_for_group(dep.tp * dep.ep),
            p2p_link=system.link_for_group(dep.replica_chips),
        )


class Engine(ABC):
    @abstractmethod
    def run_phase(
        self, name: str, ops: list[Op], system: System, dep: Deployment
    ) -> Phase: ...


class RooflineEngine(Engine):
    """Speed-of-light timing: t(op) = max(flops/peak, bytes/eff_bw)."""

    def time_op(self, op: Op, chip: Chip, comm: CommContext) -> OpTiming:
        if op.kind is OpKind.ALLREDUCE:
            one = (
                ring_allreduce_time(op.comm_bytes, comm.tp, comm.tp_link)
                if comm.tp > 1
                else 0.0
            )
            return OpTiming(op, op.count * one, 0.0, 0.0, op.count * one)
        if op.kind in (OpKind.ALLTOALL, OpKind.P2P):
            link = comm.a2a_link if op.kind is OpKind.ALLTOALL else comm.p2p_link
            if link is None:
                raise ValueError(f"{op.name}: no link available for {op.kind.value}")
            # speed of light: payload streams at line rate after one latency
            one = op.comm_bytes / link.bandwidth + link.latency_s
            return OpTiming(op, op.count * one, 0.0, 0.0, op.count * one)

        compute_t = op.flops / chip.compute.flops(op.dtype) if op.flops else 0.0
        bytes_moved = op.dram_read + op.dram_write
        mem_t = bytes_moved / chip.effective_dram_bandwidth if bytes_moved else 0.0
        one = max(compute_t, mem_t)
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
