"""Chip-graph op lowering: cost a COMPUTE Op against the expanded chip
graph's real resources instead of one lumped roofline number.

The stage-level DES (des.py) still schedules pipeline stages and links, but
its per-op unit times come from the roofline `max(flops/peak, bytes/bw)`.
This module refines that one hook: an `Op` is lowered to a tile task graph
and scheduled with the same `sched.py` core the engine already uses, so the
contention the roofline averages away becomes emergent.

Lowering (see `op_wall`): the op is split into tiles bounded by the SRAM
cap, but a compute-bearing op makes at least one tile per core so its FLOPs
distribute over the whole pool (byte count sizes memory tiles, it does not
partition compute).  Each tile streams from a DRAM bank, hop by hop along
the on-chip path (store-and-forward: one FIFO task per bandwidth-constrained
node/edge, a shared processor-sharing task for a SWITCH/NoC) into a core's
SRAM, computes its flops on that core's matrix engine, then writes results
back along the reverse path.  Tiles round-robin over banks and cores -- the
same interleave convention `Graph.expand()` wires -- so bank and core
conflicts are modelled rather than averaged, and compute/DRAM overlap falls
out of the double-buffered schedule instead of being assumed as
`max(compute, mem)`.

This module knows nothing about engines, pipelines or collectives; it may
read the graph, hardware (for dtype widening) and ops, and drives sched.py.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil, floor

from .graph import Edge, Graph, Node, NodeKind, split_endpoint
from .hardware import Compute
from .ops import Op
from .sched import Resource, ScheduleResult, Task, schedule


def _edge_res(a: str, b: str) -> str:
    """Canonical (direction-independent) resource name for a link, so reads
    and their write-backs contend on the same physical edge."""
    lo, hi = sorted((a, b))
    return f"{lo}~{hi}"


@dataclass
class _Element:
    """One bandwidth-constrained hop a tile crosses: a node or edge resource,
    its bandwidth (bytes/s) and its scheduler discipline (shared = processor
    sharing, for a SWITCH/NoC; otherwise a 1-server FIFO)."""

    resource: str
    bandwidth: float
    shared: bool


@dataclass
class OpSchedule:
    """The lowered tile schedule for one Op (count == 1): its measured wall
    time plus the task graph / result / resources, for tracing and per-chip
    utilisation."""

    wall: float
    n_tiles: int
    tasks: list[Task]
    result: ScheduleResult
    resources: dict[str, Resource]


class ChipModel:
    """A chip `Graph` lowered to a schedulable resource model.

    The graph is `expand()`ed and classified into a DRAM bank group, a
    compute-core group and (optionally) a per-core SRAM group; `op_wall`
    lowers a COMPUTE `Op` to a tile task graph over those resources.

    tile_fill is the fraction of a core's SRAM one tile may occupy; a core
    then double-buffers `max(1, floor(1/tile_fill))` tiles (0.5 -> 2 buffers:
    the next tile's read overlaps the current tile's compute).
    """

    def __init__(self, graph: Graph, tile_fill: float = 0.5):
        if not 0.0 < tile_fill <= 1.0:
            raise ValueError("tile_fill must be in (0, 1]")
        self.tile_fill = tile_fill
        g = graph.expand(deep=True)
        g.validate()
        self.graph = g

        # undirected adjacency over the expanded edge list (endpoints are
        # concrete instance names after expand)
        self._adj: dict[str, list[tuple[str, Edge]]] = {n.name: [] for n in g.nodes}
        for e in g.edges:
            self._adj[e.src].append((e.dst, e))
            self._adj[e.dst].append((e.src, e))
        self._node: dict[str, Node] = {n.name: n for n in g.nodes}

        computes = [n for n in g.nodes if n.kind is NodeKind.COMPUTE]
        if not computes:
            raise ValueError(f"{graph.name}: chip graph has no compute node")
        # A disabled instance (derate 0) exists physically but does no work:
        # drop it from the schedulable pool entirely.  Derated instances stay,
        # with their per-instance rates scaled by `derate` -- so a harvested
        # 132-of-140 die schedules on 132 cores and a throttled core paces its
        # own tiles.  (expand() has already baked instance_derates into each
        # instance's plain `derate`.)
        enabled = [n for n in computes if n.derate > 0.0]
        if not enabled:
            raise ValueError(f"{graph.name}: every compute instance is disabled")
        self.compute_instances = [n.name for n in enabled]
        self.n_cores = len(self.compute_instances)
        # per-instance effective peak rates (peak_flops * derate); non-uniform
        # once any core is throttled/harvested.
        self._core_flops: dict[str, dict[DType, float]] = {
            n.name: {d: f * n.derate for d, f in (n.peak_flops or {}).items()}
            for n in enabled
        }

        # DRAM = the memory group (instances sharing a base name) with the
        # largest total capacity (same convention as bridge.chip_from_graph).
        # Disabled banks (derate 0) leave the pool -- excluded from capacity
        # and from the round-robin alike.
        groups: dict[str, list[Node]] = {}
        for m in g.nodes:
            if m.kind is NodeKind.MEMORY and m.capacity_bytes and m.derate > 0.0:
                groups.setdefault(split_endpoint(m.name)[0], []).append(m)
        if not groups:
            raise ValueError(f"{graph.name}: chip graph has no memory with a capacity")
        dram_base, dram_members = max(
            groups.items(), key=lambda kv: sum(m.capacity_bytes or 0.0 for m in kv[1])
        )
        self.dram_base = dram_base
        self.dram_instances = [m.name for m in dram_members]
        self.n_banks = len(self.dram_instances)

        # SRAM = the memory group adjacent to the compute cores but distinct
        # from DRAM.  Absent for lumped bare-chip graphs (chip_to_graph) or
        # graphs with no SRAM: then tiling has no capacity limit and each op
        # is a single tile.
        core_to_sram: dict[str, str] = {}
        sram_seen: set[str] = set()
        for c in self.compute_instances:
            for nb, _e in self._adj[c]:
                nd = self._node[nb]
                if nd.kind is NodeKind.MEMORY and split_endpoint(nb)[0] != dram_base:
                    core_to_sram[c] = nb
                    sram_seen.add(nb)
                    break
        self.core_to_sram = core_to_sram
        self.sram_instances = [n.name for n in g.nodes if n.name in sram_seen]
        self.sram_capacity = (
            self._node[self.sram_instances[0]].capacity_bytes
            if self.sram_instances else None
        )
        self.tile_bytes = (
            self.sram_capacity * tile_fill if self.sram_capacity else None
        )
        self.n_buffers = max(1, floor(1.0 / tile_fill))

        self._chain_cache: dict[tuple[str, str], list[_Element]] = {}

    # ---- transfer path -------------------------------------------------------

    def _chain(self, bank: str, core: str) -> list[_Element]:
        """Ordered bandwidth-constrained elements a tile crosses from `bank`
        to `core`, excluding the terminal compute node.  BFS shortest path
        over the undirected expanded graph; memoised per (bank, core)."""
        cached = self._chain_cache.get((bank, core))
        if cached is not None:
            return cached
        parent: dict[str, tuple[str, Edge]] = {}
        seen = {bank}
        q = deque([bank])
        while q:
            u = q.popleft()
            if u == core:
                break
            for v, e in self._adj[u]:
                if v not in seen:
                    seen.add(v)
                    parent[v] = (u, e)
                    q.append(v)
        if core != bank and core not in parent:
            raise ValueError(f"{self.graph.name}: no path from {bank} to {core}")
        nodes = [core]
        edges: list[Edge] = []
        cur = core
        while cur != bank:
            p, e = parent[cur]
            edges.append(e)
            nodes.append(p)
            cur = p
        nodes.reverse()
        edges.reverse()
        elements: list[_Element] = []
        for i, name in enumerate(nodes[:-1]):  # skip the terminal compute core
            nd = self._node[name]
            if nd.bandwidth is not None:
                # the node figure is authoritative for derating: a derated bank
                # serves at bandwidth * derate.  Its incoming/outgoing EDGE is
                # deliberately NOT derated (the fine presets doubly-encode the
                # per-bank bandwidth on the edge too; leaving it at full rate is
                # harmless because the derated node element is the binding stage
                # of the store-and-forward chain).
                elements.append(
                    _Element(name, nd.bandwidth * nd.derate, nd.kind is NodeKind.SWITCH)
                )
            e = edges[i]
            if e.bandwidth is not None:
                elements.append(
                    _Element(_edge_res(name, nodes[i + 1]), e.bandwidth, False)
                )
        self._chain_cache[(bank, core)] = elements
        return elements

    def _n_tiles(self, total_read: float, has_flops: bool) -> int:
        """How many tiles to split an op into.

        Memory tiling bounds a tile at the SRAM cap (`tile_bytes`); it can
        only make tiles *smaller*, never coarser.  Compute is a separate
        concern: a compute-bearing op runs at least one tile per core, so
        compute-bound work distributes over the whole pool (roofline
        consistent) even when its DRAM reads are tiny -- byte count must not
        double as the compute-partitioning knob.  A pure transfer (flops == 0)
        has no compute to spread, so it keeps the memory count."""
        mem_tiles = (
            ceil(total_read / self.tile_bytes)
            if (self.tile_bytes and total_read > 0) else 1
        )
        if has_flops:
            return max(self.n_cores, mem_tiles)
        return max(1, mem_tiles)

    # ---- op lowering ---------------------------------------------------------

    def op_wall(self, op: Op) -> OpSchedule:
        """Lower a single COMPUTE op (count == 1) to a tile task graph and
        return its measured wall time and schedule."""
        n_tiles = self._n_tiles(op.dram_read, op.flops > 0)
        read_per = op.dram_read / n_tiles
        write_per = op.dram_write / n_tiles
        flops_per = op.flops / n_tiles
        # per-core rate for this dtype (already scaled by each core's derate).
        # With non-uniform rates the round-robin assignment is static work
        # distribution: the slowest core assigned a tile paces the op's tail.
        # That is the intended physical behaviour; work-stealing is future work.
        rates = {
            c: (Compute("core", pf).flops(op.dtype) if pf else 0.0)
            for c, pf in self._core_flops.items()
        }

        tasks: list[Task] = []
        resources: dict[str, Resource] = {}
        # per-core positions of the buffer-releasing task (compute, else last
        # transfer) so a later tile's first read can wait on it
        core_release: dict[str, list[int]] = {c: [] for c in self.compute_instances}

        def add(resource: str, dur: float, deps: list[int], label: str,
                shared: bool) -> int:
            key = len(tasks)
            tasks.append(Task(key, resource, dur, deps, label))
            if shared:
                resources[resource] = Resource(resource, shared=True)
            return key

        for i in range(n_tiles):
            bank = self.dram_instances[i % self.n_banks]
            core = self.compute_instances[i % self.n_cores]
            chain = self._chain(bank, core)
            rel = core_release[core]
            pos = len(rel)
            # double-buffer gate: the buffer this tile needs is freed by the
            # compute (or last transfer) of the tile n_buffers earlier on the
            # same core, so its first task waits for that
            gate: list[int] = []
            if pos >= self.n_buffers and rel[pos - self.n_buffers] is not None:
                gate = [rel[pos - self.n_buffers]]

            prev: int | None = None
            first = True

            def link(resource: str, dur: float, label: str, shared: bool) -> int:
                nonlocal prev, first
                deps = ([prev] if prev is not None else []) + (gate if first else [])
                k = add(resource, dur, deps, label, shared)
                prev, first = k, False
                return k

            if read_per > 0:
                for el in chain:
                    link(el.resource, read_per / el.bandwidth,
                         f"{op.name} t{i} rd {el.resource}", el.shared)
            comp: int | None = None
            rate = rates[core]
            if flops_per > 0 and rate > 0:
                comp = link(core, flops_per / rate, f"{op.name} t{i} cp {core}", False)
            if write_per > 0:
                for el in reversed(chain):
                    link(el.resource, write_per / el.bandwidth,
                         f"{op.name} t{i} wb {el.resource}", el.shared)
            rel.append(comp if comp is not None else prev)

        result = schedule(tasks, resources)
        return OpSchedule(result.makespan, n_tiles, tasks, result, resources)
