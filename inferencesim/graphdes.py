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

from .efficiency import Efficiency
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

    `efficiency` derates the chip's peaks consistently with the roofline /
    stage-DES: every element bandwidth (bank, NoC, SRAM, edges) is scaled by
    `efficiency.memory`, every per-core FLOP/s by `efficiency.compute`, and
    `efficiency.op_overhead_s` is added once to each op's wall (a fixed
    per-launched-op dispatch cost -- the tiles are one kernel's internal
    pipeline, so the overhead is charged per op, never per tile).  The default
    `Efficiency()` is the identity and leaves every wall bit-identical.
    """

    def __init__(self, graph: Graph, tile_fill: float = 0.5,
                 efficiency: Efficiency = Efficiency()):
        if not 0.0 < tile_fill <= 1.0:
            raise ValueError("tile_fill must be in (0, 1]")
        self.tile_fill = tile_fill
        self.efficiency = efficiency
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
        # per-instance effective peak rates (peak_flops * derate * compute
        # efficiency); non-uniform once any core is throttled/harvested.
        self._core_flops: dict[str, dict[DType, float]] = {
            n.name: {
                d: f * n.derate * efficiency.compute
                for d, f in (n.peak_flops or {}).items()
            }
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

        # 2D-mesh NoC routing: if the graph declares a mesh (meta["mesh"] =
        # {"rows", "cols", "router"}), route bank->core with deterministic XY
        # (dimension-ordered) paths that match Blackhole's NOC0 (East-then-South
        # == column-then-row) instead of an emergent BFS shortest path.  The
        # router group's instance index encodes its grid position (i -> (i//C,
        # i%C)), so coordinates survive expand()/JSON without per-instance meta.
        self._router_at: dict[tuple[int, int], str] = {}
        self._router_of: dict[str, str] = {}
        self._mesh_cols = 0
        mesh = g.meta.get("mesh")
        if mesh:
            router_base = str(mesh["router"])
            cols = int(mesh["cols"])
            self._mesh_cols = cols
            for n in g.nodes:
                if split_endpoint(n.name)[0] == router_base and "[" in n.name:
                    i = int(n.name[n.name.index("[") + 1:-1])
                    self._router_at[(i // cols, i % cols)] = n.name
            # each bank / L1 attaches to exactly one router
            for host in list(self.dram_instances) + list(self.core_to_sram.values()):
                for nb, _e in self._adj[host]:
                    if split_endpoint(nb)[0] == router_base and "[" in nb:
                        self._router_of[host] = nb
                        break

        self._chain_cache: dict[tuple[str, str], list[_Element]] = {}

    # ---- transfer path -------------------------------------------------------

    def _router_pos(self, router: str) -> tuple[int, int]:
        i = int(router[router.index("[") + 1:-1])
        return (i // self._mesh_cols, i % self._mesh_cols)

    def _xy_routers(self, entry: str, exit_: str) -> list[str]:
        """The router names on the XY (column-first, then row) path from `entry`
        to `exit_`, inclusive.  Matches Blackhole NOC0: move along the column
        axis to the destination column, then along the row axis -- a single
        turn, so it is A minimal path using only existing neighbour links."""
        (r0, c0), (r1, c1) = self._router_pos(entry), self._router_pos(exit_)
        routers = [entry]
        c = c0
        while c != c1:  # X first (change column)
            c += 1 if c1 > c else -1
            routers.append(self._router_at[(r0, c)])
        r = r0
        while r != r1:  # then Y (change row)
            r += 1 if r1 > r else -1
            routers.append(self._router_at[(r, c1)])
        return routers

    def _path_nodes(self, bank: str, core: str) -> list[str]:
        """Ordered node names a tile crosses from `bank` to `core` inclusive.

        On a declared mesh this is bank -> entry router -> XY router hops ->
        exit router -> L1 -> core; elsewhere it is a BFS shortest path over the
        undirected expanded graph (unchanged for the lumped-NoC fine presets)."""
        if (self._router_at and bank in self._router_of
                and core in self.core_to_sram
                and self.core_to_sram[core] in self._router_of):
            sram = self.core_to_sram[core]
            routers = self._xy_routers(self._router_of[bank], self._router_of[sram])
            return [bank, *routers, sram, core]
        # BFS shortest path (deterministic in adjacency order)
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
        cur = core
        while cur != bank:
            p, _e = parent[cur]
            nodes.append(p)
            cur = p
        nodes.reverse()
        return nodes

    def _edge_between(self, u: str, v: str) -> Edge:
        for w, e in self._adj[u]:
            if w == v:
                return e
        raise ValueError(f"{self.graph.name}: no edge between {u} and {v}")

    def _chain(self, bank: str, core: str) -> list[_Element]:
        """Ordered bandwidth-constrained elements a tile crosses from `bank`
        to `core`, excluding the terminal compute node.  XY mesh routing when
        the graph declares a mesh, else BFS shortest path; memoised per
        (bank, core)."""
        cached = self._chain_cache.get((bank, core))
        if cached is not None:
            return cached
        nodes = self._path_nodes(bank, core)
        edges = [self._edge_between(nodes[i], nodes[i + 1])
                 for i in range(len(nodes) - 1)]
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
                    _Element(name, nd.bandwidth * nd.derate * self.efficiency.memory,
                             nd.kind is NodeKind.SWITCH)
                )
            e = edges[i]
            if e.bandwidth is not None:
                elements.append(
                    _Element(_edge_res(name, nodes[i + 1]),
                             e.bandwidth * self.efficiency.memory, False)
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
        # fixed per-launched-op dispatch overhead (kernel launch), added once
        # per op rather than per tile: the tiles are this one kernel's internal
        # double-buffered pipeline.
        wall = result.makespan + self.efficiency.op_overhead_s
        return OpSchedule(wall, n_tiles, tasks, result, resources)
