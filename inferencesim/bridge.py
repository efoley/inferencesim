"""Bridge between the hierarchical hardware graph and the spec-sheet layer.

Two directions:

  to_graph(...)          Chip / Node / System  ->  Graph
      Every existing preset gets a graph representation for free; the graph
      is what a visual editor edits and what a discrete-event engine will
      walk.

  system_from_graph(g)   Graph  ->  System
      Aggregates a (possibly hand-built or UI-built) graph back into the
      homogeneous Chip/Node/System view that the analytic roofline engine
      consumes.  Round-tripping a preset through both directions yields
      identical simulation results (tested).

Graph conventions the extractor understands (the builders emit exactly
these):

  * role="chip" composites are processors; their inner graph holds COMPUTE,
    MEMORY and SWITCH nodes.  The chip's DRAM is the memory with the
    largest aggregate capacity; effective streaming bandwidth is the widest
    path from it to the (largest) compute node.
  * role="node" composites group chips; edges inside them (chip <-> fabric
    switch, or chip <-> chip) define the intra-node interconnect.
  * Edges at the root between "node" composites (possibly via a SWITCH)
    define the network.
  * Costs and host overhead power live in composite meta:
    {"cost_usd": ..., "overhead_power_w": ...}.
  * The interconnect topology (ring / all-to-all / mesh) lives on the fabric
    SWITCH node's meta ({"topology": ...}) -- it is the interconnect's
    property, not the chip's -- and defaults to all-to-all when absent.

Convenience fallbacks: a graph with no composites at all is treated as one
chip in a single-chip system; a graph whose root directly holds role="chip"
composites is treated as a single node.  Heterogeneous chips are rejected
(the roofline engine assumes homogeneous chips; a discrete-event engine
will lift this).
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Any

from .graph import Edge, Graph, Node, NodeKind, split_endpoint
from .hardware import Chip, Compute, DType, Link, Memory, System, Topology
from .hardware import Node as HwNode

# =============================================================================
# spec-sheet -> graph
# =============================================================================


def _unique(name: str, taken: set[str]) -> str:
    base, i = name, 2
    while name in taken:
        name = f"{base}#{i}"
        i += 1
    taken.add(name)
    return name


def chip_to_graph(chip: Chip) -> Graph:
    """Chip -> inner graph: DRAM -> (path stages) -> compute, as a chain."""
    taken: set[str] = set()
    nodes: list[Node] = []
    edges: list[Edge] = []

    dram_name = _unique(chip.dram.name, taken)
    nodes.append(Node(
        name=dram_name,
        kind=NodeKind.MEMORY,
        capacity_bytes=chip.dram.capacity_bytes,
        bandwidth=chip.dram.bandwidth,
        latency_s=chip.dram.latency_s,
        dynamic_power_w=chip.dram.power_w,
    ))
    prev = dram_name
    port = None
    for stage in chip.on_chip_path:
        if isinstance(stage, Memory):
            n = Node(
                name=_unique(stage.name, taken),
                kind=NodeKind.MEMORY,
                capacity_bytes=stage.capacity_bytes,
                bandwidth=stage.bandwidth,
                latency_s=stage.latency_s,
                dynamic_power_w=stage.power_w,
            )
        else:  # Link -> switch stage (e.g. a NoC)
            n = Node(
                name=_unique(stage.name, taken),
                kind=NodeKind.SWITCH,
                bandwidth=stage.bandwidth,
                latency_s=stage.latency_s,
                dynamic_power_w=stage.power_w,
            )
            port = n.name  # external links land at the on-chip fabric
        nodes.append(n)
        edges.append(Edge(src=prev, dst=n.name))
        prev = n.name

    compute_name = _unique(chip.compute.name, taken)
    nodes.append(Node(
        name=compute_name,
        kind=NodeKind.COMPUTE,
        peak_flops=dict(chip.compute.peak_flops),
        dynamic_power_w=chip.compute.power_w,
    ))
    edges.append(Edge(src=prev, dst=compute_name))

    g = Graph(name=chip.name, nodes=nodes, edges=edges,
              meta={"port": port or compute_name})
    return g


def _chip_composite(chip: Chip, count: int) -> Node:
    inner = chip_to_graph(chip)
    return Node(
        name="chip",
        kind=NodeKind.COMPOSITE,
        count=count,
        role="chip",
        idle_power_w=chip.idle_power_w,
        inner=inner,
        ports=(inner.meta["port"],),
        meta={"chip_name": chip.name},
    )


def node_to_graph(node: HwNode) -> Graph:
    chip_node = _chip_composite(node.chip, node.n_chips)
    nodes = [chip_node]
    edges: list[Edge] = []
    port = "chip"
    if node.n_chips > 1:
        assert node.interconnect is not None
        fabric = Node(
            name="fabric",
            kind=NodeKind.SWITCH,
            bandwidth=None,  # non-blocking; per-chip injection is the edge
            meta={"link_name": node.interconnect.name,
                  "topology": node.topology.value},
        )
        nodes.append(fabric)
        edges.append(Edge(
            src="chip",
            dst="fabric",
            bandwidth=node.interconnect.bandwidth,
            latency_s=node.interconnect.latency_s,
            power_w=node.interconnect.power_w,
            name=node.interconnect.name,
        ))
        port = "fabric"
    return Graph(name=node.name, nodes=nodes, edges=edges, meta={"port": port})


def system_to_graph(system: System) -> Graph:
    inner = node_to_graph(system.node)
    node_comp = Node(
        name="node",
        kind=NodeKind.COMPOSITE,
        count=system.n_nodes,
        role="node",
        inner=inner,
        ports=(inner.meta["port"],),
        meta={
            "node_name": system.node.name,
            "cost_usd": system.node.cost_usd,
            "overhead_power_w": system.node.overhead_power_w,
        },
    )
    nodes = [node_comp]
    edges: list[Edge] = []
    if system.n_nodes > 1:
        assert system.network is not None
        net = Node(
            name="network",
            kind=NodeKind.SWITCH,
            bandwidth=None,
            meta={"link_name": system.network.name},
        )
        nodes.append(net)
        edges.append(Edge(
            src="node",
            dst="network",
            bandwidth=system.network.bandwidth,
            latency_s=system.network.latency_s,
            power_w=system.network.power_w,
            name=system.network.name,
        ))
    return Graph(
        name=system.name,
        nodes=nodes,
        edges=edges,
        meta={"extra_cost_usd": system.extra_cost_usd,
              "description": system.description},
    )


# =============================================================================
# graph -> spec-sheet (for the analytic engine)
# =============================================================================


def chip_from_graph(g: Graph, idle_power_w: float = 0.0, name: str | None = None) -> Chip:
    """Aggregate a chip-level graph into an equivalent roofline Chip.

    Compute = sum of all COMPUTE nodes (dtype-wise); DRAM = the memory with
    the largest aggregate capacity; effective bandwidth = widest path from
    DRAM to the biggest compute node; all memory/switch/edge dynamic power
    folds into the DRAM-side figure (the roofline power model scales it by
    the memory-busy fraction either way)."""
    flat = g.flatten()
    flat.validate()

    computes = flat.find(kind=NodeKind.COMPUTE)
    if not computes:
        raise ValueError(f"{g.name}: chip graph has no compute node")
    flops: dict[DType, float] = {}
    for c in computes:
        for d, f in c.agg_flops.items():
            flops[d] = flops.get(d, 0.0) + f
    compute_power = sum(c.dynamic_power_w * c.count for c in computes)

    # DRAM = the group of memory instances (sharing a base name, e.g.
    # "gddr6-bank" / "gddr6-bank[3]") with the largest total capacity
    groups: dict[str, list[Node]] = {}
    for m in flat.find(kind=NodeKind.MEMORY):
        if m.capacity_bytes:
            groups.setdefault(split_endpoint(m.name)[0], []).append(m)
    if not groups:
        raise ValueError(f"{g.name}: chip graph has no memory with a capacity")
    dram_base, dram_members = max(
        groups.items(), key=lambda kv: sum(m.agg_capacity or 0.0 for m in kv[1])
    )
    dram_capacity = sum(m.agg_capacity or 0.0 for m in dram_members)
    dram_latency = max(m.latency_s for m in dram_members)

    # effective streaming bandwidth: max flow credits parallel routes and
    # is invariant under expand()
    eff_bw = flat.max_flow(
        [m.name for m in dram_members], [c.name for c in computes]
    )
    if eff_bw == float("inf"):
        raise ValueError(
            f"{g.name}: no bandwidth constraint anywhere between "
            f"{dram_base} and the compute nodes"
        )

    mem_side_power = sum(
        n.dynamic_power_w * n.count
        for n in flat.nodes
        if n.kind in (NodeKind.MEMORY, NodeKind.SWITCH)
    ) + sum(e.power_w * e.count for e in flat.edges)
    idle = idle_power_w + sum(n.idle_power_w * n.count for n in flat.nodes)

    return Chip(
        name=name or g.name,
        compute=Compute(name="aggregated compute", peak_flops=flops,
                        power_w=compute_power),
        dram=Memory(
            name=dram_base,
            capacity_bytes=dram_capacity,
            bandwidth=eff_bw,
            power_w=mem_side_power,
            latency_s=dram_latency,
        ),
        on_chip_path=(),
        idle_power_w=idle,
    )


def _link_from_edges(g: Graph, composite_name: str, what: str) -> Link | None:
    """The slowest edge touching `composite_name` defines the group's link."""
    touching = [e for e in g.edges if composite_name in (e.src, e.dst)]
    if not touching:
        return None
    constrained = [e for e in touching if e.bandwidth is not None]
    if not constrained:
        raise ValueError(f"{g.name}: {what} edges have no bandwidth constraint")
    slowest = min(constrained, key=lambda e: e.agg_bandwidth or 0.0)
    return Link(
        name=slowest.name or what,
        bandwidth=(slowest.agg_bandwidth or 0.0),
        latency_s=slowest.latency_s,
        power_w=slowest.power_w,
    )


def _topology_from_switches(g: Graph) -> Topology:
    """Interconnect topology from the fabric SWITCH node's meta (its single
    home); defaults to all-to-all when no switch declares one."""
    for n in g.nodes:
        if n.kind is NodeKind.SWITCH and "topology" in n.meta:
            return Topology(n.meta["topology"])
    return Topology.ALL_TO_ALL


def _node_from_graph(g: Graph, meta: dict[str, Any], name: str) -> HwNode:
    chips = g.find(role="chip")
    if not chips:
        # the whole graph is a single bare chip
        chip = chip_from_graph(g, name=name)
        return HwNode(name=name, chip=chip, n_chips=1,
                      overhead_power_w=float(meta.get("overhead_power_w", 0.0)),
                      cost_usd=float(meta.get("cost_usd", 0.0)))
    if len(chips) > 1:
        raise ValueError(
            f"{g.name}: {len(chips)} distinct chip composites; the roofline "
            f"engine needs homogeneous chips -- use count on one composite "
            f"(a discrete-event engine will lift this)"
        )
    cc = chips[0]
    assert cc.inner is not None
    chip = chip_from_graph(cc.inner, idle_power_w=cc.idle_power_w,
                           name=str(cc.meta.get("chip_name", cc.inner.name)))
    interconnect = _link_from_edges(g, cc.name, "interconnect")
    if cc.count > 1 and interconnect is None:
        # maybe chips connect pairwise chip <-> chip; a self-edge is invalid,
        # so nothing to find: require a fabric or explicit edge
        raise ValueError(f"{g.name}: {cc.count} chips but no interconnect edge")
    # topology is a property of the interconnect: node_to_graph writes it onto
    # the fabric SWITCH node's meta, so read it back from there.
    topo = _topology_from_switches(g)
    return HwNode(
        name=name,
        chip=chip,
        n_chips=cc.count,
        interconnect=interconnect,
        topology=Topology(topo),
        overhead_power_w=float(meta.get("overhead_power_w", 0.0)),
        cost_usd=float(meta.get("cost_usd", 0.0)),
    )


def system_from_graph(g: Graph) -> System:
    """Aggregate a hardware graph into the System view the roofline engine
    simulates.  See module docstring for the shapes accepted."""
    g.validate()
    node_comps = g.find(role="node")
    if len(node_comps) > 1:
        raise ValueError(
            f"{g.name}: {len(node_comps)} distinct node composites; use count "
            f"on one composite (heterogeneous systems: roadmap)"
        )
    if node_comps:
        nc = node_comps[0]
        assert nc.inner is not None
        hw_node = _node_from_graph(nc.inner, nc.meta,
                                   name=str(nc.meta.get("node_name", nc.inner.name)))
        network = _link_from_edges(g, nc.name, "network")
        if nc.count > 1 and network is None:
            raise ValueError(f"{g.name}: {nc.count} nodes but no network edge")
        return System(
            name=g.name,
            node=hw_node,
            n_nodes=nc.count,
            network=network,
            extra_cost_usd=float(g.meta.get("extra_cost_usd", 0.0)),
            description=str(g.meta.get("description", "")),
        )
    # no "node" composite: treat the root as a single node (which itself
    # falls back to a bare chip if there are no "chip" composites either)
    hw_node = _node_from_graph(g, g.meta, name=g.name)
    return System(
        name=g.name,
        node=hw_node,
        n_nodes=1,
        network=None,
        extra_cost_usd=float(g.meta.get("extra_cost_usd", 0.0)),
        description=str(g.meta.get("description", "")),
    )


def swap_chip_model(system_graph: Graph, chip_inner: Graph, port: str) -> Graph:
    """Replace the chip-level model inside a system graph with a different
    abstraction level (e.g. swap a lumped chip for a per-core model).
    Returns a new graph; the original is untouched."""
    def rewrite(g: Graph) -> Graph:
        new_nodes = []
        for n in g.nodes:
            if n.role == "chip":
                n = dc_replace(n, inner=chip_inner, ports=(port,),
                               meta={**n.meta, "chip_name": chip_inner.name})
            elif n.inner is not None:
                n = dc_replace(n, inner=rewrite(n.inner))
            new_nodes.append(n)
        return dc_replace(g, nodes=new_nodes)

    out = rewrite(system_graph)
    out.validate()
    return out
