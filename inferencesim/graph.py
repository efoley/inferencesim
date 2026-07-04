"""Hierarchical hardware graph: the structural source of truth.

A machine is a graph of nodes (things with constraints) joined by edges
(interconnects with bandwidth and latency):

  Node  -- a compute pool (constraint: which dtypes, at what FLOP/s),
           a memory (constraints: capacity, bandwidth), or a switch/router
           (constraint: aggregate bandwidth).  A node may instead be a
           COMPOSITE that nests a whole inner Graph, which is how the same
           machine can be modelled at different abstraction levels: a DGX
           Spark can be one node, or a composite containing SMs, shared
           memory and DRAM.
  Edge  -- a link between two sibling nodes: NoC hop, NVLink, PCIe,
           Ethernet.  The Tensix-Tensix connection inside a Blackhole and
           the ConnectX-7 between two Sparks are both just edges at
           different levels of the hierarchy.

Semantics:
  * `count` on a node means "this many identical instances" (140 Tensix
    cores); per-instance figures live on the node, aggregate = value*count.
    `count` on an edge means parallel links.
  * `bandwidth=None` means unconstrained (constraints live elsewhere).
  * A composite exposes `ports`: inner node names that outer edges attach
    to.  Flattening rewires edges accordingly.

Engines consume *views* of this graph (the analytic roofline engine
aggregates it via bridge.py; a discrete-event engine can walk the real
edges).  Graphs serialise to/from JSON so external tools -- e.g. a
graphical editor -- can produce them.
"""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterator

from .hardware import DType
from .units import fmt_bytes, fmt_si, fmt_time

FORMAT = "inferencesim-graph-v1"


class NodeKind(str, Enum):
    COMPUTE = "compute"
    MEMORY = "memory"
    SWITCH = "switch"
    COMPOSITE = "composite"


@dataclass
class Node:
    """A vertex with constraints.  Which constraints are meaningful depends
    on `kind`; everything is optional so a UI can build nodes incrementally
    and `Graph.validate()` reports what is missing."""

    name: str
    kind: NodeKind
    count: int = 1  # identical instances (e.g. 140 Tensix cores)
    # constraints (per instance)
    peak_flops: dict[DType, float] | None = None  # COMPUTE: supported dtypes
    capacity_bytes: float | None = None  # MEMORY
    bandwidth: float | None = None  # MEMORY/SWITCH throughput cap; None = unconstrained
    latency_s: float = 0.0
    dynamic_power_w: float = 0.0  # at full utilisation
    idle_power_w: float = 0.0
    # semantics for extraction / UI grouping: "chip", "node", ...
    role: str = ""
    # nesting
    inner: "Graph | None" = None
    ports: tuple[str, ...] = ()  # inner node names outer edges attach to
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def agg_bandwidth(self) -> float | None:
        return None if self.bandwidth is None else self.bandwidth * self.count

    @property
    def agg_capacity(self) -> float | None:
        return None if self.capacity_bytes is None else self.capacity_bytes * self.count

    @property
    def agg_flops(self) -> dict[DType, float]:
        if not self.peak_flops:
            return {}
        return {d: f * self.count for d, f in self.peak_flops.items()}


@dataclass
class Edge:
    """An interconnect between two sibling nodes (bidirectional).

    bandwidth is bytes/s per direction per link; None = unconstrained."""

    src: str
    dst: str
    bandwidth: float | None = None
    latency_s: float = 0.0
    power_w: float = 0.0
    count: int = 1  # parallel links
    name: str = ""

    @property
    def agg_bandwidth(self) -> float | None:
        return None if self.bandwidth is None else self.bandwidth * self.count


@dataclass(frozen=True)
class PathResult:
    bandwidth: float  # bottleneck along the widest path (inf if unconstrained)
    latency_s: float  # summed node + edge latency along that path
    nodes: tuple[str, ...]


@dataclass
class Graph:
    name: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- access -----------------------------------------------------------

    def node(self, name: str) -> Node:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(f"{self.name}: no node named '{name}'")

    def has_node(self, name: str) -> bool:
        return any(n.name == name for n in self.nodes)

    def find(self, kind: NodeKind | None = None, role: str | None = None) -> list[Node]:
        out = []
        for n in self.nodes:
            if kind is not None and n.kind is not kind:
                continue
            if role is not None and n.role != role:
                continue
            out.append(n)
        return out

    def walk(self) -> Iterator[tuple[str, "Node"]]:
        """Yield (path, node) for every node at every nesting level."""
        for n in self.nodes:
            yield n.name, n
            if n.inner is not None:
                for sub, m in n.inner.walk():
                    yield f"{n.name}/{sub}", m

    # ---- validation ---------------------------------------------------------

    def validate(self) -> None:
        names = [n.name for n in self.nodes]
        if len(names) != len(set(names)):
            dupes = sorted({x for x in names if names.count(x) > 1})
            raise ValueError(f"{self.name}: duplicate node names {dupes}")
        for n in self.nodes:
            if n.count < 1:
                raise ValueError(f"{self.name}/{n.name}: count must be >= 1")
            if n.kind is NodeKind.COMPOSITE:
                if n.inner is None:
                    raise ValueError(f"{self.name}/{n.name}: composite needs an inner graph")
                for p in n.ports:
                    if not n.inner.has_node(p):
                        raise ValueError(
                            f"{self.name}/{n.name}: port '{p}' not in inner graph"
                        )
                n.inner.validate()
            else:
                if n.inner is not None:
                    raise ValueError(f"{self.name}/{n.name}: only composites may nest")
                if n.kind is NodeKind.COMPUTE and not n.peak_flops:
                    raise ValueError(f"{self.name}/{n.name}: compute node needs peak_flops")
                if n.kind is NodeKind.MEMORY and (
                    n.capacity_bytes is None and n.bandwidth is None
                ):
                    raise ValueError(
                        f"{self.name}/{n.name}: memory node needs capacity or bandwidth"
                    )
        for e in self.edges:
            for end in (e.src, e.dst):
                if not self.has_node(end):
                    raise ValueError(f"{self.name}: edge endpoint '{end}' does not exist")
            if e.src == e.dst:
                raise ValueError(f"{self.name}: self-edge on '{e.src}'")
            if e.count < 1:
                raise ValueError(f"{self.name}: edge {e.src}--{e.dst} count must be >= 1")

    # ---- flattening ---------------------------------------------------------

    def flatten(self, sep: str = "/") -> "Graph":
        """Expand composites into their inner graphs (one representative
        instance each; counts on inner nodes are preserved).  Outer edges
        that touched a composite are rewired to its first port.  Composite
        power figures are dropped -- flattening is for structural queries;
        extraction reads power from the hierarchy."""
        flat, _ = self._flatten(sep)
        return flat

    def _flatten(self, sep: str) -> tuple["Graph", dict[str, str]]:
        nodes: list[Node] = []
        edges: list[Edge] = []
        port_map: dict[str, str] = {}
        for n in self.nodes:
            if n.kind is not NodeKind.COMPOSITE:
                nodes.append(replace(n))
                continue
            assert n.inner is not None
            inner_flat, inner_ports = n.inner._flatten(sep)
            for m in inner_flat.nodes:
                nodes.append(replace(m, name=f"{n.name}{sep}{m.name}"))
            for e in inner_flat.edges:
                edges.append(replace(e, src=f"{n.name}{sep}{e.src}",
                                     dst=f"{n.name}{sep}{e.dst}"))
            if n.ports:
                port = inner_ports.get(n.ports[0], n.ports[0])
                port_map[n.name] = f"{n.name}{sep}{port}"
        for e in self.edges:
            src = port_map.get(e.src, e.src)
            dst = port_map.get(e.dst, e.dst)
            for end, orig in ((src, e.src), (dst, e.dst)):
                if not any(x.name == end for x in nodes):
                    raise ValueError(
                        f"{self.name}: edge touches composite '{orig}' with no ports"
                    )
            edges.append(replace(e, src=src, dst=dst))
        return Graph(self.name, nodes, edges, dict(self.meta)), port_map

    # ---- queries ------------------------------------------------------------

    def widest_path(self, src: str, dst: str) -> PathResult:
        """Maximise the bottleneck bandwidth between two nodes (composites
        must be flattened first).  Bottleneck = min over node caps (incl.
        endpoints) and edge caps, all aggregated by count."""
        caps = {
            n.name: n.agg_bandwidth if n.agg_bandwidth is not None else float("inf")
            for n in self.nodes
        }
        lat = {n.name: n.latency_s for n in self.nodes}
        adj: dict[str, list[tuple[str, Edge]]] = {n.name: [] for n in self.nodes}
        for e in self.edges:
            adj[e.src].append((e.dst, e))
            adj[e.dst].append((e.src, e))
        if src not in caps or dst not in caps:
            missing = src if src not in caps else dst
            raise KeyError(f"{self.name}: no node named '{missing}'")

        # modified Dijkstra: widest path, latency as tie-break bookkeeping
        best: dict[str, float] = {src: caps[src]}
        heap: list[tuple[float, float, str, tuple[str, ...]]] = [
            (-caps[src], lat[src], src, (src,))
        ]
        done: set[str] = set()
        while heap:
            neg_w, latency, u, path = heapq.heappop(heap)
            w = -neg_w
            if u in done:
                continue
            done.add(u)
            if u == dst:
                return PathResult(bandwidth=w, latency_s=latency, nodes=path)
            for v, e in adj[u]:
                if v in done:
                    continue
                ecap = e.agg_bandwidth if e.agg_bandwidth is not None else float("inf")
                nw = min(w, ecap, caps[v])
                if nw > best.get(v, 0.0):
                    best[v] = nw
                    heapq.heappush(
                        heap, (-nw, latency + e.latency_s + lat[v], v, path + (v,))
                    )
        raise ValueError(f"{self.name}: no path from '{src}' to '{dst}'")

    # ---- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = self._to_dict()
        d["format"] = FORMAT
        return d

    def _to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "meta": self.meta,
            "nodes": [_node_to_dict(n) for n in self.nodes],
            "edges": [_edge_to_dict(e) for e in self.edges],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Graph":
        fmt = d.get("format", FORMAT)
        if fmt != FORMAT:
            raise ValueError(f"unsupported graph format '{fmt}' (expected {FORMAT})")
        return cls._from_dict(d)

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> "Graph":
        return cls(
            name=d["name"],
            meta=dict(d.get("meta", {})),
            nodes=[_node_from_dict(x) for x in d.get("nodes", [])],
            edges=[_edge_from_dict(x) for x in d.get("edges", [])],
        )

    @classmethod
    def from_json(cls, text: str) -> "Graph":
        return cls.from_dict(json.loads(text))

    # ---- display ------------------------------------------------------------

    def describe(self) -> str:
        lines: list[str] = []
        self._describe(lines, depth=0)
        return "\n".join(lines)

    def _describe(self, lines: list[str], depth: int) -> None:
        pad = "  " * depth
        lines.append(f"{pad}{self.name}")
        for n in self.nodes:
            mult = f" x{n.count}" if n.count > 1 else ""
            attrs = _node_attr_summary(n)
            role = f" <{n.role}>" if n.role else ""
            lines.append(f"{pad}  [{n.kind.value}]{role} {n.name}{mult}  {attrs}")
            if n.inner is not None:
                n.inner._describe(lines, depth + 2)
        for e in self.edges:
            mult = f" x{e.count}" if e.count > 1 else ""
            bw = "unconstrained" if e.bandwidth is None else fmt_si(e.bandwidth, "B/s")
            lat = f", {fmt_time(e.latency_s)}" if e.latency_s else ""
            lines.append(f"{pad}  {e.src} <--> {e.dst}{mult}  ({bw}{lat})")


def _node_attr_summary(n: Node) -> str:
    parts: list[str] = []
    if n.peak_flops:
        top = min(n.peak_flops, key=lambda d: d.bytes)
        parts.append(f"{fmt_si(n.peak_flops[top], 'FLOP/s')} @{top.value}")
    if n.capacity_bytes is not None:
        parts.append(fmt_bytes(n.capacity_bytes))
    if n.bandwidth is not None:
        parts.append(fmt_si(n.bandwidth, "B/s"))
    if n.latency_s:
        parts.append(fmt_time(n.latency_s))
    return ", ".join(parts)


def _node_to_dict(n: Node) -> dict[str, Any]:
    d: dict[str, Any] = {"name": n.name, "kind": n.kind.value}
    if n.count != 1:
        d["count"] = n.count
    if n.peak_flops:
        d["peak_flops"] = {k.value: v for k, v in n.peak_flops.items()}
    for attr in ("capacity_bytes", "bandwidth"):
        if getattr(n, attr) is not None:
            d[attr] = getattr(n, attr)
    for attr in ("latency_s", "dynamic_power_w", "idle_power_w"):
        if getattr(n, attr):
            d[attr] = getattr(n, attr)
    if n.role:
        d["role"] = n.role
    if n.inner is not None:
        d["inner"] = n.inner._to_dict()
    if n.ports:
        d["ports"] = list(n.ports)
    if n.meta:
        d["meta"] = n.meta
    return d


def _node_from_dict(d: dict[str, Any]) -> Node:
    flops = d.get("peak_flops")
    return Node(
        name=d["name"],
        kind=NodeKind(d["kind"]),
        count=d.get("count", 1),
        peak_flops={DType(k): v for k, v in flops.items()} if flops else None,
        capacity_bytes=d.get("capacity_bytes"),
        bandwidth=d.get("bandwidth"),
        latency_s=d.get("latency_s", 0.0),
        dynamic_power_w=d.get("dynamic_power_w", 0.0),
        idle_power_w=d.get("idle_power_w", 0.0),
        role=d.get("role", ""),
        inner=Graph._from_dict(d["inner"]) if d.get("inner") else None,
        ports=tuple(d.get("ports", ())),
        meta=dict(d.get("meta", {})),
    )


def _edge_to_dict(e: Edge) -> dict[str, Any]:
    d: dict[str, Any] = {"src": e.src, "dst": e.dst}
    if e.bandwidth is not None:
        d["bandwidth"] = e.bandwidth
    for attr in ("latency_s", "power_w", "name"):
        if getattr(e, attr):
            d[attr] = getattr(e, attr)
    if e.count != 1:
        d["count"] = e.count
    return d


def _edge_from_dict(d: dict[str, Any]) -> Edge:
    return Edge(
        src=d["src"],
        dst=d["dst"],
        bandwidth=d.get("bandwidth"),
        latency_s=d.get("latency_s", 0.0),
        power_w=d.get("power_w", 0.0),
        count=d.get("count", 1),
        name=d.get("name", ""),
    )
