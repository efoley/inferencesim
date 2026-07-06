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
import re
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
    # Per-instance heterogeneity: `derate` scales this instance's *rate-like*
    # figures -- effective peak_flops = peak_flops * derate, effective
    # bandwidth = bandwidth * derate.  Capacity is NOT scaled; a *disabled*
    # instance (derate == 0) exists physically but does no work and is
    # excluded from every aggregate (including capacity).  `instance_derates`
    # overrides the node-level `derate` for individual instances of a counted
    # group (index -> derate); expand() bakes it into each instance's plain
    # `derate` (a harvested 132-of-140 die, a throttled core, a dead bank).
    derate: float = 1.0
    instance_derates: dict[int, float] = field(default_factory=dict)
    dynamic_power_w: float = 0.0  # at full utilisation
    idle_power_w: float = 0.0
    # semantics for extraction / UI grouping: "chip", "node", ...
    role: str = ""
    # nesting
    inner: "Graph | None" = None
    ports: tuple[str, ...] = ()  # inner node names outer edges attach to
    meta: dict[str, Any] = field(default_factory=dict)

    def _derates(self) -> list[float]:
        """This node's per-instance derates (length == count)."""
        return [self.instance_derates.get(i, self.derate) for i in range(self.count)]

    @property
    def effective_count(self) -> float:
        """Count-equivalent for *rate-like* aggregates (peak_flops,
        bandwidth): the sum of per-instance derates.  A disabled instance
        (derate 0) contributes 0; a half-derated one contributes 0.5.  Equals
        `count` when nothing is derated."""
        return sum(self._derates())

    @property
    def enabled_count(self) -> int:
        """Count-equivalent for *capacity* (which derate does not scale): the
        number of instances that do any work at all (derate > 0).  A disabled
        instance is excluded entirely; a derated-but-live one keeps full
        capacity."""
        return sum(1 for d in self._derates() if d > 0.0)

    @property
    def agg_bandwidth(self) -> float | None:
        return None if self.bandwidth is None else self.bandwidth * self.effective_count

    @property
    def agg_capacity(self) -> float | None:
        return (
            None if self.capacity_bytes is None
            else self.capacity_bytes * self.enabled_count
        )

    @property
    def agg_flops(self) -> dict[DType, float]:
        if not self.peak_flops:
            return {}
        return {d: f * self.effective_count for d, f in self.peak_flops.items()}


class EdgePattern(str, Enum):
    """How a grouped edge wires the instances of its endpoints.

    INTERLEAVE (default): src[i % n_src] -- dst[i % n_dst] for
        i in range(max(n_src, n_dst)); covers one-to-one (equal counts) and
        star / per-instance ports (one side has count 1).
    ALL: every src instance connects to every dst instance.
    """

    INTERLEAVE = "interleave"
    ALL = "all"


_SELECTOR_RE = re.compile(r"^(?P<base>[^\[\]]+?)(?:\[(?P<sel>\*|\d+|\d+:\d+)\])?$")


def split_endpoint(endpoint: str) -> tuple[str, str | None]:
    """'sram[0:8]' -> ('sram', '0:8'); 'sram' / 'sram[*]' -> ('sram', None)."""
    m = _SELECTOR_RE.match(endpoint)
    if not m:
        raise ValueError(f"malformed endpoint '{endpoint}'")
    sel = m.group("sel")
    return m.group("base"), (None if sel in (None, "*") else sel)


def _selector_indices(sel: str | None, count: int) -> list[int]:
    if sel is None:
        return list(range(count))
    if ":" in sel:
        lo, hi = (int(x) for x in sel.split(":"))
        if not (0 <= lo < hi <= count):
            raise ValueError(f"selector [{sel}] out of range for count {count}")
        return list(range(lo, hi))
    i = int(sel)
    if not 0 <= i < count:
        raise ValueError(f"selector [{sel}] out of range for count {count}")
    return [i]


@dataclass
class Edge:
    """An interconnect between two sibling groups (bidirectional).

    bandwidth is bytes/s per direction per concrete link; `count` is
    parallel links per connected pair.  Endpoints may select instances of a
    counted group: 'sram', 'sram[*]', 'sram[3]', 'sram[0:8]'.  `pattern`
    says how selected src instances wire to selected dst instances; the
    total capacity of the grouped edge is
    bandwidth * count * n_links(pattern)."""

    src: str
    dst: str
    bandwidth: float | None = None
    latency_s: float = 0.0
    power_w: float = 0.0
    count: int = 1  # parallel links per connected pair
    pattern: EdgePattern = EdgePattern.INTERLEAVE
    name: str = ""

    @property
    def agg_bandwidth(self) -> float | None:
        """Capacity per connected pair (bandwidth x parallel links)."""
        return None if self.bandwidth is None else self.bandwidth * self.count

    def n_links(self, n_src: int, n_dst: int) -> int:
        if self.pattern is EdgePattern.ALL:
            return n_src * n_dst
        return max(n_src, n_dst)


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
            if not 0.0 <= n.derate <= 1.0:
                raise ValueError(
                    f"{self.name}/{n.name}: derate {n.derate} not in [0, 1]"
                )
            for i, d in n.instance_derates.items():
                if not 0 <= i < n.count:
                    raise ValueError(
                        f"{self.name}/{n.name}: instance_derate index {i} out of "
                        f"range for count {n.count}"
                    )
                if not 0.0 <= d <= 1.0:
                    raise ValueError(
                        f"{self.name}/{n.name}: instance_derate {d} not in [0, 1]"
                    )
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
                if self.has_node(end):  # literal name (may contain [i])
                    continue
                base, sel = split_endpoint(end)
                if not self.has_node(base):
                    raise ValueError(f"{self.name}: edge endpoint '{end}' does not exist")
                n = self.node(base)
                if sel is not None and n.kind is NodeKind.COMPOSITE:
                    raise ValueError(
                        f"{self.name}: instance selector on composite '{end}' "
                        f"is not supported"
                    )
                _selector_indices(sel, n.count)  # range check
            if self._endpoint_base(e.src) == self._endpoint_base(e.dst):
                # An edge whose endpoints resolve to the same node is normally a
                # mistake (a loop on one node).  The exception is an *intra-group*
                # wiring: a counted group joined to itself through two DISTINCT
                # selectors -- e.g. a 2D-mesh NoC wiring router[0:R*(C-1)] ~
                # router[C:R*C] for the column links, or router[j:j+C-1] ~
                # router[j+1:j+C] per row.  INTERLEAVE pairs offset selectors
                # instance-wise, so no instance is ever wired to itself; the
                # concrete edges expand() emits are between distinct instances
                # (and re-validated then).  Reject only a true self-loop: a bare
                # group (no selector) on either side, or byte-identical endpoints.
                _, src_sel = split_endpoint(e.src)
                _, dst_sel = split_endpoint(e.dst)
                if src_sel is None or dst_sel is None or e.src == e.dst:
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
        names = {x.name for x in nodes}

        def resolves(end: str) -> bool:
            # a literal node name, or a selector into a counted group that
            # survived flattening (e.g. 'gddr6-bank[0]' or 'router[17:204]' from
            # a per-router mesh, whose base group 'gddr6-bank'/'router' is a node)
            return end in names or split_endpoint(end)[0] in names

        for e in self.edges:
            src = port_map.get(e.src, e.src)
            dst = port_map.get(e.dst, e.dst)
            for end, orig in ((src, e.src), (dst, e.dst)):
                if not resolves(end):
                    raise ValueError(
                        f"{self.name}: edge touches composite '{orig}' with no ports"
                    )
            edges.append(replace(e, src=src, dst=dst))
        return Graph(self.name, nodes, edges, dict(self.meta)), port_map

    # ---- expansion ----------------------------------------------------------

    def _endpoint_base(self, endpoint: str) -> str:
        """The node an endpoint refers to: a literal name wins over selector
        interpretation (expanded instance names contain brackets)."""
        if self.has_node(endpoint):
            return endpoint
        return split_endpoint(endpoint)[0]

    def _endpoint_width(self, endpoint: str) -> int:
        if self.has_node(endpoint):
            return self.node(endpoint).count
        base, sel = split_endpoint(endpoint)
        return len(_selector_indices(sel, self.node(base).count))

    def edge_capacity(self, e: Edge) -> float | None:
        """Total bandwidth of a grouped edge: per-link bandwidth x parallel
        links per pair x number of wired pairs (per its pattern)."""
        if e.bandwidth is None:
            return None
        n = e.n_links(self._endpoint_width(e.src), self._endpoint_width(e.dst))
        return e.bandwidth * e.count * n

    def expand(self, deep: bool = True) -> "Graph":
        """Materialise counted groups into individual instances.

        'sram' x35 becomes nodes sram[0]..sram[34]; grouped edges become the
        concrete links their pattern implies.  Aggregate queries (max_flow)
        give the same answers before and after expansion; the expanded form
        is what a discrete-event engine walks and what a UI shows when a
        group is unfolded.  With deep=True, inner graphs expand too
        (composite counts themselves also expand: chip x4 -> chip[0..3])."""
        expanded, _ = self._expand(deep)
        return expanded

    def _expand(self, deep: bool) -> tuple["Graph", dict[str, list[str]]]:
        instances: dict[str, list[str]] = {}
        nodes: list[Node] = []
        for n in self.nodes:
            inner = n.inner
            ports = n.ports
            if deep and inner is not None:
                inner, inner_instances = inner._expand(deep)
                ports = tuple(
                    inner_instances.get(p, [p])[0] for p in n.ports
                )
            base = replace(n, inner=inner, ports=ports)
            derates = n._derates()
            if n.count == 1:
                instances[n.name] = [n.name]
                nodes.append(replace(base, derate=derates[0], instance_derates={}))
            else:
                names = [f"{n.name}[{i}]" for i in range(n.count)]
                instances[n.name] = names
                nodes.extend(
                    replace(base, name=nm, count=1, derate=derates[i],
                            instance_derates={})
                    for i, nm in enumerate(names)
                )
        edges: list[Edge] = []

        def concrete(endpoint: str) -> list[str]:
            if self.has_node(endpoint):  # literal name wins
                return instances[endpoint]
            base, sel = split_endpoint(endpoint)
            return [instances[base][i]
                    for i in _selector_indices(sel, self.node(base).count)]

        for e in self.edges:
            srcs = concrete(e.src)
            dsts = concrete(e.dst)
            if e.pattern is EdgePattern.ALL:
                pairs = [(s, d) for s in srcs for d in dsts]
            else:
                pairs = [
                    (srcs[i % len(srcs)], dsts[i % len(dsts)])
                    for i in range(max(len(srcs), len(dsts)))
                ]
            edges.extend(
                replace(e, src=s, dst=d, pattern=EdgePattern.INTERLEAVE)
                for s, d in pairs
            )
        return Graph(self.name, nodes, edges, dict(self.meta)), instances

    # ---- queries ------------------------------------------------------------

    def _group(self, spec: str | list[str]) -> list[str]:
        """Resolve 'sram', 'sram[0:8]', or an explicit list to node names
        present in this graph (grouped or expanded form)."""
        if isinstance(spec, list):
            return spec
        if self.has_node(spec):  # literal name (possibly an instance)
            return [spec]
        base, sel = split_endpoint(spec)
        if self.has_node(base):
            if sel is not None:
                raise ValueError(
                    f"{self.name}: '{spec}' selects instances of an unexpanded "
                    f"group; call expand() first"
                )
            return [base]
        members = [n.name for n in self.nodes
                   if split_endpoint(n.name)[0] == base]
        if not members:
            raise KeyError(f"{self.name}: no node or group named '{base}'")
        if sel is None:
            return members
        return [f"{base}[{i}]" for i in _selector_indices(sel, len(members))]

    def derate_instances(self, spec: str, derate: float) -> None:
        """Set the derate of selected instances of a node, in place.

        `spec` is a selector against a grouped (unexpanded) node --
        'tensix-fpu[132:140]' harvests the last 8 of 140 cores, 'gddr6-bank[3]'
        disables one bank, 'core' (a count-1 or literal instance node) sets the
        node-level derate.  On a counted group the selected indices get an
        `instance_derates` override; expand() later bakes it into each
        instance's plain `derate`.  Reuses the selector machinery, so an
        out-of-range selector raises."""
        if not 0.0 <= derate <= 1.0:
            raise ValueError(f"derate {derate} not in [0, 1]")
        if self.has_node(spec):  # literal name: a group with no selector, or
            n = self.node(spec)  # an already-expanded instance ('tensix-fpu[3]')
            if n.count == 1:
                n.derate = derate
            else:
                for i in range(n.count):
                    n.instance_derates[i] = derate
            return
        base, sel = split_endpoint(spec)
        n = self.node(base)  # KeyError if the group is absent
        if n.count == 1:
            _selector_indices(sel, 1)  # only [0]/[0:1] valid on a count-1 node
            n.derate = derate
            return
        for i in _selector_indices(sel, n.count):
            n.instance_derates[i] = derate

    def max_flow(self, src: str | list[str], dst: str | list[str]) -> float:
        """Aggregate bandwidth between two groups: max flow with node
        capacities (count-aggregated) and edge capacities.  Unlike
        widest_path this credits parallel routes, and it gives the same
        answer on grouped and expanded forms of a uniform graph."""
        srcs, dsts = self._group(src), self._group(dst)
        INF = float("inf")
        cap: dict[tuple[str, str], float] = {}

        def add(u: str, v: str, c: float) -> None:
            cap[(u, v)] = cap.get((u, v), 0.0) + c

        for n in self.nodes:
            c = n.agg_bandwidth if n.agg_bandwidth is not None else INF
            add(f"i:{n.name}", f"o:{n.name}", c)
        for e in self.edges:
            c = self.edge_capacity(e)
            c = INF if c is None else c
            s, d = self._endpoint_base(e.src), self._endpoint_base(e.dst)
            add(f"o:{s}", f"i:{d}", c)
            add(f"o:{d}", f"i:{s}", c)
        SRC, DST = "src!", "dst!"
        for s in srcs:
            add(SRC, f"i:{s}", INF)
        for d in dsts:
            add(f"o:{d}", DST, INF)

        adj: dict[str, list[str]] = {}
        for (u, v) in list(cap.keys()):
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
            cap.setdefault((v, u), 0.0)

        flow = 0.0
        while True:
            # BFS for an augmenting path
            parent: dict[str, str] = {SRC: SRC}
            queue = [SRC]
            while queue and DST not in parent:
                u = queue.pop(0)
                for v in adj.get(u, []):
                    if v not in parent and cap[(u, v)] > 1e-9:
                        parent[v] = u
                        queue.append(v)
            if DST not in parent:
                return flow
            bottleneck, v = INF, DST
            while v != SRC:
                u = parent[v]
                bottleneck = min(bottleneck, cap[(u, v)])
                v = u
            if bottleneck == INF:
                return INF  # unconstrained route exists
            v = DST
            while v != SRC:
                u = parent[v]
                cap[(u, v)] -= bottleneck
                cap[(v, u)] += bottleneck
                v = u
            flow += bottleneck

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
            s, d = self._endpoint_base(e.src), self._endpoint_base(e.dst)
            adj[s].append((d, e))
            adj[d].append((s, e))
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
                ec = self.edge_capacity(e)
                ecap = ec if ec is not None else float("inf")
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
            if n.count > 1 and attrs:
                agg = _node_attr_summary(n, aggregate=True)
                attrs = f"{attrs} each  (total {agg})"
            role = f" <{n.role}>" if n.role else ""
            der = _derate_summary(n)
            lines.append(f"{pad}  [{n.kind.value}]{role} {n.name}{mult}  {attrs}{der}")
            if n.inner is not None:
                n.inner._describe(lines, depth + 2)
        for e in self.edges:
            try:
                n_links = e.count * e.n_links(
                    self._endpoint_width(e.src), self._endpoint_width(e.dst)
                )
            except (KeyError, ValueError):
                n_links = e.count
            mult = f" x{n_links} links" if n_links > 1 else ""
            if e.bandwidth is None:
                bw = "unconstrained"
            else:
                bw = fmt_si(e.bandwidth, "B/s")
                if n_links > 1:
                    bw += f" each, {fmt_si(e.bandwidth * n_links, 'B/s')} total"
            lat = f", {fmt_time(e.latency_s)}" if e.latency_s else ""
            lines.append(f"{pad}  {e.src} <--> {e.dst}{mult}  ({bw}{lat})")


def _node_attr_summary(n: Node, aggregate: bool = False) -> str:
    parts: list[str] = []
    if n.peak_flops:
        top = min(n.peak_flops, key=lambda d: d.bytes)
        v = n.agg_flops[top] if aggregate else n.peak_flops[top]
        parts.append(f"{fmt_si(v, 'FLOP/s')} @{top.value}")
    if n.capacity_bytes is not None:
        v = n.agg_capacity if aggregate else n.capacity_bytes
        parts.append(fmt_bytes(v))
    if n.bandwidth is not None:
        v = n.agg_bandwidth if aggregate else n.bandwidth
        parts.append(fmt_si(v, "B/s"))
    if n.latency_s and not aggregate:
        parts.append(fmt_time(n.latency_s))
    return ", ".join(parts)


def _derate_summary(n: Node) -> str:
    """A short note on the node line when any instance is derated/disabled."""
    if n.instance_derates:
        disabled = sum(1 for d in n._derates() if d == 0.0)
        derated = sum(1 for d in n._derates() if 0.0 < d < 1.0)
        bits = []
        if disabled:
            bits.append(f"{disabled} disabled")
        if derated:
            bits.append(f"{derated} derated")
        return f"  [{n.enabled_count}/{n.count} live" + (
            f", {', '.join(bits)}]" if bits else "]")
    if n.derate != 1.0:
        return f"  [derate {n.derate:g}]"
    return ""


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
    if n.derate != 1.0:
        d["derate"] = n.derate
    if n.instance_derates:
        d["instance_derates"] = {str(k): v for k, v in n.instance_derates.items()}
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
        derate=d.get("derate", 1.0),
        instance_derates={int(k): v for k, v in d.get("instance_derates", {}).items()},
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
    if e.pattern is not EdgePattern.INTERLEAVE:
        d["pattern"] = e.pattern.value
    return d


def _edge_from_dict(d: dict[str, Any]) -> Edge:
    return Edge(
        src=d["src"],
        dst=d["dst"],
        bandwidth=d.get("bandwidth"),
        latency_s=d.get("latency_s", 0.0),
        power_w=d.get("power_w", 0.0),
        count=d.get("count", 1),
        pattern=EdgePattern(d.get("pattern", EdgePattern.INTERLEAVE.value)),
        name=d.get("name", ""),
    )
