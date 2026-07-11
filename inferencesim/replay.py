"""Replay document: the versioned data contract the viewer consumes.

`build_replay` turns a discrete-event run into a self-describing JSON document
(`"format": "inferencesim-replay-v1"`) that a front-end can single-step to
*see* tensors move through the machine.  It is deliberately independent of the
viewer: the same document would drive a native (ImGui / emscripten) front-end,
so the shape is the contract, not an implementation detail.

Document shape
--------------
```
{
  "format": "inferencesim-replay-v1",
  "meta":  {model/hardware/deployment/scenario summary strings + numbers,
            "task_kinds": [...legend...], "decode_window": {...cap doc...}},
  "levels": [ <level>, ... ]
}
```

Every **level** carries a standard hardware `Graph` (`graph.to_dict()`) so the
viewer draws every level identically, a `resource_map` resolving each DES
resource name to a graph element, and per-phase task tracks:

```
{
  "id": str, "title": str, "kind": "stage" | "chip",
  "graph": <graph-as-dict>,
  "resource_map": { resource_name: {"kind": "node"|"edge",
                                    "id": <node-name|edge-index>,
                                    "instance"?: int} },
  # stage level: one track per phase.
  "phases": { "prefill": <track>, "decode": <track> },
  # chip level (graph mode only): one track per (phase, op); `ops` orders them.
  "ops": [op_name, ...],
  "phases": { "prefill": {op_name: <track>, ...}, "decode": {...} },
}
```

A **track** is the compact, columnar encoding of one scheduled task list
(interned resource names + labels, one array row per task):

```
{
  "makespan": float,                         # seconds, window-relative
  "res":    [unique resource names],
  "labels": [unique task labels],
  "rows":   [ [res_i, start, end, kind_i, label_i, bytes], ... ],
  "n": int,          # task count BEFORE any downsample cap
  "capped": bool,    # true if `rows` is a sampled subset of `n`
  "busy": { resource_name: seconds },  # authoritative per-resource busy
}
```

`busy` is the per-resource occupancy (seconds, window-relative, sync fences
excluded) summed over the *full* task set **before** any capping, so heat/busy
meters stay honest even when `rows` is a sampled subset: a `capped` track's
stride can alias whole resource classes out of `rows` (the skewed-MoE incast),
but never out of `busy`.  Divide by `makespan` for a utilisation fraction.

`kind_i` indexes `TASK_KINDS`.  Read/writeback rows carry per-tile `bytes`
(op DRAM bytes / n_tiles) so a particle can scale with payload; sync rows
(`.bar` / `.prop` timing fences) are flagged with kind ``"sync"`` and are
rendered as fences, never link traffic.

The two expand axes the viewer offers:
  * composite nesting -- the stage level nests `u{s}` + member links inside a
    per-stage composite box;
  * count groups -- the chip level ships the *expanded* chip graph (node ids
    are exactly the DES resource names, e.g. `gddr6-bank[3]`), which the viewer
    re-groups by base name (`tensix-fpu[0..139]` -> `tensix-fpu x140`).

Decode is capped to a small steady-state window (see `build_replay`) so a file
that would otherwise hold ~10^5 tasks stays loadable; the window is documented
in `meta["decode_window"]`.
"""

from __future__ import annotations

import re
from dataclasses import replace

from .bridge import chip_graph_of
from .collectives import is_sync_resource
from .des import DESEngine
from .engine import CommContext
from .graph import Edge, Graph, Node, NodeKind
from .graphdes import _edge_res
from .hardware import System, Topology
from .ops import Op, decode_step_ops, prefill_ops
from .sched import ScheduleResult, Task
from .workload import Deployment, ModelSpec, Scenario

FORMAT = "inferencesim-replay-v1"

# Task classification legend; a track row's `kind_i` indexes this list.  Read /
# writeback move DRAM tiles (chip level), collective / hop move activations
# between chips (stage level), compute glows a unit/core, sync is a timing
# fence (barrier / propagation) carrying latency, not traffic.
TASK_KINDS = ["read", "compute", "writeback", "collective", "hop", "sync"]
_KIND = {k: i for i, k in enumerate(TASK_KINDS)}

# per-track task cap: beyond this the samplable kinds are strided down so the
# document stays loadable; compute and hop rows (the legible ones) are kept.
_TRACK_CAP = 6000

_U_RE = re.compile(r"^u(\d+)$")
# member i's link occupancy at stage s: an egress port (`.out` switched / `.cw`,
# `.ccw` ring) or -- under a skewed all-to-all's incast -- its ingress port
# (`.in`).  All resolve to the same fabric member; ingress is just the inbound
# lane the store-and-forward hop lands on (collectives.py, PR: hot-expert).
_LINK_RE = re.compile(r"^s(\d+)\.l(\d+)\.(cw|ccw|out|in)$")
_STAGE_RE = re.compile(r"^s(\d+)\.")
_HOP_LABEL_RE = re.compile(r"h\d+$")


# =============================================================================
# public entry point
# =============================================================================


def build_replay(
    engine: DESEngine,
    system: System,
    model: ModelSpec,
    scenario: Scenario,
    deployment: Deployment,
    hardware_graph: Graph | None,
    *,
    decode_warmup: int | None = None,
    decode_window: int = 2,
) -> dict:
    """Build the replay document for one deployment.

    `engine` supplies the run *configuration* (efficiency, tile-fill, whether a
    chip graph is walked); the phases are re-run here at a bounded round count
    so the document is deterministic and small regardless of the measurement
    engine's convergence.  `hardware_graph` is the graph the DES walked
    (``None`` for lumped presets); a chip level is emitted iff it is a graph.

    Decode is measured over a fixed ``decode_warmup + decode_window`` rounds and
    only the last `decode_window` steady-state rounds are kept (the fill
    transient is discarded, times are re-zeroed to the window).  `decode_warmup`
    defaults to ``max(2, pp)`` -- enough to fill a `pp`-stage pipeline.
    """
    chip_graph = chip_graph_of(hardware_graph) if hardware_graph is not None else None
    warmup = decode_warmup if decode_warmup is not None else max(2, deployment.pp)
    window = max(1, decode_window)
    tile_fill = getattr(getattr(engine, "_chip_model", None), "tile_fill", 0.5)

    replay_engine = DESEngine(
        decode_rounds=warmup + window,
        warmup=warmup,
        chip_graph=chip_graph,
        tile_fill=tile_fill,
        efficiency=engine.efficiency,
    )
    phase_ops = {
        "prefill": prefill_ops(model, scenario.prompt_len, deployment),
        "decode": decode_step_ops(model, scenario, deployment),
    }
    for name, ops in phase_ops.items():
        replay_engine.run_phase(name, ops, system, deployment)

    comm = CommContext.for_deployment(system, deployment)
    fabric_bw = comm.tp_link.bandwidth if comm.tp_link else 0.0
    phases, resources = _phase_tracks(replay_engine, warmup, fabric_bw)
    levels = [_stage_level(system, deployment, comm, phases, resources)]
    # member level: lay each stage's tp*ep*adp members out as separate nodes so
    # collective structure (TP allreduce, MoE all-to-all incast, ADP/CP rings)
    # becomes spatially visible.  Adds nothing with a single member.
    if comm.a2a > 1:
        levels.append(_member_level(system, deployment, comm, phases, resources))
    if chip_graph is not None:
        levels.append(_chip_level(replay_engine, phase_ops))

    meta = _meta(system, model, scenario, deployment, chip_graph is not None,
                 warmup, window)
    return {"format": FORMAT, "meta": meta, "levels": levels}


# =============================================================================
# meta
# =============================================================================


def _meta(system, model, scenario, deployment, graph_mode, warmup, window) -> dict:
    d = deployment
    parts = [f"TP{d.tp}", f"PP{d.pp}"]
    if d.ep > 1:
        parts.append(f"EP{d.ep}")
    if d.adp > 1:
        parts.append(f"ADP{d.adp}")
    replica = d.replica_chips
    dp = system.total_chips // replica if replica else 0
    active = f", {model.active_params / 1e9:.1f}B active" if model.moe else ""
    return {
        "model": f"{model.name} ({model.total_params / 1e9:.1f}B params{active})",
        "model_name": model.name,
        "hardware": (f"{system.name} -- {system.total_chips}x "
                     f"{system.node.chip.name}"),
        "hardware_name": system.name,
        "deployment": (f"{'  '.join(parts)}  DP{dp}  "
                       f"({replica} chips/replica), "
                       f"weights {d.weight_dtype.value}, kv {d.kv_dtype.value}"),
        "scenario": (f"batch {scenario.batch}, prompt {scenario.prompt_len}, "
                     f"output {scenario.output_len}"),
        "tp": d.tp, "pp": d.pp, "ep": d.ep, "adp": d.adp,
        "batch": scenario.batch,
        "prompt_len": scenario.prompt_len,
        "output_len": scenario.output_len,
        "total_chips": system.total_chips,
        "dp": dp,
        "graph_mode": graph_mode,
        "phases": ["prefill", "decode"],
        "task_kinds": TASK_KINDS,
        "decode_window": {
            "warmup_rounds": warmup,
            "window_rounds": window,
            "note": (f"decode re-run for {warmup + window} rounds; the last "
                     f"{window} steady-state round(s) are kept (the {warmup} "
                     f"fill rounds are dropped and times re-zeroed)."),
        },
    }


# =============================================================================
# stage level (synthesised)
# =============================================================================


def _stage_graph(system: System, dep: Deployment, comm: CommContext) -> Graph:
    """Synthesise the pipeline-stage graph in the standard format: one
    composite box per stage, holding the stage execution unit `u{s}` and its
    member fabric links as a count group, with hop edges wiring consecutive
    stages (and the wrap link) into the pipeline ring.  The count group is the
    full FFN array (`comm.a2a = tp*ep*adp`): allreduce and the pipeline hop use
    member 0, the MoE dispatch/combine all-to-all and the dense adp gather /
    cp_ring the wider `l{i}.{out,in,cw,ccw}` members."""
    pp, tp = dep.pp, dep.tp
    n_members = comm.a2a  # widest member set egressing/ingressing on this fabric
    chip = system.node.chip
    tp_link = comm.tp_link
    fabric_bw = tp_link.bandwidth if tp_link else None
    fabric_lat = tp_link.latency_s if tp_link else 0.0
    p2p = comm.p2p_link
    hop_bw = p2p.bandwidth if p2p else fabric_bw

    stages: list[Node] = []
    for s in range(pp):
        unit = Node(
            name=f"u{s}", kind=NodeKind.COMPUTE, role="compute",
            peak_flops={dt: f * tp for dt, f in chip.compute.peak_flops.items()},
            dynamic_power_w=chip.compute.power_w * tp,
            meta={"stage": s, "tp": tp, "desc": f"stage {s} execution unit "
                  f"({tp} chip{'s' if tp > 1 else ''})"},
        )
        fabric = Node(
            name=f"s{s}.fabric", kind=NodeKind.SWITCH, role="link", count=n_members,
            bandwidth=fabric_bw, latency_s=fabric_lat,
            meta={"stage": s, "topology": comm.tp_topology.value,
                  "desc": f"stage {s} member fabric links (egress + ingress)"},
        )
        inner_edges = [Edge(src=f"u{s}", dst=f"s{s}.fabric",
                            bandwidth=fabric_bw, name="egress")]
        inner = Graph(name=f"stage{s}", nodes=[unit, fabric], edges=inner_edges)
        stages.append(Node(
            name=f"stage{s}", kind=NodeKind.COMPOSITE, role="stage",
            inner=inner, ports=(f"u{s}",), meta={"stage": s},
        ))

    edges: list[Edge] = []
    if pp > 1:
        for s in range(pp):
            edges.append(Edge(src=f"stage{s}", dst=f"stage{(s + 1) % pp}",
                              bandwidth=hop_bw, name="hop"))
    return Graph(name="pipeline", nodes=stages, edges=edges,
                 meta={"kind": "stage"})


def _stage_resource_map(resources: set[str]) -> dict:
    """Resolve every stage-DES resource name to a synthesised graph element.
    `u{s}` -> the stage unit node; `s{s}.l{i}.*` -> member i of the stage's
    fabric count group; sync resources (`.bar`/`.prop`) -> the stage fabric
    (rendered as a fence, not traffic)."""
    out: dict[str, dict] = {}
    for r in resources:
        m = _U_RE.match(r)
        if m:
            out[r] = {"kind": "node", "id": f"u{m.group(1)}"}
            continue
        m = _LINK_RE.match(r)
        if m:
            out[r] = {"kind": "node", "id": f"s{m.group(1)}.fabric",
                      "instance": int(m.group(2))}
            continue
        m = _STAGE_RE.match(r)  # sync: s{s}.bar.. / s{s}.prop..
        if m:
            out[r] = {"kind": "node", "id": f"s{m.group(1)}.fabric",
                      "instance": 0}
    return out


def _classify_stage(resource: str, label: str) -> int:
    if ".bar" in resource or ".prop" in resource:
        return _KIND["sync"]
    if _U_RE.match(resource):
        return _KIND["compute"]
    if _LINK_RE.match(resource):  # member link occupancy (egress or ingress)
        # only the pipeline hop rides an egress port with an `h{s}` label; every
        # other link task (allreduce / all-to-all egress+ingress / cp_ring) is
        # collective traffic.
        return _KIND["hop"] if _HOP_LABEL_RE.search(label) else _KIND["collective"]
    return _KIND["compute"]


def _phase_tracks(engine: DESEngine, warmup: int,
                  fabric_bw: float) -> tuple[dict[str, dict], set[str]]:
    """Extract and encode one stage-DES task track per phase, returning the
    encoded tracks and the set of every resource they mention.  The stage and
    member levels are two *views* of the same run (same tasks, different graph +
    resource_map), so the tracks are built once here and shared by both."""
    resources: set[str] = set()
    phases: dict[str, dict] = {}
    for phase in ("prefill", "decode"):
        tasks, result = engine.last_runs[phase]
        rows = _stage_rows(
            tasks, result,
            keep_from_round=warmup if phase == "decode" else None,
            fabric_bw=fabric_bw,
        )
        for res, *_ in rows:
            resources.add(res)
        phases[phase] = _encode_track(rows, _rows_busy(rows))
    return phases, resources


def _stage_level(system: System, dep: Deployment, comm: CommContext,
                 phases: dict[str, dict], resources: set[str]) -> dict:
    graph = _stage_graph(system, dep, comm)
    return {
        "id": "stage",
        "title": "Pipeline stages",
        "kind": "stage",
        "graph": graph.to_dict(),
        "resource_map": _stage_resource_map(resources),
        "phases": phases,
    }


def _stage_rows(tasks: list[Task], result: ScheduleResult,
                keep_from_round: int | None, fabric_bw: float) -> list[list]:
    """Extract (resource, start, end, kind, label, bytes) rows for a stage
    phase.  When `keep_from_round` is set (decode), only tasks in rounds >=
    that index survive and times are shifted so the window starts at 0."""
    kept: list[Task] = []
    if keep_from_round is None:
        kept = tasks
    else:
        for t in tasks:
            m = re.match(r"^r(\d+)", t.label)
            if m and int(m.group(1)) >= keep_from_round:
                kept.append(t)
    if not kept:
        return []
    t0 = min(result.start[t.key] for t in kept)
    rows: list[list] = []
    for t in kept:
        start = result.start[t.key] - t0
        end = result.finish[t.key] - t0
        kind = _classify_stage(t.resource, t.label)
        payload = 0.0
        if kind in (_KIND["collective"], _KIND["hop"]) and fabric_bw:
            payload = (end - start) * fabric_bw
        rows.append([t.resource, start, end, kind, t.label, payload])
    return rows


# =============================================================================
# member level (synthesised; emitted only when a stage has > 1 member)
# =============================================================================
#
# The stage level pools each stage's tp*ep*adp chips into one `u{s}` box, so a
# collective reads only as an aggregate port meter.  The member level lays each
# stage's members out as *separate* nodes -- so an all-to-all's incast piles
# visibly onto the hot member's ingress, a ring's steps hop member-to-member,
# etc.  It is the same DES run as the stage level (same task tracks), re-graphed.
#
# Member index -> (tp, ep, adp) coordinate.  The a2a group is the tp*ep*adp
# array with **tp innermost, then ep, then adp** (engine.CommContext: "tp
# innermost, then ep/adp"; the CP ring's members "span the tp*adp array with
# stride tp" == tp innermost; MoE expert blocks are placed contiguously over the
# flat array indexed by this member, member 0 the hottest -- workload.py).  So:
#     tp_coord  = i % tp
#     ep_coord  = (i // tp) % ep
#     adp_coord =  i // (tp * ep)
_DOT = "·"  # middle dot, matches the viewer's axis separators


def _member_decomp(i: int, dep: Deployment) -> tuple[int, int, int]:
    return (i % dep.tp, (i // dep.tp) % dep.ep, i // (dep.tp * dep.ep))


def _member_label(i: int, dep: Deployment) -> str:
    """Short, human-meaningful member label -- only axes with degree > 1 shown
    (pure TP -> `tp3`; MoE EP -> `ep5`; tp*ep -> `tp1·ep0`)."""
    tp_c, ep_c, adp_c = _member_decomp(i, dep)
    parts: list[str] = []
    if dep.tp > 1:
        parts.append(f"tp{tp_c}")
    if dep.ep > 1:
        parts.append(f"ep{ep_c}")
    if dep.adp > 1:
        parts.append(f"adp{adp_c}")
    return _DOT.join(parts) if parts else f"m{i}"


def _member_meta(i: int, s: int, dep: Deployment) -> dict:
    tp_c, ep_c, adp_c = _member_decomp(i, dep)
    hot = i == 0 and dep.ep > 1  # member 0 owns the lowest (hottest) expert block
    desc = (f"stage {s} member {i} ({_member_label(i, dep)}) -- one chip of the "
            f"stage's {dep.tp * dep.ep * dep.adp}-chip fabric; compute is pooled "
            f"in u{s}, this node carries its collective egress/ingress"
            + (". Hottest expert-owner: incast lands here." if hot else "."))
    return {"stage": s, "member": i, "tp": tp_c, "ep": ep_c, "adp": adp_c,
            "label": _member_label(i, dep), "hot": hot, "desc": desc}


def _member_graph(system: System, dep: Deployment, comm: CommContext) -> Graph:
    """Synthesise the member-level graph: one composite per stage holding the
    pooled compute unit `u{s}`, its `n = tp*ep*adp` member nodes laid out
    individually, and the stage's fabric structure -- a central switch hub (edges
    member<->hub) on a switched fabric, or a member-to-member ring (cw + ccw
    edges) on a ring.  Hop edges wire consecutive stage composites when pp > 1.

    The fabric edges are top-level (referencing the nested member/hub nodes) so
    the viewer's `drawEdges` renders them and collective particles ride them; the
    hop edges join the composites exactly as at the stage level."""
    pp, tp = dep.pp, dep.tp
    n_members = comm.a2a
    use_ring = comm.a2a_topology is Topology.RING
    chip = system.node.chip
    tp_link = comm.tp_link
    fabric_bw = tp_link.bandwidth if tp_link else None
    fabric_lat = tp_link.latency_s if tp_link else 0.0
    p2p = comm.p2p_link
    hop_bw = p2p.bandwidth if p2p else fabric_bw

    stages: list[Node] = []
    edges: list[Edge] = []
    for s in range(pp):
        unit = Node(
            name=f"u{s}", kind=NodeKind.COMPUTE, role="compute",
            peak_flops={dt: f * tp for dt, f in chip.compute.peak_flops.items()},
            dynamic_power_w=chip.compute.power_w * tp,
            meta={"stage": s, "tp": tp, "label": f"u{s} (pooled)",
                  "desc": f"stage {s} execution unit -- its {n_members} member "
                  f"chips' compute pooled as one block (the stage DES has no "
                  f"per-member compute split; kernels serialise on u{s})"},
        )
        members = [
            Node(name=f"s{s}.m{i}", kind=NodeKind.SWITCH, role="link",
                 bandwidth=fabric_bw, latency_s=fabric_lat,
                 meta=_member_meta(i, s, dep))
            for i in range(n_members)
        ]
        inner_nodes = [unit, *members]
        if not use_ring:
            hub = Node(
                name=f"s{s}.hub", kind=NodeKind.SWITCH, role="switch",
                bandwidth=fabric_bw, latency_s=fabric_lat,
                meta={"stage": s, "topology": comm.a2a_topology.value,
                      "label": "switch", "desc": f"stage {s} switched fabric "
                      f"({comm.a2a_topology.value}); members egress and ingress "
                      f"through it -- an all-to-all's messages funnel here"},
            )
            inner_nodes.append(hub)
            for i in range(n_members):
                edges.append(Edge(src=f"s{s}.m{i}", dst=f"s{s}.hub",
                                  bandwidth=fabric_bw, name="egress"))
                edges.append(Edge(src=f"s{s}.hub", dst=f"s{s}.m{i}",
                                  bandwidth=fabric_bw, name="ingress"))
        else:
            for i in range(n_members):
                nxt = (i + 1) % n_members
                edges.append(Edge(src=f"s{s}.m{i}", dst=f"s{s}.m{nxt}",
                                  bandwidth=fabric_bw, name="cw"))
                edges.append(Edge(src=f"s{s}.m{nxt}", dst=f"s{s}.m{i}",
                                  bandwidth=fabric_bw, name="ccw"))
        inner = Graph(name=f"stage{s}", nodes=inner_nodes, edges=[],
                      meta={"topology": comm.a2a_topology.value})
        stages.append(Node(
            name=f"stage{s}", kind=NodeKind.COMPOSITE, role="stage",
            inner=inner, ports=(f"u{s}",),
            meta={"stage": s, "members": n_members,
                  "topology": comm.a2a_topology.value},
        ))

    if pp > 1:
        for s in range(pp):
            edges.append(Edge(src=f"stage{s}", dst=f"stage{(s + 1) % pp}",
                              bandwidth=hop_bw, name="hop"))
    return Graph(name="members", nodes=stages, edges=edges,
                 meta={"kind": "member", "topology": comm.a2a_topology.value})


def _member_resource_map(resources: set[str], dep: Deployment) -> dict:
    """Resolve every stage-DES resource to a member-level graph element.
    `u{s}` -> the pooled compute node; `s{s}.l{i}.{out,in,cw,ccw}` -> member i's
    own node (egress and ingress of one member both resolve to that member, as at
    the stage level); sync (`.bar`/`.prop`) -> member 0, matching the stage
    level's fence policy of resolving to the first fabric member."""
    out: dict[str, dict] = {}
    for r in resources:
        m = _U_RE.match(r)
        if m:
            out[r] = {"kind": "node", "id": f"u{m.group(1)}"}
            continue
        m = _LINK_RE.match(r)
        if m:
            out[r] = {"kind": "node", "id": f"s{m.group(1)}.m{int(m.group(2))}"}
            continue
        m = _STAGE_RE.match(r)  # sync: s{s}.bar.. / s{s}.prop..
        if m:
            out[r] = {"kind": "node", "id": f"s{m.group(1)}.m0"}
    return out


def _member_level(system: System, dep: Deployment, comm: CommContext,
                  phases: dict[str, dict], resources: set[str]) -> dict:
    graph = _member_graph(system, dep, comm)
    return {
        "id": "member",
        "title": "Stage members",
        "kind": "member",
        "graph": graph.to_dict(),
        "resource_map": _member_resource_map(resources, dep),
        "phases": phases,
    }


# =============================================================================
# chip level (graph mode only)
# =============================================================================


def _chip_level(engine: DESEngine, phase_ops: dict[str, list[Op]]) -> dict:
    expanded = engine._chip_model.graph  # the expand()ed chip graph the tiles ran on
    edge_index = {_edge_res(e.src, e.dst): i for i, e in enumerate(expanded.edges)}
    node_names = {n.name for n in expanded.nodes}

    resources: set[str] = set()
    ops_seen: list[str] = []
    phases: dict[str, dict] = {}
    for phase in ("prefill", "decode"):
        op_by_name = {op.name: op for op in phase_ops[phase]}
        tracks: dict[str, dict] = {}
        for op_name, sched in engine.last_op_runs[phase].items():
            op = op_by_name.get(op_name)
            n_tiles = max(1, sched.n_tiles)
            read_per = (op.dram_read / n_tiles) if op else 0.0
            write_per = (op.dram_write / n_tiles) if op else 0.0
            rows = _chip_rows(sched.tasks, sched.result, read_per, write_per)
            for res, *_ in rows:
                resources.add(res)
            # chip resources are processor-shared, so busy is union-of-intervals,
            # not a row-duration sum -- carry the scheduler's authoritative figure.
            busy = {r: b for r, b in sched.result.busy.items()
                    if b > 0.0 and not is_sync_resource(r)}
            tracks[op_name] = _encode_track(rows, busy)
            if op_name not in ops_seen:
                ops_seen.append(op_name)
        phases[phase] = tracks

    return {
        "id": "chip",
        "title": f"Chip -- {expanded.name}",
        "kind": "chip",
        "graph": expanded.to_dict(),
        "resource_map": _chip_resource_map(resources, node_names, edge_index),
        "ops": ops_seen,
        "phases": phases,
    }


def _chip_resource_map(resources: set[str], node_names: set[str],
                       edge_index: dict[str, int]) -> dict:
    out: dict[str, dict] = {}
    for r in resources:
        if r in edge_index:  # `lo~hi` link resource
            out[r] = {"kind": "edge", "id": edge_index[r]}
        elif r in node_names:  # instance node (bank / core / sram / noc)
            out[r] = {"kind": "node", "id": r}
    return out


def _chip_rows(tasks: list[Task], result: ScheduleResult,
               read_per: float, write_per: float) -> list[list]:
    rows: list[list] = []
    for t in tasks:
        kind = _classify_chip(t.label)
        if kind == _KIND["read"]:
            payload = read_per
        elif kind == _KIND["writeback"]:
            payload = write_per
        else:
            payload = 0.0
        rows.append([t.resource, result.start[t.key], result.finish[t.key],
                     kind, t.label, payload])
    return rows


def _classify_chip(label: str) -> int:
    """Chip tile labels are `<op> t<i> <verb> <resource>` with verb rd/cp/wb."""
    toks = label.split()
    verb = toks[2] if len(toks) > 2 else ""
    if verb == "rd":
        return _KIND["read"]
    if verb == "wb":
        return _KIND["writeback"]
    return _KIND["compute"]


# =============================================================================
# compact track encoding
# =============================================================================


def _rows_busy(rows: list[list]) -> dict[str, float]:
    """Authoritative per-resource busy (seconds) over the *full* row set, sync
    fences excluded (`is_sync_resource`).  Stage/member resources schedule FIFO
    on a single server, so a row's ``end - start`` is its task's whole duration
    and the per-resource sum equals `ScheduleResult.busy` exactly -- but summed
    here rather than read from the result because the decode rows are a re-zeroed
    steady-state *window* of a longer schedule, so the whole-run `result.busy`
    would not match the windowed track makespan.  (Processor-shared chip
    resources are *not* summed this way -- their busy is union-of-intervals, so
    `_chip_level` carries `ScheduleResult.busy` directly.)"""
    busy: dict[str, float] = {}
    for resource, start, end, _kind, _label, _payload in rows:
        if is_sync_resource(resource):
            continue
        busy[resource] = busy.get(resource, 0.0) + (end - start)
    return busy


def _encode_track(rows: list[list], busy: dict[str, float]) -> dict:
    """Intern resource names and labels and emit array rows.  If the row count
    exceeds `_TRACK_CAP`, the samplable kinds (read / writeback / collective /
    sync) are strided down while compute and hop rows are kept in full.

    `busy` is the authoritative per-resource busy map (seconds, sync excluded),
    computed by the caller over the *full* task set before any capping; the
    viewer reads its heat/busy meters from it rather than re-summing the (possibly
    sampled) rows, whose stride can alias whole resource classes out of a capped
    track (the DEP8 skewed-MoE incast bug)."""
    n = len(rows)
    if n > _TRACK_CAP:
        rows, capped = _downsample(rows, _TRACK_CAP)
    else:
        capped = False

    res_ids: dict[str, int] = {}
    label_ids: dict[str, int] = {}
    out_rows: list[list] = []
    makespan = 0.0
    for resource, start, end, kind, label, payload in rows:
        ri = res_ids.setdefault(resource, len(res_ids))
        li = label_ids.setdefault(label, len(label_ids))
        # round to trim JSON: picosecond time resolution, whole bytes.
        out_rows.append([ri, round(start, 12), round(end, 12), kind, li,
                         int(round(payload))])
        makespan = max(makespan, end)
    return {
        "makespan": makespan,
        "res": list(res_ids),
        "labels": list(label_ids),
        "rows": out_rows,
        "n": n,
        "capped": capped,
        "busy": {r: b for r, b in busy.items() if b > 0.0},
    }


_KEEP_ALWAYS = {_KIND["compute"], _KIND["hop"]}


def _downsample(rows: list[list], cap: int) -> tuple[list[list], bool]:
    keep = [r for r in rows if r[3] in _KEEP_ALWAYS]
    samplable = [r for r in rows if r[3] not in _KEEP_ALWAYS]
    budget = max(0, cap - len(keep))
    if budget and samplable:
        stride = max(1, len(samplable) // budget + (1 if len(samplable) % budget else 0))
        keep.extend(samplable[::stride])
    keep.sort(key=lambda r: r[1])  # by start time
    return keep, True
