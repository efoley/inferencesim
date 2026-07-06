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
}
```

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
from .des import DESEngine
from .engine import CommContext
from .graph import Edge, Graph, Node, NodeKind
from .graphdes import _edge_res
from .hardware import System
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
_LINK_RE = re.compile(r"^s(\d+)\.l(\d+)\.(cw|ccw|out)$")
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
    levels = [_stage_level(replay_engine, system, deployment, comm, warmup)]
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
    `tp` member outbound links as a count group, with hop edges wiring
    consecutive stages (and the wrap link) into the pipeline ring."""
    pp, tp = dep.pp, dep.tp
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
            name=f"s{s}.fabric", kind=NodeKind.SWITCH, role="link", count=tp,
            bandwidth=fabric_bw, latency_s=fabric_lat,
            meta={"stage": s, "topology": comm.tp_topology.value,
                  "desc": f"stage {s} member outbound links"},
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
    if _LINK_RE.match(resource):  # member outbound link occupancy
        return _KIND["hop"] if _HOP_LABEL_RE.search(label) else _KIND["collective"]
    return _KIND["compute"]


def _stage_level(engine: DESEngine, system: System, dep: Deployment,
                 comm: CommContext, warmup: int) -> dict:
    graph = _stage_graph(system, dep, comm)
    fabric_bw = comm.tp_link.bandwidth if comm.tp_link else 0.0
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
        phases[phase] = _encode_track(rows)
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
            tracks[op_name] = _encode_track(rows)
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


def _encode_track(rows: list[list]) -> dict:
    """Intern resource names and labels and emit array rows.  If the row count
    exceeds `_TRACK_CAP`, the samplable kinds (read / writeback / collective /
    sync) are strided down while compute and hop rows are kept in full."""
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
