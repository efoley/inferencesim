"""Discrete-event engine.

Where the roofline engine sums op times analytically (assuming perfectly
balanced pipeline stages and formulaic overlap), the DES engine builds a
real task graph -- one task per (round, microbatch, pipeline stage, layer
block) with explicit dependencies -- and simulates it against resources
with FIFO queues:

    u{s}           the stage-s execution unit (its tp chips' compute+DRAM,
                   kernels serialise on it)
    s{s}.l{i}.out  member i's outbound link at stage s, named for its fabric:
    s{s}.l{i}.cw   `.out` (one egress port) on a switched fabric, or the two
    s{s}.l{i}.ccw  distinct `.cw`/`.ccw` cables on a ring.  Collectives are
                   expanded to their per-step transfers over these links
                   (collectives.py), and the pipeline hop rides member 0's
                   link -- so collective/hop and collective/collective
                   contention emerges here.

Because tasks only run when their inputs are ready *and* their resource is
free, the interesting behaviour is emergent rather than assumed:

  * pipeline microbatches overlap across stages; the steady-state round
    period is measured, not derived from a balanced-stages formula;
  * unbalanced stages (n_layers % pp != 0) genuinely cost throughput here,
    where the analytic engine only warns;
  * pipeline hops and collectives on different resources overlap with
    other microbatches' compute; overlap_comm is emergent, so the flag is
    ignored;
  * prefill (a single request walking the stages) shows fill/drain serial
    behaviour automatically.

Per-task service times reuse the same speed-of-light math as the roofline
engine (max(flops/peak, bytes/bw), ring collectives), so any difference
between the two engines is pure scheduling/contention, never unit costs.

Current granularity: pipeline stages and their links.  Passing a
`chip_graph` refines just the per-op unit cost: instead of the roofline
`max(flops/peak, bytes/bw)`, a COMPUTE op is lowered to a tile task graph
over the expanded chip's DRAM banks, NoC and per-core SRAM/matrix engines
(graphdes.ChipModel) and its wall time measured.  Collectives are always
expanded per-step over the fabric topology (collectives.py); MESH_2D still
falls back to the closed form.  Because a link carries only bandwidth
occupancy and latency rides the dependency chain, an isolated collective
reproduces its closed form exactly, so the expansion changes results only
where a collective genuinely contends for link bandwidth (with a hop or a
concurrent collective).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from . import collectives
from .engine import CommContext, Engine, Phase, RooflineEngine
from .graph import Graph
from .graphdes import ChipModel, OpSchedule
from .hardware import System, Topology
from .ops import Op, OpKind
from .sched import ScheduleResult, Task, schedule
from .workload import Deployment


def _split_layers(n_layers: int, pp: int) -> list[int]:
    base, extra = divmod(n_layers, pp)
    return [base + (1 if s < extra else 0) for s in range(pp)]


@dataclass
class _LayerCosts:
    """One-instance service times for the repeating per-layer structure,
    recovered from the lowered op list."""

    attn: float = 0.0  # qkv + attention + out_proj on the stage unit
    ffn: float = 0.0  # ffn / moe_routed / moe_shared on the stage unit
    allreduce: float = 0.0  # per collective (0, 1 or 2 per layer)
    n_allreduce: int = 0
    dispatch: float = 0.0  # MoE all-to-alls (ep > 1)
    combine: float = 0.0
    hop: float = 0.0  # pipeline p2p
    edge: float = 0.0  # embed + lm_head on the edge stages
    n_layers: int = 1


@dataclass
class _CommPlan:
    """Raw per-collective transfer parameters (payload, group size, link
    bandwidth/latency, fabric topology) recovered from the op list + comm
    context, so `collectives.py` can expand each collective into its per-step
    link transfers.  `_LayerCosts` keeps the closed-form *unit times*; this
    keeps the *ingredients* of those closed forms.  Links are None-safe: a
    group of size <= 1 is never expanded, so its 0.0 bw/lat is unused."""

    # allreduce: ring over the tp group
    ar_group: int = 1
    ar_bytes: float = 0.0
    ar_bw: float = 0.0
    ar_lat: float = 0.0
    ar_topo: Topology = Topology.ALL_TO_ALL
    # MoE dispatch/combine: all-to-all over the tp*ep array
    a2a_group: int = 1
    dispatch_bytes: float = 0.0
    combine_bytes: float = 0.0
    a2a_bw: float = 0.0
    a2a_lat: float = 0.0
    a2a_topo: Topology = Topology.ALL_TO_ALL
    # pipeline p2p hop: split into bandwidth occupancy (on the link) and flight
    # time (on the dependency chain), so hop latency does not occupy the link.
    hop_bytes: float = 0.0
    hop_bw: float = 0.0
    hop_lat: float = 0.0


_ATTN_OPS = {"qkv_proj", "attention", "out_proj"}
_FFN_OPS = {"ffn", "moe_routed", "moe_shared"}
_EDGE_OPS = {"embed", "lm_head"}


class DESEngine(Engine):
    """Event-driven engine at pipeline-stage granularity.

    decode_rounds/warmup control the steady-state measurement: the pipeline
    is simulated for `decode_rounds` full rounds and TPOT is the mean round
    period after discarding `warmup` rounds (fill transient).

    Both default to None, which selects *convergence control*: the round
    count is grown automatically (start small, then double, rebuilding the
    task graph each time) until two successive period estimates agree within
    `rtol`, or `max_rounds` is hit.  Passing an explicit `decode_rounds`
    pins a fixed run instead (byte-identical to the historical engine); with
    `warmup` left None it defaults to `decode_rounds // 2` (the historical
    16/8 ratio).  The auto-mode outcome (rounds used, whether it converged,
    the final relative delta) is recorded on `self.last_convergence` -- the
    engine has no channel to the report's warnings list, so a cap hit without
    convergence is surfaced there rather than printed.

    chip_graph (optional) switches per-op COMPUTE unit costs from the roofline
    formula to a tile schedule over the expanded chip graph; tile_fill is
    forwarded to the ChipModel (fraction of per-core SRAM a tile uses)."""

    def __init__(self, decode_rounds: int | None = None,
                 warmup: int | None = None, rtol: float = 1e-3,
                 max_rounds: int = 256, chip_graph: Graph | None = None,
                 tile_fill: float = 0.5):
        self._auto = decode_rounds is None
        if self._auto:
            if warmup is not None:
                raise ValueError("warmup requires an explicit decode_rounds")
            # decode_rounds/warmup are chosen per-iteration by the convergence
            # loop in _decode_wall; they are unset in auto mode.
            self.decode_rounds: int | None = None
            self.warmup: int | None = None
        else:
            if warmup is None:
                warmup = decode_rounds // 2  # historical ratio
            if decode_rounds <= warmup:
                raise ValueError("decode_rounds must exceed warmup")
            self.decode_rounds = decode_rounds
            self.warmup = warmup
        self.rtol = rtol
        self.max_rounds = max_rounds
        self._roofline = RooflineEngine()
        self._chip_model = (
            ChipModel(chip_graph, tile_fill) if chip_graph is not None else None
        )
        # (tasks, result) from the last run of each phase, for observability
        # (per-resource utilisation, Chrome-trace export).
        self.last_runs: dict[str, tuple[list[Task], ScheduleResult]] = {}
        # per distinct COMPUTE op, its chip-level tile schedule (graph mode
        # only), keyed by phase then op name -- for chip utilisation and trace.
        self.last_op_runs: dict[str, dict[str, OpSchedule]] = {}
        # convergence outcome of the last auto-mode decode measurement
        # ({"rounds", "converged", "rel_delta"}); None after a fixed run.
        self.last_convergence: dict | None = None

    # ---- costs from the lowered ops -----------------------------------------

    def op_time(self, op: Op, system: System, dep: Deployment) -> float:
        """Unit cost of one `Op` (all `count` instances) under this engine.

        Public hook for callers that cost ops outside a phase -- notably the
        request-level serving simulator (serve.py), which needs a single op's
        time at an arbitrary batch/context.  It makes the same per-op choice
        `run_phase` makes internally: in graph mode a COMPUTE op is refined to
        its expanded-chip tile schedule (op_wall), otherwise (and always for
        comm ops) it stays the roofline closed form.  This lets graph-refined
        chip costs flow into serving numbers without duplicating the lowering.
        """
        comm = CommContext.for_deployment(system, dep)
        chip = system.node.chip
        if self._chip_model is not None and op.kind is OpKind.COMPUTE:
            return op.count * self._chip_model.op_wall(replace(op, count=1)).wall
        return self._roofline.time_op(op, chip, comm).time

    def _costs(
        self, ops: list[Op], system: System, dep: Deployment, comm: CommContext
    ) -> tuple[_LayerCosts, dict[str, OpSchedule], dict[str, str]]:
        """Recover the repeating per-layer service times.  In graph mode a
        COMPUTE op's unit time is the measured wall of its chip-graph tile
        schedule (recorded in `op_runs`); comm ops stay closed-form.  `buckets`
        maps each COMPUTE op onto the stage-task family it feeds (attn / ffn /
        head), for the per-chip utilisation weighting."""
        chip = system.node.chip
        op_runs: dict[str, OpSchedule] = {}
        buckets: dict[str, str] = {}

        def one(op: Op) -> float:
            if self._chip_model is not None and op.kind is OpKind.COMPUTE:
                sched = self._chip_model.op_wall(replace(op, count=1))
                op_runs[op.name] = sched
                return sched.wall
            return self._roofline.time_op(replace(op, count=1), chip, comm).time

        c = _LayerCosts()
        for op in ops:
            if op.name in _ATTN_OPS:
                c.attn += one(op)
                buckets[op.name] = "attn"
                if op.name == "qkv_proj":
                    c.n_layers = op.count
            elif op.name in _FFN_OPS:
                c.ffn += one(op)
                buckets[op.name] = "ffn"
            elif op.name == "allreduce":
                c.allreduce = one(op)
                c.n_allreduce = op.count // max(c.n_layers, 1)
            elif op.name == "moe_dispatch":
                c.dispatch = one(op)
            elif op.name == "moe_combine":
                c.combine = one(op)
            elif op.name == "pp_hop":
                c.hop = one(op)
            elif op.name in _EDGE_OPS:
                c.edge += one(op)
                buckets[op.name] = "head"
            elif op.kind is OpKind.COMPUTE:
                c.ffn += one(op)  # anything unrecognised runs on the unit
                buckets[op.name] = "ffn"
        # allreduce count may not divide evenly if it was recovered from a
        # differently-shaped op list; recompute defensively
        for op in ops:
            if op.name == "allreduce" and c.n_layers:
                c.n_allreduce = max(1, round(op.count / c.n_layers))
        return c, op_runs, buckets

    def _comm_plan(
        self, ops: list[Op], dep: Deployment, comm: CommContext
    ) -> _CommPlan:
        """Recover the per-collective transfer ingredients from the op list."""
        payload = {op.name: op.comm_bytes for op in ops if op.is_comm}

        def bwlat(link) -> tuple[float, float]:
            return (link.bandwidth, link.latency_s) if link else (0.0, 0.0)

        ar_bw, ar_lat = bwlat(comm.tp_link)
        a2a_bw, a2a_lat = bwlat(comm.a2a_link)
        hop_bw, hop_lat = bwlat(comm.p2p_link)
        return _CommPlan(
            ar_group=comm.tp,
            ar_bytes=payload.get("allreduce", 0.0),
            ar_bw=ar_bw, ar_lat=ar_lat, ar_topo=comm.tp_topology,
            a2a_group=dep.tp * dep.ep,
            dispatch_bytes=payload.get("moe_dispatch", 0.0),
            combine_bytes=payload.get("moe_combine", 0.0),
            a2a_bw=a2a_bw, a2a_lat=a2a_lat, a2a_topo=comm.a2a_topology,
            hop_bytes=payload.get("pp_hop", 0.0),
            hop_bw=hop_bw, hop_lat=hop_lat,
        )

    def _emit_hop(self, tasks: list[Task], prev: int, s: int,
                  plan: _CommPlan, label: str) -> int:
        """A pipeline hop rides the sending chip's (member 0) outbound link,
        carrying only its bandwidth occupancy; the flight time follows on a
        per-hop propagation resource so it never occupies the link.  The link
        is named for the stage's fabric -- by convention the tp-group topology
        (`.out` on a switched fabric, `.cw` on a ring)."""
        link = collectives.egress(f"s{s}", 0, plan.ar_topo)
        occ = plan.hop_bytes / plan.hop_bw if plan.hop_bw else 0.0
        prev = collectives._add(tasks, link, occ, [prev], label)
        if plan.hop_lat > 0.0:
            prev = collectives._add(
                tasks, f"s{s}.prop_h{len(tasks)}", plan.hop_lat, [prev], label)
        return prev

    # ---- task-graph construction ---------------------------------------------

    def _stage_tasks(self, tasks: list[Task], c: _LayerCosts, plan: _CommPlan,
                     s: int, n_layers_here: int, prev: int | None, tag: str,
                     is_first: bool, is_last: bool) -> int:
        """Append the task chain for one (microbatch, round) passing through
        stage s; returns the key of its last task.

        Collectives are expanded to their per-step link transfers (see
        collectives.py) on the stage's per-member outbound link resources
        (`s{s}.l{i}.cw` / `.ccw`), which also carry the pipeline hops on
        member 0 -- so collective/hop contention emerges here rather than
        being averaged into a lumped `c{s}` fabric."""
        prefix = f"s{s}"

        def add(resource: str, duration: float, label: str) -> None:
            nonlocal prev_key
            key = len(tasks)
            deps = [prev_key] if prev_key is not None else []
            tasks.append(Task(key, resource, duration, deps, label))
            prev_key = key

        def allreduce() -> None:
            nonlocal prev_key
            prev_key = collectives.allreduce(
                tasks, prev_key, plan.ar_group, plan.ar_bytes, plan.ar_bw,
                plan.ar_lat, plan.ar_topo, prefix, f"{tag} ar")

        def all_to_all(payload: float) -> None:
            nonlocal prev_key
            prev_key = collectives.all_to_all(
                tasks, prev_key, plan.a2a_group, payload, plan.a2a_bw,
                plan.a2a_lat, plan.a2a_topo, prefix, f"{tag} a2a")

        prev_key = prev
        if is_first and c.edge:
            add(f"u{s}", 0.0, f"{tag} embed")  # embed cost folded
        for _ in range(n_layers_here):
            add(f"u{s}", c.attn, f"{tag} attn")
            if c.n_allreduce >= 2 and c.allreduce:
                allreduce()
            if c.dispatch:
                all_to_all(plan.dispatch_bytes)
            add(f"u{s}", c.ffn, f"{tag} ffn")
            if c.combine:
                all_to_all(plan.combine_bytes)
            if c.n_allreduce >= 1 and c.allreduce:
                allreduce()
        if is_last and c.edge:
            add(f"u{s}", c.edge, f"{tag} head")
        assert prev_key is not None
        return prev_key

    def _round_period(
        self, c: _LayerCosts, plan: _CommPlan, dep: Deployment,
        decode_rounds: int, warmup: int,
    ) -> tuple[float, list[Task], ScheduleResult]:
        """Build and schedule `decode_rounds` pipeline rounds and return the
        mean steady-state round period after discarding `warmup` fill rounds,
        with the tasks and schedule (for observability)."""
        pp = dep.pp
        layers = _split_layers(c.n_layers, pp)
        tasks: list[Task] = []
        token_done: list[list[int]] = [[] for _ in range(decode_rounds)]
        tail: dict[int, int] = {}  # microbatch -> last task key of prev round
        for r in range(decode_rounds):
            for m in range(pp):
                prev = tail.get(m)
                for s in range(pp):
                    last = self._stage_tasks(
                        tasks, c, plan, s, layers[s], prev, f"r{r}m{m}s{s}",
                        is_first=(s == 0), is_last=(s == pp - 1),
                    )
                    token_done[r].append(last) if s == pp - 1 else None
                    if c.hop and pp > 1:
                        # hop to the next stage on the sending chip's (member 0)
                        # outbound link, so its bandwidth contends with that
                        # chip's collective sends.  Wrap-around feeds the next
                        # round.
                        last = self._emit_hop(tasks, last, s, plan, f"r{r}m{m}h{s}")
                    prev = last
                tail[m] = prev
        result = schedule(tasks)
        round_end = [max(result.finish[k] for k in keys) for keys in token_done]
        wall = (round_end[-1] - round_end[warmup - 1]) / (decode_rounds - warmup)
        return wall, tasks, result

    def _decode_wall(
        self, c: _LayerCosts, plan: _CommPlan, dep: Deployment
    ) -> tuple[float, list[Task], ScheduleResult]:
        """Steady-state pipeline round period: every microbatch advances one
        token per round.

        Fixed mode (explicit `decode_rounds`) measures once over the requested
        rounds -- byte-identical to the historical engine.  Auto mode (the
        default) grows the round count until the estimate stabilises: start at
        max(8, 2*pp) rounds with half warmup, then double and re-measure
        (rebuilding the graph, cheap relative to getting the period wrong)
        until two successive estimates agree within `rtol` or `max_rounds` is
        reached.  The outcome is recorded on `self.last_convergence`."""
        if not self._auto:
            self.last_convergence = None
            return self._round_period(
                c, plan, dep, self.decode_rounds, self.warmup)

        rounds = min(max(8, 2 * dep.pp), self.max_rounds)
        prev_est: float | None = None
        rel = float("inf")
        while True:
            wall, tasks, result = self._round_period(
                c, plan, dep, rounds, rounds // 2)
            if prev_est is not None:
                rel = abs(wall - prev_est) / (abs(wall) or 1.0)
            converged = rel <= self.rtol
            if converged or rounds >= self.max_rounds:
                self.last_convergence = {
                    "rounds": rounds,
                    "converged": converged,
                    "rel_delta": rel,
                }
                return wall, tasks, result
            prev_est = wall
            rounds = min(rounds * 2, self.max_rounds)

    def _prefill_wall(
        self, c: _LayerCosts, plan: _CommPlan, dep: Deployment
    ) -> tuple[float, list[Task], ScheduleResult]:
        """A single request walks the stages sequentially (no other work in
        flight during TTFT measurement)."""
        pp = dep.pp
        layers = _split_layers(c.n_layers, pp)
        tasks: list[Task] = []
        prev: int | None = None
        for s in range(pp):
            prev = self._stage_tasks(tasks, c, plan, s, layers[s], prev, f"s{s}",
                                     is_first=(s == 0), is_last=(s == pp - 1))
            if c.hop and s < pp - 1:
                prev = self._emit_hop(tasks, prev, s, plan, f"h{s}")
        result = schedule(tasks)
        return result.makespan, tasks, result

    def _chip_resource_busy(
        self, tasks: list[Task], op_runs: dict[str, OpSchedule],
        buckets: dict[str, str], pp: int,
    ) -> dict[str, float]:
        """Per-chip resource busy over the phase, keyed `chip:<resource>`.

        Each COMPUTE op's tile schedule reports how long each chip resource
        (bank, NoC, SRAM, core) is busy for one execution; weight that by how
        many stage-level tasks the op feeds (one per bucket task it maps to)
        and divide by pp, since the pp pipeline stages are distinct chips and
        we want a per-chip figure.  simulate.py turns busy/span into a
        fraction, so `chip:gddr6-bank[3]` reads as that bank's utilisation
        over the phase."""
        counts = {"attn": 0, "ffn": 0, "head": 0}
        for t in tasks:
            tok = t.label.rsplit(" ", 1)[-1] if t.label else ""
            if tok in counts:
                counts[tok] += 1
        busy: dict[str, float] = {}
        for op_name, sched in op_runs.items():
            n = counts.get(buckets.get(op_name, ""), 0)
            if not n:
                continue
            for res, b in sched.result.busy.items():
                busy[f"chip:{res}"] = busy.get(f"chip:{res}", 0.0) + n * b / pp
        return busy

    # ---- Engine interface -----------------------------------------------------

    def run_phase(self, name: str, ops: list[Op], system: System,
                  dep: Deployment) -> Phase:
        comm = CommContext.for_deployment(system, dep)
        if dep.tp > 1 and comm.tp_link is None:
            raise ValueError("TP > 1 requires an interconnect for collectives")
        chip = system.node.chip
        timings = [self._roofline.time_op(op, chip, comm) for op in ops]
        costs, op_runs, buckets = self._costs(ops, system, dep, comm)
        plan = self._comm_plan(ops, dep, comm)
        if name == "decode":
            wall, tasks, result = self._decode_wall(costs, plan, dep)
        else:
            wall, tasks, result = self._prefill_wall(costs, plan, dep)
        self.last_runs[name] = (tasks, result)
        self.last_op_runs[name] = op_runs
        # drop the collective sync tasks (barriers / propagation) from the
        # reported utilisation: they carry dependency-chain latency, not
        # physical link occupancy.  They stay in `last_runs`/the trace.
        resource_busy = {
            r: b for r, b in result.busy.items()
            if b > 0.0 and not collectives.is_sync_resource(r)
        }
        if op_runs:  # graph mode: merge in per-chip resource utilisation
            resource_busy = {
                **resource_busy,
                **self._chip_resource_busy(tasks, op_runs, buckets, dep.pp),
            }
        return Phase(name, timings, wall_time=wall,
                     resource_busy=resource_busy, resource_span=result.makespan)
