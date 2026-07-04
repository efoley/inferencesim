"""Discrete-event engine.

Where the roofline engine sums op times analytically (assuming perfectly
balanced pipeline stages and formulaic overlap), the DES engine builds a
real task graph -- one task per (round, microbatch, pipeline stage, layer
block) with explicit dependencies -- and simulates it against resources
with FIFO queues:

    u{s}   the stage-s execution unit (its tp chips' compute+DRAM,
           kernels serialise on it)
    c{s}   the stage-s collective fabric (tp allreduces, MoE all-to-alls)
    h{s}   the point-to-point link out of stage s (pipeline hops)

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
(graphdes.ChipModel) and its wall time measured.  Comm ops stay closed-form
(collective internals are a later roadmap item).  Behaviour is byte-for-byte
unchanged without a chip_graph.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .engine import CommContext, Engine, Phase, RooflineEngine
from .graph import Graph
from .graphdes import ChipModel, OpSchedule
from .hardware import System
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


_ATTN_OPS = {"qkv_proj", "attention", "out_proj"}
_FFN_OPS = {"ffn", "moe_routed", "moe_shared"}
_EDGE_OPS = {"embed", "lm_head"}


class DESEngine(Engine):
    """Event-driven engine at pipeline-stage granularity.

    decode_rounds/warmup control the steady-state measurement: the pipeline
    is simulated for `decode_rounds` full rounds and TPOT is the mean round
    period after discarding `warmup` rounds (fill transient).

    chip_graph (optional) switches per-op COMPUTE unit costs from the roofline
    formula to a tile schedule over the expanded chip graph; tile_fill is
    forwarded to the ChipModel (fraction of per-core SRAM a tile uses)."""

    def __init__(self, decode_rounds: int = 16, warmup: int = 8,
                 chip_graph: Graph | None = None, tile_fill: float = 0.5):
        if decode_rounds <= warmup:
            raise ValueError("decode_rounds must exceed warmup")
        self.decode_rounds = decode_rounds
        self.warmup = warmup
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

    # ---- costs from the lowered ops -----------------------------------------

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

    # ---- task-graph construction ---------------------------------------------

    def _stage_tasks(self, tasks: list[Task], c: _LayerCosts, s: int,
                     n_layers_here: int, prev: int | None, tag: str,
                     is_first: bool, is_last: bool) -> int:
        """Append the serial task chain for one (microbatch, round) passing
        through stage s; returns the key of its last task."""

        def add(resource: str, duration: float, label: str) -> int:
            key = len(tasks)
            deps = [prev_key] if prev_key is not None else []
            tasks.append(Task(key, resource, duration, deps, label))
            return key

        prev_key = prev
        if is_first and c.edge:
            prev_key = add(f"u{s}", 0.0, f"{tag} embed")  # embed cost folded
        for _ in range(n_layers_here):
            prev_key = add(f"u{s}", c.attn, f"{tag} attn")
            if c.n_allreduce >= 2 and c.allreduce:
                prev_key = add(f"c{s}", c.allreduce, f"{tag} ar")
            if c.dispatch:
                prev_key = add(f"c{s}", c.dispatch, f"{tag} a2a")
            prev_key = add(f"u{s}", c.ffn, f"{tag} ffn")
            if c.combine:
                prev_key = add(f"c{s}", c.combine, f"{tag} a2a")
            if c.n_allreduce >= 1 and c.allreduce:
                prev_key = add(f"c{s}", c.allreduce, f"{tag} ar")
        if is_last and c.edge:
            prev_key = add(f"u{s}", c.edge, f"{tag} head")
        assert prev_key is not None
        return prev_key

    def _decode_wall(
        self, c: _LayerCosts, dep: Deployment
    ) -> tuple[float, list[Task], ScheduleResult]:
        """Steady-state pipeline round period: every microbatch advances one
        token per round."""
        pp = dep.pp
        layers = _split_layers(c.n_layers, pp)
        tasks: list[Task] = []
        token_done: list[list[int]] = [[] for _ in range(self.decode_rounds)]
        tail: dict[int, int] = {}  # microbatch -> last task key of prev round
        for r in range(self.decode_rounds):
            for m in range(pp):
                prev = tail.get(m)
                for s in range(pp):
                    last = self._stage_tasks(
                        tasks, c, s, layers[s], prev, f"r{r}m{m}s{s}",
                        is_first=(s == 0), is_last=(s == pp - 1),
                    )
                    token_done[r].append(last) if s == pp - 1 else None
                    if c.hop and pp > 1:
                        # hop to the next stage (wrap-around feeds the next
                        # round's first stage)
                        key = len(tasks)
                        tasks.append(Task(key, f"h{s}", c.hop, [last],
                                          f"r{r}m{m}h{s}"))
                        last = key
                    prev = last
                tail[m] = prev
        result = schedule(tasks)
        round_end = [max(result.finish[k] for k in keys) for keys in token_done]
        w = self.warmup
        wall = (round_end[-1] - round_end[w - 1]) / (self.decode_rounds - w)
        return wall, tasks, result

    def _prefill_wall(
        self, c: _LayerCosts, dep: Deployment
    ) -> tuple[float, list[Task], ScheduleResult]:
        """A single request walks the stages sequentially (no other work in
        flight during TTFT measurement)."""
        pp = dep.pp
        layers = _split_layers(c.n_layers, pp)
        tasks: list[Task] = []
        prev: int | None = None
        for s in range(pp):
            prev = self._stage_tasks(tasks, c, s, layers[s], prev, f"s{s}",
                                     is_first=(s == 0), is_last=(s == pp - 1))
            if c.hop and s < pp - 1:
                key = len(tasks)
                tasks.append(Task(key, f"h{s}", c.hop, [prev], f"h{s}"))
                prev = key
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
        if name == "decode":
            wall, tasks, result = self._decode_wall(costs, dep)
        else:
            wall, tasks, result = self._prefill_wall(costs, dep)
        self.last_runs[name] = (tasks, result)
        self.last_op_runs[name] = op_runs
        resource_busy = result.busy
        if op_runs:  # graph mode: merge in per-chip resource utilisation
            resource_busy = {
                **result.busy,
                **self._chip_resource_busy(tasks, op_runs, buckets, dep.pp),
            }
        return Phase(name, timings, wall_time=wall,
                     resource_busy=resource_busy, resource_span=result.makespan)
