"""Request-level, iteration-granular serving simulation for ONE replica.

Where `simulate.py` composes prefill and decode *analytically* -- one exclusive
prefill plus a steady-state decode round, combined into throughput/latency
averages -- this module runs a discrete event loop that interleaves prefill and
decode iterations in time under a real arrival process, so the numbers it
reports (TTFT/TPOT percentiles, inter-token gaps, batch occupancy, peak KV) all
carry queueing and prefill/decode interference rather than assuming them away.
It covers DES_todo.md section 4 at engine-iteration granularity.

Scope and why there is no scheduler here:

  * ONE replica, pp == 1.  At pp=1 an engine iteration is a *serial* chain of
    ops, so its duration is a closed-form sum -- there is nothing to schedule
    *within* an iteration (sched.py earns its keep only when pipeline stages
    overlap, pp>1).  The interesting dynamics are all *between* iterations:
    when a request is admitted, when its prefill preempts decode, how the KV
    cache grows.  So the loop is a plain event clock advanced by iteration
    durations, not a task graph.
  * Data parallelism is handled exactly as `simulate.py` composes it: a
    dp-way system serves dp independent replicas, so a whole-system arrival
    rate is divided by dp to get one replica's load (explicit `arrivals`
    traces are taken as already per-replica).  Whole-system throughput is the
    per-replica result times dp.
  * Prefill is *exclusive*: one request, its whole prompt, one iteration
    (matching `simulate.py`'s single-exclusive-prefill assumption).  While it
    runs, the decoding batch stalls -- that stall is exactly the interference
    the inter-token-gap metric exposes.  Chunked prefill and task-level pp>1
    serving remain future work.

Per-iteration cost model (see `_DecodeCost`): the loop runs thousands of
iterations, so ops are not re-lowered every step.  A decode iteration's cost is
the serial sum of its lowered ops; every op except attention depends only on
the running-batch size, so that part is tabulated over batch = 1..max once (the
table also captures the MoE expected-active-experts nonlinearity exactly, since
it is evaluated per batch).  Attention -- the only context-dependent op -- is
recost per iteration from the live (batch, total-context), which keeps per-token
KV growth exact and is cheap (one op).  The `engine` hook lets graph-refined
per-op chip costs (DESEngine graph mode) flow into these costs.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from math import ceil, floor
from typing import Callable

from .engine import CommContext, Engine, RooflineEngine
from .hardware import System
from .ops import (
    Op,
    decode_attention_op,
    decode_ops,
    kv_cache_bytes_per_chip,
    prefill_ops,
    validate_deployment,
)
from .simulate import weight_bytes_per_chip
from .workload import Deployment, ModelSpec, Scenario


# ---- configuration ----------------------------------------------------------


@dataclass(frozen=True)
class ServeConfig:
    """A serving run: an arrival process plus batching/admission knobs.

    Exactly one of `arrival_rate` (requests/s, Poisson) and `arrivals` (explicit
    arrival times, for tests/traces) must be set.  Prompt and output lengths
    come from the `Scenario`.
    """

    arrival_rate: float | None = None  # whole-system requests/s (Poisson)
    arrivals: list[float] | None = None  # explicit per-replica arrival times
    n_requests: int = 200  # simulate until this many complete (arrival_rate mode)
    max_batch: int = 64  # continuous-batching slots
    seed: int = 0
    prefill_first: bool = True  # vLLM-like: waiting prefills preempt decode

    def __post_init__(self) -> None:
        if (self.arrival_rate is None) == (self.arrivals is None):
            raise ValueError(
                "set exactly one of ServeConfig.arrival_rate or .arrivals"
            )
        if self.arrival_rate is not None and self.arrival_rate <= 0:
            raise ValueError("arrival_rate must be > 0")
        if self.max_batch < 1:
            raise ValueError("max_batch must be >= 1")


# ---- per-iteration cost model -----------------------------------------------


def _op_coster(system: System, dep: Deployment, engine: Engine) -> Callable[[Op], float]:
    """A `cost(op) -> seconds` for one whole op (all `count` instances) under
    `engine`.  DESEngine exposes `op_time` (graph-refined COMPUTE costs flow in
    there); a plain RooflineEngine costs via `time_op`."""
    if hasattr(engine, "op_time"):
        return lambda op: engine.op_time(op, system, dep)  # type: ignore[attr-defined]
    chip = system.node.chip
    comm = CommContext.for_deployment(system, dep)
    return lambda op: engine.time_op(op, chip, comm).time  # type: ignore[attr-defined]


@dataclass
class _DecodeCost:
    """Precomputed decode-iteration cost for a fixed (system, model, dep).

    `base[n]` is the summed cost of every decode op *except* attention at batch
    n (context-independent, so exact once tabulated -- including the MoE active-
    expert term).  `iter_time(n, sum_ctx)` adds the one attention op, recost
    from the live total context, so the KV-growth term is exact per iteration.
    """

    base: list[float]  # index by batch size; base[0] unused
    _attn_cost: Callable[[Op], float]
    model: ModelSpec
    dep: Deployment

    def iter_time(self, n: int, sum_ctx: float) -> float:
        att = self._attn_cost(decode_attention_op(self.model, self.dep, n, sum_ctx / n))
        return self.base[n] + att


def _build_decode_cost(
    system: System, model: ModelSpec, dep: Deployment, max_batch: int,
    cost_op: Callable[[Op], float],
) -> _DecodeCost:
    base = [0.0] * (max_batch + 1)
    for n in range(1, max_batch + 1):
        # context is irrelevant to the non-attention ops; pass a dummy 1.0
        base[n] = sum(
            cost_op(op) for op in decode_ops(model, dep, n, 1.0) if op.name != "attention"
        )
    return _DecodeCost(base, cost_op, model, dep)


def decode_iteration_time(
    system: System, model: ModelSpec, dep: Deployment, batch: float, sum_ctx: float,
    engine: Engine | None = None,
) -> float:
    """Serial-chain cost of one decode iteration: `batch` sequences each emit a
    token while sharing a total live context of `sum_ctx` tokens.  A direct op
    lowering (no table), public so callers/tests can oracle the loop's
    per-iteration cost."""
    engine = engine or RooflineEngine()
    cost_op = _op_coster(system, dep, engine)
    return sum(cost_op(op) for op in decode_ops(model, dep, batch, sum_ctx / batch))


def prefill_iteration_time(
    system: System, model: ModelSpec, dep: Deployment, prompt_len: int,
    engine: Engine | None = None,
) -> float:
    """Serial-chain cost of one exclusive prefill of `prompt_len` tokens (the
    TTFT of a request that never waits).  Equals `simulate`'s analytic TTFT for
    the roofline engine (overlap_comm off)."""
    engine = engine or RooflineEngine()
    cost_op = _op_coster(system, dep, engine)
    return sum(cost_op(op) for op in prefill_ops(model, prompt_len, dep))


# ---- records & report -------------------------------------------------------


@dataclass(frozen=True)
class RequestRecord:
    """One completed request's timeline (all times relative to its arrival)."""

    idx: int
    arrival: float
    prompt_len: int
    output_len: int
    ttft: float  # arrival -> first token (prefill completion)
    completion: float  # arrival -> last token
    tpot: float  # mean seconds per output token over the decode phase


@dataclass
class _Req:
    """Mutable in-flight request state."""

    idx: int
    arrival: float
    prompt_len: int
    output_len: int
    reserved_kv: float
    needs_prefill: bool = True
    gen: int = 0  # output tokens produced so far
    context: int = 0  # current KV length in tokens
    ttft: float = 0.0  # arrival -> first token
    first_token_time: float = 0.0  # absolute time of the first token
    last_emit: float = 0.0  # absolute time of the last token emitted


@dataclass
class ServeReport:
    system: System
    model: ModelSpec
    scenario: Scenario
    deployment: Deployment
    config: ServeConfig

    dp: int
    idle_chips: int

    # offered vs achieved load (per replica and whole-system = x dp).
    # `achieved` divides completions by (last completion - first arrival), so
    # it includes the post-arrival drain tail and under-reads sustained
    # capacity on short runs; `backlog_at_last_arrival` is the direct
    # queue-growth evidence (O(1) excursion when stable, grows linearly with
    # the run length past capacity).
    offered_rate_replica: float | None
    achieved_rate_replica: float
    backlog_at_last_arrival: int
    output_tokens_per_s_replica: float
    input_tokens_per_s_replica: float

    # latency (seconds), over completed requests -- includes queueing
    ttft_mean: float
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    tpot_mean: float  # per-request mean seconds/token, averaged

    # inter-token gap (seconds): the interference metric
    itg_mean: float
    itg_p50: float
    itg_p99: float

    # occupancy & memory
    mean_batch: float
    peak_batch: int
    kv_feasible_batch: int
    peak_kv_bytes: float
    kv_budget_bytes: float

    # counts / timing
    n_completed: int
    n_prefill_iters: int
    n_decode_iters: int
    makespan: float
    saturated: bool

    requests: list[RequestRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def offered_rate_system(self) -> float | None:
        return None if self.offered_rate_replica is None else self.offered_rate_replica * self.dp

    @property
    def achieved_rate_system(self) -> float:
        return self.achieved_rate_replica * self.dp

    @property
    def output_tokens_per_s_system(self) -> float:
        return self.output_tokens_per_s_replica * self.dp


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile of an already-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q / 100.0 * (len(sorted_vals) - 1)
    lo = floor(pos)
    hi = ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


# ---- the loop ---------------------------------------------------------------


def serve(
    system: System,
    model: ModelSpec,
    scenario: Scenario,
    deployment: Deployment = Deployment(),
    config: ServeConfig = ServeConfig(arrival_rate=1.0),
    engine: Engine | None = None,
) -> ServeReport:
    """Simulate continuous batching on one replica under `config`'s arrivals.

    Restricted to `deployment.pp == 1` (see module docstring): at pp>1 an
    iteration is no longer a serial chain and needs the pipeline scheduler,
    which is future work.
    """
    validate_deployment(model, deployment)
    if deployment.pp != 1:
        raise ValueError(
            f"serve() supports pp=1 only (got pp={deployment.pp}); pipeline-"
            "parallel serving needs task-level fill/drain across stages and is "
            "future work. Use tp/ep to scale a replica."
        )
    engine = engine or RooflineEngine()

    replica = deployment.replica_chips
    if replica > system.total_chips:
        raise ValueError(
            f"replica needs tp*ep={replica} chips but {system.name} has "
            f"{system.total_chips}"
        )
    dp = system.total_chips // replica
    idle_chips = system.total_chips - dp * replica
    warnings: list[str] = []

    chip = system.node.chip
    cost_op = _op_coster(system, deployment, engine)

    # ---- memory budget & KV feasibility ---------------------------------
    weights = weight_bytes_per_chip(model, deployment)
    microbatch = config.max_batch / (deployment.pp * deployment.ep)
    activations = 4 * microbatch * model.d_model * deployment.act_dtype.bytes
    kv_budget = chip.dram.capacity_bytes - weights - activations
    per_req_kv = kv_cache_bytes_per_chip(model, scenario.max_context, deployment)
    if per_req_kv <= 0:
        kv_feasible = config.max_batch
    elif kv_budget <= 0:
        kv_feasible = 0
    else:
        kv_feasible = int(floor(kv_budget / per_req_kv))
    if kv_feasible < 1:
        raise ValueError(
            f"no KV headroom on {chip.name}: weights {weights / 1e9:.1f} GB + "
            f"activations leave {kv_budget / 1e9:.2f} GB, but one request needs "
            f"{per_req_kv / 1e9:.2f} GB of KV at context {scenario.max_context} "
            "(raise tp, shrink context, or use a smaller kv-dtype)"
        )
    admit_cap = min(config.max_batch, kv_feasible)
    if kv_feasible < config.max_batch:
        warnings.append(
            f"KV budget caps concurrency at {kv_feasible} < max_batch="
            f"{config.max_batch}"
        )

    # ---- arrivals -------------------------------------------------------
    rng = random.Random(config.seed)
    if config.arrivals is not None:
        times = sorted(config.arrivals)
        offered_rate_replica: float | None = (
            len(times) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0]
            else None
        )
    else:
        per_replica_rate = config.arrival_rate / dp  # type: ignore[operator]
        t = 0.0
        times = []
        for _ in range(config.n_requests):
            t += rng.expovariate(per_replica_rate)
            times.append(t)
        offered_rate_replica = per_replica_rate

    reqs_in = [
        _Req(i, at, scenario.prompt_len, scenario.output_len, per_req_kv)
        for i, at in enumerate(times)
    ]
    n_total = len(reqs_in)
    if n_total == 0:
        raise ValueError("no requests to serve")

    # ---- precomputed costs ----------------------------------------------
    dec = _build_decode_cost(system, model, deployment, admit_cap, cost_op)
    prefill_cache: dict[int, float] = {}

    def prefill_cost(prompt_len: int) -> float:
        c = prefill_cache.get(prompt_len)
        if c is None:
            c = sum(cost_op(op) for op in prefill_ops(model, prompt_len, deployment))
            prefill_cache[prompt_len] = c
        return c

    kv_per_token = kv_cache_bytes_per_chip(model, 1, deployment)

    # ---- event loop -----------------------------------------------------
    clock = 0.0
    arr_i = 0
    waiting: deque[_Req] = deque()
    running: list[_Req] = []
    kv_reserved = 0.0
    kv_actual = 0.0

    completed: list[RequestRecord] = []
    gaps: list[float] = []  # inter-token gaps across all requests
    batch_samples: list[int] = []  # decode batch size per decode iteration
    peak_batch = 0
    peak_kv = 0.0
    n_prefill = 0
    n_decode = 0

    backlog_at_last_arrival = 0

    while len(completed) < n_total:
        # intake: newly-arrived requests join the FIFO waiting queue
        while arr_i < n_total and reqs_in[arr_i].arrival <= clock:
            waiting.append(reqs_in[arr_i])
            arr_i += 1
            if arr_i == n_total:
                backlog_at_last_arrival = len(waiting)
        # admission: FIFO, gated by the batch slot count and the KV budget
        while (
            waiting
            and len(running) < admit_cap
            and kv_reserved + waiting[0].reserved_kv <= kv_budget + 1e-6
        ):
            r = waiting.popleft()
            r.needs_prefill = True
            running.append(r)
            kv_reserved += r.reserved_kv

        pending = [r for r in running if r.needs_prefill]
        decoders = [r for r in running if not r.needs_prefill]

        if pending and (config.prefill_first or not decoders):
            do_prefill = True
        elif decoders:
            do_prefill = False
        elif pending:
            do_prefill = True
        else:
            # nothing runnable: jump the clock to the next arrival
            if arr_i >= n_total:
                break  # defensive: should not happen while completed < n_total
            clock = max(clock, reqs_in[arr_i].arrival)
            continue

        if do_prefill:
            r = pending[0]  # FIFO: earliest-admitted pending request
            clock += prefill_cost(r.prompt_len)
            n_prefill += 1
            r.needs_prefill = False
            r.context = r.prompt_len
            r.gen = 0
            r.ttft = clock - r.arrival
            r.first_token_time = clock
            r.last_emit = clock
            kv_actual += kv_per_token * r.prompt_len
            peak_kv = max(peak_kv, kv_actual)
            peak_batch = max(peak_batch, len(running))
        else:
            n = len(decoders)
            sum_ctx = sum(r.context for r in decoders)
            clock += dec.iter_time(n, sum_ctx)
            n_decode += 1
            batch_samples.append(n)
            peak_batch = max(peak_batch, len(running))
            finished: list[_Req] = []
            for r in decoders:
                r.gen += 1
                gaps.append(clock - r.last_emit)
                r.last_emit = clock
                r.context += 1
                kv_actual += kv_per_token
                if r.gen >= r.output_len:
                    finished.append(r)
            peak_kv = max(peak_kv, kv_actual)
            for r in finished:
                tpot = (clock - r.first_token_time) / r.output_len if r.output_len else 0.0
                completed.append(
                    RequestRecord(
                        idx=r.idx, arrival=r.arrival, prompt_len=r.prompt_len,
                        output_len=r.output_len, ttft=r.ttft,
                        completion=clock - r.arrival, tpot=tpot,
                    )
                )
                running.remove(r)
                kv_reserved -= r.reserved_kv
                kv_actual -= kv_per_token * r.context

    # ---- aggregate metrics ----------------------------------------------
    completed.sort(key=lambda rec: rec.idx)
    first_arrival = times[0]
    makespan = clock - first_arrival
    ttfts = sorted(rec.ttft for rec in completed)
    tpots = [rec.tpot for rec in completed]
    gaps_sorted = sorted(gaps)
    total_output = sum(rec.output_len for rec in completed)
    total_input = sum(rec.prompt_len for rec in completed)

    achieved = n_total / makespan if makespan > 0 else 0.0
    # Saturated = the waiting queue GREW over the arrival window -- the direct
    # queueing-theory definition of not keeping up.  Throughput shortfall and
    # TTFT inflation are both unreliable here: `achieved` under-reads on short
    # runs (drain tail), and p99 TTFT spikes transiently on stable-but-loaded
    # systems.  A stable system shows an O(1) backlog excursion at the last
    # arrival; past capacity the backlog scales with the run, so a threshold
    # that grows with the per-replica request count separates them.
    saturated = backlog_at_last_arrival > max(4, ceil(0.1 * n_total))

    return ServeReport(
        system=system, model=model, scenario=scenario, deployment=deployment,
        config=config, dp=dp, idle_chips=idle_chips,
        offered_rate_replica=offered_rate_replica,
        achieved_rate_replica=achieved,
        backlog_at_last_arrival=backlog_at_last_arrival,
        output_tokens_per_s_replica=total_output / makespan if makespan > 0 else 0.0,
        input_tokens_per_s_replica=total_input / makespan if makespan > 0 else 0.0,
        ttft_mean=sum(ttfts) / len(ttfts) if ttfts else 0.0,
        ttft_p50=_percentile(ttfts, 50), ttft_p95=_percentile(ttfts, 95),
        ttft_p99=_percentile(ttfts, 99),
        tpot_mean=sum(tpots) / len(tpots) if tpots else 0.0,
        itg_mean=sum(gaps) / len(gaps) if gaps else 0.0,
        itg_p50=_percentile(gaps_sorted, 50), itg_p99=_percentile(gaps_sorted, 99),
        mean_batch=sum(batch_samples) / len(batch_samples) if batch_samples else 0.0,
        peak_batch=peak_batch, kv_feasible_batch=admit_cap,
        peak_kv_bytes=peak_kv, kv_budget_bytes=kv_budget,
        n_completed=len(completed), n_prefill_iters=n_prefill, n_decode_iters=n_decode,
        makespan=makespan, saturated=saturated,
        requests=completed, warnings=warnings,
    )


# ---- rendering --------------------------------------------------------------


def format_serve_report(r: ServeReport) -> str:
    """Plain-text serve report, in the style of report.format_report."""
    from .units import fmt_bytes, fmt_si, fmt_time

    s, m, sc, d, cfg = r.system, r.model, r.scenario, r.deployment, r.config
    lines: list[str] = []
    add = lines.append

    add("=" * 72)
    add(f"inferencesim serve  |  {s.name}  x  {m.name}")
    add("=" * 72)
    add(f"Parallelism  : TP={d.tp}  EP={d.ep}  DP={r.dp}  ({d.replica_chips} chips/replica)"
        + (f"  ({r.idle_chips} chips idle)" if r.idle_chips else ""))
    add(f"Scenario     : prompt={sc.prompt_len}, output={sc.output_len}, "
        f"max_batch={cfg.max_batch}"
        + (f" (KV-capped to {r.kv_feasible_batch})" if r.kv_feasible_batch < cfg.max_batch else ""))
    arrival = (f"Poisson {cfg.arrival_rate} req/s system" if cfg.arrival_rate is not None
               else f"trace of {r.n_completed} arrivals")
    add(f"Arrivals     : {arrival}, prefill_first={cfg.prefill_first}, seed={cfg.seed}")
    add("-" * 72)
    off_r = "n/a" if r.offered_rate_replica is None else f"{r.offered_rate_replica:.3f}"
    off_s = "n/a" if r.offered_rate_system is None else f"{r.offered_rate_system:.3f}"
    add(f"Offered load : {off_r} req/s/replica  ({off_s} req/s system)")
    add(f"Achieved     : {r.achieved_rate_replica:.3f} req/s/replica  "
        f"({r.achieved_rate_system:.3f} req/s system, incl. drain tail)"
        + ("   ** SATURATED **" if r.saturated else ""))
    if r.backlog_at_last_arrival:
        add(f"Backlog      : {r.backlog_at_last_arrival} waiting at last arrival "
            f"(stable systems show an O(1) excursion here)")
    add(f"Throughput   : {fmt_si(r.output_tokens_per_s_replica, 'tok/s')} output/replica  "
        f"({fmt_si(r.output_tokens_per_s_system, 'tok/s')} system)")
    add("-" * 72)
    add(f"TTFT         : mean {fmt_time(r.ttft_mean)}  p50 {fmt_time(r.ttft_p50)}  "
        f"p95 {fmt_time(r.ttft_p95)}  p99 {fmt_time(r.ttft_p99)}")
    add(f"TPOT         : mean {fmt_time(r.tpot_mean)}/token"
        + (f"  ->  {1.0 / r.tpot_mean:.1f} tok/s/request" if r.tpot_mean > 0 else ""))
    add(f"Inter-token  : mean {fmt_time(r.itg_mean)}  p50 {fmt_time(r.itg_p50)}  "
        f"p99 {fmt_time(r.itg_p99)}"
        + (f"  (p99/median {r.itg_p99 / r.itg_p50:.1f}x = prefill interference)"
           if r.itg_p50 > 0 else ""))
    add("-" * 72)
    add(f"Batch occ.   : mean {r.mean_batch:.1f}, peak {r.peak_batch} "
        f"(feasible {r.kv_feasible_batch})")
    add(f"KV cache     : peak {fmt_bytes(r.peak_kv_bytes)} / {fmt_bytes(r.kv_budget_bytes)} "
        f"budget per chip")
    add(f"Iterations   : {r.n_prefill_iters} prefill + {r.n_decode_iters} decode, "
        f"{r.n_completed} completed over {fmt_time(r.makespan)}")
    for w in r.warnings:
        add(f"WARNING      : {w}")
    add("=" * 72)
    return "\n".join(lines)
