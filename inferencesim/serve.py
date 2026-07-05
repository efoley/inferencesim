"""Request-level, iteration-granular serving simulation for ONE replica.

Where `simulate.py` composes prefill and decode *analytically* -- one exclusive
prefill plus a steady-state decode round, combined into throughput/latency
averages -- this module runs a discrete event loop that interleaves prefill and
decode iterations in time under a real arrival process, so the numbers it
reports (TTFT/TPOT percentiles, inter-token gaps, batch occupancy, peak KV) all
carry queueing and prefill/decode interference rather than assuming them away.
It covers DES_todo.md section 4 at engine-iteration granularity.

`serve_disagg` extends the same iteration granularity to *disaggregated* serving
(DistServe / NVIDIA Dynamo): the chips split into a prefill pool and a decode
pool, the KV cache streams between them, and decode runs pure (never
prefill-stalled).  It is an event-driven multi-replica loop -- each replica
keeps its own clock, a global heap advances the earliest completion -- but every
replica's per-iteration cost is still the closed-form serial-op sum, so only the
*placement* of iterations across replicas is scheduled.  It reuses this module's
request construction (mixed lengths), KV policies (on_demand preemption becomes a
cross-pool re-prefill + re-transfer), and cost tables; chunked prefill is N/A
(exclusive prefill replicas make it moot).  A lone request through a zero-cost
link reproduces the aggregated loop exactly, under either KV policy.

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

Three admission/prefill regimes, all off one loop:

  * Exclusive prefill (`prefill_chunk=None`, the historical default): a waiting
    request's whole prompt is one iteration that stalls the decoding batch --
    the interference the inter-token-gap metric exposes.  `prefill_chunk=K`
    instead mixes a K-token prefill chunk into each decode iteration
    (Sarathi-style): gaps shrink, TTFT stretches.
  * KV policy.  `on_demand` (default) admits a request against only its prompt
    KV plus a `kv_watermark` fraction of the budget, grows KV per token, and
    PREEMPTS the latest-admitted decoder (vLLM recompute) when a step would
    overflow -- the victim's KV is freed, it returns to the front of the queue,
    and on re-admission its prefill recomputes prompt + tokens-generated-so-far
    before decoding resumes.  `reserve` is the conservative policy: admit only
    if the request's full prompt+output KV fits, so it never preempts but
    admits fewer.
  * Mixed request lengths: prompt/output are per request (explicit lists,
    parallel to `arrivals`, or a `LengthDist` sampled in Poisson mode).

Per-iteration cost model (see `_DecodeCost`): the loop runs thousands of
iterations, so ops are not re-lowered every step.  A decode iteration's cost is
the serial sum of its lowered ops; every op except attention depends only on
the running-batch size, so that part is tabulated over batch = 1..max_batch
once (the table also captures the MoE expected-active-experts nonlinearity
exactly, since it is evaluated per batch).  Attention -- the only
context-dependent op -- is recost per iteration from the live (batch,
total-context), which keeps per-token KV growth exact and is cheap (one op).
Prefill and prefill-chunk costs are cached (keys grid-rounded so a continuous
length distribution can't grow the cache without bound).  The `engine` hook
lets graph-refined per-op chip costs (DESEngine graph mode) flow into all of
these.
"""

from __future__ import annotations

import heapq
import math
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
    prefill_chunk_ops,
    prefill_ops,
    validate_deployment,
)
from .simulate import weight_bytes_per_chip
from .workload import Deployment, ModelSpec, Scenario

_EPS = 1e-6  # bytes slack when comparing KV footprints to the budget


# ---- configuration ----------------------------------------------------------


@dataclass(frozen=True)
class LengthDist:
    """A prompt/output length distribution for Poisson-mode sampling.

    kind="uniform": integer in [a, b].
    kind="lognormal": round(a * exp(N(0, b))) -- `a` is the median, `b` the
    sigma of the underlying normal.  Samples are clamped to >= 1 and rounded to
    the caller's token grid so the prefill-cost cache stays bounded.
    """

    kind: str
    a: float
    b: float

    @classmethod
    def uniform(cls, lo: int, hi: int) -> "LengthDist":
        return cls("uniform", float(lo), float(hi))

    @classmethod
    def lognormal(cls, median: float, sigma: float) -> "LengthDist":
        return cls("lognormal", float(median), float(sigma))

    def sample(self, rng: random.Random, grid: int) -> int:
        if self.kind == "uniform":
            v = rng.uniform(self.a, self.b)
        elif self.kind == "lognormal":
            v = self.a * math.exp(rng.gauss(0.0, self.b))
        else:
            raise ValueError(f"unknown LengthDist kind {self.kind!r}")
        return _round_grid(v, grid)


@dataclass(frozen=True)
class ServeConfig:
    """A serving run: an arrival process, request lengths, and admission knobs.

    Exactly one of `arrival_rate` (requests/s, Poisson) and `arrivals` (explicit
    per-replica arrival times) must be set.

    Lengths default to the `Scenario`'s prompt_len/output_len.  Override them
    per request with `prompt_lens`/`output_lens` (lists parallel to `arrivals`)
    or, in Poisson mode, with `prompt_dist`/`output_dist` (`LengthDist`s sampled
    from the same seeded RNG).

    KV: `kv_policy="on_demand"` charges admission only the prompt footprint (up
    to `kv_watermark` of the budget) and preempts on overflow; `"reserve"`
    charges the full prompt+output footprint and never preempts.

    `prefill_chunk` (tokens): None = exclusive whole-prompt prefill; an int
    mixes that many prefill tokens into each decode iteration.
    """

    arrival_rate: float | None = None  # whole-system requests/s (Poisson)
    arrivals: list[float] | None = None  # explicit per-replica arrival times
    prompt_lens: list[int] | None = None  # per-request, parallel to arrivals
    output_lens: list[int] | None = None
    prompt_dist: LengthDist | None = None  # Poisson-mode length sampling
    output_dist: LengthDist | None = None
    n_requests: int = 200  # simulate until this many complete (arrival_rate mode)
    max_batch: int = 64  # continuous-batching slots
    seed: int = 0
    prefill_first: bool = True  # exclusive mode: waiting prefills preempt decode
    kv_policy: str = "on_demand"  # "on_demand" | "reserve"
    kv_watermark: float = 0.95  # usable fraction of the KV budget (on_demand)
    prefill_chunk: int | None = None  # tokens/iteration; None = exclusive
    length_grid: int = 64  # round sampled/recompute token counts for cost caching

    def __post_init__(self) -> None:
        if (self.arrival_rate is None) == (self.arrivals is None):
            raise ValueError(
                "set exactly one of ServeConfig.arrival_rate or .arrivals"
            )
        if self.arrival_rate is not None and self.arrival_rate <= 0:
            raise ValueError("arrival_rate must be > 0")
        if self.max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        if self.kv_policy not in ("on_demand", "reserve"):
            raise ValueError("kv_policy must be 'on_demand' or 'reserve'")
        if not 0.0 < self.kv_watermark <= 1.0:
            raise ValueError("kv_watermark must be in (0, 1]")
        if self.prefill_chunk is not None and self.prefill_chunk < 1:
            raise ValueError("prefill_chunk must be >= 1 (or None)")
        if self.arrivals is not None:
            n = len(self.arrivals)
            for name, lst in (("prompt_lens", self.prompt_lens),
                              ("output_lens", self.output_lens)):
                if lst is not None and len(lst) != n:
                    raise ValueError(f"{name} must be parallel to arrivals ({n})")


@dataclass(frozen=True)
class DisaggConfig:
    """The pool structure of a prefill/decode *disaggregated* run.

    The chips are partitioned into a prefill pool (`n_prefill_replicas` replicas
    of `prefill_deployment`) and a decode pool (`n_decode_replicas` replicas of
    `decode_deployment`), with
    `n_p x prefill.replica_chips + n_d x decode.replica_chips <= total_chips`.
    A waiting request runs its whole prompt on any free prefill replica
    (exclusive -- prefill replicas never batch decode, which is the point); on
    completion its KV cache streams to the least-loaded decode replica with
    headroom and decode runs there as pure decode iterations, never stalled by a
    prefill (the architectural win of DistServe / NVIDIA Dynamo).  First token
    lands at prefill completion + KV transfer, so TTFT includes the transfer.

    Everything *else* -- the arrival process, mixed request lengths, `max_batch`,
    seed, and the KV policy -- comes from the companion `ServeConfig` passed to
    `serve_disagg`, so the whole polished length/admission surface applies
    unchanged.  The one exception: `prefill_chunk` must be None (chunked prefill
    is moot with exclusive prefill replicas), and `prefill_first` is ignored.

    KV transfer cost is `kv_bytes(context) / transfer_bw + transfer_latency`.  By
    default the link resolves from the system with the same node-vs-network rule
    as `System.link_for_group`: all pool chips within one node ride
    `node.interconnect`, a spanning pool the `network` link.  `transfer_bw` /
    `transfer_latency` override it (e.g. a zero-cost link for the oracle).  The
    transfer is a delay on the request's timeline: the prefill replica is freed
    at prefill completion (pipelined) and the decode replica is not occupied
    during it.  Link *contention* between concurrent transfers is future work.
    """

    prefill_deployment: Deployment
    decode_deployment: Deployment
    n_prefill_replicas: int
    n_decode_replicas: int
    transfer_bw: float | None = None  # override pool-to-pool bandwidth (bytes/s)
    transfer_latency: float | None = None  # override pool-to-pool latency (s)

    def __post_init__(self) -> None:
        if self.n_prefill_replicas < 1 or self.n_decode_replicas < 1:
            raise ValueError("n_prefill_replicas and n_decode_replicas must be >= 1")


def _round_grid(v: float, grid: int) -> int:
    """Round a token count to the nearest positive multiple of `grid`."""
    if grid <= 1:
        return max(1, int(round(v)))
    return max(grid, int(round(v / grid)) * grid)


def _ceil_grid(n: int, grid: int) -> int:
    """Round a token count up to a multiple of `grid` (cost-cache key)."""
    if grid <= 1:
        return n
    return int(ceil(n / grid)) * grid


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


def chunked_prefill_ttft(
    system: System, model: ModelSpec, dep: Deployment, prompt_len: int, chunk: int,
    engine: Engine | None = None,
) -> float:
    """TTFT of a lone request prefilled `chunk` tokens per iteration: the sum of
    the per-chunk iteration costs.  Exceeds the exclusive prefill time because
    each chunk re-streams the weights and re-reads the growing KV cache."""
    engine = engine or RooflineEngine()
    cost_op = _op_coster(system, dep, engine)
    total = 0.0
    done = 0
    while done < prompt_len:
        c = min(chunk, prompt_len - done)
        last = done + c == prompt_len
        total += sum(cost_op(op) for op in prefill_chunk_ops(model, dep, c, done, last))
        done += c
    return total


# ---- records & report -------------------------------------------------------


@dataclass(frozen=True)
class RequestRecord:
    """One completed request's timeline (all times relative to its arrival)."""

    idx: int
    arrival: float
    prompt_len: int
    output_len: int
    ttft: float  # arrival -> first token (prefill completion, + KV transfer in disagg)
    completion: float  # arrival -> last token
    tpot: float  # mean seconds per output token over the decode phase
    n_preemptions: int = 0  # times this request was preempted (recompute)
    transfer: float = 0.0  # first KV prefill->decode transfer delay (disagg only)


@dataclass
class _Req:
    """Mutable in-flight request state."""

    idx: int
    arrival: float
    prompt_len: int
    output_len: int
    needs_prefill: bool = True
    prefill_done: int = 0  # prompt(+recompute) tokens prefilled so far
    started: bool = False  # first token emitted (TTFT recorded)?
    gen: int = 0  # output tokens produced so far
    context: int = 0  # current KV length in tokens
    ttft: float = 0.0  # arrival -> first token
    first_token_time: float = 0.0  # absolute time of the first token
    last_emit: float = 0.0  # absolute time of the last token emitted
    preemptions: int = 0
    transfer_delay: float = 0.0  # disagg: the first (TTFT-contributing) transfer

    @property
    def prefill_target(self) -> int:
        """KV length to (re)build: prompt for a fresh request, prompt + already-
        generated tokens for a recompute after preemption."""
        return self.prompt_len + self.gen


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
    # queue-growth evidence (O(1) excursion when stable, grows past capacity).
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
    n_preemptions: int

    # request-length distribution (p50/p99 of prompt & output over completions)
    prompt_p50: int
    prompt_p99: int
    output_p50: int
    output_p99: int
    mixed_lengths: bool

    # counts / timing
    n_completed: int
    n_prefill_iters: int
    n_decode_iters: int
    makespan: float
    saturated: bool

    # ---- disaggregated serving (all None/0 for the aggregated path) ----------
    # When `disagg` is set the run is prefill/decode disaggregated: the fields
    # above are whole-system (dp == n_decode_replicas, so the per-"replica"
    # figures are whole-system / decode-replica count; `deployment` echoes the
    # decode pool, `prefill_deployment` the prefill pool).
    disagg: DisaggConfig | None = None
    prefill_deployment: Deployment | None = None
    n_prefill_replicas: int = 0
    n_decode_replicas: int = 0
    prefill_util: float = 0.0  # prefill replicas' mean busy fraction
    decode_util: float = 0.0  # decode replicas' mean busy fraction
    transfer_mean: float = 0.0  # mean KV transfer delay over all transfers (s)
    transfer_p99: float = 0.0
    transfer_bytes_total: float = 0.0  # total KV bytes streamed (incl. re-transfers)
    transfer_bw: float = 0.0  # resolved pool-to-pool bandwidth (bytes/s)
    transfer_latency: float = 0.0  # resolved pool-to-pool latency (s)

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


# ---- request construction ---------------------------------------------------


def _build_requests(
    config: ServeConfig, scenario: Scenario, dp: int, rng: random.Random,
) -> list[_Req]:
    """Arrival times and per-request prompt/output lengths."""
    grid = config.length_grid
    reqs: list[_Req] = []
    if config.arrivals is not None:
        times = list(config.arrivals)
        order = sorted(range(len(times)), key=lambda i: times[i])
        for new_i, i in enumerate(order):
            prompt = (config.prompt_lens[i] if config.prompt_lens is not None
                      else scenario.prompt_len)
            output = (config.output_lens[i] if config.output_lens is not None
                      else scenario.output_len)
            reqs.append(_Req(new_i, times[i], int(prompt), int(output)))
    else:
        per_replica_rate = config.arrival_rate / dp  # type: ignore[operator]
        t = 0.0
        for i in range(config.n_requests):
            t += rng.expovariate(per_replica_rate)  # arrival gap first...
            prompt = (config.prompt_dist.sample(rng, grid)
                      if config.prompt_dist is not None else scenario.prompt_len)
            output = (config.output_dist.sample(rng, grid)
                      if config.output_dist is not None else scenario.output_len)
            reqs.append(_Req(i, t, int(prompt), int(output)))
    return reqs


# ---- the loop ---------------------------------------------------------------


def serve(
    system: System,
    model: ModelSpec,
    scenario: Scenario,
    deployment: Deployment = Deployment(),
    config: ServeConfig = ServeConfig(arrival_rate=1.0),
    engine: Engine | None = None,
) -> ServeReport:
    """Simulate continuous batching on one replica under `config`.

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
            f"replica needs tp*ep*adp={replica} chips but {system.name} has "
            f"{system.total_chips}"
        )
    dp = system.total_chips // replica
    idle_chips = system.total_chips - dp * replica
    warnings: list[str] = []

    chip = system.node.chip
    cost_op = _op_coster(system, deployment, engine)
    on_demand = config.kv_policy == "on_demand"

    # ---- memory budget --------------------------------------------------
    weights = weight_bytes_per_chip(model, deployment)
    microbatch = config.max_batch / (deployment.pp * deployment.ep * deployment.adp)
    activations = 4 * microbatch * model.d_model * deployment.act_dtype.bytes
    kv_budget = chip.dram.capacity_bytes - weights - activations
    kv_per_token = kv_cache_bytes_per_chip(model, 1, deployment)
    # usable KV: on_demand keeps a watermark of headroom; reserve uses it all
    usable = (config.kv_watermark * kv_budget) if on_demand else kv_budget
    if usable <= 0:
        raise ValueError(
            f"no KV headroom on {chip.name}: weights {weights / 1e9:.1f} GB + "
            f"activations exceed the {chip.dram.capacity_bytes / 1e9:.1f} GB DRAM "
            "(raise tp, or use a smaller weight/kv dtype)"
        )

    # ---- arrivals & lengths ---------------------------------------------
    rng = random.Random(config.seed)
    reqs_in = _build_requests(config, scenario, dp, rng)
    n_total = len(reqs_in)
    if n_total == 0:
        raise ValueError("no requests to serve")
    times = [r.arrival for r in reqs_in]

    # livelock guard #1: a request whose full context can't fit the HARD budget
    # can never complete under either policy -- fail loudly rather than thrash.
    # (A request that fits the hard budget but not the on_demand watermark is
    # still served: it force-admits when the replica is idle and runs over the
    # watermark up to the hard budget -- guard #2 below, a warning not an error.)
    for r in reqs_in:
        full = (r.prompt_len + r.output_len) * kv_per_token
        if full > kv_budget + _EPS:
            raise ValueError(
                f"request {r.idx} needs {full / 1e9:.2f} GB of KV at full context "
                f"{r.prompt_len + r.output_len} but only {kv_budget / 1e9:.2f} GB of "
                "DRAM is free per chip; it can never fit (raise tp, shrink lengths, "
                "or use a smaller kv-dtype)"
            )

    if config.arrival_rate is not None:
        offered_rate_replica: float | None = config.arrival_rate / dp
    else:
        offered_rate_replica = (
            len(times) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0]
            else None
        )

    # representative "how many max-context requests fit" figure for the report
    per_req_max = kv_cache_bytes_per_chip(model, scenario.max_context, deployment)
    kv_feasible = (min(config.max_batch, int(floor(usable / per_req_max)))
                   if per_req_max > 0 else config.max_batch)

    # ---- precomputed costs ----------------------------------------------
    dec = _build_decode_cost(system, model, deployment, config.max_batch, cost_op)
    prefill_cache: dict[int, float] = {}
    chunk_cache: dict[tuple[int, int, bool], float] = {}

    def prefill_cost(n_tokens: int) -> float:
        key = _ceil_grid(n_tokens, config.length_grid)
        c = prefill_cache.get(key)
        if c is None:
            c = sum(cost_op(op) for op in prefill_ops(model, key, deployment))
            prefill_cache[key] = c
        return c

    def chunk_cost(chunk: int, prior: int, produce: bool) -> float:
        key = (chunk, prior, produce)
        c = chunk_cache.get(key)
        if c is None:
            c = sum(cost_op(op) for op in prefill_chunk_ops(model, deployment, chunk,
                                                            prior, produce))
            chunk_cache[key] = c
        return c

    # ---- event loop -----------------------------------------------------
    clock = 0.0
    arr_i = 0
    waiting: deque[_Req] = deque()
    running: list[_Req] = []
    kv_used = 0.0  # actual KV in use across running requests
    kv_reserved = 0.0  # reserve policy: committed full footprints

    completed: list[RequestRecord] = []
    gaps: list[float] = []
    batch_samples: list[int] = []
    peak_batch = 0
    peak_kv = 0.0
    n_prefill = 0
    n_decode = 0
    n_preempt = 0
    forced_over_budget = 0
    backlog_at_last_arrival = 0

    def full_fp(r: _Req) -> float:
        return (r.prompt_len + r.output_len) * kv_per_token

    def can_admit(r: _Req) -> bool:
        if len(running) >= config.max_batch:
            return False
        if on_demand:
            # commit each running request to its prompt footprint (pending) or
            # current context (decoders, which grow -- preemption handles that);
            # admit only if the newcomer's own footprint also fits.  Charging
            # prompts, not full prompt+output, is what lets on_demand pack more
            # requests than reserve.
            committed = sum(
                (x.prefill_target if x.needs_prefill else x.context) * kv_per_token
                for x in running
            )
            return committed + r.prefill_target * kv_per_token <= usable + _EPS
        return kv_reserved + full_fp(r) <= kv_budget + _EPS

    def preempt(victim: _Req) -> None:
        nonlocal kv_used, n_preempt
        kv_used -= victim.context * kv_per_token
        running.remove(victim)
        victim.needs_prefill = True
        victim.prefill_done = 0
        victim.context = 0
        victim.preemptions += 1
        waiting.appendleft(victim)  # recompute priority: front of the queue
        n_preempt += 1

    def latest_decoder(exclude: _Req | None = None) -> _Req | None:
        cands = [r for r in running if not r.needs_prefill and r is not exclude]
        return max(cands, key=lambda r: r.idx) if cands else None

    def finish(r: _Req) -> None:
        nonlocal kv_used, kv_reserved
        tpot = (clock - r.first_token_time) / r.output_len if r.output_len else 0.0
        completed.append(RequestRecord(
            idx=r.idx, arrival=r.arrival, prompt_len=r.prompt_len,
            output_len=r.output_len, ttft=r.ttft, completion=clock - r.arrival,
            tpot=tpot, n_preemptions=r.preemptions,
        ))
        running.remove(r)
        kv_used -= r.context * kv_per_token
        if not on_demand:
            kv_reserved -= full_fp(r)

    def complete_prefill(r: _Req) -> None:
        """Mark a (re)prefill done; record TTFT on the first prefill only."""
        r.needs_prefill = False
        r.context = r.prefill_target
        if not r.started:
            r.started = True
            r.ttft = clock - r.arrival
            r.first_token_time = clock
            r.last_emit = clock
        # recompute: keep ttft/first_token_time and DO NOT reset last_emit, so
        # the victim's next inter-token gap includes the whole preemption stall.

    chunked = config.prefill_chunk is not None
    iter_cap = 1000 * n_total + 50 * sum(r.prompt_len + r.output_len for r in reqs_in)
    iters = 0

    def step_decoders(decoders_now: list[_Req]) -> list[_Req]:
        """Advance one token for each decoder; return the ones that completed.
        `clock` and `kv_used` have already been updated by the caller for this
        iteration, so token emission times and gaps are recorded against it."""
        nonlocal kv_used
        done: list[_Req] = []
        for r in decoders_now:
            r.gen += 1
            gaps.append(clock - r.last_emit)
            r.last_emit = clock
            r.context += 1
            kv_used += kv_per_token
            if r.gen >= r.output_len:
                done.append(r)
        return done

    while len(completed) < n_total:
        iters += 1
        if iters > iter_cap:
            raise ValueError(
                "serve() did not terminate within the iteration budget -- likely "
                "a preemption livelock; inspect the construction"
            )
        # intake
        while arr_i < n_total and reqs_in[arr_i].arrival <= clock:
            waiting.append(reqs_in[arr_i])
            arr_i += 1
            if arr_i == n_total:
                backlog_at_last_arrival = len(waiting)
        # admission (FIFO, head-of-line)
        while waiting and can_admit(waiting[0]):
            r = waiting.popleft()
            r.needs_prefill = True
            r.prefill_done = 0
            r.context = 0
            running.append(r)
            if not on_demand:
                kv_reserved += full_fp(r)
        # progress guarantee: if the replica is idle but the head can't clear the
        # watermark gate, force-admit it anyway (it fits the hard budget by guard
        # #1, so it runs alone).  Without this a request whose footprint exceeds
        # the on_demand watermark -- or a preempted request whose recompute
        # target does -- could wait forever.
        if not running and waiting:
            r = waiting.popleft()
            r.needs_prefill = True
            r.prefill_done = 0
            r.context = 0
            running.append(r)
            if not on_demand:
                kv_reserved += full_fp(r)

        # the active prefiller: an in-progress chunked prefill first, else the
        # oldest pending request (at most one prefills at a time, vLLM default)
        pending = [r for r in running if r.needs_prefill]
        if pending:
            in_progress = [r for r in pending if r.prefill_done > 0]
            prefiller: _Req | None = min(in_progress or pending, key=lambda r: r.idx)
        else:
            prefiller = None
        decoders = [r for r in running if not r.needs_prefill]

        if prefiller is None and not decoders:
            if arr_i >= n_total:
                break  # defensive: nothing runnable and no more arrivals
            clock = max(clock, reqs_in[arr_i].arrival)
            continue

        if not chunked:
            do_prefill = prefiller is not None and (config.prefill_first or not decoders)
            if do_prefill:
                # ---- exclusive prefill: the whole (re)prefill in one iteration,
                # stalling the decode batch ----
                r = prefiller
                need = r.prefill_target * kv_per_token  # r holds 0 (prefill_done=0)
                if on_demand:
                    while kv_used + need > usable + _EPS:
                        v = latest_decoder(exclude=r)
                        if v is None:
                            break  # guard #1: r alone fits, so kv_used(0)+need ok
                        preempt(v)
                clock += prefill_cost(r.prefill_target)
                n_prefill += 1
                kv_used += need
                complete_prefill(r)
                peak_kv = max(peak_kv, kv_used)
                peak_batch = max(peak_batch, len(running))
                continue
            # ---- decode step ----
            decoders_now = list(decoders)
            if on_demand:
                while len(decoders_now) > 1 and \
                        kv_used + len(decoders_now) * kv_per_token > usable + _EPS:
                    v = max(decoders_now, key=lambda x: x.idx)
                    decoders_now.remove(v)
                    preempt(v)
                if len(decoders_now) == 1 and kv_used + kv_per_token > usable + _EPS:
                    forced_over_budget += 1  # guard #2: can't preempt the last one
            n = len(decoders_now)
            clock += dec.iter_time(n, sum(r.context for r in decoders_now))
            n_decode += 1
            batch_samples.append(n)
            for r in step_decoders(decoders_now):
                finish(r)
            peak_kv = max(peak_kv, kv_used)
            peak_batch = max(peak_batch, len(running))
            continue

        # ---- chunked prefill: mix one prefill chunk into a decode step ----
        decoders_now = list(decoders)
        chunk_size = 0
        prior = produce = 0
        if prefiller is not None:
            target = prefiller.prefill_target
            chunk_size = min(config.prefill_chunk, target - prefiller.prefill_done)
            prior = prefiller.prefill_done
            produce = prefiller.prefill_done + chunk_size == target
        if on_demand:
            while decoders_now and \
                    kv_used + (len(decoders_now) + chunk_size) * kv_per_token > usable + _EPS:
                v = max(decoders_now, key=lambda x: x.idx)
                decoders_now.remove(v)
                preempt(v)
            if not decoders_now and chunk_size and kv_used + chunk_size * kv_per_token > usable + _EPS:
                forced_over_budget += 1  # guard #2
        n = len(decoders_now)
        step_cost = dec.iter_time(n, sum(r.context for r in decoders_now)) if n else 0.0
        if chunk_size:
            step_cost += chunk_cost(chunk_size, prior, bool(produce))
        clock += step_cost
        finished: list[_Req] = []
        if n:
            n_decode += 1
            batch_samples.append(n)
            finished = step_decoders(decoders_now)
        if prefiller is not None and chunk_size:
            n_prefill += 1
            prefiller.prefill_done += chunk_size
            prefiller.context = prefiller.prefill_done
            kv_used += chunk_size * kv_per_token
            if prefiller.prefill_done >= target:
                complete_prefill(prefiller)
        peak_kv = max(peak_kv, kv_used)
        for r in finished:
            finish(r)
        peak_batch = max(peak_batch, len(running))

    if forced_over_budget:
        warnings.append(
            f"{forced_over_budget} decode step(s) ran a single request over the "
            "KV watermark (nothing left to preempt); results assume it fit the "
            "hard budget"
        )

    # ---- aggregate metrics ----------------------------------------------
    completed.sort(key=lambda rec: rec.idx)
    first_arrival = times[0]
    makespan = clock - first_arrival
    ttfts = sorted(rec.ttft for rec in completed)
    tpots = [rec.tpot for rec in completed]
    gaps_sorted = sorted(gaps)
    prompts = sorted(rec.prompt_len for rec in completed)
    outputs = sorted(rec.output_len for rec in completed)
    total_output = sum(rec.output_len for rec in completed)
    total_input = sum(rec.prompt_len for rec in completed)
    mixed = len(set(prompts)) > 1 or len(set(outputs)) > 1

    achieved = n_total / makespan if makespan > 0 else 0.0
    # Saturated = the waiting queue GREW over the arrival window -- the direct
    # queueing-theory definition of not keeping up (achieved under-reads on
    # short runs; a stable system shows an O(1) backlog excursion here).
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
        peak_batch=peak_batch, kv_feasible_batch=kv_feasible,
        peak_kv_bytes=peak_kv, kv_budget_bytes=kv_budget, n_preemptions=n_preempt,
        prompt_p50=int(_percentile(prompts, 50)), prompt_p99=int(_percentile(prompts, 99)),
        output_p50=int(_percentile(outputs, 50)), output_p99=int(_percentile(outputs, 99)),
        mixed_lengths=mixed,
        n_completed=len(completed), n_prefill_iters=n_prefill, n_decode_iters=n_decode,
        makespan=makespan, saturated=saturated,
        requests=completed, warnings=warnings,
    )


# ---- disaggregated serving --------------------------------------------------


def kv_transfer_bytes(model: ModelSpec, n_tokens: int, kv_dtype) -> float:
    """The KV cache bytes `n_tokens` occupy -- the payload that streams from the
    prefill pool to the decode pool.  The whole logical cache (all layers, all kv
    heads), independent of how either pool shards it, so it is a single
    well-defined transfer size."""
    return model.kv_bytes_per_token(kv_dtype) * n_tokens


def kv_transfer_time(bytes_: float, bandwidth: float, latency: float) -> float:
    """One-latency transfer of `bytes_` over a link: bytes/bw + latency (the same
    occupancy-plus-one-flight-time rule the collective closed forms use)."""
    return (bytes_ / bandwidth if bandwidth else 0.0) + latency


def _resolve_transfer(
    system: System, n_pool_chips: int, disagg: DisaggConfig
) -> tuple[float, float]:
    """(bandwidth, latency) of the prefill<->decode link.  Explicit overrides
    win; otherwise resolve from the system with `link_for_group` semantics."""
    if disagg.transfer_bw is not None:
        return disagg.transfer_bw, (disagg.transfer_latency or 0.0)
    link = system.link_for_group(n_pool_chips) if n_pool_chips > 1 else None
    if link is None:
        raise ValueError(
            "cannot resolve a prefill<->decode link for the KV transfer "
            f"({n_pool_chips} pool chips on {system.name} has no interconnect); "
            "pass transfer_bw (and transfer_latency) explicitly"
        )
    lat = disagg.transfer_latency if disagg.transfer_latency is not None else link.latency_s
    return link.bandwidth, lat


class _PrefillReplica:
    __slots__ = ("current", "started", "busy_total")

    def __init__(self) -> None:
        self.current: _Req | None = None
        self.started: float = 0.0
        self.busy_total: float = 0.0


class _DecodeReplica:
    __slots__ = ("running", "iter_batch", "iterating", "iter_start",
                 "kv_used", "kv_reserved", "busy_total")

    def __init__(self) -> None:
        self.running: list[_Req] = []
        self.iter_batch: list[_Req] = []
        self.iterating: bool = False
        self.iter_start: float = 0.0
        self.kv_used: float = 0.0  # actual KV resident (both policies)
        self.kv_reserved: float = 0.0  # reserve policy: committed full footprints
        self.busy_total: float = 0.0


def serve_disagg(
    system: System,
    model: ModelSpec,
    scenario: Scenario,
    config: ServeConfig = ServeConfig(arrival_rate=1.0),
    disagg: DisaggConfig | None = None,
    engine: Engine | None = None,
) -> ServeReport:
    """Simulate prefill/decode disaggregated serving (see `DisaggConfig`).

    `config` supplies the arrival process, mixed request lengths, `max_batch`,
    seed, and KV policy (all reused from the aggregated loop); `disagg` supplies
    the pool structure.  Event-driven multi-replica loop: each prefill/decode
    replica keeps its own clock and a global heap advances the earliest
    completion.  Requests flow arrival -> prefill queue (FIFO to the first free
    prefill replica) -> KV transfer -> decode replica (least-loaded with
    headroom, else a decode-waiting queue) -> pure decode until done.  Under
    `kv_policy="on_demand"` a decode replica that would overflow preempts its
    latest decoder (vLLM recompute); in disagg the victim returns to the FRONT of
    the *prefill* pool to recompute prompt+generated tokens, then re-transfers --
    so a preemption honestly costs a re-prefill AND a re-transfer.
    """
    if disagg is None:
        raise ValueError("serve_disagg requires a DisaggConfig")
    engine = engine or RooflineEngine()
    p_dep, d_dep = disagg.prefill_deployment, disagg.decode_deployment
    validate_deployment(model, p_dep)
    validate_deployment(model, d_dep)
    for label, dep in (("prefill", p_dep), ("decode", d_dep)):
        if dep.pp != 1:
            raise ValueError(
                f"serve_disagg() supports pp=1 only (got {label} pp={dep.pp}); "
                "pipeline-parallel serving is future work. Use tp/ep/adp."
            )
    if config.prefill_chunk is not None:
        raise ValueError(
            "chunked prefill (prefill_chunk) is not supported with --disagg: "
            "exclusive prefill replicas make chunking moot (chunked prefill caps "
            "the interference stall that disaggregation removes outright)"
        )

    n_p, n_d = disagg.n_prefill_replicas, disagg.n_decode_replicas
    pool_chips = n_p * p_dep.replica_chips + n_d * d_dep.replica_chips
    if pool_chips > system.total_chips:
        raise ValueError(
            f"disagg pools need {n_p}x{p_dep.replica_chips} (prefill) + "
            f"{n_d}x{d_dep.replica_chips} (decode) = {pool_chips} chips but "
            f"{system.name} has {system.total_chips}"
        )
    idle_chips = system.total_chips - pool_chips
    warnings: list[str] = []
    if idle_chips:
        warnings.append(
            f"{idle_chips} chip(s) idle: {pool_chips} of {system.total_chips} "
            "partitioned into prefill/decode pools"
        )

    transfer_bw, transfer_lat = _resolve_transfer(system, pool_chips, disagg)
    on_demand = config.kv_policy == "on_demand"

    # ---- decode-replica memory budget -----------------------------------
    chip = system.node.chip
    d_weights = weight_bytes_per_chip(model, d_dep)
    microbatch = config.max_batch / (d_dep.pp * d_dep.ep * d_dep.adp)
    d_activations = 4 * microbatch * model.d_model * d_dep.act_dtype.bytes
    kv_budget = chip.dram.capacity_bytes - d_weights - d_activations
    kv_per_token = kv_cache_bytes_per_chip(model, 1, d_dep)
    usable = (config.kv_watermark * kv_budget) if on_demand else kv_budget
    if usable <= 0:
        raise ValueError(
            f"no KV headroom on a decode replica ({chip.name}): weights "
            f"{d_weights / 1e9:.1f} GB + activations exceed the "
            f"{chip.dram.capacity_bytes / 1e9:.1f} GB DRAM (raise decode tp/adp)"
        )

    # ---- arrivals & lengths (whole-system: dp=1) ------------------------
    rng = random.Random(config.seed)
    reqs_in = _build_requests(config, scenario, 1, rng)
    n_total = len(reqs_in)
    if n_total == 0:
        raise ValueError("no requests to serve")
    times = [r.arrival for r in reqs_in]

    # a request whose full context can't fit a decode replica's hard budget can
    # never complete -- fail loudly (mirrors serve()'s guard #1)
    for r in reqs_in:
        full = (r.prompt_len + r.output_len) * kv_per_token
        if full > kv_budget + _EPS:
            raise ValueError(
                f"request {r.idx} needs {full / 1e9:.2f} GB of KV at full context "
                f"{r.prompt_len + r.output_len} but a decode replica has only "
                f"{kv_budget / 1e9:.2f} GB (raise decode tp/adp or shrink lengths)"
            )
    # a prefill replica must hold one prompt's KV alongside its weights
    p_prompt_kv = kv_cache_bytes_per_chip(model, max(r.prompt_len for r in reqs_in), p_dep)
    if weight_bytes_per_chip(model, p_dep) + p_prompt_kv > chip.dram.capacity_bytes:
        warnings.append(
            "a prefill replica may not fit the largest prompt's KV (weights + "
            "prompt KV exceed per-chip DRAM); raise prefill tp"
        )

    offered_rate_system: float | None = (
        config.arrival_rate if config.arrival_rate is not None
        else (len(times) / (times[-1] - times[0])
              if len(times) > 1 and times[-1] > times[0] else None)
    )
    per_req_max = kv_cache_bytes_per_chip(model, scenario.max_context, d_dep)
    kv_feasible = (min(config.max_batch, int(floor(usable / per_req_max)))
                   if per_req_max > 0 else config.max_batch)

    # ---- precomputed costs ----------------------------------------------
    p_cost_op = _op_coster(system, p_dep, engine)
    p_prefill_cache: dict[int, float] = {}

    def prefill_cost(n_tokens: int) -> float:
        key = _ceil_grid(n_tokens, config.length_grid)
        c = p_prefill_cache.get(key)
        if c is None:
            c = sum(p_cost_op(op) for op in prefill_ops(model, key, p_dep))
            p_prefill_cache[key] = c
        return c

    d_cost_op = _op_coster(system, d_dep, engine)
    dec = _build_decode_cost(system, model, d_dep, config.max_batch, d_cost_op)
    d_kv_dtype = d_dep.kv_dtype

    # ---- event loop -----------------------------------------------------
    preps = [_PrefillReplica() for _ in range(n_p)]
    dreps = [_DecodeReplica() for _ in range(n_d)]
    waiting_prefill: deque[_Req] = deque()
    waiting_decode: deque[_Req] = deque()

    heap: list[tuple[float, int, str, int]] = []
    seq = 0

    def push(when: float, tag: str, ref: int) -> None:
        nonlocal seq
        heapq.heappush(heap, (when, seq, tag, ref))
        seq += 1

    for req in reqs_in:  # arrivals seed the loop (already in time order)
        push(req.arrival, "arrival", req.idx)

    completed: list[RequestRecord] = []
    gaps: list[float] = []
    batch_samples: list[int] = []
    transfer_delays: list[float] = []
    stat = {"n_prefill": 0, "n_decode": 0, "n_preempt": 0, "peak_batch": 0,
            "peak_kv": 0.0, "transfer_bytes": 0.0, "forced": 0,
            "arrivals_seen": 0, "backlog": 0, "victims": False}

    def full_fp(r: _Req) -> float:
        return (r.prompt_len + r.output_len) * kv_per_token

    def dispatch_prefill(now: float) -> None:
        for i, pr in enumerate(preps):
            if pr.current is not None:
                continue
            if not waiting_prefill:
                break
            r = waiting_prefill.popleft()
            r.needs_prefill = True
            r.prefill_done = 0
            r.context = 0
            pr.current = r
            pr.started = now
            stat["n_prefill"] += 1
            push(now + prefill_cost(r.prefill_target), "prefill_done", i)

    def can_admit(dr: _DecodeReplica, r: _Req) -> bool:
        if len(dr.running) >= config.max_batch:
            return False
        if on_demand:
            committed = sum(x.context for x in dr.running) * kv_per_token
            return committed + r.context * kv_per_token <= usable + _EPS
        return dr.kv_reserved + full_fp(r) <= kv_budget + _EPS

    def admit_to_decode(dr: _DecodeReplica, r: _Req) -> None:
        dr.running.append(r)
        dr.kv_used += r.context * kv_per_token
        if not on_demand:
            dr.kv_reserved += full_fp(r)
        stat["peak_kv"] = max(stat["peak_kv"], dr.kv_used)
        stat["peak_batch"] = max(stat["peak_batch"], len(dr.running))

    def preempt_from_decode(dr: _DecodeReplica, v: _Req) -> None:
        # free the victim's KV and send it to the FRONT of the PREFILL pool to
        # recompute prompt+generated tokens, then re-transfer (the honest disagg
        # preemption penalty).  gen is kept, so prefill_target = prompt + gen.
        dr.kv_used -= v.context * kv_per_token
        dr.running.remove(v)
        v.needs_prefill = True
        v.prefill_done = 0
        v.context = 0
        v.preemptions += 1
        waiting_prefill.appendleft(v)
        stat["n_preempt"] += 1
        stat["victims"] = True

    def start_decode_iter(dr: _DecodeReplica, i: int, now: float) -> None:
        if on_demand:
            while (len(dr.running) > 1
                   and dr.kv_used + len(dr.running) * kv_per_token > usable + _EPS):
                preempt_from_decode(dr, max(dr.running, key=lambda x: x.idx))
            if len(dr.running) == 1 and dr.kv_used + kv_per_token > usable + _EPS:
                stat["forced"] += 1  # nothing left to preempt; run over the watermark
        if not dr.running:
            return
        n = len(dr.running)
        dr.iter_batch = list(dr.running)
        dr.iterating = True
        dr.iter_start = now
        stat["peak_batch"] = max(stat["peak_batch"], n)
        push(now + dec.iter_time(n, sum(r.context for r in dr.running)), "decode_done", i)

    def dispatch_decode(now: float) -> None:
        # admit waiting requests onto the least-loaded replica with headroom
        while waiting_decode:
            r = waiting_decode[0]
            best_i, best_load = -1, -1
            for i, dr in enumerate(dreps):
                if can_admit(dr, r) and (best_i < 0 or len(dr.running) < best_load):
                    best_i, best_load = i, len(dr.running)
            if best_i < 0:
                break
            admit_to_decode(dreps[best_i], waiting_decode.popleft())
        # progress guarantee: force the head onto an empty replica (it fits the
        # hard budget by guard #1) so a watermark-gated request can't wait forever
        if waiting_decode:
            for dr in dreps:
                if not dr.running:
                    admit_to_decode(dr, waiting_decode.popleft())
                    if not waiting_decode:
                        break
        # start every non-empty idle replica (on_demand preemption happens inside)
        stat["victims"] = False
        for i, dr in enumerate(dreps):
            if dr.running and not dr.iterating:
                start_decode_iter(dr, i, now)
        if stat["victims"]:
            dispatch_prefill(now)  # pick up any preempted victims

    def finish(dr: _DecodeReplica, r: _Req, now: float) -> None:
        tpot = (now - r.first_token_time) / r.output_len if r.output_len else 0.0
        completed.append(RequestRecord(
            idx=r.idx, arrival=r.arrival, prompt_len=r.prompt_len,
            output_len=r.output_len, ttft=r.ttft, completion=now - r.arrival,
            tpot=tpot, n_preemptions=r.preemptions, transfer=r.transfer_delay,
        ))
        dr.running.remove(r)
        dr.kv_used -= r.context * kv_per_token
        if not on_demand:
            dr.kv_reserved -= full_fp(r)

    # livelock guard: every prefill/decode/transfer step advances time, but a
    # pathological preemption loop should still bail rather than spin forever.
    event_cap = 1000 * n_total + 200 * sum(r.prompt_len + r.output_len for r in reqs_in)
    events = 0
    last_time = times[0]

    while len(completed) < n_total and heap:
        events += 1
        if events > event_cap:
            raise ValueError(
                "serve_disagg() did not terminate within the event budget -- "
                "likely a preemption livelock; inspect the construction"
            )
        now, _, tag, ref = heapq.heappop(heap)
        last_time = now
        if tag == "arrival":
            waiting_prefill.append(reqs_in[ref])
            stat["arrivals_seen"] += 1
            if stat["arrivals_seen"] == n_total:
                stat["backlog"] = len(waiting_prefill) + len(waiting_decode)
            dispatch_prefill(now)
        elif tag == "prefill_done":
            pr = preps[ref]
            r = pr.current
            assert r is not None
            r.needs_prefill = False
            r.context = r.prefill_target  # KV rebuilt: prompt (+ gen on recompute)
            pr.busy_total += now - pr.started
            pr.current = None
            tbytes = kv_transfer_bytes(model, r.context, d_kv_dtype)
            tdelay = kv_transfer_time(tbytes, transfer_bw, transfer_lat)
            transfer_delays.append(tdelay)
            stat["transfer_bytes"] += tbytes
            if not r.started:
                r.transfer_delay = tdelay  # the TTFT-contributing transfer
            push(now + tdelay, "transfer_done", r.idx)
            dispatch_prefill(now)
        elif tag == "transfer_done":
            r = reqs_in[ref]
            if not r.started:
                r.started = True
                r.ttft = now - r.arrival  # arrival -> prefill done -> transfer done
                r.first_token_time = now
                r.last_emit = now
            # recompute: keep ttft/first_token_time and DO NOT reset last_emit, so
            # the victim's next inter-token gap includes the whole re-prefill +
            # re-transfer stall.
            waiting_decode.append(r)
            dispatch_decode(now)
        else:  # decode_done
            dr = dreps[ref]
            batch = dr.iter_batch
            n = len(batch)
            stat["n_decode"] += 1
            batch_samples.append(n)
            dr.busy_total += now - dr.iter_start
            done: list[_Req] = []
            for r in batch:
                r.gen += 1
                gaps.append(now - r.last_emit)
                r.last_emit = now
                r.context += 1
                dr.kv_used += kv_per_token
                if r.gen >= r.output_len:
                    done.append(r)
            stat["peak_kv"] = max(stat["peak_kv"], dr.kv_used)
            for r in done:
                finish(dr, r, now)
            dr.iterating = False
            dr.iter_batch = []
            dispatch_decode(now)

    if stat["forced"]:
        warnings.append(
            f"{stat['forced']} decode step(s) ran a single request over the KV "
            "watermark (nothing left to preempt); assumed it fit the hard budget"
        )

    # ---- aggregate metrics ----------------------------------------------
    completed.sort(key=lambda rec: rec.idx)
    first_arrival = times[0]
    last_completion = max((rec.arrival + rec.completion for rec in completed),
                          default=last_time)
    makespan = last_completion - first_arrival
    ttfts = sorted(rec.ttft for rec in completed)
    tpots = [rec.tpot for rec in completed]
    gaps_sorted = sorted(gaps)
    transfers_sorted = sorted(transfer_delays)
    prompts = sorted(rec.prompt_len for rec in completed)
    outputs = sorted(rec.output_len for rec in completed)
    total_output = sum(rec.output_len for rec in completed)
    total_input = sum(rec.prompt_len for rec in completed)
    mixed = len(set(prompts)) > 1 or len(set(outputs)) > 1

    achieved_system = n_total / makespan if makespan > 0 else 0.0
    prefill_busy = sum(pr.busy_total for pr in preps)
    decode_busy = sum(dr.busy_total for dr in dreps)
    saturated = stat["backlog"] > max(4, ceil(0.1 * n_total))

    def per_replica(v: float) -> float:
        return v / n_d

    return ServeReport(
        system=system, model=model, scenario=scenario, deployment=d_dep,
        config=config, dp=n_d, idle_chips=idle_chips,
        offered_rate_replica=(None if offered_rate_system is None
                              else per_replica(offered_rate_system)),
        achieved_rate_replica=per_replica(achieved_system),
        backlog_at_last_arrival=stat["backlog"],
        output_tokens_per_s_replica=per_replica(
            total_output / makespan if makespan > 0 else 0.0),
        input_tokens_per_s_replica=per_replica(
            total_input / makespan if makespan > 0 else 0.0),
        ttft_mean=sum(ttfts) / len(ttfts) if ttfts else 0.0,
        ttft_p50=_percentile(ttfts, 50), ttft_p95=_percentile(ttfts, 95),
        ttft_p99=_percentile(ttfts, 99),
        tpot_mean=sum(tpots) / len(tpots) if tpots else 0.0,
        itg_mean=sum(gaps) / len(gaps) if gaps else 0.0,
        itg_p50=_percentile(gaps_sorted, 50), itg_p99=_percentile(gaps_sorted, 99),
        mean_batch=sum(batch_samples) / len(batch_samples) if batch_samples else 0.0,
        peak_batch=stat["peak_batch"], kv_feasible_batch=kv_feasible,
        peak_kv_bytes=stat["peak_kv"], kv_budget_bytes=kv_budget,
        n_preemptions=stat["n_preempt"],
        prompt_p50=int(_percentile(prompts, 50)), prompt_p99=int(_percentile(prompts, 99)),
        output_p50=int(_percentile(outputs, 50)), output_p99=int(_percentile(outputs, 99)),
        mixed_lengths=mixed,
        n_completed=len(completed), n_prefill_iters=stat["n_prefill"],
        n_decode_iters=stat["n_decode"], makespan=makespan, saturated=saturated,
        disagg=disagg, prefill_deployment=p_dep,
        n_prefill_replicas=n_p, n_decode_replicas=n_d,
        prefill_util=(prefill_busy / (makespan * n_p) if makespan > 0 else 0.0),
        decode_util=(decode_busy / (makespan * n_d) if makespan > 0 else 0.0),
        transfer_mean=sum(transfer_delays) / len(transfer_delays) if transfer_delays else 0.0,
        transfer_p99=_percentile(transfers_sorted, 99),
        transfer_bytes_total=stat["transfer_bytes"],
        transfer_bw=transfer_bw, transfer_latency=transfer_lat,
        requests=completed, warnings=warnings,
    )


# ---- rendering --------------------------------------------------------------


def format_serve_report(r: ServeReport) -> str:
    """Plain-text serve report, in the style of report.format_report."""
    from .units import fmt_bytes, fmt_si, fmt_time

    s, m, sc, d, cfg = r.system, r.model, r.scenario, r.deployment, r.config
    lines: list[str] = []
    add = lines.append
    disagg = r.disagg is not None

    def _pool(dep: Deployment) -> str:
        return (f"TP={dep.tp}" + (f" EP={dep.ep}" if dep.ep > 1 else "")
                + (f" ADP={dep.adp}" if dep.adp > 1 else "")
                + f" ({dep.replica_chips} chips/replica)")

    add("=" * 72)
    add(f"inferencesim serve{' (disagg)' if disagg else ''}  |  {s.name}  x  {m.name}")
    add("=" * 72)
    if disagg:
        add(f"Prefill pool : {r.n_prefill_replicas}x  {_pool(r.prefill_deployment)}")
        add(f"Decode pool  : {r.n_decode_replicas}x  {_pool(d)}"
            + (f"  ({r.idle_chips} chips idle)" if r.idle_chips else ""))
    else:
        add(f"Parallelism  : TP={d.tp}  EP={d.ep}"
            + (f"  ADP={d.adp}" if d.adp > 1 else "")
            + f"  DP={r.dp}  ({d.replica_chips} chips/replica)"
            + (f"  ({r.idle_chips} chips idle)" if r.idle_chips else ""))
    if r.mixed_lengths:
        lengths = (f"prompt p50/p99 {r.prompt_p50}/{r.prompt_p99}, "
                   f"output p50/p99 {r.output_p50}/{r.output_p99} (mixed)")
    else:
        lengths = f"prompt={sc.prompt_len}, output={sc.output_len}"
    add(f"Scenario     : {lengths}, max_batch={cfg.max_batch}")
    if disagg:
        add(f"Admission    : kv_policy={cfg.kv_policy}"
            + (f" (watermark {cfg.kv_watermark:.2f})" if cfg.kv_policy == "on_demand" else "")
            + ", prefill=disaggregated (exclusive per prefill replica)")
    else:
        prefill_mode = ("exclusive" if cfg.prefill_chunk is None
                        else f"chunked/{cfg.prefill_chunk}")
        add(f"Admission    : kv_policy={cfg.kv_policy}"
            + (f" (watermark {cfg.kv_watermark:.2f})" if cfg.kv_policy == "on_demand" else "")
            + f", prefill={prefill_mode}, prefill_first={cfg.prefill_first}")
    arrival = (f"Poisson {cfg.arrival_rate} req/s system" if cfg.arrival_rate is not None
               else f"trace of {r.n_completed} arrivals")
    add(f"Arrivals     : {arrival}, seed={cfg.seed}")
    add("-" * 72)
    unit = "decode-replica" if disagg else "replica"
    off_r = "n/a" if r.offered_rate_replica is None else f"{r.offered_rate_replica:.3f}"
    off_s = "n/a" if r.offered_rate_system is None else f"{r.offered_rate_system:.3f}"
    add(f"Offered load : {off_r} req/s/{unit}  ({off_s} req/s system)")
    add(f"Achieved     : {r.achieved_rate_replica:.3f} req/s/{unit}  "
        f"({r.achieved_rate_system:.3f} req/s system, incl. drain tail)"
        + ("   ** SATURATED **" if r.saturated else ""))
    if r.backlog_at_last_arrival:
        add(f"Backlog      : {r.backlog_at_last_arrival} waiting at last arrival "
            f"(stable systems show an O(1) excursion here)")
    add(f"Throughput   : {fmt_si(r.output_tokens_per_s_replica, 'tok/s')} output/{unit}  "
        f"({fmt_si(r.output_tokens_per_s_system, 'tok/s')} system)")
    if disagg:
        add(f"Pool util    : prefill {r.prefill_util * 100:.1f}%  "
            f"decode {r.decode_util * 100:.1f}%  "
            f"(prefill~1 & decode low = prefill-starved; the reverse = decode-bound)")
    add("-" * 72)
    add(f"TTFT         : mean {fmt_time(r.ttft_mean)}  p50 {fmt_time(r.ttft_p50)}  "
        f"p95 {fmt_time(r.ttft_p95)}  p99 {fmt_time(r.ttft_p99)}"
        + ("  (incl. KV transfer)" if disagg else ""))
    if disagg:
        add(f"KV transfer  : mean {fmt_time(r.transfer_mean)}  p99 {fmt_time(r.transfer_p99)}  "
            f"({fmt_bytes(r.transfer_bytes_total)} total @ "
            f"{fmt_si(r.transfer_bw, 'B/s')}, {fmt_time(r.transfer_latency)} lat)")
    add(f"TPOT         : mean {fmt_time(r.tpot_mean)}/token"
        + (f"  ->  {1.0 / r.tpot_mean:.1f} tok/s/request" if r.tpot_mean > 0 else ""))
    itg_tail = "  (interference eliminated: pure decode)" if disagg else ""
    add(f"Inter-token  : mean {fmt_time(r.itg_mean)}  p50 {fmt_time(r.itg_p50)}  "
        f"p99 {fmt_time(r.itg_p99)}"
        + (f"  (p99/median {r.itg_p99 / r.itg_p50:.1f}x){itg_tail}" if r.itg_p50 > 0 else ""))
    add("-" * 72)
    add(f"Batch occ.   : mean {r.mean_batch:.1f}, peak {r.peak_batch} "
        + (f"per decode replica (feasible {r.kv_feasible_batch})" if disagg
           else f"(feasible {r.kv_feasible_batch})"))
    add(f"KV cache     : peak {fmt_bytes(r.peak_kv_bytes)} / {fmt_bytes(r.kv_budget_bytes)} "
        f"budget per {'decode ' if disagg else ''}chip"
        + (f",  {r.n_preemptions} preemption(s)" if r.n_preemptions else ""))
    add(f"Iterations   : {r.n_prefill_iters} prefill + {r.n_decode_iters} decode, "
        f"{r.n_completed} completed over {fmt_time(r.makespan)}")
    for w in r.warnings:
        add(f"WARNING      : {w}")
    add("=" * 72)
    return "\n".join(lines)
