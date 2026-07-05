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
    ttft: float  # arrival -> first token (prefill completion)
    completion: float  # arrival -> last token
    tpot: float  # mean seconds per output token over the decode phase
    n_preemptions: int = 0  # times this request was preempted (recompute)


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
            f"replica needs tp*ep={replica} chips but {system.name} has "
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
    microbatch = config.max_batch / (deployment.pp * deployment.ep)
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
    if r.mixed_lengths:
        lengths = (f"prompt p50/p99 {r.prompt_p50}/{r.prompt_p99}, "
                   f"output p50/p99 {r.output_p50}/{r.output_p99} (mixed)")
    else:
        lengths = f"prompt={sc.prompt_len}, output={sc.output_len}"
    add(f"Scenario     : {lengths}, max_batch={cfg.max_batch}")
    prefill_mode = ("exclusive" if cfg.prefill_chunk is None
                    else f"chunked/{cfg.prefill_chunk}")
    add(f"Admission    : kv_policy={cfg.kv_policy}"
        + (f" (watermark {cfg.kv_watermark:.2f})" if cfg.kv_policy == "on_demand" else "")
        + f", prefill={prefill_mode}, prefill_first={cfg.prefill_first}")
    arrival = (f"Poisson {cfg.arrival_rate} req/s system" if cfg.arrival_rate is not None
               else f"trace of {r.n_completed} arrivals")
    add(f"Arrivals     : {arrival}, seed={cfg.seed}")
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
        + (f"  (p99/median {r.itg_p99 / r.itg_p50:.1f}x)" if r.itg_p50 > 0 else ""))
    add("-" * 72)
    add(f"Batch occ.   : mean {r.mean_batch:.1f}, peak {r.peak_batch} "
        f"(feasible {r.kv_feasible_batch})")
    add(f"KV cache     : peak {fmt_bytes(r.peak_kv_bytes)} / {fmt_bytes(r.kv_budget_bytes)} "
        f"budget per chip"
        + (f",  {r.n_preemptions} preemption(s)" if r.n_preemptions else ""))
    add(f"Iterations   : {r.n_prefill_iters} prefill + {r.n_decode_iters} decode, "
        f"{r.n_completed} completed over {fmt_time(r.makespan)}")
    for w in r.warnings:
        add(f"WARNING      : {w}")
    add("=" * 72)
    return "\n".join(lines)
