"""Request-level serving simulation (serve.py).

Validation philosophy (DES_todo.md): every metric is pinned by a degenerate
oracle that reproduces the analytic model, plus monotonicity/interference
checks for the behaviour that only the event loop can produce.
"""

from math import floor

import pytest

from inferencesim.engine import RooflineEngine
from inferencesim.hardware import DType
from inferencesim.ops import kv_cache_bytes_per_chip
from inferencesim.presets import GB300_NVL72, GPT_OSS_120B, H100_SINGLE, LLAMA_3_1_70B
from inferencesim.serve import (
    LengthDist,
    ServeConfig,
    _build_decode_cost,
    _op_coster,
    chunked_prefill_ttft,
    decode_iteration_time,
    prefill_iteration_time,
    serve,
)
from inferencesim.simulate import simulate, weight_bytes_per_chip
from inferencesim.workload import Deployment, Scenario

GB300 = GB300_NVL72
DEP = Deployment(tp=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8)


# ---- 1. single-request oracle (both KV policies) ---------------------------


@pytest.mark.parametrize("policy", ["on_demand", "reserve"])
def test_single_request_matches_analytic(policy):
    """A lone request never queues, so its TTFT must equal simulate.py's
    analytic prefill time, and its total time must equal ttft plus the
    telescoped decode cost.  The per-step attention cost is affine in context,
    so the sum of O growing-context steps collapses to O x the mean-context
    step exactly.  With no memory pressure the identity must hold under BOTH KV
    policies (admission accounting differs, timing does not)."""
    scen = Scenario(batch=64, prompt_len=4096, output_len=200)
    r = serve(GB300, LLAMA_3_1_70B, scen, DEP,
              ServeConfig(arrivals=[0.0], max_batch=64, kv_policy=policy))
    rec = r.requests[0]
    assert rec.n_preemptions == 0

    ttft_analytic = simulate(GB300, LLAMA_3_1_70B, scen, DEP).ttft_s
    assert rec.ttft == pytest.approx(ttft_analytic, rel=1e-9)

    mean_ctx = scen.prompt_len + (scen.output_len - 1) / 2.0
    decode_total = scen.output_len * decode_iteration_time(
        GB300, LLAMA_3_1_70B, DEP, 1, mean_ctx
    )
    assert rec.completion == pytest.approx(rec.ttft + decode_total, rel=1e-9)


def test_chunked_single_request_ttft_is_sum_of_chunks():
    """A lone chunked-prefill request emits its first token when the last chunk
    lands, so its TTFT is exactly the sum of the per-chunk iteration costs -- and
    that sum EXCEEDS the exclusive prefill time, because every chunk re-streams
    the weights and re-reads the growing KV cache."""
    prompt, chunk = 4096, 512
    scen = Scenario(batch=64, prompt_len=prompt, output_len=8)
    r = serve(GB300, LLAMA_3_1_70B, scen, DEP,
              ServeConfig(arrivals=[0.0], max_batch=64, prefill_chunk=chunk))
    rec = r.requests[0]
    expected = chunked_prefill_ttft(GB300, LLAMA_3_1_70B, DEP, prompt, chunk)
    assert rec.ttft == pytest.approx(expected, rel=1e-9)
    assert rec.ttft > prefill_iteration_time(GB300, LLAMA_3_1_70B, DEP, prompt)


# ---- 2. independent requests far apart --------------------------------------


def test_two_requests_far_apart_are_independent():
    """Two requests separated by a huge gap never share the batch, so each
    reproduces the single-request numbers exactly."""
    scen = Scenario(batch=64, prompt_len=2048, output_len=128)
    solo = serve(GB300, LLAMA_3_1_70B, scen, DEP, ServeConfig(arrivals=[0.0], max_batch=64))
    # A gap of 100 s (>> the ~0.4 s a single request takes) guarantees no
    # overlap without the fp cancellation a huge offset like 1e6 would inject
    # into the second request's ttft = (arrival + prefill) - arrival.
    pair = serve(GB300, LLAMA_3_1_70B, scen, DEP,
                 ServeConfig(arrivals=[0.0, 100.0], max_batch=64))
    s = solo.requests[0]
    for rec in pair.requests:
        assert rec.ttft == pytest.approx(s.ttft, rel=1e-9)
        assert rec.completion == pytest.approx(s.completion, rel=1e-9)
        assert rec.tpot == pytest.approx(s.tpot, rel=1e-9)


# ---- 3. saturation vs the decode-only ceiling -------------------------------


def test_saturation_approaches_decode_only_ceiling():
    """Firehose arrivals (all at t=0) and n >> max_batch drive the replica to a
    near-full decoding batch.  Achieved output throughput lands just below
    simulate.py's decode-only ceiling: it is NOT exact because prefill
    iterations steal time from decode (and the run ramps up/drains).  The
    observed shortfall here is ~9%; the band is loose enough to absorb the ramp
    but still asserts the loop can't beat the analytic ceiling."""
    scen = Scenario(batch=64, prompt_len=256, output_len=2048)
    rep = simulate(GB300, LLAMA_3_1_70B, scen, DEP)
    ceiling_replica = rep.decode_only_tokens_per_s / rep.dp
    r = serve(GB300, LLAMA_3_1_70B, scen, DEP,
              ServeConfig(arrivals=[0.0] * 300, max_batch=64, seed=0))
    ratio = r.output_tokens_per_s_replica / ceiling_replica
    assert r.peak_batch == 64
    assert 0.75 < ratio < 1.0  # observed ~0.91: prefill steals the rest


# ---- 4. determinism ---------------------------------------------------------


def _fingerprint(r):
    return (r.ttft_p99, r.achieved_rate_replica, r.itg_p99, r.peak_kv_bytes,
            r.output_tokens_per_s_replica)


def test_determinism():
    scen = Scenario(batch=64, prompt_len=1024, output_len=256)
    cfg = dict(arrival_rate=30.0, n_requests=80, max_batch=64)
    a = serve(GB300, LLAMA_3_1_70B, scen, DEP, ServeConfig(seed=0, **cfg))
    b = serve(GB300, LLAMA_3_1_70B, scen, DEP, ServeConfig(seed=0, **cfg))
    c = serve(GB300, LLAMA_3_1_70B, scen, DEP, ServeConfig(seed=1, **cfg))
    assert _fingerprint(a) == _fingerprint(b)  # same seed -> identical
    assert _fingerprint(a) != _fingerprint(c)  # different seed -> different


# ---- 5. queueing monotonicity -----------------------------------------------


def test_ttft_p99_grows_with_load():
    """p99 TTFT (which includes queue waiting) is strictly larger near
    saturation than at a light load -- the whole point of a request-level
    arrival process."""
    scen = Scenario(batch=64, prompt_len=4096, output_len=1024)
    sat = simulate(GB300, LLAMA_3_1_70B, scen, DEP).requests_per_s  # system req/s
    low = serve(GB300, LLAMA_3_1_70B, scen, DEP,
                ServeConfig(arrival_rate=0.4 * sat, n_requests=200, max_batch=64, seed=1))
    high = serve(GB300, LLAMA_3_1_70B, scen, DEP,
                 ServeConfig(arrival_rate=0.9 * sat, n_requests=200, max_batch=64, seed=1))
    assert high.ttft_p99 > low.ttft_p99


# ---- 6. prefill/decode interference -----------------------------------------


def test_prefill_interference_spikes_inter_token_gap():
    """With prefill_first, a prefill iteration stalls the decoding batch, so
    some inter-token gaps are ~prefill-sized (>> mean TPOT).  A stream that
    keeps a small batch decoding while fresh prefills land shows p99 gap many x
    the median; a single request with nothing to interrupt it does not."""
    scen = Scenario(batch=64, prompt_len=4096, output_len=64)
    inter = serve(GB300, LLAMA_3_1_70B, scen, DEP,
                  ServeConfig(arrival_rate=40.0, n_requests=120, max_batch=64, seed=0))
    assert inter.itg_p99 > 3.0 * inter.itg_p50  # observed ~11.6x

    control = serve(GB300, LLAMA_3_1_70B, scen, DEP,
                    ServeConfig(arrivals=[0.0], max_batch=64))
    assert control.itg_p99 < 1.5 * control.itg_p50  # uninterrupted -> flat


# ---- 7. KV-cap admission ----------------------------------------------------


def _h100_kv_setup():
    dep = Deployment(tp=1, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    scen = Scenario(batch=64, prompt_len=2048, output_len=512)
    chip = H100_SINGLE.node.chip
    weights = weight_bytes_per_chip(LLAMA_3_1_70B, dep)
    act = 4 * 64 * LLAMA_3_1_70B.d_model * dep.act_dtype.bytes  # pp=ep=1
    budget = chip.dram.capacity_bytes - weights - act
    per_req = kv_cache_bytes_per_chip(LLAMA_3_1_70B, scen.max_context, dep)
    return dep, scen, budget, per_req


def test_reserve_kv_budget_caps_concurrency_and_never_preempts():
    """A single H100 holds ~70 GB of llama-70b weights, leaving only a few GB
    for KV.  With `reserve`, admission charges the full prompt+output footprint,
    so concurrency is capped at floor(budget / full-footprint) and the loop
    never preempts."""
    dep, scen, budget, per_req = _h100_kv_setup()
    expected = int(floor(budget / per_req))
    assert 1 < expected < 64  # genuinely memory-limited

    r = serve(H100_SINGLE, LLAMA_3_1_70B, scen, dep,
              ServeConfig(arrivals=[0.0] * 40, max_batch=64, seed=0, kv_policy="reserve"))
    assert r.kv_feasible_batch == expected
    assert r.peak_batch == expected  # demand saturates the KV-feasible max
    assert r.peak_batch < 64
    assert r.n_preemptions == 0


def test_on_demand_admits_more_than_reserve():
    """On the same tight construction, `on_demand` admits against only the
    prompt footprint, so it packs strictly more requests concurrently than
    `reserve` -- at the cost of preempting as the decoders grow."""
    dep, scen, _, _ = _h100_kv_setup()
    reserve = serve(H100_SINGLE, LLAMA_3_1_70B, scen, dep,
                    ServeConfig(arrivals=[0.0] * 40, max_batch=64, kv_policy="reserve"))
    on_demand = serve(H100_SINGLE, LLAMA_3_1_70B, scen, dep,
                      ServeConfig(arrivals=[0.0] * 40, max_batch=64, kv_policy="on_demand"))
    assert on_demand.peak_batch > reserve.peak_batch
    assert on_demand.n_preemptions > 0
    assert on_demand.n_completed == reserve.n_completed == 40


def test_on_demand_preempts_completes_and_delays_the_victim():
    """Under memory pressure on_demand preempts (recompute), yet every request
    completes with the full output, and a preempted request finishes later than
    it would uncontended -- the recompute penalty is real."""
    dep, scen, _, _ = _h100_kv_setup()
    contended = serve(H100_SINGLE, LLAMA_3_1_70B, scen, dep,
                      ServeConfig(arrivals=[0.0] * 40, max_batch=64, seed=0))
    assert contended.n_preemptions > 0
    assert contended.n_completed == 40
    assert all(rec.output_len == scen.output_len for rec in contended.requests)

    solo = serve(H100_SINGLE, LLAMA_3_1_70B, scen, dep,
                 ServeConfig(arrivals=[0.0], max_batch=64))
    lone_completion = solo.requests[0].completion
    preempted = [rec for rec in contended.requests if rec.n_preemptions > 0]
    assert preempted  # some request was recomputed
    # every recomputed request finishes far later than an uncontended run
    assert all(rec.completion > lone_completion for rec in preempted)


def test_full_context_over_hard_budget_raises():
    """Guard #1: a request whose full prompt+output KV exceeds the hard DRAM
    budget can never complete -- fail loudly instead of thrashing."""
    dep = Deployment(tp=1, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    scen = Scenario(batch=1, prompt_len=2048, output_len=200_000)  # ~33 GB of KV
    with pytest.raises(ValueError, match="can never fit"):
        serve(H100_SINGLE, LLAMA_3_1_70B, scen, dep,
              ServeConfig(arrivals=[0.0], max_batch=8))


def test_over_watermark_single_request_force_completes_with_warning():
    """Guard #2: a request that fits the hard budget but exceeds the on_demand
    watermark has nothing to preempt when it runs alone, so it force-completes
    over the watermark and the run carries a warning (not an error)."""
    dep = Deployment(tp=1, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    chip = H100_SINGLE.node.chip
    budget = chip.dram.capacity_bytes - weight_bytes_per_chip(LLAMA_3_1_70B, dep) \
        - 4 * 8 * LLAMA_3_1_70B.d_model * dep.act_dtype.bytes
    per_tok = kv_cache_bytes_per_chip(LLAMA_3_1_70B, 1, dep)
    tokens = int(0.7 * budget / per_tok)  # between watermark (0.5) and hard budget
    scen = Scenario(batch=1, prompt_len=1024, output_len=tokens - 1024)
    r = serve(H100_SINGLE, LLAMA_3_1_70B, scen, dep,
              ServeConfig(arrivals=[0.0], max_batch=8, kv_watermark=0.5))
    assert r.n_completed == 1
    assert any("over the KV watermark" in w for w in r.warnings)


# ---- mixed request lengths + chunked prefill --------------------------------


def _long_among_short(prefill_chunk):
    """24 short requests decoding when one 16k-prompt request lands."""
    scen = Scenario(batch=64, prompt_len=512, output_len=64)
    n_short = 24
    arrivals = [0.0] * n_short + [0.08]
    prompts = [512] * n_short + [16384]
    outputs = [64] * (n_short + 1)
    return serve(GB300, LLAMA_3_1_70B, scen, DEP, ServeConfig(
        arrivals=arrivals, prompt_lens=prompts, output_lens=outputs,
        max_batch=64, prefill_chunk=prefill_chunk))


def test_chunked_prefill_collapses_the_long_prefill_stall():
    """A 16k-prompt request among short ones: exclusive prefill freezes the whole
    decode batch for the length of the long prefill (a huge inter-token gap),
    while chunked prefill smears it across decode iterations -- so the itg p99
    collapses.  The price is the long request's own TTFT, which stretches because
    each chunk re-reads its growing KV."""
    exclusive = _long_among_short(prefill_chunk=None)
    chunked = _long_among_short(prefill_chunk=512)
    assert exclusive.mixed_lengths and chunked.mixed_lengths
    # itg p99 collapses (observed ~17x); require a comfortable margin
    assert exclusive.itg_p99 > 5.0 * chunked.itg_p99
    long_excl = max(exclusive.requests, key=lambda r: r.prompt_len)
    long_chunk = max(chunked.requests, key=lambda r: r.prompt_len)
    assert long_excl.prompt_len == long_chunk.prompt_len == 16384
    assert long_chunk.ttft > long_excl.ttft  # the TTFT price of chunking


def test_determinism_with_length_distributions():
    """Sampling prompt/output lengths from the same seeded RNG is reproducible;
    a different seed changes the draw."""
    cfg = dict(arrival_rate=25.0, n_requests=60, max_batch=64,
               prompt_dist=LengthDist.uniform(256, 4096),
               output_dist=LengthDist.lognormal(256, 0.6))
    a = serve(GB300, LLAMA_3_1_70B, Scenario(batch=64, prompt_len=1, output_len=1),
              DEP, ServeConfig(seed=0, **cfg))
    b = serve(GB300, LLAMA_3_1_70B, Scenario(batch=64, prompt_len=1, output_len=1),
              DEP, ServeConfig(seed=0, **cfg))
    c = serve(GB300, LLAMA_3_1_70B, Scenario(batch=64, prompt_len=1, output_len=1),
              DEP, ServeConfig(seed=1, **cfg))
    assert a.mixed_lengths
    assert _fingerprint(a) == _fingerprint(b)
    assert _fingerprint(a) != _fingerprint(c)
    assert (a.prompt_p50, a.prompt_p99) != (a.output_p50, a.output_p99)


# ---- 8. MoE per-iteration expected-active-experts nonlinearity ---------------


def test_moe_active_experts_nonlinearity_is_live():
    """gpt-oss decode cost is dominated by streaming the *activated* experts,
    and the expected number of distinct experts is nonlinear in batch (it
    saturates).  So an 8-token step costs far less than 8 single-token steps --
    proof the loop evaluates expected_active_experts at the actual batch each
    iteration rather than scaling a fixed unit cost linearly."""
    dep = Deployment(tp=4, ep=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
    ctx = 2048.0
    c1 = decode_iteration_time(GB300, GPT_OSS_120B, dep, 1, 1 * ctx)
    c8 = decode_iteration_time(GB300, GPT_OSS_120B, dep, 8, 8 * ctx)
    assert c8 < 2.0 * c1  # strongly sublinear (observed c8 ~ 1.1 x c1)
    assert c8 < 0.5 * (8 * c1)  # nowhere near a linear 8x scale-up

    scen = Scenario(batch=64, prompt_len=2048, output_len=256)
    r = serve(GB300, GPT_OSS_120B, scen, dep,
              ServeConfig(arrival_rate=20.0, n_requests=80, max_batch=64, seed=0))
    assert r.output_tokens_per_s_system > 0
    assert r.n_completed == 80


# ---- 9. pp > 1 is rejected --------------------------------------------------


def test_pipeline_parallel_is_rejected():
    scen = Scenario(batch=8, prompt_len=512, output_len=64)
    dep = Deployment(tp=2, pp=2, weight_dtype=DType.FP8)
    with pytest.raises(ValueError, match="pp=1 only"):
        serve(GB300, LLAMA_3_1_70B, scen, dep, ServeConfig(arrivals=[0.0], max_batch=8))


# ---- cost-model exactness (precomputed table == direct lowering) ------------


def test_decode_cost_table_matches_direct_lowering():
    """The loop uses a per-batch table for the context-independent decode ops
    plus a per-iteration attention recost; that must reproduce a fresh full
    lowering of the decode step to full precision, for any (batch, context)."""
    max_batch = 32
    cost_op = _op_coster(GB300, DEP, RooflineEngine())
    table = _build_decode_cost(GB300, LLAMA_3_1_70B, DEP, max_batch, cost_op)
    for n, mean_ctx in [(1, 3000.0), (7, 1500.0), (32, 5000.0)]:
        direct = decode_iteration_time(GB300, LLAMA_3_1_70B, DEP, n, n * mean_ctx)
        assert table.iter_time(n, n * mean_ctx) == pytest.approx(direct, rel=1e-9)


# ---- MoE expert-load skew ---------------------------------------------------


def test_moe_serve_skew_lowers_throughput():
    """A MoE serve run under expert-load skew completes, but with lower
    throughput than the uniform (skew=0) run: the hot member paces moe_routed's
    weight streaming, so every decode iteration is slower.  The skew flows in
    automatically through the per-batch decode cost table (which re-lowers the
    MoE ops), no serve-loop change needed."""
    from dataclasses import replace

    dep = Deployment(tp=1, ep=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
    scen = Scenario(batch=64, prompt_len=1024, output_len=256)
    base = GPT_OSS_120B
    skewed = replace(base, moe=replace(base.moe, skew=1.0))
    cfg = ServeConfig(arrivals=[0.0] * 120, max_batch=64, seed=0)  # firehose -> saturated
    r0 = serve(GB300, base, scen, dep, cfg)
    r1 = serve(GB300, skewed, scen, dep, cfg)
    assert r1.n_completed == r0.n_completed == 120  # both complete
    assert r1.output_tokens_per_s_system < r0.output_tokens_per_s_system
    assert r1.tpot_mean > r0.tpot_mean  # slower per-token under skew
