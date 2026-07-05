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
    ServeConfig,
    _build_decode_cost,
    _op_coster,
    decode_iteration_time,
    serve,
)
from inferencesim.simulate import simulate, weight_bytes_per_chip
from inferencesim.workload import Deployment, Scenario

GB300 = GB300_NVL72
DEP = Deployment(tp=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8)


# ---- 1. single-request oracle ----------------------------------------------


def test_single_request_matches_analytic():
    """A lone request never queues, so its TTFT must equal simulate.py's
    analytic prefill time, and its total time must equal ttft plus the
    telescoped decode cost.  The per-step attention cost is affine in context,
    so the sum of O growing-context steps collapses to O x the mean-context
    step exactly -- the identity that says the cost model is affine in context
    for a dense model."""
    scen = Scenario(batch=64, prompt_len=4096, output_len=200)
    r = serve(GB300, LLAMA_3_1_70B, scen, DEP, ServeConfig(arrivals=[0.0], max_batch=64))
    rec = r.requests[0]

    ttft_analytic = simulate(GB300, LLAMA_3_1_70B, scen, DEP).ttft_s
    assert rec.ttft == pytest.approx(ttft_analytic, rel=1e-9)

    mean_ctx = scen.prompt_len + (scen.output_len - 1) / 2.0
    decode_total = scen.output_len * decode_iteration_time(
        GB300, LLAMA_3_1_70B, DEP, 1, mean_ctx
    )
    assert rec.completion == pytest.approx(rec.ttft + decode_total, rel=1e-9)


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


def test_kv_budget_caps_concurrency():
    """A single H100 holds ~70 GB of llama-70b weights, leaving only a few GB
    for KV: the loop must never admit more concurrent requests than the KV
    budget allows, even with max_batch far higher."""
    dep = Deployment(tp=1, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    scen = Scenario(batch=64, prompt_len=2048, output_len=512)
    chip = H100_SINGLE.node.chip
    weights = weight_bytes_per_chip(LLAMA_3_1_70B, dep)
    act = 4 * 64 * LLAMA_3_1_70B.d_model * dep.act_dtype.bytes  # pp=ep=1
    budget = chip.dram.capacity_bytes - weights - act
    per_req = kv_cache_bytes_per_chip(LLAMA_3_1_70B, scen.max_context, dep)
    expected = int(floor(budget / per_req))
    assert 1 < expected < 64  # genuinely memory-limited

    r = serve(H100_SINGLE, LLAMA_3_1_70B, scen, dep,
              ServeConfig(arrivals=[0.0] * 40, max_batch=64, seed=0))
    assert r.kv_feasible_batch == expected
    assert r.peak_batch == expected  # demand saturates the KV-feasible max
    assert r.peak_batch < 64


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
