"""Prefill/decode disaggregated serving (serve_disagg).

Validation philosophy (DES_todo.md): every metric is pinned by a degenerate
oracle that reproduces the aggregated model, plus the interference-elimination,
pool-starvation, and cross-pool preemption behaviours only the two-pool
architecture can produce.

The lineage is DistServe / NVIDIA Dynamo disaggregation: prefill runs on its own
exclusive replicas, the KV cache streams to a decode pool, and decode never
stalls on a prefill.  serve_disagg reuses serve()'s request construction (mixed
lengths) and BOTH KV policies -- so a lone request through a zero-cost link is
bit-for-bit the aggregated single-request path under either policy.
"""

import pytest

from inferencesim.hardware import DType
from inferencesim.presets import DGX_H100, GB300_NVL72, LLAMA_3_1_70B, NVLINK5
from inferencesim.serve import (
    DisaggConfig,
    ServeConfig,
    kv_transfer_bytes,
    kv_transfer_time,
    prefill_iteration_time,
    serve,
    serve_disagg,
)
from inferencesim.simulate import simulate
from inferencesim.workload import Deployment, Scenario

GB300 = GB300_NVL72
DEP = Deployment(tp=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8)


def _disagg(system, scen, cfg, *, n_p=1, n_d=1, prefill=DEP, decode=DEP,
            transfer_bw=None, transfer_latency=None):
    return serve_disagg(
        system, LLAMA_3_1_70B, scen, cfg,
        DisaggConfig(prefill_deployment=prefill, decode_deployment=decode,
                     n_prefill_replicas=n_p, n_decode_replicas=n_d,
                     transfer_bw=transfer_bw, transfer_latency=transfer_latency),
    )


# ---- 1. degenerate anchor: a lone request through a zero-cost link ----------


@pytest.mark.parametrize("policy", ["on_demand", "reserve"])
def test_single_request_matches_aggregated(policy):
    """One request, 1 prefill + 1 decode replica, zero-cost transfer: the pools
    change nothing for a lone request (which never preempts), so its TTFT and
    completion must equal the aggregated serve()'s to full precision under either
    KV policy -- the transfer, the only new term, is zero here."""
    scen = Scenario(batch=64, prompt_len=4096, output_len=200)
    cfg = ServeConfig(arrivals=[0.0], max_batch=64, kv_policy=policy)
    agg = serve(GB300, LLAMA_3_1_70B, scen, DEP, cfg)
    dis = _disagg(GB300, scen, cfg, transfer_bw=float("inf"), transfer_latency=0.0)
    a, d = agg.requests[0], dis.requests[0]
    assert d.transfer == 0.0
    assert d.n_preemptions == 0
    assert d.ttft == pytest.approx(a.ttft, rel=1e-9)
    assert d.completion == pytest.approx(a.completion, rel=1e-9)
    assert d.tpot == pytest.approx(a.tpot, rel=1e-9)


# ---- 2. transfer accounting is exact ----------------------------------------


def test_transfer_accounting_exact():
    """With a finite-bandwidth link the first token lands at prefill completion
    plus the KV transfer, so TTFT == prefill_cost + kv_bytes/bw + latency to full
    precision (the request never queues)."""
    scen = Scenario(batch=64, prompt_len=4096, output_len=64)
    bw, lat = 900e9, 1e-6
    cfg = ServeConfig(arrivals=[0.0], max_batch=64)
    dis = _disagg(GB300, scen, cfg, transfer_bw=bw, transfer_latency=lat)
    kvb = kv_transfer_bytes(LLAMA_3_1_70B, scen.prompt_len, DEP.kv_dtype)
    tt = kv_transfer_time(kvb, bw, lat)
    pc = prefill_iteration_time(GB300, LLAMA_3_1_70B, DEP, scen.prompt_len)
    rec = dis.requests[0]
    assert rec.transfer == pytest.approx(tt, rel=1e-9)
    assert rec.ttft == pytest.approx(pc + tt, rel=1e-9)
    assert dis.transfer_bytes_total == pytest.approx(kvb, rel=1e-12)


def test_default_link_is_the_node_interconnect():
    """GB300 NVL72 is a single node, so both pools sit inside it and an unset
    transfer link resolves to the node interconnect (NVLink 5)."""
    scen = Scenario(batch=64, prompt_len=2048, output_len=32)
    dis = _disagg(GB300, scen, ServeConfig(arrivals=[0.0]), n_p=2, n_d=2)
    assert dis.transfer_bw == NVLINK5.bandwidth
    assert dis.transfer_latency == NVLINK5.latency_s


# ---- 3. interference elimination (the architectural win) --------------------


def test_disagg_eliminates_prefill_interference():
    """A stream of long prompts among a small decoding batch makes the aggregated
    loop spike its inter-token-gap p99 (a prefill stalls the batch).  Run
    disaggregated, decode replicas do pure decode, so the itg p99 collapses to
    ~the decode iteration time and drops below the aggregated exclusive p99 --
    without paying chunking's TTFT price."""
    scen = Scenario(batch=64, prompt_len=4096, output_len=64)
    cfg = ServeConfig(arrival_rate=40.0, n_requests=120, max_batch=64, seed=0,
                      kv_policy="reserve")
    agg = serve(GB300, LLAMA_3_1_70B, scen, DEP, cfg)
    dis = _disagg(GB300, scen, cfg, n_p=2, n_d=5)

    assert agg.itg_p99 > 3.0 * agg.itg_p50  # aggregated: prefill-stall spike
    assert dis.itg_p99 < agg.itg_p99  # disagg decode pool < aggregated exclusive
    assert dis.itg_p99 < 1.5 * dis.itg_p50  # flat: pure decode, no stall
    assert dis.itg_p50 == pytest.approx(dis.tpot_mean, rel=0.3)


# ---- 4. capacity: decode-bound vs prefill-starved ---------------------------


def test_saturated_throughput_tracks_decode_ceiling():
    """With an ample prefill pool feeding it, the decode pool is the bottleneck,
    so saturated system throughput approaches n_decode x the per-replica
    decode-only ceiling.  Because decode is pure (no prefill stealing time), the
    disagg pool tracks the ceiling more tightly than the aggregated loop does."""
    scen = Scenario(batch=64, prompt_len=256, output_len=2048)
    rep = simulate(GB300, LLAMA_3_1_70B, scen, DEP)
    per_replica_ceiling = rep.decode_only_tokens_per_s / rep.dp
    n_d = 2
    cfg = ServeConfig(arrivals=[0.0] * 300, max_batch=64, kv_policy="reserve")
    dis = _disagg(GB300, scen, cfg, n_p=4, n_d=n_d, transfer_bw=float("inf"))
    ratio = dis.output_tokens_per_s_system / (n_d * per_replica_ceiling)
    assert dis.peak_batch == 64  # decode replicas fill up
    assert 0.75 < ratio < 1.02  # observed ~0.81: only ramp/drain lost
    assert dis.decode_util > 0.9  # decode is the bottleneck
    assert dis.prefill_util < 0.2  # prefill is over-provisioned


def test_prefill_starved_leaves_decode_idle():
    """The dual: a single prefill replica behind long prompts cannot feed the
    decode pool, so TTFT queues up (saturated) while the decode replicas sit
    mostly idle -- prefill util ~1.0, decode util low.  This is the pool-sizing
    tension the report warns about."""
    scen = Scenario(batch=64, prompt_len=16384, output_len=16)
    cfg = ServeConfig(arrivals=[0.0] * 160, max_batch=64, kv_policy="reserve")
    dis = _disagg(GB300, scen, cfg, n_p=1, n_d=4, transfer_bw=float("inf"))
    assert dis.prefill_util > 0.95  # prefill pinned
    assert dis.decode_util < 0.30  # decode starved
    assert dis.saturated
    assert dis.backlog_at_last_arrival > 100  # prefill queue grows unbounded


# ---- 5. on_demand preemption is a cross-pool re-prefill ----------------------


def test_on_demand_preempts_via_the_prefill_pool():
    """Under KV pressure a decode replica preempts its newest decoder; in disagg
    the victim has no prefill hardware, so it returns to the prefill pool, is
    recomputed (prompt + generated) and re-transferred.  Every request still
    completes with its full output, and the preemption count is positive."""
    dep = Deployment(tp=1, weight_dtype=DType.FP8, kv_dtype=DType.FP8)  # tight KV
    scen = Scenario(batch=64, prompt_len=2048, output_len=512)
    on_demand = _disagg(DGX_H100, scen,
                        ServeConfig(arrivals=[0.0] * 40, max_batch=64, seed=0,
                                    kv_policy="on_demand"),
                        n_p=2, n_d=1, prefill=dep, decode=dep, transfer_bw=float("inf"))
    reserve = _disagg(DGX_H100, scen,
                     ServeConfig(arrivals=[0.0] * 40, max_batch=64, seed=0,
                                 kv_policy="reserve"),
                     n_p=2, n_d=1, prefill=dep, decode=dep, transfer_bw=float("inf"))
    assert on_demand.n_preemptions > 0  # decode replica overflowed and preempted
    assert reserve.n_preemptions == 0  # reserve never preempts
    assert on_demand.peak_batch > reserve.peak_batch  # on_demand packs more
    assert on_demand.n_completed == reserve.n_completed == 40
    assert all(rec.output_len == scen.output_len for rec in on_demand.requests)
    # a preempted request paid re-prefill + re-transfer -> extra prefill iters
    assert on_demand.n_prefill_iters > 40


# ---- 6. mixed request lengths flow through the pools ------------------------


def test_mixed_lengths_through_the_pools():
    """A 32k prompt among short chats: per-request lengths flow through prefill
    (each request's own prompt), the transfer (its own KV), and decode (its own
    context).  The report flags mixed lengths and the run completes."""
    n = 60
    arrivals = [i * 0.05 for i in range(n)]
    prompts = [512] * n
    prompts[10] = 32768  # one long-context request among the chats
    outputs = [128] * n
    cfg = ServeConfig(arrivals=arrivals, prompt_lens=prompts, output_lens=outputs,
                      max_batch=64, kv_policy="reserve")
    dis = _disagg(GB300, scen := Scenario(batch=64, prompt_len=512, output_len=128),
                  cfg, n_p=2, n_d=4, transfer_bw=900e9, transfer_latency=1e-6)
    assert dis.mixed_lengths
    assert dis.n_completed == n
    assert dis.prompt_p99 > dis.prompt_p50  # the 32k prompt lifts the tail
    # the 32k request's KV transfer is far larger than a 512-token chat's
    long_rec = next(r for r in dis.requests if r.prompt_len == 32768)
    chat_rec = next(r for r in dis.requests if r.prompt_len == 512)
    assert long_rec.transfer > 10 * chat_rec.transfer


# ---- 7. determinism ---------------------------------------------------------


def _fingerprint(r):
    return (r.ttft_p99, r.achieved_rate_replica, r.itg_p99, r.peak_kv_bytes,
            r.output_tokens_per_s_replica, r.transfer_mean, r.decode_util,
            r.n_preemptions)


def test_determinism():
    scen = Scenario(batch=64, prompt_len=1024, output_len=256)
    def run(seed):
        return _disagg(GB300, scen,
                       ServeConfig(arrival_rate=30.0, n_requests=80, max_batch=64,
                                   seed=seed),
                       n_p=2, n_d=4)
    assert _fingerprint(run(0)) == _fingerprint(run(0))  # same seed -> identical
    assert _fingerprint(run(0)) != _fingerprint(run(1))  # different seed differs


# ---- 8. rejections & scope --------------------------------------------------


def test_pipeline_parallel_pools_are_rejected():
    scen = Scenario(batch=8, prompt_len=512, output_len=64)
    cfg = ServeConfig(arrivals=[0.0], max_batch=8)
    pp = Deployment(tp=2, pp=2, weight_dtype=DType.FP8)
    with pytest.raises(ValueError, match="pp=1 only"):
        _disagg(GB300, scen, cfg, prefill=pp, decode=DEP)
    with pytest.raises(ValueError, match="pp=1 only"):
        _disagg(GB300, scen, cfg, prefill=DEP, decode=pp)


def test_chunked_prefill_is_rejected():
    """Chunked prefill is moot with exclusive prefill replicas -- reject it."""
    scen = Scenario(batch=8, prompt_len=512, output_len=64)
    cfg = ServeConfig(arrivals=[0.0], max_batch=8, prefill_chunk=256)
    with pytest.raises(ValueError, match="chunked prefill"):
        _disagg(GB300, scen, cfg, transfer_bw=float("inf"))


def test_over_partition_is_rejected():
    """The two pools cannot claim more chips than the system has."""
    scen = Scenario(batch=8, prompt_len=512, output_len=64)
    cfg = ServeConfig(arrivals=[0.0], max_batch=8)
    with pytest.raises(ValueError, match="chips but"):
        _disagg(GB300, scen, cfg, n_p=8, n_d=4, transfer_bw=float("inf"))


def test_idle_chips_reported():
    scen = Scenario(batch=8, prompt_len=512, output_len=64)
    dis = _disagg(GB300, scen, ServeConfig(arrivals=[0.0], max_batch=8),
                  n_p=2, n_d=5, transfer_bw=float("inf"))
    assert dis.idle_chips == 72 - (2 + 5) * DEP.replica_chips  # 72 - 56 = 16
    assert any("idle" in w for w in dis.warnings)


# ---- 9. adp composes on the decode side -------------------------------------


def test_decode_adp_composes():
    """The decode pool can run attention-DP (KV cut by adp); the run completes
    and the report echoes both pool deployments."""
    decode = Deployment(tp=4, adp=2, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
    scen = Scenario(batch=64, prompt_len=2048, output_len=128)
    dis = _disagg(GB300, scen,
                  ServeConfig(arrival_rate=20.0, n_requests=80, max_batch=64, seed=0),
                  n_p=2, n_d=4, prefill=DEP, decode=decode, transfer_bw=float("inf"))
    assert dis.n_completed == 80
    assert dis.output_tokens_per_s_system > 0
    assert dis.deployment == decode  # report echoes the decode pool
    assert dis.prefill_deployment == DEP
