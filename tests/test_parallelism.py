"""Behavioral checks for the tp/pp/ep mapping."""

import pytest

from inferencesim.hardware import DType
from inferencesim.presets import (
    DGX_H100,
    GB300_NVL72,
    GPT_OSS_120B,
    LLAMA_3_1_70B,
    TT_QUIETBOX,
)
from inferencesim.simulate import simulate, weight_bytes_per_chip
from inferencesim.workload import Deployment, Scenario


SCEN = Scenario(batch=32, prompt_len=2048, output_len=512)


def test_ep_requires_moe():
    with pytest.raises(ValueError, match="dense"):
        simulate(GB300_NVL72, LLAMA_3_1_70B, SCEN, Deployment(tp=2, ep=4))


def test_replica_must_fit_chip_count():
    with pytest.raises(ValueError, match="tp\\*pp\\*ep"):
        simulate(DGX_H100, LLAMA_3_1_70B, SCEN, Deployment(tp=4, pp=4))


def test_pp_cuts_per_chip_memory():
    base = Deployment(tp=2, weight_dtype=DType.FP8)
    piped = Deployment(tp=2, pp=4, weight_dtype=DType.FP8)
    r_base = simulate(DGX_H100, LLAMA_3_1_70B, SCEN, base)
    r_pipe = simulate(DGX_H100, LLAMA_3_1_70B, SCEN, piped)
    # layer weights and kv split 4 ways (embeddings stay on edge stages)
    assert r_pipe.memory.weights < 0.3 * r_base.memory.weights
    assert r_pipe.memory.kv_cache == pytest.approx(r_base.memory.kv_cache / 4)


def test_pp_ttft_close_to_tp_only_and_hops_appear():
    base = Deployment(tp=2, weight_dtype=DType.FP8)
    piped = Deployment(tp=2, pp=4, weight_dtype=DType.FP8)
    r_base = simulate(DGX_H100, LLAMA_3_1_70B, SCEN, base)
    r_pipe = simulate(DGX_H100, LLAMA_3_1_70B, SCEN, piped)
    # a single request traverses stages sequentially: same math, plus 3 hops
    assert r_pipe.ttft_s == pytest.approx(r_base.ttft_s, rel=0.05)
    assert r_pipe.ttft_s > r_base.ttft_s
    names = {t.op.name for t in r_pipe.decode.timings}
    assert "pp_hop" in names


def test_pp_scales_capacity_at_constant_per_chip_efficiency():
    """PP at batch scaled by pp keeps the same microbatch, so per-chip
    throughput stays ~flat while the replica serves 4x the sequences in
    1/4 the per-chip memory -- capacity, not speed, is PP's product."""
    tp_only = simulate(
        DGX_H100, LLAMA_3_1_70B, Scenario(batch=32, prompt_len=2048, output_len=512),
        Deployment(tp=2, weight_dtype=DType.FP8),
    )
    piped = simulate(
        DGX_H100, LLAMA_3_1_70B, Scenario(batch=128, prompt_len=2048, output_len=512),
        Deployment(tp=2, pp=4, weight_dtype=DType.FP8),
    )
    assert piped.memory.fits
    assert piped.memory.total < 0.5 * tp_only.memory.total
    per_chip_base = tp_only.decode_only_tokens_per_s / (tp_only.dp * 2)
    per_chip_pipe = piped.decode_only_tokens_per_s / (piped.dp * 8)
    # parity within ~10% (extra KV traffic and hops cost a little)
    assert per_chip_pipe == pytest.approx(per_chip_base, rel=0.10)


def test_ep_shards_expert_weights_and_uses_alltoall():
    tp_only = Deployment(tp=4, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    ep_map = Deployment(tp=1, ep=4, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    w_tp = weight_bytes_per_chip(GPT_OSS_120B, tp_only)
    w_ep = weight_bytes_per_chip(GPT_OSS_120B, ep_map)
    # expert weights shard the same 4 ways; attention/embeddings replicate,
    # so EP holds a bit more per chip
    assert w_ep == pytest.approx(w_tp, rel=0.15)
    assert w_ep > w_tp

    r = simulate(TT_QUIETBOX, GPT_OSS_120B, Scenario(batch=16, prompt_len=2048,
                                                     output_len=512), ep_map)
    names = {t.op.name for t in r.decode.timings}
    assert {"moe_dispatch", "moe_combine"} <= names
    # attention-DP means no tp allreduce at tp=1
    assert all(t.time == 0 for t in r.decode.timings if t.op.name == "allreduce")
    assert r.output_tokens_per_s > 0


def test_ep_amortizes_expert_reads_better_than_pure_dp():
    """4 QuietBox cards as one ep=4 replica sharing experts should beat
    4 independent dp replicas when the same total batch can't amortize
    expert streaming alone -- the classic reason EP exists."""
    scen_ep = Scenario(batch=32, prompt_len=2048, output_len=512)
    scen_dp = Scenario(batch=8, prompt_len=2048, output_len=512)  # 32 total / 4 replicas
    dep_ep = Deployment(tp=1, ep=4, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    dep_dp = Deployment(tp=1, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    r_ep = simulate(TT_QUIETBOX, GPT_OSS_120B, scen_ep, dep_ep)
    r_dp = simulate(TT_QUIETBOX, GPT_OSS_120B, scen_dp, dep_dp)
    assert not r_dp.memory.fits  # 117 GB fp8 doesn't fit one 32 GB card anyway
    assert r_ep.memory.fits
    assert r_ep.decode_only_tokens_per_s > 0


def test_starved_pipeline_warns():
    r = simulate(GB300_NVL72, GPT_OSS_120B,
                 Scenario(batch=4, prompt_len=1024, output_len=256),
                 Deployment(tp=1, pp=4, ep=4, weight_dtype=DType.FP8))
    assert any("starved" in w for w in r.warnings)
