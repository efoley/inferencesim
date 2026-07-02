"""End-to-end sanity checks of the roofline engine against back-of-envelope
numbers for well-understood configurations."""

import pytest

from inferencesim.hardware import DType
from inferencesim.presets import (
    DGX_SPARK,
    GB300_NVL72,
    H100_SINGLE,
    LLAMA_3_1_8B,
    LLAMA_3_1_70B,
    GPT_OSS_120B,
    TT_QUIETBOX,
)
from inferencesim.simulate import simulate
from inferencesim.workload import Deployment, Scenario


def test_h100_llama70b_fp8_batch1_decode_is_weight_bound():
    """Single H100, fp8 llama-70b, batch 1: decode is famously DRAM-bound.
    Speed of light = weights (~70.6 GB) / 3.35 TB/s ~= 21 ms/token."""
    dep = Deployment(tp=1, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    scen = Scenario(batch=1, prompt_len=1024, output_len=256)
    r = simulate(H100_SINGLE, LLAMA_3_1_70B, scen, dep)
    assert r.memory.fits
    sol = LLAMA_3_1_70B.total_params / H100_SINGLE.node.chip.dram.bandwidth
    assert r.tpot_s == pytest.approx(sol, rel=0.15)  # kv + activations add a bit
    assert 35 < 1 / r.tpot_s < 50  # tok/s ballpark
    # and it should indeed be memory-bound
    assert r.decode.category_bounds()["linear"] == "memory"


def test_h100_llama70b_bf16_does_not_fit():
    dep = Deployment(tp=1, weight_dtype=DType.BF16)
    scen = Scenario(batch=1, prompt_len=1024, output_len=256)
    r = simulate(H100_SINGLE, LLAMA_3_1_70B, scen, dep)
    assert not r.memory.fits
    assert any("does not fit" in w for w in r.warnings)


def test_batching_amortizes_weight_reads():
    """tok/s should scale strongly with batch while decode is weight-bound."""
    dep = Deployment(tp=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
    thr = []
    for b in (1, 16, 64):
        scen = Scenario(batch=b, prompt_len=2048, output_len=512)
        thr.append(simulate(GB300_NVL72, LLAMA_3_1_70B, scen, dep).decode_only_tokens_per_s)
    assert thr[1] > 8 * thr[0]
    assert thr[2] > thr[1]


def test_prefill_is_compute_bound_on_gpus():
    dep = Deployment(tp=8, weight_dtype=DType.FP8)
    scen = Scenario(batch=32, prompt_len=8192, output_len=512)
    r = simulate(GB300_NVL72, LLAMA_3_1_70B, scen, dep)
    assert r.prefill.category_bounds()["linear"] == "compute"


def test_spark_llama8b_ballpark():
    """DGX Spark, llama-8b fp8, batch 1: ~8 GB weights / 273 GB/s ~= 29 ms."""
    dep = Deployment(tp=1, weight_dtype=DType.FP8)
    scen = Scenario(batch=1, prompt_len=1024, output_len=256)
    r = simulate(DGX_SPARK, LLAMA_3_1_8B, scen, dep)
    assert r.memory.fits
    assert 25 < 1 / r.tpot_s < 40  # tok/s at speed of light


def test_quietbox_moe_runs_and_tp_comm_appears():
    dep = Deployment(tp=4, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    scen = Scenario(batch=8, prompt_len=2048, output_len=512)
    r = simulate(TT_QUIETBOX, GPT_OSS_120B, scen, dep)
    assert r.memory.fits  # 120B fp8 over 4x 32GB is tight but fits
    assert r.decode.category_times().get("comm", 0) > 0
    assert r.output_tokens_per_s > 0


def test_tp_reduces_tpot():
    scen = Scenario(batch=16, prompt_len=2048, output_len=512)
    r1 = simulate(GB300_NVL72, LLAMA_3_1_70B, scen, Deployment(tp=1, weight_dtype=DType.FP8))
    r8 = simulate(GB300_NVL72, LLAMA_3_1_70B, scen, Deployment(tp=8, weight_dtype=DType.FP8))
    # sub-linear: per-layer allreduce latency eats into the 8x memory speedup
    assert r8.tpot_s < r1.tpot_s / 2
    assert r8.decode.category_times()["comm"] > 0


def test_comm_overlap_reduces_latency():
    scen = Scenario(batch=64, prompt_len=4096, output_len=1024)
    base = Deployment(tp=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
    over = Deployment(tp=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8,
                      overlap_comm=True)
    r_base = simulate(GB300_NVL72, LLAMA_3_1_70B, scen, base)
    r_over = simulate(GB300_NVL72, LLAMA_3_1_70B, scen, over)
    assert r_over.tpot_s < r_base.tpot_s
    assert r_over.ttft_s < r_base.ttft_s
    # overlap can never beat the slower of the two streams
    comm = r_base.decode.category_times()["comm"]
    assert r_over.tpot_s >= max(comm, r_base.tpot_s - comm) * 0.999


def test_power_and_cost_are_sane():
    dep = Deployment(tp=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
    scen = Scenario(batch=64, prompt_len=4096, output_len=1024)
    r = simulate(GB300_NVL72, LLAMA_3_1_70B, scen, dep)
    max_power = (
        GB300_NVL72.n_nodes * GB300_NVL72.node.overhead_power_w
        + GB300_NVL72.total_chips * GB300_NVL72.node.chip.max_power_w
    )
    idle_power = (
        GB300_NVL72.n_nodes * GB300_NVL72.node.overhead_power_w
        + GB300_NVL72.total_chips * GB300_NVL72.node.chip.idle_power_w
    )
    assert idle_power < r.system_power_w < max_power
    assert 0 < r.joules_per_output_token < 100
    assert 0 < r.usd_per_m_output_tokens < 100
    assert 0 < r.capex_share < 1
