"""Attention variants: sliding-window (SWA) and multi-head latent (MLA).

Validation philosophy mirrors the rest of the suite: a degenerate oracle pins
each variant against the dense baseline it must reduce to, exact hand-computed
accounting pins the new machinery, and DES==roofline serial oracles show the
lowering stays consistent across engines.
"""

from dataclasses import replace

import pytest

from inferencesim.des import DESEngine
from inferencesim.engine import RooflineEngine
from inferencesim.hardware import DType
from inferencesim.ops import (
    decode_attention_ops,
    kv_cache_bytes_per_chip,
    prefill_ops,
)
from inferencesim.presets import (
    DEEPSEEK_V3,
    DGX_H100,
    GB300_NVL72,
    GPT_OSS_120B,
    H100_SINGLE,
)
from inferencesim.serve import ServeConfig, serve
from inferencesim.simulate import simulate
from inferencesim.workload import Deployment, MLAConfig, ModelSpec, MoEConfig, Scenario


# ---- small hand-checkable models --------------------------------------------


def _dense(n_layers=6, **kw):
    return ModelSpec(name="dense", n_layers=n_layers, d_model=512, n_heads=8,
                     n_kv_heads=8, d_head=64, d_ff=2048, vocab_size=2000, **kw)


def _swa(window, every, n_layers=6):
    return _dense(n_layers=n_layers, swa_window=window, swa_every=every)


def _attn_totals(ops):
    """(flops, dram_read, dram_write) summed over the attention op class(es),
    weighted by each op's layer count."""
    a = [o for o in ops if o.category == "attention"]
    return (sum(o.flops * o.count for o in a),
            sum(o.dram_read * o.count for o in a),
            sum(o.dram_write * o.count for o in a))


DEP1 = Deployment(tp=1)


# ---- 1. SWA degenerate: window >= context, all layers == dense --------------


def test_swa_window_beyond_context_is_dense_ops():
    """A window at least as large as the context, on every layer, reduces the
    attention op flops/bytes and the KV cache to the dense baseline exactly."""
    S = 512
    dense = _dense()
    swa = _swa(window=100_000, every=1)  # W >> S, every layer windowed
    for causal, ops_d, ops_s in [
        (True, prefill_ops(dense, S, DEP1), prefill_ops(swa, S, DEP1)),
    ]:
        fd, rd, wd = _attn_totals(ops_d)
        fs, rs, ws = _attn_totals(ops_s)
        assert fs == pytest.approx(fd, rel=1e-9)
        assert rs == pytest.approx(rd, rel=1e-9)
        assert ws == pytest.approx(wd, rel=1e-9)
    # decode attention too
    fd, rd, wd = _attn_totals(decode_attention_ops(dense, DEP1, 4, 4 * S))
    fs, rs, ws = _attn_totals(decode_attention_ops(swa, DEP1, 4, 4 * S))
    assert (fs, rs, ws) == pytest.approx((fd, rd, wd), rel=1e-9)
    # KV cache footprint
    assert kv_cache_bytes_per_chip(swa, S, DEP1) == pytest.approx(
        kv_cache_bytes_per_chip(dense, S, DEP1), rel=1e-9)


def test_swa_window_beyond_context_is_dense_simulate():
    """The whole roofline pipeline collapses to the dense numbers."""
    scen = Scenario(batch=8, prompt_len=384, output_len=64)  # max ctx 448 < W
    dense = _dense()
    swa = _swa(window=100_000, every=1)
    dep = Deployment(tp=1, weight_dtype=DType.FP8)
    a = simulate(H100_SINGLE, dense, scen, dep)
    b = simulate(H100_SINGLE, swa, scen, dep)
    assert b.tpot_s == pytest.approx(a.tpot_s, rel=1e-9)
    assert b.ttft_s == pytest.approx(a.ttft_s, rel=1e-9)
    assert b.memory.kv_cache == pytest.approx(a.memory.kv_cache, rel=1e-9)


# ---- 2. SWA exact accounting (swa_every=2, ctx > W) -------------------------


def test_swa_kv_and_decode_bytes_exact():
    m = _swa(window=128, every=2, n_layers=4)  # 2 windowed + 2 full
    assert m.n_swa_layers == 2 and m.n_full_attn_layers == 2
    ctx = 512  # > W
    per_layer = 2 * m.n_kv_heads * m.d_head * DType.BF16.bytes  # 2*8*64*2 = 2048
    # cache: 2 full layers hold `ctx`, 2 windowed layers hold min(ctx, W)=128
    cached = 2 * ctx + 2 * 128
    assert kv_cache_bytes_per_chip(m, ctx, DEP1) == per_layer * cached
    # decode attention DRAM reads: full op reads ctx, windowed op reads W
    ops = decode_attention_ops(m, DEP1, 1, ctx)
    full = next(o for o in ops if o.name == "attention")
    win = next(o for o in ops if o.name == "attention_swa")
    assert full.count == 2 and win.count == 2
    assert full.dram_read == 1 * ctx * 2 * m.n_kv_heads * m.d_head * DType.BF16.bytes
    assert win.dram_read == 1 * 128 * 2 * m.n_kv_heads * m.d_head * DType.BF16.bytes


# ---- 3. banded prefill formula vs brute-force -------------------------------


def test_banded_prefill_matches_bruteforce():
    """The windowed causal score count is the closed form
    0.5*(S^2 - max(0, S-W)^2).  For integer S, W this is the exact trapezoidal
    integral of the per-position window size min(x, W), so it must match a
    brute-force sum over positions to full precision (and reduce to the dense
    triangle S^2/2 when W >= S)."""
    for S, W in [(8, 3), (128, 128), (300, 64), (5, 10), (2048, 128)]:
        brute = sum((min(i, W) + min(i + 1, W)) / 2.0 for i in range(S))
        closed = 0.5 * (S * S - max(0, S - W) ** 2)
        assert closed == pytest.approx(brute, rel=1e-12)
    # and the op itself carries that count
    m = _swa(window=128, every=1)  # all layers windowed
    for S in (64, 200, 512):
        win = next(o for o in prefill_ops(m, S, DEP1) if o.name == "attention_swa")
        score_pairs = win.flops / (1 * m.n_heads * 2 * 2.0 * m.d_head)
        brute = sum((min(i, 128) + min(i + 1, 128)) / 2.0 for i in range(S))
        assert score_pairs == pytest.approx(brute, rel=1e-9)


# ---- 4. MLA accounting + DeepSeek-V3 param count ----------------------------


def test_mla_kv_per_token_is_compressed_latent():
    m = DEEPSEEK_V3
    latent = m.mla.kv_lora_rank + m.mla.qk_rope_head_dim  # 512 + 64 = 576
    expect = latent * m.n_layers * DType.BF16.bytes
    assert kv_cache_bytes_per_chip(m, 1, DEP1) == expect
    assert m.kv_bytes_per_token(DType.BF16) == expect
    # the latent is replicated across tp (tiny) -- tp does NOT cut per-chip KV
    assert kv_cache_bytes_per_chip(m, 1, Deployment(tp=8)) == expect
    # ep batch-shards it, though
    assert kv_cache_bytes_per_chip(m, 1, Deployment(tp=8, ep=8)) == pytest.approx(
        expect / 8)


def test_mla_decode_attention_reads_compressed_cache():
    m = DEEPSEEK_V3
    latent = m.mla.kv_lora_rank + m.mla.qk_rope_head_dim
    ops = decode_attention_ops(m, DEP1, 1, 1000)
    assert len(ops) == 1  # MLA is a single op
    op = ops[0]
    assert op.dram_read == 1 * 1000 * latent * DType.BF16.bytes


def test_deepseek_v3_param_counts():
    # published: 671B total, 37B active
    assert abs(DEEPSEEK_V3.total_params - 671e9) / 671e9 < 0.02
    assert abs(DEEPSEEK_V3.active_params - 37e9) / 37e9 < 0.05


# ---- 5. gpt-oss corrected preset --------------------------------------------


def test_gpt_oss_still_validates_and_kv_halves():
    m = GPT_OSS_120B
    assert m.n_swa_layers == 18 and m.n_full_attn_layers == 18
    # params unchanged by SWA (attention-only feature)
    assert abs(m.total_params - 117e9) / 117e9 < 0.05
    ctx = 2048
    per_layer = 2 * m.n_kv_heads * m.d_head * DType.BF16.bytes
    expect = per_layer * (18 * ctx + 18 * 128)  # 18 full + 18 windowed@128
    assert kv_cache_bytes_per_chip(m, ctx, DEP1) == expect
    # ~half the naive all-dense figure
    dense = per_layer * (36 * ctx)
    assert kv_cache_bytes_per_chip(m, ctx, DEP1) < 0.6 * dense


def test_gpt_oss_swa_lowers_decode_attention_vs_uncorrected():
    uncorrected = replace(GPT_OSS_120B, swa_window=None, swa_every=0)
    dep = Deployment(tp=4, ep=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
    scen = Scenario(batch=32, prompt_len=8192, output_len=4096)  # big context
    sw = simulate(GB300_NVL72, GPT_OSS_120B, scen, dep)
    de = simulate(GB300_NVL72, uncorrected, scen, dep)
    assert (sw.decode.category_times()["attention"]
            < de.decode.category_times()["attention"])
    assert sw.tpot_s < de.tpot_s  # directional TPOT win
    assert sw.memory.kv_cache < de.memory.kv_cache


# ---- 6. serve: SWA KV plateau; MLA huge batch fits --------------------------


def test_serve_swa_kv_plateaus_at_window():
    """An all-windowed model's per-request KV footprint saturates at the window,
    so the peak KV is independent of how many tokens the request generates."""
    W = 256
    m = _swa(window=W, every=1)
    dep = Deployment(tp=1, weight_dtype=DType.FP8, kv_dtype=DType.BF16)
    kpt = kv_cache_bytes_per_chip(m, 1, dep)
    scen = Scenario(batch=8, prompt_len=64, output_len=1000)  # ctx grows past W

    def peak(output_len):
        s = Scenario(batch=8, prompt_len=64, output_len=output_len)
        return serve(H100_SINGLE, m, s, dep,
                     ServeConfig(arrivals=[0.0], max_batch=8)).peak_kv_bytes

    p1, p2 = peak(1000), peak(3000)
    assert p1 == pytest.approx(W * kpt, rel=1e-9)   # plateau at the window
    assert p1 == pytest.approx(p2, rel=1e-9)        # independent of output_len
    # far below the dense (uncapped) footprint at that context
    assert p1 < 0.4 * (scen.max_context * kpt)


def test_serve_mla_tiny_kv_fits_huge_batch():
    """MLA's compressed cache is so small that a big batch at long context is
    KV-feasible where a per-head cache never would be."""
    dep = Deployment(tp=8, ep=8, weight_dtype=DType.FP8, kv_dtype=DType.BF16)
    scen = Scenario(batch=64, prompt_len=4096, output_len=256)
    r = serve(GB300_NVL72, DEEPSEEK_V3, scen, dep,
              ServeConfig(arrivals=[0.0] * 64, max_batch=64, seed=0))
    assert r.n_completed == 64
    assert r.kv_feasible_batch == 64  # KV is never the limiter
    assert r.peak_batch == 64
    assert r.n_preemptions == 0


# ---- 7. DES == roofline serial oracle for MLA and SWA -----------------------


@pytest.mark.parametrize("model,dep", [
    (DEEPSEEK_V3, Deployment(tp=8, ep=8, weight_dtype=DType.FP8, kv_dtype=DType.BF16)),
    (GPT_OSS_120B, Deployment(tp=4, ep=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8)),
])
def test_des_matches_roofline_serial_mla_and_swa(model, dep):
    """pp=1 is one serial op chain, so the DES must equal the analytic sum to
    full precision -- including the split (SWA) or compressed (MLA) attention
    ops routed through the count-weighted per-layer cost."""
    scen = Scenario(batch=64, prompt_len=2048, output_len=512)
    a = simulate(GB300_NVL72, model, scen, dep, engine=RooflineEngine())
    d = simulate(GB300_NVL72, model, scen, dep, engine=DESEngine())
    assert d.tpot_s == pytest.approx(a.tpot_s, rel=1e-9)
    assert d.ttft_s == pytest.approx(a.ttft_s, rel=1e-9)


def test_des_matches_roofline_dense_swa():
    """SWA in a *dense* model (isolating the two-attention-op split from MoE):
    the serial pp=1 chain still matches roofline exactly."""
    m = _swa(window=128, every=2, n_layers=8)
    dep = Deployment(tp=2, weight_dtype=DType.FP8)
    scen = Scenario(batch=16, prompt_len=1024, output_len=256)
    a = simulate(DGX_H100, m, scen, dep, engine=RooflineEngine())
    d = simulate(DGX_H100, m, scen, dep, engine=DESEngine())
    assert d.tpot_s == pytest.approx(a.tpot_s, rel=1e-9)
    assert d.ttft_s == pytest.approx(a.ttft_s, rel=1e-9)


# ---- 8. MLA and SWA are mutually exclusive ----------------------------------


def test_mla_and_swa_rejected():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ModelSpec(name="bad", n_layers=4, d_model=512, n_heads=8, n_kv_heads=8,
                  d_head=64, d_ff=2048, vocab_size=1000,
                  swa_window=128, swa_every=1,
                  mla=MLAConfig(kv_lora_rank=512, qk_rope_head_dim=64,
                                qk_nope_head_dim=128, v_head_dim=128))


def test_swa_config_requires_window_and_stride():
    with pytest.raises(ValueError, match="swa_window set but swa_every"):
        _dense(swa_window=128, swa_every=0)
    with pytest.raises(ValueError, match="swa_every set but no swa_window"):
        _dense(swa_every=2)
