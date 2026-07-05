"""Explicit attention-DP + expert-parallel ("DEP") deployments.

Two patterns share this file:

  * dense **attention-DP** (`adp`): DP attention + TP FFN (DeepSeek-V3 style,
    TRT-LLM's dense DEPn).  Attention weights replicated across adp groups
    (per-chip KV divides by adp); the FFN shards over the whole tp*adp array,
    its allreduce replaced by an allgather + reduce-scatter.
  * MoE **expert-parallel** (`ep`): validated here to be structurally TRT-LLM's
    DEPn for MoE -- replicated attention, batch/KV sharded by ep, experts over
    ep, dispatch/combine all-to-alls, and a zero-cost (tp=1) attention allreduce.

Validation philosophy mirrors the rest of the suite: degenerate oracles pin the
new machinery against closed forms / the pre-feature path, and conservation laws
pin what must not change.
"""

import pytest

from inferencesim import collectives
from inferencesim.des import DESEngine
from inferencesim.efficiency import Efficiency
from inferencesim.engine import (
    CommContext,
    RooflineEngine,
    ring_allreduce_time,
    ring_gather_time,
)
from inferencesim.hardware import DType, Link, Topology
from inferencesim.ops import (
    OpKind,
    decode_attention_op,
    decode_ops,
    kv_cache_bytes_per_chip,
    prefill_ops,
    validate_deployment,
)
from inferencesim.presets import DGX_H100, GB300_NVL72, GPT_OSS_120B, LLAMA_3_1_70B
from inferencesim.sched import Task, schedule
from inferencesim.serve import ServeConfig, prefill_iteration_time, serve
from inferencesim.simulate import simulate, weight_bytes_per_chip
from inferencesim.workload import Deployment, ModelSpec, Scenario


def _small_dense() -> ModelSpec:
    """A tiny dense model with clean power-of-two dims for exact arithmetic."""
    return ModelSpec(
        name="tiny-dense", n_layers=4, d_model=512, n_heads=8, n_kv_heads=8,
        d_head=64, d_ff=2048, vocab_size=1000, gated_mlp=True,
    )


SCEN = Scenario(batch=32, prompt_len=2048, output_len=512)


# ---- 1. adp == 1 is the degenerate anchor: bit-identical to no-adp -----------


def _report_fields(r):
    return (r.ttft_s, r.tpot_s, r.output_tokens_per_s, r.decode_only_tokens_per_s,
            r.memory.weights, r.memory.kv_cache, r.system_power_w,
            r.usd_per_m_output_tokens)


def test_adp1_is_bit_identical_to_no_adp():
    """A dense Deployment with an explicit adp=1 reproduces the default (no-adp)
    deployment to the last bit, and lowers NO gather/scatter ops -- the adp
    machinery must vanish at adp=1."""
    base = Deployment(tp=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
    adp1 = Deployment(tp=8, adp=1, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
    scen = Scenario(batch=64, prompt_len=4096, output_len=1024)
    a = simulate(GB300_NVL72, LLAMA_3_1_70B, scen, base)
    b = simulate(GB300_NVL72, LLAMA_3_1_70B, scen, adp1)
    assert _report_fields(a) == _report_fields(b)
    # no gather/scatter machinery is lowered at adp=1, and the dense FFN stays
    # tp-sharded (tp*adp == tp)
    dec = decode_ops(LLAMA_3_1_70B, adp1, 64, 4096.0)
    pre = prefill_ops(LLAMA_3_1_70B, 4096, adp1)
    assert not any(op.kind is OpKind.HALFRING for op in (*dec, *pre))
    assert next(op for op in dec if op.name == "ffn").flops == pytest.approx(
        2 * 64 * LLAMA_3_1_70B.ffn_params_total / 8)
    assert next(op for op in pre if op.name == "ffn").flops == pytest.approx(
        2 * 4096 * LLAMA_3_1_70B.ffn_params_total / 8)


def test_adp_introduces_gather_scatter():
    """adp > 1 lowers exactly one allgather and one reduce-scatter per layer,
    both HALFRING comm ops over the tp*adp array, and drops the FFN allreduce
    (only the attention allreduce survives)."""
    dep = Deployment(tp=2, adp=4, weight_dtype=DType.FP8)
    ops = decode_ops(LLAMA_3_1_70B, dep, 32, 2048.0)
    gathers = [op for op in ops if op.name == "ffn_gather"]
    scatters = [op for op in ops if op.name == "ffn_scatter"]
    assert len(gathers) == len(scatters) == 1
    g = gathers[0]
    assert g.kind is OpKind.HALFRING and g.count == LLAMA_3_1_70B.n_layers
    # payload is the FULL batch's hidden state (B x d_model x act_bytes)
    assert g.comm_bytes == pytest.approx(32 * LLAMA_3_1_70B.d_model * DType.BF16.bytes)
    # one allreduce per layer now (attention only); its payload is B/adp tokens
    ar = next(op for op in ops if op.name == "allreduce")
    assert ar.count == LLAMA_3_1_70B.n_layers
    assert ar.comm_bytes == pytest.approx((32 / 4) * LLAMA_3_1_70B.d_model * DType.BF16.bytes)


# ---- 2. weight & KV accounting (exact numbers on a small model) --------------


def test_attention_weight_bytes_unchanged_ffn_bytes_divide_by_tp_adp():
    m = _small_dense()
    L, tp = m.n_layers, 4
    wb = DType.FP8.bytes
    non_ffn = (m.embedding_params / tp + L * m.attn_params / tp) * wb
    ffn1 = L * m.ffn_params_total / (tp * 1) * wb
    ffn4 = L * m.ffn_params_total / (tp * 4) * wb

    w1 = weight_bytes_per_chip(m, Deployment(tp=tp, adp=1, weight_dtype=DType.FP8))
    w4 = weight_bytes_per_chip(m, Deployment(tp=tp, adp=4, weight_dtype=DType.FP8))
    # attention + embedding replicated across adp -> unchanged; FFN divides by adp
    assert w1 == pytest.approx(non_ffn + ffn1, rel=1e-12)
    assert w4 == pytest.approx(non_ffn + ffn4, rel=1e-12)
    assert (w1 - non_ffn) == pytest.approx(4 * (w4 - non_ffn), rel=1e-12)  # FFN / 4


def test_kv_divides_by_adp_exactly():
    m = _small_dense()
    tp, n_tokens = 4, 10_000
    kvh = m.n_kv_heads / min(tp, m.n_kv_heads)
    per = m.n_layers * 2 * kvh * m.d_head * DType.FP8.bytes * n_tokens
    kv1 = kv_cache_bytes_per_chip(m, n_tokens, Deployment(tp=tp, adp=1, kv_dtype=DType.FP8))
    kv4 = kv_cache_bytes_per_chip(m, n_tokens, Deployment(tp=tp, adp=4, kv_dtype=DType.FP8))
    assert kv1 == pytest.approx(per, rel=1e-12)
    assert kv4 == pytest.approx(per / 4, rel=1e-12)


# ---- 3. FLOP conservation: per-chip FLOPs x chips is invariant in adp --------


def test_decode_flops_conserved_across_adp_at_fixed_array():
    """At a fixed tp*adp array the full decode batch does the same math, and
    every op runs on all tp*adp chips, so per-chip FLOPs are invariant across
    how the array is split between tensor- and attention-parallelism -- i.e.
    total replica FLOPs (per-chip x tp*adp chips) is conserved.

    (Prefill is single-request: only one adp group runs its attention, so the
    attention path there scales with 1/tp, not 1/(tp*adp) -- not conserved, and
    correctly so; the FFN, sequence-parallel over tp*adp, still is.)"""
    m = LLAMA_3_1_70B

    def per_chip_flops(tp, adp):
        dep = Deployment(tp=tp, adp=adp, weight_dtype=DType.FP8)
        return sum(op.flops * op.count for op in decode_ops(m, dep, 32, 2048.0))

    f41 = per_chip_flops(4, 1)
    assert per_chip_flops(2, 2) == pytest.approx(f41, rel=1e-12)
    assert per_chip_flops(1, 4) == pytest.approx(f41, rel=1e-12)


# ---- 4. comm oracles: half-ring expansion == closed form at rel 1e-9 ---------


@pytest.mark.parametrize("g", [2, 4, 8])
@pytest.mark.parametrize("lat", [0.0, 1e-6])
@pytest.mark.parametrize("topo", [Topology.RING, Topology.ALL_TO_ALL])
def test_half_ring_matches_closed_form(g, lat, topo):
    """The expanded allgather / reduce-scatter, scheduled in isolation, has
    makespan exactly ring_gather_time -- and exactly half a ring allreduce --
    while each link carries pure occupancy (g-1)*payload/(g*bw)."""
    payload, bw = 4e6, 100e9
    tasks: list[Task] = []
    exit_key = collectives.half_ring(tasks, None, g, payload, bw, lat, topo, "s0", "gs")
    r = schedule(tasks)
    link = Link("l", bw, lat)
    assert exit_key is not None
    assert r.makespan == pytest.approx(ring_gather_time(payload, g, link), rel=1e-9)
    assert r.makespan == pytest.approx(ring_allreduce_time(payload, g, link) / 2, rel=1e-9)
    res = "s0.l0.cw" if topo is Topology.RING else "s0.l0.out"
    assert r.busy[res] == pytest.approx((g - 1) * payload / (g * bw), rel=1e-9)


def test_half_ring_group_one_is_noop():
    tasks: list[Task] = []
    assert collectives.half_ring(tasks, 5, 1, 1e6, 100e9, 1e-6,
                                 Topology.ALL_TO_ALL, "s0", "gs") == 5
    assert tasks == []


# ---- 5. DES == roofline on the serial adp chain, at any efficiency -----------

_EFFS = [
    Efficiency(),
    Efficiency(compute=0.7, memory=0.85, collective=0.6, op_overhead_s=2e-6),
]


@pytest.mark.parametrize("eff", _EFFS)
def test_des_matches_roofline_serial_dense_adp_at_any_efficiency(eff):
    """tp=2, adp=2, pp=1: one serial op chain, so the DES (with the gather /
    reduce-scatter half-ring expansions and their per-op overhead) must equal
    the roofline sum to full precision -- for any efficiency."""
    dep = Deployment(tp=2, adp=2, weight_dtype=DType.FP8)
    a = simulate(DGX_H100, LLAMA_3_1_70B, SCEN, dep, engine=RooflineEngine(eff))
    d = simulate(DGX_H100, LLAMA_3_1_70B, SCEN, dep, engine=DESEngine(efficiency=eff))
    assert d.tpot_s == pytest.approx(a.tpot_s, rel=1e-9)
    assert d.ttft_s == pytest.approx(a.ttft_s, rel=1e-9)


# ---- 6. memory: adp buys KV feasibility -------------------------------------


def test_adp_makes_an_infeasible_batch_fit():
    """A batch/context that overflows per-chip memory at adp=1 fits once adp
    shards the KV (and the FFN weights) across more chips -- the reason the
    pattern exists."""
    scen = Scenario(batch=256, prompt_len=8192, output_len=1)
    tp = 2
    r1 = simulate(DGX_H100, LLAMA_3_1_70B, scen,
                  Deployment(tp=tp, adp=1, weight_dtype=DType.FP8, kv_dtype=DType.FP8))
    r4 = simulate(DGX_H100, LLAMA_3_1_70B, scen,
                  Deployment(tp=tp, adp=4, weight_dtype=DType.FP8, kv_dtype=DType.FP8))
    assert not r1.memory.fits
    assert r4.memory.fits
    assert r4.memory.kv_cache == pytest.approx(r1.memory.kv_cache / 4, rel=1e-12)


# ---- 7. serve() works with adp > 1; the single-request oracle still holds ----


def test_serve_runs_with_adp_and_single_request_oracle_holds():
    dep = Deployment(tp=2, adp=2, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
    scen = Scenario(batch=32, prompt_len=2048, output_len=64)
    r = serve(GB300_NVL72, LLAMA_3_1_70B, scen, dep,
              ServeConfig(arrivals=[0.0], max_batch=32))
    assert r.deployment.adp == 2
    assert r.n_completed == 1
    ttft_direct = prefill_iteration_time(GB300_NVL72, LLAMA_3_1_70B, dep, 2048)
    assert r.requests[0].ttft == pytest.approx(ttft_direct, rel=1e-9)
    # the analytic ttft must agree too (a lone request never queues)
    assert r.requests[0].ttft == pytest.approx(
        simulate(GB300_NVL72, LLAMA_3_1_70B, scen, dep).ttft_s, rel=1e-9)


# ---- 8. MoE ep IS structurally TRT-LLM's DEPn (validation, not new code) -----


def test_moe_ep_is_dep_structure():
    """gpt-oss at tp=1, ep=4 == TRT-LLM DEP4: attention data-parallel over 4
    (b_att = B/4, replicated attention weights), experts sharded over 4,
    dispatch/combine all-to-alls, and a zero-cost attention allreduce (tp=1)."""
    m, B = GPT_OSS_120B, 32
    dep = Deployment(tp=1, ep=4, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
    ops = decode_ops(m, dep, B, 2048.0)

    # the FFN array group is 4 (tp*ep*adp) -- what DEP4 shards the experts over
    comm = CommContext.for_deployment(GB300_NVL72, dep)
    assert comm.a2a == 4
    assert {"moe_dispatch", "moe_combine"} <= {op.name for op in ops}

    # attention runs data-parallel over 4: b_att = B/4 (compare flops to ep=1)
    att4 = decode_attention_op(m, dep, B, 2048.0)
    att1 = decode_attention_op(m, Deployment(tp=1, ep=1, weight_dtype=DType.FP4),
                               B, 2048.0)
    assert att4.flops == pytest.approx(att1.flops / 4, rel=1e-12)

    # the single (attention) allreduce costs zero at tp=1 -- no FFN allreduce
    engine = RooflineEngine()
    chip = GB300_NVL72.node.chip
    for op in ops:
        if op.name == "allreduce":
            assert engine.time_op(op, chip, comm).time == 0.0

    # experts shard over the array of 4; the full batch amortizes the reads
    routed = next(op for op in ops if op.name == "moe_routed")
    assert routed.flops == pytest.approx(
        2 * B * m.moe.top_k * m.expert_params / 4, rel=1e-12)


def test_moe_rejects_adp():
    """adp is dense-only: MoE attention-DP is exactly what ep provides."""
    with pytest.raises(ValueError, match="dense-only"):
        validate_deployment(GPT_OSS_120B, Deployment(tp=1, adp=2))
    with pytest.raises(ValueError, match="dense-only"):
        simulate(GB300_NVL72, GPT_OSS_120B, SCEN, Deployment(tp=1, adp=2))
