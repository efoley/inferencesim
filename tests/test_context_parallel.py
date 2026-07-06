"""Context-parallel prefill (CP): the adp attention-DP groups double as
context-parallel position shards during prefill, so a single long prompt's
attention parallelises across the whole tp*adp array (ring / striped attention,
arXiv:2310.01889; DeepSeek-V3 context-parallel prefill) instead of running on
one adp group -- the recorded DEP-PR gap (PARALLELISM.md section 6).

Validation philosophy mirrors the rest of the suite: degenerate oracles pin the
new machinery against the pre-feature path, conservation laws pin what must not
change, and DES==roofline serial oracles show the lowering is engine-consistent.
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
    prefill_chunk_ops,
    prefill_ops,
)
from inferencesim.presets import (
    DGX_H100,
    GB300_NVL72,
    LLAMA_3_1_70B,
    TT_QUIETBOX,
)
from inferencesim.sched import Task, schedule
from inferencesim.serve import (
    ServeConfig,
    chunked_prefill_ttft,
    prefill_iteration_time,
    serve,
)
from inferencesim.simulate import simulate
from inferencesim.workload import Deployment, MLAConfig, ModelSpec, Scenario


# ---- small hand-checkable models --------------------------------------------


def _dense(n_layers=6, **kw):
    return ModelSpec(name="dense", n_layers=n_layers, d_model=512, n_heads=16,
                     n_kv_heads=16, d_head=64, d_ff=2048, vocab_size=2000, **kw)


def _swa(window, every, n_layers=6):
    return _dense(n_layers=n_layers, swa_window=window, swa_every=every)


def _dense_mla(n_layers=6):
    """A DENSE MLA model -- MLA without a MoE FFN, so `adp` (dense-only) is legal
    and the MLA context-parallel latent ring can be exercised."""
    return ModelSpec(
        name="dense-mla", n_layers=n_layers, d_model=512, n_heads=16,
        n_kv_heads=16, d_head=128, d_ff=2048, vocab_size=2000,
        mla=MLAConfig(kv_lora_rank=128, qk_rope_head_dim=32, qk_nope_head_dim=64,
                      v_head_dim=64),
    )


def _attn_flops(ops):
    return sum(o.flops * o.count for o in ops if o.category == "attention")


SCEN = Scenario(batch=32, prompt_len=2048, output_len=512)


# ---- 1. degenerate anchors ---------------------------------------------------


def test_adp1_prefill_emits_no_cp_ops():
    """adp == 1: CP vanishes -- no cp_kv_ring op, attention undivided."""
    dep = Deployment(tp=8, adp=1, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
    ops = prefill_ops(LLAMA_3_1_70B, 4096, dep)
    assert not any(op.kind is OpKind.CP_RING for op in ops)
    # attention is the full triangle over tp only (heads/tp), no adp division
    att = _attn_flops(ops)
    m = LLAMA_3_1_70B
    expect = m.n_layers * (m.n_heads / 8) * 2 * 2.0 * 4096 * 4096 * 0.5 * m.d_head
    assert att == pytest.approx(expect, rel=1e-12)


def test_cp_prefill_false_is_the_pre_pr_single_group_prefill():
    """cp_prefill=False pins the historical single-group prefill exactly: the
    attention runs on ONE adp group (1/tp, NOT 1/(tp*adp)) and no CP ring is
    emitted -- while the FFN still shards over tp*adp (that pre-dates CP)."""
    m = _dense()
    S, tp, adp = 512, 2, 4
    off = Deployment(tp=tp, adp=adp, weight_dtype=DType.FP8, cp_prefill=False)
    ops = prefill_ops(m, S, off)
    assert not any(op.kind is OpKind.CP_RING for op in ops)
    # attention: single group, heads/tp, the full S^2/2 triangle -- the pinned
    # pre-PR value (no adp in the denominator)
    expect = m.n_layers * (m.n_heads / tp) * 2 * 2.0 * S * S * 0.5 * m.d_head
    assert _attn_flops(ops) == pytest.approx(expect, rel=1e-12)
    # qkv_proj still processes the whole S (one group holds the sequence)
    qkv = next(op for op in ops if op.name == "qkv_proj")
    assert qkv.flops == pytest.approx(2.0 * S * m.attn_qkv_params / tp, rel=1e-12)
    # ...but the FFN gather/scatter over tp*adp is present (that is the DEP FFN,
    # not CP): CP is strictly additive to the existing adp prefill lowering
    halfrings = [op for op in ops if op.kind is OpKind.HALFRING]
    assert {op.name for op in halfrings} == {"ffn_gather", "ffn_scatter"}
    assert all(op.count == m.n_layers for op in halfrings)


def test_cp_prefill_off_vs_on_only_differs_in_attention_and_ring():
    """Turning CP on divides the attention by adp and adds the ring; everything
    else (qkv/out at S/adp, FFN, gather/scatter) is a token relabel of the same
    work, so the FFN GEMM is untouched."""
    m = _dense()
    S = 512
    on = Deployment(tp=2, adp=4, weight_dtype=DType.FP8, cp_prefill=True)
    off = Deployment(tp=2, adp=4, weight_dtype=DType.FP8, cp_prefill=False)
    o_on, o_off = prefill_ops(m, S, on), prefill_ops(m, S, off)
    # CP divides attention by adp exactly
    assert _attn_flops(o_on) == pytest.approx(_attn_flops(o_off) / 4, rel=1e-12)
    # the FFN GEMM (weights TP over tp*adp) is identical -- CP re-assembles S
    ffn_on = next(op for op in o_on if op.name == "ffn")
    ffn_off = next(op for op in o_off if op.name == "ffn")
    assert ffn_on.flops == ffn_off.flops == pytest.approx(
        2 * S * m.ffn_params_total / (2 * 4), rel=1e-12)


# ---- 2. FLOP conservation across the adp split (striped) ---------------------


@pytest.mark.parametrize("model", [_dense(), _swa(window=128, every=2), _dense_mla()])
def test_prefill_attention_flops_conserved_across_cp_split(model):
    """Striped CP: at a fixed tp*adp array every chip does an equal 1/(tp*adp) of
    the causal attention triangle, so per-chip prefill attention flops are
    invariant across how the 4-chip array splits between tp and adp -- i.e. total
    replica attention flops (per-chip x tp*adp) is conserved.  Holds per variant
    (dense GQA, SWA banded, MLA decompressed)."""
    S = 1024

    def per_chip_attn(tp, adp):
        dep = Deployment(tp=tp, adp=adp, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
        return _attn_flops(prefill_ops(model, S, dep))

    base = per_chip_attn(4, 1)
    assert per_chip_attn(2, 2) == pytest.approx(base, rel=1e-9)
    assert per_chip_attn(1, 4) == pytest.approx(base, rel=1e-9)


def test_prefill_attention_dram_conserved_across_cp_split():
    """The once-streamed prompt K/V also divides evenly under striping, so
    per-chip attention DRAM is invariant at a fixed tp*adp array too."""
    m = _dense()
    S = 1024

    def per_chip_dram(tp, adp):
        dep = Deployment(tp=tp, adp=adp, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
        ops = prefill_ops(m, S, dep)
        return sum((o.dram_read + o.dram_write) * o.count
                   for o in ops if o.category == "attention")

    base = per_chip_dram(4, 1)
    assert per_chip_dram(2, 2) == pytest.approx(base, rel=1e-9)
    assert per_chip_dram(1, 4) == pytest.approx(base, rel=1e-9)


# ---- 3. the DEP-PR prefill caveat is resolved (TTFT) -------------------------


def test_cp_resolves_the_dep_prefill_ttft_gap():
    """On a FIXED 8-chip array, the DEP PR's tp=2,adp=4 prefill was WORSE than
    tp=8 (its attention ran on one adp group = 1/tp).  Context parallelism makes
    that attention run over the whole tp*adp=8 array, so CP-on TTFT beats CP-off
    (the pinned pre-PR value) and is competitive with tp=8 -- the caveat closed."""
    m = LLAMA_3_1_70B
    scen = Scenario(batch=1, prompt_len=32768, output_len=1)
    fp = dict(weight_dtype=DType.FP4, kv_dtype=DType.FP8)
    t_tp8 = simulate(GB300_NVL72, m, scen, Deployment(tp=8, **fp)).ttft_s
    t_off = simulate(GB300_NVL72, m, scen,
                     Deployment(tp=2, adp=4, cp_prefill=False, **fp)).ttft_s
    t_on = simulate(GB300_NVL72, m, scen,
                    Deployment(tp=2, adp=4, cp_prefill=True, **fp)).ttft_s
    assert t_off > t_tp8          # the recorded gap: adp prefill was worse
    assert t_on < t_off           # CP improves it
    assert t_on < 1.1 * t_tp8     # ...to competitive-or-better than tp=8


# ---- 4. the CP ring-pass oracle: expansion == closed form -------------------

_EFFS = [
    Efficiency(),
    Efficiency(compute=0.7, memory=0.85, collective=0.6, op_overhead_s=2e-6),
]


@pytest.mark.parametrize("eff", _EFFS)
@pytest.mark.parametrize("system,dep", [
    (GB300_NVL72, Deployment(tp=2, adp=4, weight_dtype=DType.FP8, kv_dtype=DType.FP8)),
    (TT_QUIETBOX, Deployment(tp=1, adp=4, weight_dtype=DType.FP8, kv_dtype=DType.FP8)),
])
def test_cp_ring_roofline_is_ring_gather_over_adp(system, dep, eff):
    """The cp_kv_ring op costs ring_gather_time with group = adp (cp-1 steps),
    NOT the tp*adp a2a group, over the a2a fabric -- on switched (GB300) and ring
    (QuietBox) alike, at any efficiency.  It is exactly half a ring allreduce
    over adp."""
    S = 4096
    ring = next(op for op in prefill_ops(LLAMA_3_1_70B, S, dep)
                if op.kind is OpKind.CP_RING)
    comm = CommContext.for_deployment(system, dep)
    assert comm.cp == dep.adp
    scaled = Link(comm.a2a_link.name, comm.a2a_link.bandwidth * eff.collective,
                  comm.a2a_link.latency_s)
    one = ring_gather_time(ring.comm_bytes, dep.adp, scaled) + eff.op_overhead_s
    got = RooflineEngine(eff).time_op(ring, system.node.chip, comm).time
    assert got == pytest.approx(ring.count * one, rel=1e-9)
    # exactly half a ring allreduce over the adp groups
    assert ring_gather_time(ring.comm_bytes, dep.adp, scaled) == pytest.approx(
        ring_allreduce_time(ring.comm_bytes, dep.adp, scaled) / 2, rel=1e-9)


@pytest.mark.parametrize("topo", [Topology.RING, Topology.ALL_TO_ALL])
def test_cp_ring_expansion_matches_closed_form(topo):
    """The DES expands the cp ring via `half_ring` with group = adp, so its
    isolated makespan is exactly ring_gather_time(payload, adp) and each link
    carries pure occupancy -- both fabrics."""
    payload, bw, lat, adp = 8e6, 200e9, 1e-6, 4
    tasks: list[Task] = []
    exit_key = collectives.half_ring(tasks, None, adp, payload, bw, lat, topo,
                                     "s0", "cp")
    r = schedule(tasks)
    link = Link("l", bw, lat)
    assert exit_key is not None
    assert r.makespan == pytest.approx(ring_gather_time(payload, adp, link), rel=1e-9)


# ---- 5. MLA sends compressed-latent blocks, GQA sends K/V heads --------------


def test_cp_ring_payload_is_latent_for_mla_kvheads_for_gqa():
    """The CP ring circulates the model's cached K/V form: GQA sends this chip's
    KV-head shard (2*kvh*d_head/token); MLA sends only the shared compressed
    latent (latent_dim/token) -- 1-2 orders of magnitude smaller."""
    S = 2048
    # GQA
    gqa = _dense()
    dep = Deployment(tp=2, adp=4, kv_dtype=DType.FP8)
    ring = next(op for op in prefill_ops(gqa, S, dep) if op.kind is OpKind.CP_RING)
    kvh = gqa.n_kv_heads / min(2, gqa.n_kv_heads)
    assert ring.comm_bytes == pytest.approx(
        S * 2 * kvh * gqa.d_head * DType.FP8.bytes, rel=1e-12)
    # MLA: only the latent (kv_lora_rank + qk_rope_head_dim) per token
    mla = _dense_mla()
    ring_m = next(op for op in prefill_ops(mla, S, dep) if op.kind is OpKind.CP_RING)
    assert ring_m.comm_bytes == pytest.approx(
        S * mla.mla.latent_dim * DType.FP8.bytes, rel=1e-12)
    # the MLA latent ring is far cheaper than the GQA K/V ring
    assert ring_m.comm_bytes < 0.2 * ring.comm_bytes


# ---- 6. DES == roofline on the serial CP prefill chain, any efficiency -------


@pytest.mark.parametrize("eff", _EFFS)
@pytest.mark.parametrize("system,dep", [
    (DGX_H100, Deployment(tp=2, adp=2, weight_dtype=DType.FP8)),
    (TT_QUIETBOX, Deployment(tp=1, adp=4, weight_dtype=DType.FP8, kv_dtype=DType.FP8)),
])
def test_des_matches_roofline_serial_cp_prefill(system, dep, eff):
    """tp*adp, pp=1: prefill is one serial op chain, so the DES (with the cp
    ring's half-ring expansion and its per-op overhead) must equal the roofline
    sum to full precision -- switched (DGX) and ring (QuietBox) fabrics, any
    efficiency.  The cp ring adds NO store-and-forward fill (it is a ring pass,
    exact in isolation), unlike the switched MoE all-to-all."""
    a = simulate(system, LLAMA_3_1_70B, SCEN, dep, engine=RooflineEngine(eff))
    d = simulate(system, LLAMA_3_1_70B, SCEN, dep, engine=DESEngine(efficiency=eff))
    assert d.ttft_s == pytest.approx(a.ttft_s, rel=1e-9)
    assert d.tpot_s == pytest.approx(a.tpot_s, rel=1e-9)


# ---- 7. serve consistency + chunked x CP ------------------------------------


@pytest.mark.parametrize("cp_prefill", [True, False])
def test_serve_single_request_ttft_equals_simulate_with_adp(cp_prefill):
    """A lone request never queues, so serve's TTFT, the direct
    prefill_iteration_time, and simulate's analytic TTFT must all agree at
    rel=1e-9 under adp>1 -- both CP settings (they all route through prefill_ops,
    so CP flows through consistently)."""
    dep = Deployment(tp=2, adp=2, weight_dtype=DType.FP4, kv_dtype=DType.FP8,
                     cp_prefill=cp_prefill)
    scen = Scenario(batch=32, prompt_len=8192, output_len=64)
    r = serve(GB300_NVL72, LLAMA_3_1_70B, scen, dep,
              ServeConfig(arrivals=[0.0], max_batch=32))
    assert r.n_completed == 1
    direct = prefill_iteration_time(GB300_NVL72, LLAMA_3_1_70B, dep, 8192)
    assert r.requests[0].ttft == pytest.approx(direct, rel=1e-9)
    assert r.requests[0].ttft == pytest.approx(
        simulate(GB300_NVL72, LLAMA_3_1_70B, scen, dep).ttft_s, rel=1e-9)


def test_chunked_prefill_cp_semantics_pinned():
    """Chunked prefill under CP: each chunk is a mini exclusive prefill, so its
    `chunk` query positions split into cp = adp striped blocks (chunk attention
    divides by adp) and a cp_kv_ring re-circulates the growing kv_len context.
    Pin exactly what a chunk lowers to, and that CP parallelises it (lower TTFT
    than CP off)."""
    m = LLAMA_3_1_70B
    on = Deployment(tp=2, adp=4, weight_dtype=DType.FP4, kv_dtype=DType.FP8,
                    cp_prefill=True)
    off = Deployment(tp=2, adp=4, weight_dtype=DType.FP4, kv_dtype=DType.FP8,
                     cp_prefill=False)
    chunk, prior = 512, 2048
    kv_len = prior + chunk
    ops_on = prefill_chunk_ops(m, on, chunk, prior, produce_logits=False)
    ops_off = prefill_chunk_ops(m, off, chunk, prior, produce_logits=False)
    # chunk attention divides by adp under CP
    assert _attn_flops(ops_on) == pytest.approx(_attn_flops(ops_off) / 4, rel=1e-12)
    # a cp_kv_ring re-circulates the growing kv_len (prior+chunk) context
    ring = next(op for op in ops_on if op.kind is OpKind.CP_RING)
    kvh = m.n_kv_heads / min(2, m.n_kv_heads)
    assert ring.comm_bytes == pytest.approx(
        kv_len * 2 * kvh * m.d_head * DType.FP8.bytes, rel=1e-12)
    assert not any(op.kind is OpKind.CP_RING for op in ops_off)
    # end to end: CP parallelises the chunk stream, so chunked TTFT drops
    ttft_on = chunked_prefill_ttft(GB300_NVL72, m, on, 8192, chunk=512)
    ttft_off = chunked_prefill_ttft(GB300_NVL72, m, off, 8192, chunk=512)
    assert ttft_on < ttft_off
