"""Efficiency factors: the derating knobs and their consistent application.

The governing invariants:
  * `Efficiency()` (all 1.0, zero overhead) is the identity -- every engine
    must produce bit-identical numbers with and without it (the full suite
    passing unchanged is the broad guard; here is one explicit report compare).
  * each factor scales exactly the stream it names (compute FLOP/s, memory
    bandwidth, collective bandwidth) and the overhead adds count x overhead;
  * the same Efficiency applied to the roofline and the DES agrees to full
    precision on a serial (pp=1) chain -- for ANY efficiency, including the
    expanded collectives and per-op overhead.
"""

import argparse
from statistics import median

import pytest

from inferencesim.calibration import (
    ANCHORS,
    Anchor,
    calibrate_report,
    optimism_ratio,
    run_anchor,
)
from inferencesim.des import DESEngine
from inferencesim.efficiency import (
    PROFILES,
    Efficiency,
    profile_for,
    vendor_profile_name,
)
from inferencesim.engine import CommContext, RooflineEngine, ring_allreduce_time
from inferencesim.graph import Edge, Graph, Node, NodeKind
from inferencesim.graphdes import ChipModel
from inferencesim.hardware import DType
from inferencesim.ops import Op, OpKind
from inferencesim.presets import DGX_H100, GB300_NVL72, GPT_OSS_120B, H100_SINGLE, LLAMA_3_1_70B
from inferencesim.simulate import simulate
from inferencesim.workload import Deployment, Scenario


# ---- default Efficiency is the identity -------------------------------------


def _report_fields(r):
    return (r.ttft_s, r.tpot_s, r.output_tokens_per_s, r.decode_only_tokens_per_s,
            r.system_power_w, r.usd_per_m_output_tokens)


def test_default_efficiency_is_bit_identical_roofline():
    """A report from RooflineEngine() and RooflineEngine(Efficiency()) is
    byte-for-byte identical -- the default derating changes no number."""
    dep = Deployment(tp=8, weight_dtype=DType.FP8)
    scen = Scenario(batch=32, prompt_len=2048, output_len=512)
    a = simulate(DGX_H100, LLAMA_3_1_70B, scen, dep, engine=RooflineEngine())
    b = simulate(DGX_H100, LLAMA_3_1_70B, scen, dep,
                 engine=RooflineEngine(Efficiency()))
    assert _report_fields(a) == _report_fields(b)


def test_default_efficiency_is_bit_identical_des():
    """Same guarantee for the discrete-event engine."""
    dep = Deployment(tp=2, pp=2, weight_dtype=DType.FP8)
    scen = Scenario(batch=32, prompt_len=2048, output_len=512)
    a = simulate(DGX_H100, LLAMA_3_1_70B, scen, dep, engine=DESEngine())
    b = simulate(DGX_H100, LLAMA_3_1_70B, scen, dep,
                 engine=DESEngine(efficiency=Efficiency()))
    assert _report_fields(a) == _report_fields(b)


# ---- each factor scales its own stream --------------------------------------


def _chip_and_comm(system=H100_SINGLE, dep=Deployment(tp=1)):
    return system.node.chip, CommContext.for_deployment(system, dep)


def test_halved_compute_doubles_compute_time():
    chip, comm = _chip_and_comm()
    op = Op("x", OpKind.COMPUTE, DType.FP8, "linear", 1, flops=1e15,
            dram_read=0.0, dram_write=0.0)
    sol = RooflineEngine().time_op(op, chip, comm)
    half = RooflineEngine(Efficiency(compute=0.5)).time_op(op, chip, comm)
    assert half.compute_time == pytest.approx(2 * sol.compute_time, rel=1e-9)
    assert half.time == pytest.approx(2 * sol.time, rel=1e-9)  # compute-bound


def test_halved_memory_doubles_mem_time():
    chip, comm = _chip_and_comm()
    op = Op("x", OpKind.COMPUTE, DType.FP8, "linear", 1, flops=0.0,
            dram_read=1e9, dram_write=0.0)
    sol = RooflineEngine().time_op(op, chip, comm)
    half = RooflineEngine(Efficiency(memory=0.5)).time_op(op, chip, comm)
    assert half.mem_time == pytest.approx(2 * sol.mem_time, rel=1e-9)
    assert half.time == pytest.approx(2 * sol.time, rel=1e-9)  # mem-bound


def test_op_overhead_adds_count_times_overhead():
    chip, comm = _chip_and_comm()
    op = Op("x", OpKind.COMPUTE, DType.FP8, "linear", count=5, flops=1e15,
            dram_read=1e9, dram_write=0.0)
    base = RooflineEngine().time_op(op, chip, comm).time
    with_oh = RooflineEngine(Efficiency(op_overhead_s=1e-6)).time_op(op, chip, comm).time
    assert with_oh - base == pytest.approx(5 * 1e-6, rel=1e-9)


def test_collective_factor_scales_bw_term_not_latency():
    """The collective factor scales only the ring allreduce's bandwidth term;
    the latency term (physical flight time) is untouched."""
    chip, comm = _chip_and_comm(DGX_H100, Deployment(tp=8))
    op = Op("allreduce", OpKind.ALLREDUCE, DType.BF16, "comm", 1, comm_bytes=1e7)
    sol = RooflineEngine().time_op(op, chip, comm).time
    half = RooflineEngine(Efficiency(collective=0.5)).time_op(op, chip, comm).time

    link = comm.tp_link
    assert sol == pytest.approx(ring_allreduce_time(1e7, 8, link), rel=1e-9)
    lat_term = 2 * (8 - 1) * link.latency_s
    bw_term = sol - lat_term
    # bandwidth term doubles at half efficiency, latency term unchanged
    assert half == pytest.approx(2 * bw_term + lat_term, rel=1e-9)
    assert half - 2 * bw_term == pytest.approx(lat_term, rel=1e-9)


# ---- roofline / DES agree at ANY efficiency (serial pp=1) --------------------

_EFFS = [
    Efficiency(),
    Efficiency(compute=0.7, memory=0.85, collective=0.6, op_overhead_s=2e-6),
]


@pytest.mark.parametrize("eff", _EFFS)
def test_des_matches_roofline_serial_dense_at_any_efficiency(eff):
    """tp=8, pp=1: one serial chain, so DES must equal the roofline sum to full
    precision -- proving compute/memory/collective and per-op overhead thread
    identically through both engines (unit costs and expanded collectives)."""
    dep = Deployment(tp=8, weight_dtype=DType.FP8)
    scen = Scenario(batch=32, prompt_len=2048, output_len=512)
    a = simulate(DGX_H100, LLAMA_3_1_70B, scen, dep, engine=RooflineEngine(eff))
    d = simulate(DGX_H100, LLAMA_3_1_70B, scen, dep, engine=DESEngine(efficiency=eff))
    assert d.tpot_s == pytest.approx(a.tpot_s, rel=1e-9)
    assert d.ttft_s == pytest.approx(a.ttft_s, rel=1e-9)


@pytest.mark.parametrize("eff", _EFFS)
def test_des_matches_roofline_serial_moe_ep_at_any_efficiency(eff):
    """MoE + EP, pp=1: the all-to-all dispatch/combine expansions plus their
    per-op overhead must also match the roofline exactly on the serial chain."""
    dep = Deployment(tp=4, ep=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
    scen = Scenario(batch=128, prompt_len=2048, output_len=512)
    a = simulate(GB300_NVL72, GPT_OSS_120B, scen, dep, engine=RooflineEngine(eff))
    d = simulate(GB300_NVL72, GPT_OSS_120B, scen, dep, engine=DESEngine(efficiency=eff))
    assert d.tpot_s == pytest.approx(a.tpot_s, rel=1e-9)


# ---- graph mode -------------------------------------------------------------


def _single_bank_chip(bank_bw: float) -> Graph:
    """bank -> noc -> sram -> core, only the bank constrained (single tile
    stream serialises on it)."""
    return Graph(
        name="degenerate",
        nodes=[
            Node("bank", NodeKind.MEMORY, count=1, capacity_bytes=1e12, bandwidth=bank_bw),
            Node("noc", NodeKind.SWITCH),
            Node("sram", NodeKind.MEMORY, count=1, capacity_bytes=1e6),
            Node("core", NodeKind.COMPUTE, count=1, peak_flops={DType.FP16: 1e12}),
        ],
        edges=[Edge("bank", "noc"), Edge("noc", "sram"), Edge("sram", "core")],
    )


def test_graph_mode_memory_efficiency_scales_bandwidth():
    """A mem-bound op through one bank at memory efficiency 0.5 is exactly
    bytes / (bandwidth x 0.5)."""
    B, R = 1e11, 1e7
    m = ChipModel(_single_bank_chip(B), tile_fill=0.5,
                  efficiency=Efficiency(memory=0.5))
    s = m.op_wall(Op("x", OpKind.COMPUTE, DType.FP16, "linear", 1, 0.0, R, 0.0))
    assert s.wall == pytest.approx(R / (B * 0.5), rel=1e-9)


def test_graph_mode_default_efficiency_is_identity():
    """ChipModel with the explicit default matches the no-arg construction
    bit-for-bit."""
    B, R, F = 1e11, 1e7, 5e11
    op = Op("x", OpKind.COMPUTE, DType.FP16, "linear", 1, F, R, 0.0)
    a = ChipModel(_single_bank_chip(B), tile_fill=0.5).op_wall(op).wall
    b = ChipModel(_single_bank_chip(B), tile_fill=0.5,
                  efficiency=Efficiency()).op_wall(op).wall
    assert a == b


def test_graph_mode_op_overhead_added_once_per_op():
    """op_overhead_s is charged once per op (not per tile): the wall grows by
    exactly the overhead even for a many-tile op."""
    B, R = 1e11, 1e7  # 20 tiles at tile_fill 0.5
    op = Op("x", OpKind.COMPUTE, DType.FP16, "linear", 1, 0.0, R, 0.0)
    base = ChipModel(_single_bank_chip(B), tile_fill=0.5).op_wall(op)
    with_oh = ChipModel(_single_bank_chip(B), tile_fill=0.5,
                        efficiency=Efficiency(op_overhead_s=3e-6)).op_wall(op)
    assert base.n_tiles > 1
    assert with_oh.wall - base.wall == pytest.approx(3e-6, rel=1e-9)


# ---- validation -------------------------------------------------------------


def test_efficiency_validation_rejects_bad_factors():
    with pytest.raises(ValueError):
        Efficiency(compute=0.0)         # factor must be > 0
    with pytest.raises(ValueError):
        Efficiency(memory=1.5)          # factor must be <= 1
    with pytest.raises(ValueError):
        Efficiency(collective=0.0)
    with pytest.raises(ValueError):
        Efficiency(op_overhead_s=-1.0)  # overhead must be >= 0
    # the boundaries are valid
    Efficiency(compute=1.0, memory=1.0, collective=1.0, op_overhead_s=0.0)


def test_profiles_present_and_sol_is_identity():
    assert PROFILES["sol"] == Efficiency()
    typ = PROFILES["typical"]
    assert 0.0 < typ.compute <= 1.0 and 0.0 < typ.memory <= 1.0
    assert 0.0 < typ.collective <= 1.0 and typ.op_overhead_s >= 0.0


def test_every_profile_validates():
    """Every named profile is a valid Efficiency (its __post_init__ enforced the
    (0,1] / >=0 invariants at construction)."""
    for name, eff in PROFILES.items():
        assert isinstance(eff, Efficiency), name
        assert 0.0 < eff.compute <= 1.0 and 0.0 < eff.memory <= 1.0, name
        assert 0.0 < eff.collective <= 1.0 and eff.op_overhead_s >= 0.0, name


# ---- per-vendor profiles ----------------------------------------------------


def test_typical_nv_aliases_the_global_typical():
    """typical-nv == typical today (the global fit was driven by the NVIDIA
    anchors), kept separate so the two can diverge later."""
    assert PROFILES["typical-nv"] == PROFILES["typical"]


def test_typical_tt_values_in_cited_bands():
    """Tenstorrent memory is re-fitted to 0.40 (qb2 decode residual 0.57/1.41),
    near The Register's 41-50%-of-peak band; the other knobs are kept from the
    global fit for lack of tt-specific evidence."""
    tt, nv = PROFILES["typical-tt"], PROFILES["typical-nv"]
    # memory: anchor-derived 0.57/1.41 = 0.40, just under The Register 0.41-0.50
    assert tt.memory == pytest.approx(0.40, abs=5e-3)
    assert tt.memory < nv.memory  # tt derates harder than NVIDIA
    # compute/collective/op_overhead kept from the global fit
    assert tt.compute == nv.compute
    assert tt.collective == nv.collective
    assert tt.op_overhead_s == nv.op_overhead_s


def test_profile_for_auto_maps_vendor():
    """'auto' routes tt-* keys (and the explicit fine-preset set) to typical-tt,
    everything else to typical-nv."""
    for tt_key in ("tt-quietbox-2", "tt-quietbox", "tt-quietbox-fine",
                   "blackhole-p150-fine"):
        assert profile_for(tt_key) is PROFILES["typical-tt"], tt_key
        assert vendor_profile_name(tt_key) == "typical-tt", tt_key
    for nv_key in ("h100", "dgx-h100", "gb300-nvl72", "dgx-spark", "dgx-h100-fine"):
        assert profile_for(nv_key) is PROFILES["typical-nv"], nv_key
        assert vendor_profile_name(nv_key) == "typical-nv", nv_key


def test_profile_for_explicit_name_bypasses_mapping():
    """An explicit profile name is honoured even for tt hardware (no vendor
    remapping)."""
    assert profile_for("tt-quietbox-2", "sol") is PROFILES["sol"]
    assert profile_for("tt-quietbox-2", "typical-nv") is PROFILES["typical-nv"]
    assert profile_for("h100", "typical-tt") is PROFILES["typical-tt"]


def test_profile_for_unknown_name_raises_with_available_list():
    with pytest.raises(ValueError) as excinfo:
        profile_for("h100", "bogus")
    msg = str(excinfo.value)
    assert "bogus" in msg
    assert "typical-tt" in msg and "typical-nv" in msg  # lists the available set


def test_qb2_decode_brackets_one_under_typical_tt():
    """The tt decode/memory anchor -- a 1.41x residual under the global typical
    (the motivation for per-vendor profiles) -- brackets 1 under typical-tt."""
    dec = next(a for a in ANCHORS if a.name == "qb2-qwen32b-ttmetal-decode")
    _, _, glob = run_anchor(dec, PROFILES["typical"])
    assert glob > 1.3, "the global-typical residual this PR fixes"
    _, _, tt = run_anchor(dec, PROFILES["typical-tt"])
    assert 0.9 <= tt <= 1.1, f"qb2 decode under typical-tt: {tt:.3f}x"


def test_calibrate_report_auto_labels_each_anchor_vendor():
    """calibrate_report(resolve=...) scores each anchor under its vendor profile
    and shows a profile column naming it -- tt and nv rows in one table."""
    report = calibrate_report(resolve=lambda hw: profile_for(hw, "auto"))
    assert "profile" in report
    assert "typical-tt" in report and "typical-nv" in report


def test_cli_efficiency_default_is_sol_auto_is_opt_in():
    """The CLI default stays sol (identity) for every vendor -- 'auto' vendor
    derating is strictly opt-in (this is the CLI counterpart to the bit-identical
    guards above).  Explicit names bypass the mapping; --eff-* overrides layer on
    top of the auto-resolved base."""
    from inferencesim.cli import _efficiency_from_args

    def ns(**kw):
        base = dict(efficiency=None, eff_compute=None, eff_memory=None,
                    eff_collective=None, op_overhead_s=None)
        base.update(kw)
        return argparse.Namespace(**base)

    # default (no --efficiency): sol regardless of hardware -- zero behaviour change
    assert _efficiency_from_args(ns(), "tt-quietbox-2") == PROFILES["sol"]
    assert _efficiency_from_args(ns(), "h100") == PROFILES["sol"]
    # opt-in: --efficiency auto vendor-resolves per hardware key
    assert _efficiency_from_args(ns(efficiency="auto"), "tt-quietbox-2") \
        == PROFILES["typical-tt"]
    assert _efficiency_from_args(ns(efficiency="auto"), "h100") == PROFILES["typical-nv"]
    # an explicit named profile bypasses the vendor mapping
    assert _efficiency_from_args(ns(efficiency="typical"), "tt-quietbox-2") \
        == PROFILES["typical"]
    # --eff-* overrides layer onto the auto-resolved base
    got = _efficiency_from_args(ns(efficiency="auto", eff_memory=0.5), "tt-quietbox-2")
    assert got.memory == 0.5
    assert got.compute == PROFILES["typical-tt"].compute


# ---- calibration anchors ----------------------------------------------------


def test_every_anchor_resolves_and_runs():
    """Every ANCHOR's hardware/model keys resolve and run_anchor executes,
    returning a positive simulated value, its measured value, and a finite
    positive optimism ratio."""
    assert ANCHORS  # the set is populated
    for a in ANCHORS:
        simulated, measured, ratio = run_anchor(a, Efficiency())
        assert simulated > 0, a.name
        assert measured == a.measured > 0, a.name
        assert ratio == ratio and ratio > 0, a.name  # finite (not NaN), positive


def test_sol_is_optimistic_for_every_anchor():
    """The roofline is an upper bound: at `sol` every anchor's optimism ratio
    must be >= ~1.  A sub-1 sol ratio is a preset/spec or unit error, not a fit
    target -- this guards the calibration invariant."""
    for a in ANCHORS:
        _, _, ratio = run_anchor(a, PROFILES["sol"])
        assert ratio >= 0.95, f"{a.name}: sol optimism {ratio:.2f}x < 1 (investigate)"
    ratios = [run_anchor(a, PROFILES["sol"])[2] for a in ANCHORS]
    assert median(ratios) > 1.2  # aggregate optimism is robustly > 1


def test_typical_brackets_one():
    """The fitted `typical` profile brings the optimism ratios to bracket 1."""
    ratios = [run_anchor(a, PROFILES["typical"])[2] for a in ANCHORS]
    assert 0.8 <= median(ratios) <= 1.2


def test_optimism_ratio_inverts_for_latency_metrics():
    """A faster (lower) simulated latency reads as optimistic (>1); a higher
    simulated throughput reads as optimistic (>1)."""
    assert optimism_ratio(200.0, 100.0, "output_tok_s") == pytest.approx(2.0)
    assert optimism_ratio(50.0, 100.0, "tpot_ms") == pytest.approx(2.0)  # sim faster
    assert optimism_ratio(200.0, 100.0, "tpot_ms") == pytest.approx(0.5)  # sim slower


def test_anchor_validation_rejects_bad_metric_and_regime():
    with pytest.raises(ValueError):
        Anchor(name="x", hardware_key="h100", model_key="llama-3.1-8b",
               metric="bogus", measured=1.0, source="")
    with pytest.raises(ValueError):
        Anchor(name="x", hardware_key="h100", model_key="llama-3.1-8b",
               metric="tpot_ms", measured=1.0, source="", regime="bogus")
