"""Discrete-event engine vs the analytic roofline engine."""

import pytest

from inferencesim.des import DESEngine
from inferencesim.engine import RooflineEngine
from inferencesim.hardware import DType
from inferencesim.presets import DGX_H100, GB300_NVL72, LLAMA_3_1_70B
from inferencesim.report import format_report
from inferencesim.simulate import simulate
from inferencesim.workload import Deployment, Scenario


def _run(dep, scen, engine, system=DGX_H100, model=LLAMA_3_1_70B):
    return simulate(system, model, scen, dep, engine=engine)


# ---- DES vs analytic --------------------------------------------------------


def test_des_matches_roofline_when_everything_is_serial():
    """tp-only (pp=1): the task graph is one serial chain, so DES must equal
    the analytic sum exactly."""
    dep = Deployment(tp=8, weight_dtype=DType.FP8)
    scen = Scenario(batch=32, prompt_len=2048, output_len=512)
    a = _run(dep, scen, RooflineEngine())
    d = _run(dep, scen, DESEngine())
    assert d.tpot_s == pytest.approx(a.tpot_s, rel=1e-9)
    assert d.ttft_s == pytest.approx(a.ttft_s, rel=1e-9)


def _tiny_model():
    """8 layers, tiny vocab (so the LM head doesn't dominate a stage), but
    wide enough that a layer's weight streaming (~70us) dwarfs link latency
    (~1us) -- the per-layer pipeline structure is what these tests isolate."""
    from inferencesim.workload import ModelSpec

    return ModelSpec(
        name="tiny", n_layers=8, d_model=4096, n_heads=32, n_kv_heads=8,
        d_head=128, d_ff=16384, vocab_size=1000,
    )


def test_des_balanced_pipeline_close_to_analytic():
    """pp=4 divides 8 layers evenly: the simulated round period should match
    the analytic balanced-stage formula closely."""
    dep = Deployment(tp=1, pp=4, weight_dtype=DType.FP8)
    scen = Scenario(batch=8, prompt_len=512, output_len=128)
    a = _run(dep, scen, RooflineEngine(), model=_tiny_model())
    d = _run(dep, scen, DESEngine(), model=_tiny_model())
    assert d.tpot_s == pytest.approx(a.tpot_s, rel=0.03)


def test_des_charges_for_unbalanced_stages():
    """pp=3 on 8 layers -> stages of 3/3/2.  The analytic engine assumes
    balance (and warns); in the DES the 3-layer stages set the round period:
    3 stages x 3 layers = 9 layer-times, vs 4 x 2 = 8 for balanced pp=4 at
    the same microbatch size."""
    scen3 = Scenario(batch=6, prompt_len=512, output_len=128)   # microbatch 2
    scen4 = Scenario(batch=8, prompt_len=512, output_len=128)   # microbatch 2
    a3 = _run(Deployment(tp=1, pp=3, weight_dtype=DType.FP8), scen3,
              RooflineEngine(), model=_tiny_model())
    d3 = _run(Deployment(tp=1, pp=3, weight_dtype=DType.FP8), scen3,
              DESEngine(), model=_tiny_model())
    d4 = _run(Deployment(tp=1, pp=4, weight_dtype=DType.FP8), scen4,
              DESEngine(), model=_tiny_model())
    assert any("does not divide" in w for w in a3.warnings)
    assert d3.tpot_s > a3.tpot_s  # unbalance costs real time
    # not exactly 9/8: the (small) LM head rides the last stage, which is
    # the bottleneck stage under pp=4 but not under pp=3
    assert d3.tpot_s / d4.tpot_s == pytest.approx(9 / 8, rel=0.04)


def test_des_overlaps_the_lm_head_the_analytic_engine_serialises():
    """With a real vocab the LM head is ~a layer's worth of weight streaming
    on the last stage only.  The analytic engine adds all pp executions of it
    to the round; the DES overlaps them with the other stages' work, so DES
    is faster here despite charging for the 12/11 stage imbalance."""
    dep = Deployment(tp=1, pp=7, weight_dtype=DType.FP8)
    scen = Scenario(batch=28, prompt_len=2048, output_len=512)
    a = _run(dep, scen, RooflineEngine(), system=GB300_NVL72)
    d = _run(dep, scen, DESEngine(), system=GB300_NVL72)
    assert d.tpot_s < a.tpot_s
    assert d.tpot_s == pytest.approx(a.tpot_s, rel=0.10)  # same ballpark


def test_des_prefill_walks_stages_serially():
    """A single request can't pipeline: prefill time should match the
    analytic sum (plus nothing -- hops are on the serial path)."""
    dep = Deployment(tp=2, pp=4, weight_dtype=DType.FP8)
    scen = Scenario(batch=64, prompt_len=4096, output_len=512)
    a = _run(dep, scen, RooflineEngine())
    d = _run(dep, scen, DESEngine())
    assert d.ttft_s == pytest.approx(a.ttft_s, rel=1e-9)


def test_des_with_moe_and_ep():
    """MoE + EP, pp=1 -> a serial chain.  The switched all-to-all is now
    store-and-forward (egress `.out` then ingress `.in`), so each dispatch and
    combine costs the closed form the roofline charges PLUS exactly one message
    of fill (comm_bytes/((g-1)*bw)); everything else costs identically.  With no
    pipeline overlap or contention (pp=1), the DES decode round therefore equals
    the roofline round plus 2*L such fills (dispatch + combine per layer) -- the
    documented, bounded a2a gap, asserted here to full precision as the new
    exact anchor (skew=0 -> uniform messages, no incast)."""
    from inferencesim.engine import CommContext
    from inferencesim.ops import decode_step_ops
    from inferencesim.presets import GPT_OSS_120B
    dep = Deployment(tp=4, ep=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
    scen = Scenario(batch=128, prompt_len=2048, output_len=512)
    d = simulate(GB300_NVL72, GPT_OSS_120B, scen, dep, engine=DESEngine())
    a = simulate(GB300_NVL72, GPT_OSS_120B, scen, dep, engine=RooflineEngine())
    comm = CommContext.for_deployment(GB300_NVL72, dep)
    disp = next(o for o in decode_step_ops(GPT_OSS_120B, scen, dep)
                if o.name == "moe_dispatch")
    occ = (disp.comm_bytes / (comm.a2a - 1)) / comm.a2a_link.bandwidth  # one-message fill
    fill = 2 * GPT_OSS_120B.n_layers * occ  # dispatch + combine, once per layer
    assert d.tpot_s == pytest.approx(a.tpot_s + fill, rel=1e-9)  # the new exact value
    assert a.tpot_s < d.tpot_s <= a.tpot_s + fill * (1 + 1e-9)  # bounded by the fill
    assert d.output_tokens_per_s > 0


def _gpt_oss_skew(skew):
    from dataclasses import replace

    from inferencesim.presets import GPT_OSS_120B
    return replace(GPT_OSS_120B, moe=replace(GPT_OSS_120B.moe, skew=skew))


def _des2():
    # pp=1 is a serial chain, so 2 rounds measure the exact period (fast).
    return DESEngine(decode_rounds=2, warmup=1)


def test_moe_skew0_is_bit_identical_anchor():
    """Explicit skew=0 reproduces the historical numbers exactly on BOTH engines:
    the analytic roofline takes the uniform lowering branch (bit-identical), and
    the DES builds the identical task graph as the base preset (uniform a2a, no
    incast) -- the degenerate anchor."""
    from inferencesim.presets import GPT_OSS_120B
    dep = Deployment(tp=1, ep=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
    scen = Scenario(batch=128, prompt_len=2048, output_len=512)
    base_a = simulate(GB300_NVL72, GPT_OSS_120B, scen, dep, engine=RooflineEngine())
    skew0_a = simulate(GB300_NVL72, _gpt_oss_skew(0.0), scen, dep, engine=RooflineEngine())
    assert skew0_a.tpot_s == base_a.tpot_s  # bit-identical analytic anchor
    base_d = simulate(GB300_NVL72, GPT_OSS_120B, scen, dep, engine=_des2())
    skew0_d = simulate(GB300_NVL72, _gpt_oss_skew(0.0), scen, dep, engine=_des2())
    assert skew0_d.tpot_s == base_d.tpot_s


def test_moe_hot_expert_worsens_tpot_and_lights_ingress():
    """gpt-oss DEP8 at skew=1.0: the hot experts cost real TPOT (weight-streaming
    pacing + all-to-all incast), and the hottest member's ingress port `.in`
    tops the fabric ingress utilisation -- the observable of the incast."""
    dep = Deployment(tp=1, ep=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
    scen = Scenario(batch=128, prompt_len=2048, output_len=512)
    d0 = simulate(GB300_NVL72, _gpt_oss_skew(0.0), scen, dep, engine=_des2())
    d1 = simulate(GB300_NVL72, _gpt_oss_skew(1.0), scen, dep, engine=_des2())
    assert d1.tpot_s > d0.tpot_s  # hot experts cost real TPOT
    util = d1.resource_util["decode"]
    ins = {k: v for k, v in util.items() if k.endswith(".in")}
    assert ins  # switched a2a -> ingress ports present, surfaced (not sync-filtered)
    assert max(ins, key=ins.get) == "s0.l0.in"  # hot member (block 0) tops ingress
    assert util["s0.l0.in"] > util["s0.l7.in"]  # coldest member's ingress lighter


def test_moe_tpot_monotone_in_skew():
    """TPOT increases monotonically with skew (two interior points suffice)."""
    dep = Deployment(tp=1, ep=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
    scen = Scenario(batch=128, prompt_len=2048, output_len=512)
    tpots = [simulate(GB300_NVL72, _gpt_oss_skew(s), scen, dep, engine=_des2()).tpot_s
             for s in (0.0, 0.6, 1.2)]
    assert tpots[0] < tpots[1] < tpots[2]


def test_des_ring_fabric_moe_free_is_sane():
    """tt-quietbox is a RING of Blackholes.  A dense tp=2 pp=2 run exercises
    the ring allreduce (g=2) and pipeline hops sharing member 0's link, with
    no MoE all-to-all.  The DES stays finite and positive and in the same
    ballpark as the analytic engine.

    This pp=2 pipeline fills slowly: convergence control grows the measurement
    out to ~256 rounds, where the steady-state period settles at ~roofline (a
    hair below it -- microbatch overlap lets the DES beat the serial roofline
    sum slightly).  A fixed 16-round run instead reads ~1.65x higher purely
    because the fill transient has not yet decayed -- exactly the artefact
    convergence control removes.  The g=2 allreduce expansion itself still
    reproduces the old closed-form timing exactly, so the collective model is
    unchanged here."""
    from inferencesim.presets import TT_QUIETBOX
    dep = Deployment(tp=2, pp=2, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    scen = Scenario(batch=32, prompt_len=2048, output_len=512)
    d = simulate(TT_QUIETBOX, LLAMA_3_1_70B, scen, dep, engine=DESEngine())
    a = simulate(TT_QUIETBOX, LLAMA_3_1_70B, scen, dep, engine=RooflineEngine())
    assert d.tpot_s > 0 and d.ttft_s > 0
    assert 0.9 * a.tpot_s <= d.tpot_s <= 3 * a.tpot_s


# ---- convergence control ----------------------------------------------------


def test_auto_convergence_pp1_serial_is_exact_and_immediate():
    """Auto mode (the new default) on a serial pp=1 chain: the round period is
    exact from the first round, so successive estimates agree immediately, the
    engine reports converged, and TPOT equals roofline to full precision."""
    dep = Deployment(tp=8, weight_dtype=DType.FP8)
    scen = Scenario(batch=32, prompt_len=2048, output_len=512)
    a = _run(dep, scen, RooflineEngine())
    engine = DESEngine()
    d = _run(dep, scen, engine)
    assert d.tpot_s == pytest.approx(a.tpot_s, rel=1e-9)
    assert engine.last_convergence["converged"] is True
    assert engine.last_convergence["rel_delta"] < 1e-9  # exact from round 1


def test_auto_convergence_unbalanced_pipeline_matches_long_fixed_run():
    """Auto mode on an unbalanced pp=4 pipeline (10 layers -> 3/3/2/2) grows
    the round count past the small start, reports converged, and lands within
    rtol of a long fixed 128-round run."""
    from inferencesim.workload import ModelSpec

    model = ModelSpec(name="tiny", n_layers=10, d_model=4096, n_heads=32,
                      n_kv_heads=8, d_head=128, d_ff=16384, vocab_size=1000)
    dep = Deployment(tp=1, pp=4, weight_dtype=DType.FP8)
    scen = Scenario(batch=8, prompt_len=512, output_len=128)
    engine = DESEngine()  # auto, rtol=1e-3
    auto = _run(dep, scen, engine, model=model)
    fixed = _run(dep, scen, DESEngine(decode_rounds=128), model=model)
    assert engine.last_convergence["converged"] is True
    assert 8 < engine.last_convergence["rounds"] <= 128  # actually grew
    assert auto.tpot_s == pytest.approx(fixed.tpot_s, rel=engine.rtol)


def test_explicit_rounds_preserve_historical_fixed_behaviour():
    """Explicit decode_rounds/warmup pins a fixed run: warmup left unset
    defaults to decode_rounds // 2 (the historical 16/8 ratio) and gives a
    byte-identical result, the convergence bookkeeping stays empty, and the
    fixed-16 value is distinct from the new auto default -- which is the
    behaviour change this guards."""
    dep = Deployment(tp=1, pp=4, weight_dtype=DType.FP8)
    scen = Scenario(batch=8, prompt_len=512, output_len=128)
    explicit = DESEngine(decode_rounds=16, warmup=8)
    r_explicit = _run(dep, scen, explicit, model=_tiny_model())
    r_implicit = _run(dep, scen, DESEngine(decode_rounds=16), model=_tiny_model())
    assert r_explicit.tpot_s == r_implicit.tpot_s   # byte-identical fixed path
    assert explicit.last_convergence is None         # nothing auto-grown
    # the new default (auto) genuinely differs from the old fixed-16 default
    r_auto = _run(dep, scen, DESEngine(), model=_tiny_model())
    assert abs(r_auto.tpot_s - r_explicit.tpot_s) / r_explicit.tpot_s > 1e-4


def test_auto_convergence_reports_cap_hit_without_converging():
    """A slow-filling pipeline capped below its settling point reports
    converged=False and the round count it stopped at, rather than printing or
    raising (the engine has no channel to the report's warnings list)."""
    from inferencesim.presets import TT_QUIETBOX
    dep = Deployment(tp=2, pp=2, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    scen = Scenario(batch=32, prompt_len=2048, output_len=512)
    engine = DESEngine(max_rounds=32)  # far below where this pp=2 ring settles
    simulate(TT_QUIETBOX, LLAMA_3_1_70B, scen, dep, engine=engine)
    assert engine.last_convergence["converged"] is False
    assert engine.last_convergence["rounds"] == 32
    assert engine.last_convergence["rel_delta"] > engine.rtol


def test_explicit_path_validation_still_enforced():
    """The historical guard survives on the explicit path: decode_rounds must
    exceed warmup, and warmup is meaningless without an explicit decode_rounds
    (auto mode chooses it per iteration)."""
    with pytest.raises(ValueError):
        DESEngine(decode_rounds=8, warmup=8)
    with pytest.raises(ValueError):
        DESEngine(warmup=4)


# ---- observability: per-resource utilisation --------------------------------


def test_des_populates_resource_utilisation():
    """A tp>1, pp>1 run exercises the stage execution units (u{s}) and the
    per-member outbound links that now carry both the expanded collectives and
    the pipeline hops; the DES surfaces per-resource utilisation in (0, 1].
    DGX_H100 is a switched (ALL_TO_ALL) fabric, so the links are `.out` egress
    ports.  The dependency-chain sync tasks (barriers / propagation, which
    carry latency rather than link occupancy) are filtered out of the
    report."""
    dep = Deployment(tp=2, pp=2, weight_dtype=DType.FP8)
    scen = Scenario(batch=32, prompt_len=2048, output_len=512)
    engine = DESEngine()
    r = _run(dep, scen, engine)
    assert r.resource_util is not None
    decode_util = r.resource_util["decode"]
    assert any(k.startswith("u") for k in decode_util)     # stage execution units
    assert any(".l0.out" in k for k in decode_util)        # member-0 egress (collective + hops)
    assert any(".l1.out" in k for k in decode_util)        # member-1 egress port
    assert not any(".bar" in k or ".prop" in k for k in decode_util)  # sync filtered
    assert all(0.0 < f <= 1.0 for f in decode_util.values())


def test_report_shows_util_block_for_des_not_roofline():
    dep = Deployment(tp=8, weight_dtype=DType.FP8)
    scen = Scenario(batch=32, prompt_len=2048, output_len=512)
    des_text = format_report(_run(dep, scen, DESEngine()))
    roof_text = format_report(_run(dep, scen, RooflineEngine()))
    assert "resource util" in des_text
    assert "resource util" not in roof_text


# ---- graph mode: walking the expanded chip graph ----------------------------


def _fine_quietbox():
    from inferencesim.bridge import system_from_graph
    from inferencesim.presets import LLAMA_3_1_70B
    from inferencesim.presets_fine import tt_quietbox_fine

    return system_from_graph(tt_quietbox_fine()), LLAMA_3_1_70B


def _graph_engine():
    from inferencesim.presets_fine import blackhole_p150_fine

    return DESEngine(chip_graph=blackhole_p150_fine())


def test_graph_des_refines_lumped_des_never_optimistic():
    """The graph-DES on the fine chip is a strict refinement of the lumped
    stage-DES on the same aggregated system: it runs, TPOT is positive, and
    it is never faster (the tile schedule adds fill/drain, NoC sharing and
    per-core granularity on top of the same roofline totals).  The gap stays
    modest because decode's weight-streaming ops span thousands of tiles, so
    fill/drain amortises away."""
    system, model = _fine_quietbox()
    scen = Scenario(batch=16, prompt_len=2048, output_len=512)
    dep = Deployment(tp=2, pp=2, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    graph = simulate(system, model, scen, dep, engine=_graph_engine())
    lumped = simulate(system, model, scen, dep, engine=DESEngine())
    assert graph.tpot_s > 0
    assert graph.tpot_s >= lumped.tpot_s * (1 - 1e-9)
    assert graph.tpot_s == pytest.approx(lumped.tpot_s, rel=0.25)


def test_graph_des_reports_chip_resource_utilisation():
    """Graph mode surfaces per-chip resources (DRAM banks, NoC, SRAM, cores)
    alongside the stage-level u/c/h entries."""
    system, model = _fine_quietbox()
    scen = Scenario(batch=16, prompt_len=2048, output_len=512)
    dep = Deployment(tp=2, pp=2, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    r = simulate(system, model, scen, dep, engine=_graph_engine())
    decode = r.resource_util["decode"]
    assert any("gddr6-bank" in k for k in decode)   # chip resources present
    assert any(k in ("u0", "u1") for k in decode)   # stage resources retained
    assert all(f > 0.0 for f in decode.values())


def test_lumped_des_has_no_chip_resources():
    """Without a chip_graph the DES is byte-for-byte the old engine: no
    chip-namespaced resources appear."""
    dep = Deployment(tp=2, pp=2, weight_dtype=DType.FP8)
    scen = Scenario(batch=32, prompt_len=2048, output_len=512)
    r = _run(dep, scen, DESEngine())
    assert not any(k.startswith("chip:") for k in r.resource_util["decode"])


def test_graph_des_trace_emits_per_op_tracks():
    import json

    from inferencesim.sched import chrome_trace

    system, model = _fine_quietbox()
    scen = Scenario(batch=16, prompt_len=2048, output_len=512)
    dep = Deployment(tp=2, pp=2, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    engine = _graph_engine()
    simulate(system, model, scen, dep, engine=engine)
    op_runs = engine.last_op_runs["decode"]
    assert op_runs  # graph mode recorded per-op chip schedules
    name, sched = next(iter(op_runs.items()))
    trace = chrome_trace(sched.tasks, sched.result, prefix=f"decode/op:{name}/")
    json.loads(json.dumps(trace))  # valid, serialisable JSON
    procs = [e for e in trace["traceEvents"] if e["ph"] == "M"]
    assert procs and all(
        e["args"]["name"].startswith(f"decode/op:{name}/") for e in procs
    )
