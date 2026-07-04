"""Discrete-event engine vs the analytic roofline engine."""

import pytest

from inferencesim.des import DESEngine, _Task, schedule
from inferencesim.engine import RooflineEngine
from inferencesim.hardware import DType
from inferencesim.presets import DGX_H100, GB300_NVL72, LLAMA_3_1_70B
from inferencesim.simulate import simulate
from inferencesim.workload import Deployment, Scenario


def _run(dep, scen, engine, system=DGX_H100, model=LLAMA_3_1_70B):
    return simulate(system, model, scen, dep, engine=engine)


# ---- scheduler core ---------------------------------------------------------


def test_scheduler_serialises_on_resources_and_respects_deps():
    tasks = [
        _Task(0, "u", 2.0, []),
        _Task(1, "u", 3.0, []),        # same resource: queues behind 0
        _Task(2, "link", 1.0, [0]),    # different resource: overlaps with 1
        _Task(3, "u", 1.0, [2]),
    ]
    finish = schedule(tasks)
    assert finish[0] == 2.0
    assert finish[1] == 5.0
    assert finish[2] == 3.0   # started at 2 on the free link
    assert finish[3] == 6.0   # waited for the unit (busy till 5)


def test_scheduler_detects_cycles():
    with pytest.raises(ValueError, match="cycle"):
        schedule([_Task(0, "u", 1.0, [1]), _Task(1, "u", 1.0, [0])])


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
    from inferencesim.presets import GPT_OSS_120B
    dep = Deployment(tp=4, ep=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
    scen = Scenario(batch=128, prompt_len=2048, output_len=512)
    d = simulate(GB300_NVL72, GPT_OSS_120B, scen, dep, engine=DESEngine())
    a = simulate(GB300_NVL72, GPT_OSS_120B, scen, dep, engine=RooflineEngine())
    # serial chain again (pp=1): must agree
    assert d.tpot_s == pytest.approx(a.tpot_s, rel=1e-9)
    assert d.output_tokens_per_s > 0
