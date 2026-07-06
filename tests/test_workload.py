import pytest

from inferencesim.hardware import DType
from inferencesim.presets import GPT_OSS_120B, LLAMA_3_1_8B, LLAMA_3_1_70B
from inferencesim.workload import MoEConfig


def test_llama_70b_param_count():
    # published: 70.6B
    assert abs(LLAMA_3_1_70B.total_params - 70.6e9) / 70.6e9 < 0.02


def test_llama_8b_param_count():
    # published: 8.03B
    assert abs(LLAMA_3_1_8B.total_params - 8.03e9) / 8.03e9 < 0.02


def test_gpt_oss_120b_counts():
    # published: ~117B total, ~5.1B active
    assert abs(GPT_OSS_120B.total_params - 117e9) / 117e9 < 0.05
    assert abs(GPT_OSS_120B.active_params - 5.1e9) / 5.1e9 < 0.15


def test_kv_bytes_per_token():
    # llama-70b, bf16: 80 layers * 2 * 8 heads * 128 dim * 2 B = 327,680 B
    assert LLAMA_3_1_70B.kv_bytes_per_token(DType.BF16) == 80 * 2 * 8 * 128 * 2


def test_expected_active_experts_bounds():
    moe = MoEConfig(n_experts=128, top_k=4, d_ff_expert=1024)
    assert moe.expected_active_experts(0) == 0
    assert abs(moe.expected_active_experts(1) - 4.0) < 0.07  # ~top_k for one token
    big = moe.expected_active_experts(10_000)
    assert 127.9 < big <= 128.0  # saturates at n_experts
    # monotone in tokens
    assert moe.expected_active_experts(8) < moe.expected_active_experts(64)


# ---- expert-load skew (EP hot-expert imbalance) -----------------------------


def test_moe_skew_validation():
    MoEConfig(n_experts=8, top_k=2, d_ff_expert=64, skew=0.0)  # ok
    MoEConfig(n_experts=8, top_k=2, d_ff_expert=64, skew=1.5)  # ok
    with pytest.raises(ValueError):
        MoEConfig(n_experts=8, top_k=2, d_ff_expert=64, skew=-0.1)


def test_skew0_popularity_is_uniform_and_factor_one():
    """skew=0 is the degenerate anchor: uniform popularity, unit pacing factor,
    and balanced per-member shares."""
    moe = MoEConfig(n_experts=128, top_k=4, d_ff_expert=64, skew=0.0)
    assert moe._popularity() == [1.0 / 128] * 128
    for g in (1, 8, 32):
        mp = moe.member_popularity(g)
        assert sum(mp) == pytest.approx(1.0, rel=1e-12)
        assert mp == pytest.approx([1.0 / g] * g, rel=1e-12)
        assert moe.hot_member_factor(g) == pytest.approx(1.0, rel=1e-12)


def test_skew_concentrates_and_factor_monotone():
    """Positive skew concentrates popularity onto member 0 and raises the pacing
    factor above 1, monotonically in skew."""
    moe0 = MoEConfig(n_experts=128, top_k=4, d_ff_expert=64, skew=0.0)
    moe_a = MoEConfig(n_experts=128, top_k=4, d_ff_expert=64, skew=0.6)
    moe_b = MoEConfig(n_experts=128, top_k=4, d_ff_expert=64, skew=1.2)
    g = 8
    # member 0 (hottest block) carries more than its uniform share, and the tail
    # members less
    mp = moe_b.member_popularity(g)
    assert mp[0] > 1.0 / g > mp[-1]
    assert mp == sorted(mp, reverse=True)  # contiguous blocks -> decreasing
    f0, fa, fb = (m.hot_member_factor(g) for m in (moe0, moe_a, moe_b))
    assert f0 == pytest.approx(1.0)
    assert 1.0 < fa < fb  # strictly increasing in skew


def test_expected_active_on_member_sums_to_global():
    """Per-block expected-active sums to the global (skewed) expectation, and at
    skew=0 reproduces expected_active_experts exactly (each member == global/g)."""
    n_tokens, g = 256, 8
    # skew=0: bit-identical to the uniform closed form
    moe0 = MoEConfig(n_experts=128, top_k=4, d_ff_expert=64, skew=0.0)
    per_member = [moe0.expected_active_on_member(m, n_tokens, g) for m in range(g)]
    assert sum(per_member) == pytest.approx(
        moe0.expected_active_experts(n_tokens), rel=1e-12)
    assert per_member == pytest.approx(
        [moe0.expected_active_experts(n_tokens) / g] * g, rel=1e-12)
    # skew>0: still sums to the global skewed expected-active (sum over experts
    # of 1-(1-min(1,top_k*pop_e))^n), and the hot block has MORE distinct experts
    # saturated than a cold one
    moe = MoEConfig(n_experts=128, top_k=4, d_ff_expert=64, skew=1.2)
    pop = moe._popularity()
    global_active = sum(1.0 - (1.0 - min(1.0, 4 * p)) ** n_tokens for p in pop)
    per_member = [moe.expected_active_on_member(m, n_tokens, g) for m in range(g)]
    assert sum(per_member) == pytest.approx(global_active, rel=1e-12)
    assert per_member[0] >= per_member[-1]


def test_tokens_to_member_matches_popularity():
    moe = MoEConfig(n_experts=64, top_k=4, d_ff_expert=64, skew=0.8)
    g, n = 8, 100
    for m in range(g):
        assert moe.tokens_to_member(m, n, g) == pytest.approx(
            moe.member_popularity(g)[m] * n * 4)


def test_moe_routed_paced_by_hot_member():
    """The moe_routed lowering is bit-identical at skew=0 and paced by the
    hottest member at skew>0: activation flops/writes scale by hot_member_factor
    and the weight read reflects the hot block's expected-active experts.  The
    dispatch/combine ops carry the popularity vector under skew, None without."""
    from dataclasses import replace

    from inferencesim.ops import decode_ops
    from inferencesim.workload import Deployment

    dep = Deployment(tp=1, ep=8)  # DEP8: tp*ep = 8 members
    g, batch = 8, 128
    base = GPT_OSS_120B                                    # skew=0 preset
    skewed = replace(base, moe=replace(base.moe, skew=1.0))
    o0 = {o.name: o for o in decode_ops(base, dep, batch, 2000)}
    os_ = {o.name: o for o in decode_ops(skewed, dep, batch, 2000)}
    hot = skewed.moe.hot_member_factor(g)
    assert hot > 1.0
    # activation flops / writes pace by the hot member
    assert os_["moe_routed"].flops == pytest.approx(o0["moe_routed"].flops * hot)
    assert os_["moe_routed"].dram_write == pytest.approx(
        o0["moe_routed"].dram_write * hot)
    # weight read >= the uniform average (hot block streams its active experts)
    assert os_["moe_routed"].dram_read > o0["moe_routed"].dram_read
    # dispatch/combine carry the popularity vector under skew, None at skew=0
    assert o0["moe_dispatch"].member_weights is None
    assert os_["moe_dispatch"].member_weights == pytest.approx(
        tuple(skewed.moe.member_popularity(g)))
    assert os_["moe_combine"].member_weights == os_["moe_dispatch"].member_weights
    # comm_bytes (per-chip average payload) is unchanged by skew
    assert os_["moe_dispatch"].comm_bytes == pytest.approx(o0["moe_dispatch"].comm_bytes)
