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
