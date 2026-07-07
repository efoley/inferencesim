"""Extract KV-cache anatomy data (Panel A: bytes/token across attention
variants; Panel B: per-chip KV wall under tp vs adp) from inferencesim.

Read-only: does not modify the repo.
"""
import json
import traceback

from inferencesim.hardware import DType
from inferencesim.presets import MODELS
from inferencesim.workload import Deployment
from inferencesim.ops import kv_cache_bytes_per_chip, validate_deployment

FP8 = DType.FP8
LONG_CTX = 131072

out = {"panel_a": {}, "panel_b": {}, "notes": []}

# ---------------------------------------------------------------------------
# PANEL A: KV bytes/token across attention variants
# ---------------------------------------------------------------------------
panel_a_models = ["llama-3.1-8b", "llama-3.1-70b", "gpt-oss-120b", "deepseek-v3"]

for key in panel_a_models:
    m = MODELS[key]
    uncapped = m.kv_bytes_per_token(FP8)
    dep1 = Deployment(tp=1, kv_dtype=FP8)
    effective_total = kv_cache_bytes_per_chip(m, n_tokens=LONG_CTX, dep=dep1)
    effective_per_token = effective_total / LONG_CTX

    # concrete workload: batch=32, context=32768 (total KV, not per-chip;
    # dep=Deployment(tp=1) so per-chip == total for one replica)
    ctx32k = 32768
    batch = 32
    total_kv_32k_1seq = kv_cache_bytes_per_chip(m, n_tokens=ctx32k, dep=dep1)
    total_kv_32k_workload = total_kv_32k_1seq * batch

    entry = {
        "name": m.name,
        "n_layers": m.n_layers,
        "n_kv_heads": m.n_kv_heads,
        "d_head": m.d_head,
        "kv_bytes_per_token_uncapped_fp8": uncapped,
        "kv_bytes_per_token_effective_at_131072ctx_fp8": effective_per_token,
        "ratio_effective_over_uncapped": effective_per_token / uncapped,
        "swa_window": m.swa_window,
        "swa_every": m.swa_every,
        "n_swa_layers": m.n_swa_layers,
        "n_full_attn_layers": m.n_full_attn_layers,
        "mla": None,
        "workload_batch32_ctx32768_total_kv_bytes_fp8": total_kv_32k_workload,
        "workload_batch32_ctx32768_total_kv_bytes_fp8_human_GB": total_kv_32k_workload / 1e9,
    }
    if m.mla is not None:
        entry["mla"] = {
            "kv_lora_rank": m.mla.kv_lora_rank,
            "qk_rope_head_dim": m.mla.qk_rope_head_dim,
            "qk_nope_head_dim": m.mla.qk_nope_head_dim,
            "v_head_dim": m.mla.v_head_dim,
            "q_lora_rank": m.mla.q_lora_rank,
            "latent_dim": m.mla.latent_dim,
        }
    out["panel_a"][key] = entry

# cross-model headline ratios
llama70 = out["panel_a"]["llama-3.1-70b"]
llama8 = out["panel_a"]["llama-3.1-8b"]
gptoss = out["panel_a"]["gpt-oss-120b"]
dsv3 = out["panel_a"]["deepseek-v3"]

out["panel_a_ratios"] = {
    "deepseek_mla_vs_llama70_gqa_uncapped_bytes_per_token": (
        dsv3["kv_bytes_per_token_uncapped_fp8"] / llama70["kv_bytes_per_token_uncapped_fp8"]
    ),
    "deepseek_mla_vs_llama70_gqa_effective_bytes_per_token": (
        dsv3["kv_bytes_per_token_effective_at_131072ctx_fp8"]
        / llama70["kv_bytes_per_token_effective_at_131072ctx_fp8"]
    ),
    "gptoss_effective_over_uncapped": gptoss["ratio_effective_over_uncapped"],
    "gptoss_uncapped_vs_llama70_uncapped": (
        gptoss["kv_bytes_per_token_uncapped_fp8"] / llama70["kv_bytes_per_token_uncapped_fp8"]
    ),
    "gptoss_effective_vs_llama70_effective": (
        gptoss["kv_bytes_per_token_effective_at_131072ctx_fp8"]
        / llama70["kv_bytes_per_token_effective_at_131072ctx_fp8"]
    ),
    "llama70_vs_llama8_uncapped_bytes_per_token": (
        llama70["kv_bytes_per_token_uncapped_fp8"] / llama8["kv_bytes_per_token_uncapped_fp8"]
    ),
}

# ---------------------------------------------------------------------------
# PANEL B: per-chip KV wall: tp vs adp (llama-3.1-70b) + MLA flatness (deepseek-v3)
# ---------------------------------------------------------------------------
llama = MODELS["llama-3.1-70b"]
ctx = 131072
batch = 32

series1 = []  # tp sweep, adp=1
for T in [1, 2, 4, 8, 16, 32]:
    dep = Deployment(tp=T, kv_dtype=FP8)
    rec = {"tp": T}
    try:
        validate_deployment(llama, dep)
        rec["validate_deployment_ok"] = True
        per_seq = kv_cache_bytes_per_chip(llama, n_tokens=ctx, dep=dep)
        rec["per_chip_kv_bytes_1seq"] = per_seq
        rec["per_chip_kv_bytes_batch32"] = per_seq * batch
        rec["kv_heads_per_chip"] = llama.n_kv_heads / min(T, llama.n_kv_heads)
    except Exception as e:
        rec["validate_deployment_ok"] = False
        rec["error_type"] = type(e).__name__
        rec["error_text"] = str(e)
    series1.append(rec)

series2 = []  # tp=8 fixed, adp sweep
for A in [1, 2, 4, 8]:
    dep = Deployment(tp=8, adp=A, kv_dtype=FP8)
    rec = {"tp": 8, "adp": A}
    try:
        validate_deployment(llama, dep)
        rec["validate_deployment_ok"] = True
        per_seq = kv_cache_bytes_per_chip(llama, n_tokens=ctx, dep=dep)
        rec["per_chip_kv_bytes_1seq"] = per_seq
        rec["per_chip_kv_bytes_batch32"] = per_seq * batch
    except Exception as e:
        rec["validate_deployment_ok"] = False
        rec["error_type"] = type(e).__name__
        rec["error_text"] = str(e)
    series2.append(rec)

# deepseek-v3 MLA under tp sweep: latent replicated, should be flat
dsv3_model = MODELS["deepseek-v3"]
series3 = []
for T in [1, 2, 4, 8]:
    dep = Deployment(tp=T, kv_dtype=FP8)
    rec = {"tp": T}
    try:
        validate_deployment(dsv3_model, dep)
        rec["validate_deployment_ok"] = True
        per_seq = kv_cache_bytes_per_chip(dsv3_model, n_tokens=ctx, dep=dep)
        rec["per_chip_kv_bytes_1seq"] = per_seq
        rec["per_chip_kv_bytes_batch32"] = per_seq * batch
    except Exception as e:
        rec["validate_deployment_ok"] = False
        rec["error_type"] = type(e).__name__
        rec["error_text"] = str(e)
    series3.append(rec)

# Also confirm adp>1 on an MoE model (deepseek) is rejected -- documents why
# ADP series 2 in panel B is llama-only (dense). Included as a note/check.
adp_on_moe_check = {}
try:
    validate_deployment(dsv3_model, Deployment(tp=1, adp=2))
    adp_on_moe_check["ok"] = True
except Exception as e:
    adp_on_moe_check["ok"] = False
    adp_on_moe_check["error_type"] = type(e).__name__
    adp_on_moe_check["error_text"] = str(e)

out["panel_b"] = {
    "model_dense_gqa": "llama-3.1-70b",
    "n_kv_heads": llama.n_kv_heads,
    "context_tokens": ctx,
    "batch": batch,
    "kv_dtype": "fp8",
    "series1_tp_sweep_adp1": series1,
    "series2_tp8_adp_sweep": series2,
    "model_mla": "deepseek-v3",
    "series3_mla_tp_sweep": series3,
    "adp_on_moe_model_rejected_check": adp_on_moe_check,
}

# headline numbers for series1: plateau ratio past tp=8
tp8 = next(r for r in series1 if r["tp"] == 8)
tp16 = next(r for r in series1 if r["tp"] == 16)
tp32 = next(r for r in series1 if r["tp"] == 32)
out["panel_b"]["tp_wall_summary"] = {
    "tp8_per_chip_kv_bytes_batch32": tp8.get("per_chip_kv_bytes_batch32"),
    "tp16_per_chip_kv_bytes_batch32": tp16.get("per_chip_kv_bytes_batch32"),
    "tp32_per_chip_kv_bytes_batch32": tp32.get("per_chip_kv_bytes_batch32"),
    "tp16_over_tp8_ratio": (
        tp16.get("per_chip_kv_bytes_batch32", 0) / tp8.get("per_chip_kv_bytes_batch32", 1)
        if tp8.get("validate_deployment_ok") and tp16.get("validate_deployment_ok") else None
    ),
    "tp32_over_tp8_ratio": (
        tp32.get("per_chip_kv_bytes_batch32", 0) / tp8.get("per_chip_kv_bytes_batch32", 1)
        if tp8.get("validate_deployment_ok") and tp32.get("validate_deployment_ok") else None
    ),
}

mla_tp1 = series3[0]
mla_tp8 = series3[-1]
out["panel_b"]["mla_flatness_check"] = {
    "tp1_per_chip_kv_bytes_batch32": mla_tp1.get("per_chip_kv_bytes_batch32"),
    "tp8_per_chip_kv_bytes_batch32": mla_tp8.get("per_chip_kv_bytes_batch32"),
    "ratio_tp8_over_tp1": (
        mla_tp8.get("per_chip_kv_bytes_batch32", 0) / mla_tp1.get("per_chip_kv_bytes_batch32", 1)
    ),
}

# ---------------------------------------------------------------------------
# notes
# ---------------------------------------------------------------------------
out["notes"] = [
    "kv_bytes_per_token(dtype) [workload.py:246] is the UNCAPPED whole-model "
    "per-token growth rate: n_layers * 2 * n_kv_heads * d_head * dtype.bytes "
    "for GQA/dense/SWA models (SWA's cap is NOT applied here -- it assumes "
    "every layer is full-context); for MLA it is n_layers * latent_dim * "
    "dtype.bytes where latent_dim = kv_lora_rank + qk_rope_head_dim.",
    "kv_cache_bytes_per_chip(model, n_tokens, dep) [ops.py:101] IS where the "
    "SWA cap bites: for SWA models, cached = n_full_attn_layers * n_tokens + "
    "n_swa_layers * min(n_tokens, swa_window); at n_tokens=131072 >> "
    "swa_window=128, the windowed layers contribute a constant 128 tokens' "
    "worth instead of growing, so effective bytes/token trends toward the "
    "full-attention-layers-only rate as context grows.",
    "gpt-oss-120b has swa_every=2 (18 of 36 layers windowed at window=128); "
    "at 131072 ctx the 18 windowed layers cache ~128 tokens each instead of "
    "131072, collapsing their contribution to ~0.1% of the unwindowed value -- "
    "this is why effective bytes/token is roughly half the uncapped rate at "
    "long context (the other 18 full-attention layers still scale with ctx).",
    "For GQA/dense models (including SWA variants), kv_cache_bytes_per_chip "
    "DOES divide by tp via _kv_heads_per_chip = n_kv_heads / min(tp, "
    "n_kv_heads) [ops.py:95-98]: KV heads shard up to n_kv_heads ways, then "
    "further tp beyond n_kv_heads REPLICATES the (already-owned) KV heads "
    "instead of sharding further -- so per-chip KV plateaus, it does not "
    "error. validate_deployment [ops.py:80] has no tp-vs-n_kv_heads check at "
    "all (only checks tp,pp,ep,adp >= 1 and the ep/adp <-> MoE/dense "
    "exclusivity) -- so tp=16 and tp=32 on an 8-KV-head model run without "
    "error or warning, silently wasting the extra TP ranks on KV memory (each "
    "extra doubling beyond tp=8 buys zero further KV reduction).",
    "For MLA (deepseek-v3), kv_cache_bytes_per_chip does NOT divide by tp at "
    "all [ops.py:119-122]: 'per_token = n_layers * latent_dim * kv_dtype.bytes; "
    "return per_token * n_tokens / groups' where groups = pp*ep*adp -- tp is "
    "absent from the formula entirely because the compressed latent is "
    "replicated across the tp group (comment: 'replicated across tp (it is "
    "tiny)'). Confirmed empirically: per-chip KV bytes are byte-for-byte "
    "identical across tp=1,2,4,8.",
    "adp>1 is dense-only; calling validate_deployment on the MoE deepseek-v3 "
    "model with adp=2 raises ValueError (text captured in "
    "panel_b.adp_on_moe_model_rejected_check) -- this is why panel B's "
    "adp sweep (series2) uses llama-3.1-70b (dense), not deepseek-v3.",
    "Deployment() default kv_dtype is bf16; all figures here explicitly pass "
    "kv_dtype=fp8 (1 byte/element) per the task spec, so 'fp8' KV bytes are "
    "already 1 byte/scalar (no extra dtype conversion needed).",
    "workload_batch32_ctx32768 total KV figures use dep=Deployment(tp=1) so "
    "per-chip == whole-model total KV for one replica (no sharding applied); "
    "this is meant as an unsharded reference figure, not a deployment "
    "recommendation.",
]

with open(
    "/private/tmp/claude-501/-Users-eric-development-inferencesim2/"
    "aa8b7baa-3a34-46c9-86fc-a58b5f7d2892/scratchpad/kv_anatomy.json",
    "w",
) as f:
    json.dump(out, f, indent=2)

print("wrote kv_anatomy.json")
print(json.dumps(out["panel_a_ratios"], indent=2))
print(json.dumps(out["panel_b"]["tp_wall_summary"], indent=2))
print(json.dumps(out["panel_b"]["mla_flatness_check"], indent=2))
print(json.dumps(out["panel_b"]["adp_on_moe_model_rejected_check"], indent=2))
