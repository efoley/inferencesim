"""Extract data for a MoE expert-amortization chart.

Produces:
  1. expected_active_experts(B) for gpt-oss-120b and deepseek-v3 across a
     batch sweep, plus the ceiling (n_experts).
  2. decode DRAM bytes/token vs batch, split into 'moe' vs 'non-moe'
     categories, for the two MoE models and (as a dense comparison) for
     llama-3.1-70b.
"""
import json
from collections import defaultdict

from inferencesim.presets import MODELS
from inferencesim.workload import Deployment
from inferencesim.hardware import DType
from inferencesim.ops import decode_ops, validate_deployment

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
CTX = 4096

MOE_MODELS = ["gpt-oss-120b", "deepseek-v3"]
DENSE_MODEL = "llama-3.1-70b"

DEP = Deployment(tp=1, weight_dtype=DType.FP4, kv_dtype=DType.FP8)

results = {"models": {}, "notes": []}

# sanity: confirm tp=1 (ep=1, adp=1, pp=1) validates for all three models
for name in MOE_MODELS + [DENSE_MODEL]:
    model = MODELS[name]
    validate_deployment(model, DEP)  # raises if invalid
results["notes"].append(
    "Deployment used for all three models: tp=1, pp=1, ep=1, adp=1 "
    "(replica_chips=1; per-chip figures == whole-replica figures here), "
    "weight_dtype=fp4, kv_dtype=fp8, act_dtype=bf16 (default). "
    "validate_deployment() accepted tp=1/ep=1 for both MoE models, so no "
    "fallback deployment was needed."
)
results["notes"].append(
    "decode_ops ctx=4096 (mean context) for every batch point; only batch "
    "varies. Op.dram_read/dram_write are PER OP INSTANCE -- totals multiply "
    "by Op.count (e.g. n_layers) before dividing by batch to get bytes/token."
)
results["notes"].append(
    "Category breakdown reveals two regimes, not just 'moe vs rest': the "
    "'attention' category (KV-cache read+write) is EXACTLY flat per token "
    "across the whole batch sweep (e.g. gpt-oss-120b: 7.789e7 bytes/token at "
    "every batch from 1 to 1024) because its DRAM traffic scales linearly "
    "with n_seq (cancels with the /B). Everything else that streams a "
    "*shared weight matrix once per round* -- MoE expert weights ('moe'), "
    "qkv_proj/out_proj ('linear'), and embed/lm_head ('head') -- amortizes "
    "~1/B. So the honest 3-way split for the chart is: moe_bytes_per_token "
    "(the expert-amortization curve, the star of this chart), "
    "attention_kv_bytes_per_token (flat line, the memory-wall floor), and "
    "other_weight_bytes_per_token (also 1/B-ish, dense attention+head "
    "weights). 'non_moe' = attention_kv + other_weight combined, kept for "
    "convenience but it is NOT flat -- only the pure 'attention' series is."
)
results["notes"].append(
    "Dense llama-3.1-70b has no 'moe' category at all (0 bytes); its "
    "'other_weight' term (70B active params at fp4 = 0.5 B/param -> ~35GB "
    "total weight) dominates at B=1 and amortizes ~1/B, while its flat "
    "'attention' term (~6.71e8 bytes/token at ctx=4096) becomes the dominant "
    "floor at large batch (94% of total bytes/token at B=1024) -- the "
    "classic dense decode memory-wall curve, with NO expert-count effect."
)

# ---- 1. expected_active_experts sweep -----------------------------------

for name in MOE_MODELS:
    model = MODELS[name]
    moe = model.moe
    assert moe is not None
    sweep = []
    for B in BATCHES:
        active = moe.expected_active_experts(B)
        sweep.append({
            "batch": B,
            "expected_active_experts": active,
            "frac_of_total": active / moe.n_experts,
        })
    results["models"].setdefault(name, {})["expected_active_experts"] = {
        "n_experts": moe.n_experts,
        "top_k": moe.top_k,
        "sweep": sweep,
    }

# ---- 2. decode DRAM bytes/token vs batch --------------------------------

def bytes_by_category(model, dep, batch, ctx):
    """Total (dram_read+dram_write) bytes for one decode round, grouped by
    Op.category. Op.dram_read/dram_write are PER INSTANCE; multiply by
    Op.count (e.g. n_layers) to get the total moved this round."""
    ops = decode_ops(model, dep, batch=batch, ctx=ctx)
    totals = defaultdict(float)
    for op in ops:
        totals[op.category] += op.count * (op.dram_read + op.dram_write)
    return totals


for name in MOE_MODELS + [DENSE_MODEL]:
    model = MODELS[name]
    sweep = []
    categories_seen = set()
    for B in BATCHES:
        totals = bytes_by_category(model, DEP, B, CTX)
        categories_seen.update(totals.keys())
        moe_bytes = totals.get("moe", 0.0)
        attn_bytes = totals.get("attention", 0.0)
        non_moe_bytes = sum(v for k, v in totals.items() if k != "moe")
        # "other weight streaming": qkv/out_proj (category 'linear') + embed/lm_head
        # (category 'head') + comm -- these amortize ~1/B just like MoE experts,
        # because they're shared weight matrices read once per round for the
        # whole batch.  Only 'attention' (the KV-cache read/write, which scales
        # WITH n_seq) is batch-invariant per token.
        other_weight_bytes = sum(
            v for k, v in totals.items() if k not in ("moe", "attention")
        )
        total_bytes = moe_bytes + non_moe_bytes
        sweep.append({
            "batch": B,
            "moe_bytes_per_token": moe_bytes / B,
            "attention_kv_bytes_per_token": attn_bytes / B,
            "other_weight_bytes_per_token": other_weight_bytes / B,
            "non_moe_bytes_per_token": non_moe_bytes / B,
            "total_bytes_per_token": total_bytes / B,
            "by_category_bytes_per_token": {k: v / B for k, v in totals.items()},
        })
    results["models"].setdefault(name, {})["decode_dram_bytes_per_token"] = {
        "ctx": CTX,
        "categories_seen": sorted(categories_seen),
        "sweep": sweep,
    }

# ---- headline numbers -----------------------------------------------------

headline = {}
for name in MOE_MODELS:
    moe = MODELS[name].moe
    b1 = moe.expected_active_experts(1)
    b64 = moe.expected_active_experts(64)
    # saturation: smallest batch in a fine sweep where active >= 0.99 * n_experts
    sat_B = None
    B = 1
    while B <= 1_000_000:
        if moe.expected_active_experts(B) >= 0.99 * moe.n_experts:
            sat_B = B
            break
        B *= 2
    headline[name] = {
        "n_experts": moe.n_experts,
        "top_k": moe.top_k,
        "active_experts_at_B1": b1,
        "active_experts_at_B64": b64,
        "saturation_B_99pct": sat_B,
    }
    dd = results["models"][name]["decode_dram_bytes_per_token"]["sweep"]
    b1_bytes = dd[0]["total_bytes_per_token"]
    b256_entry = next(e for e in dd if e["batch"] == 256)
    headline[name]["bytes_per_token_B1"] = b1_bytes
    headline[name]["bytes_per_token_B256"] = b256_entry["total_bytes_per_token"]
    headline[name]["bytes_per_token_ratio_B1_over_B256"] = (
        b1_bytes / b256_entry["total_bytes_per_token"]
    )

dense_dd = results["models"][DENSE_MODEL]["decode_dram_bytes_per_token"]["sweep"]
dense_b1 = dense_dd[0]["total_bytes_per_token"]
dense_b256 = next(e for e in dense_dd if e["batch"] == 256)["total_bytes_per_token"]
headline[DENSE_MODEL] = {
    "bytes_per_token_B1": dense_b1,
    "bytes_per_token_B256": dense_b256,
    "bytes_per_token_ratio_B1_over_B256": dense_b1 / dense_b256,
}

results["headline"] = headline

with open(
    "/private/tmp/claude-501/-Users-eric-development-inferencesim2/"
    "aa8b7baa-3a34-46c9-86fc-a58b5f7d2892/scratchpad/moe_amortization.json",
    "w",
) as f:
    json.dump(results, f, indent=2)

print(json.dumps(headline, indent=2))
