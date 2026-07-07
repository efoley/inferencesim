"""Extract data for the "who can shard MLA" panel: per-chip KV and per-chip
weight bytes for DeepSeek-V3 as a function of parallelism dimension (ep vs
tp), contrasted with llama-3.1-70b's GQA per-chip KV under tp.

Read-only: does not modify the repo.
"""
import json

from inferencesim.hardware import DType
from inferencesim.presets import MODELS, GB300_NVL72, B300
from inferencesim.workload import Deployment
from inferencesim.ops import kv_cache_bytes_per_chip, validate_deployment
from inferencesim.simulate import weight_bytes_per_chip

FP8 = DType.FP8
FP4 = DType.FP4

CTX = 131072
BATCH = 32

ds = MODELS["deepseek-v3"]
llama70 = MODELS["llama-3.1-70b"]

out = {"meta": {}, "series": {}, "notes": [], "headline": {}}

out["meta"] = {
    "model_mla": "deepseek-v3",
    "model_gqa_contrast": "llama-3.1-70b",
    "context_tokens": CTX,
    "batch": BATCH,
    "kv_dtype": "fp8",
    "weight_dtype_for_weight_series": "fp4",
    "GB_definition": "1 GB = 1e9 bytes (matches inferencesim.units.GB)",
    "deepseek_kv_bytes_per_token_fp8": ds.kv_bytes_per_token(FP8),
    "deepseek_mla_latent_dim": ds.mla.latent_dim,
    "deepseek_mla_kv_lora_rank": ds.mla.kv_lora_rank,
    "deepseek_mla_qk_rope_head_dim": ds.mla.qk_rope_head_dim,
    "deepseek_n_layers": ds.n_layers,
    "deepseek_n_experts": ds.moe.n_experts,
    "deepseek_top_k": ds.moe.top_k,
    "deepseek_n_dense_layers": ds.moe.n_dense_layers,
    "deepseek_total_params": ds.total_params,
    "deepseek_active_params": ds.active_params,
    "gb300_chip_name": B300.name,
    "gb300_chip_hbm_capacity_bytes": B300.dram.capacity_bytes,
    "gb300_chip_hbm_capacity_GB": B300.dram.capacity_bytes / 1e9,
    "gb300_system_total_chips": GB300_NVL72.total_chips,
}

# ---------------------------------------------------------------------------
# 1. Per-chip KV bytes, deepseek-v3, tp=1, ep sweep (MLA / groups = pp*ep*adp)
# ---------------------------------------------------------------------------
ep_values = [1, 2, 4, 8, 16, 32, 64]
series_ep_kv = []
for E in ep_values:
    dep = Deployment(tp=1, ep=E, kv_dtype=FP8)
    validate_deployment(ds, dep)  # ep>1 requires MoE model -- ds qualifies; no error expected
    per_seq = kv_cache_bytes_per_chip(ds, n_tokens=CTX, dep=dep)
    per_chip_batch = per_seq * BATCH
    series_ep_kv.append({
        "ep": E,
        "tp": 1,
        "per_chip_kv_bytes_1seq": per_seq,
        "per_chip_kv_bytes_batch32": per_chip_batch,
        "per_chip_kv_GB_batch32": per_chip_batch / 1e9,
    })

# ---------------------------------------------------------------------------
# 2. Per-chip KV bytes, deepseek-v3, tp sweep, ep=1 (should be FLAT: MLA
#    latent is replicated across tp, not sharded)
# ---------------------------------------------------------------------------
tp_values = [1, 2, 4, 8, 16, 32]
series_tp_kv = []
for T in tp_values:
    dep = Deployment(tp=T, ep=1, kv_dtype=FP8)
    validate_deployment(ds, dep)
    per_seq = kv_cache_bytes_per_chip(ds, n_tokens=CTX, dep=dep)
    per_chip_batch = per_seq * BATCH
    fits_gb300 = per_chip_batch <= B300.dram.capacity_bytes
    series_tp_kv.append({
        "tp": T,
        "ep": 1,
        "per_chip_kv_bytes_1seq": per_seq,
        "per_chip_kv_bytes_batch32": per_chip_batch,
        "per_chip_kv_GB_batch32": per_chip_batch / 1e9,
    })

flat_values = {r["per_chip_kv_bytes_batch32"] for r in series_tp_kv}
is_flat = len(flat_values) == 1

# ---------------------------------------------------------------------------
# 3. Per-chip WEIGHT bytes at fp4, deepseek-v3, tp=1, ep sweep
#    (weight_bytes_per_chip from simulate.py -- NOT model.weight_bytes(),
#    which is the whole-model, unsharded total; see notes below)
# ---------------------------------------------------------------------------
series_ep_weights = []
for E in ep_values:
    dep = Deployment(tp=1, ep=E, weight_dtype=FP4)
    per_chip = weight_bytes_per_chip(ds, dep)
    series_ep_weights.append({
        "ep": E,
        "tp": 1,
        "per_chip_weight_bytes_fp4": per_chip,
        "per_chip_weight_GB_fp4": per_chip / 1e9,
        "fits_gb300_alone": per_chip <= B300.dram.capacity_bytes,
    })

# the non-expert "floor" that does NOT shard with ep (replicated attn +
# shared-expert + dense-prefix-FFN weights, at tp=pp=1): what per-chip weight
# bytes converges to as ep -> infinity
nd = ds.moe.n_dense_layers
moe_layers = ds.n_layers - nd
floor_params = (
    ds.embedding_params
    + ds.n_layers * ds.attn_params
    + moe_layers * ds.shared_expert_params
    + nd * ds._dense_ffn_params
)
floor_bytes_fp4 = floor_params * FP4.bytes

# whole-model (unsharded) weight bytes at fp4, for contrast with the
# naive "model.weight_bytes(fp4)" the task description proposed
whole_model_weight_bytes_fp4 = ds.weight_bytes(FP4)

# ---------------------------------------------------------------------------
# 4. Chip capacity line
# ---------------------------------------------------------------------------
gb300_capacity_GB = B300.dram.capacity_bytes / 1e9

# ---------------------------------------------------------------------------
# Contrast series: llama-3.1-70b per-chip KV under tp (same workload)
# ---------------------------------------------------------------------------
series_llama_tp_kv = []
for T in tp_values:
    dep = Deployment(tp=T, kv_dtype=FP8)
    validate_deployment(llama70, dep)
    per_seq = kv_cache_bytes_per_chip(llama70, n_tokens=CTX, dep=dep)
    per_chip_batch = per_seq * BATCH
    series_llama_tp_kv.append({
        "tp": T,
        "per_chip_kv_bytes_1seq": per_seq,
        "per_chip_kv_bytes_batch32": per_chip_batch,
        "per_chip_kv_GB_batch32": per_chip_batch / 1e9,
    })

out["series"] = {
    "deepseek_kv_vs_ep_tp1": series_ep_kv,
    "deepseek_kv_vs_tp_ep1": series_tp_kv,
    "deepseek_weights_fp4_vs_ep_tp1": series_ep_weights,
    "llama70_kv_vs_tp_contrast": series_llama_tp_kv,
}

out["headline"] = {
    "kv_per_chip_GB_ep1": series_ep_kv[0]["per_chip_kv_GB_batch32"],
    "kv_per_chip_GB_ep64": series_ep_kv[-1]["per_chip_kv_GB_batch32"],
    "kv_ep1_over_ep64_ratio": (
        series_ep_kv[0]["per_chip_kv_bytes_batch32"]
        / series_ep_kv[-1]["per_chip_kv_bytes_batch32"]
    ),
    "tp_series_all_equal_bytes": is_flat,
    "tp_series_flat_value_GB": (
        next(iter(flat_values)) / 1e9 if is_flat else None
    ),
    "weights_fp4_per_chip_GB_ep1": series_ep_weights[0]["per_chip_weight_GB_fp4"],
    "weights_fp4_per_chip_GB_ep8": series_ep_weights[3]["per_chip_weight_GB_fp4"],
    "weights_fp4_per_chip_GB_ep64": series_ep_weights[-1]["per_chip_weight_GB_fp4"],
    "weights_fp4_floor_GB_as_ep_to_infinity": floor_bytes_fp4 / 1e9,
    "whole_model_weight_bytes_fp4_GB_unsharded": whole_model_weight_bytes_fp4 / 1e9,
    "gb300_chip_hbm_capacity_GB": gb300_capacity_GB,
    "ep1_weights_exceed_gb300_capacity": (
        series_ep_weights[0]["per_chip_weight_GB_fp4"] > gb300_capacity_GB
    ),
    "smallest_ep_that_fits_weights_alone_in_gb300": next(
        (r["ep"] for r in series_ep_weights if r["fits_gb300_alone"]), None
    ),
    "llama70_kv_GB_tp1": series_llama_tp_kv[0]["per_chip_kv_GB_batch32"],
    "llama70_kv_GB_tp32": series_llama_tp_kv[-1]["per_chip_kv_GB_batch32"],
    "llama70_kv_GB_tp8": series_llama_tp_kv[3]["per_chip_kv_GB_batch32"],
    "llama70_plateaus_at_tp": llama70.n_kv_heads,
}

out["notes"] = [
    "kv_cache_bytes_per_chip [ops.py:101-128]: for MLA (model.mla is not "
    "None), per_token = n_layers * mla.latent_dim * kv_dtype.bytes and the "
    "result divides by groups = pp*ep*adp ONLY -- tp never appears in the "
    "formula [ops.py:118-122] because the compressed latent (kv_lora_rank + "
    "qk_rope_head_dim = 512+64 = 576 dims/layer/token) is small and "
    "replicated across the tp group rather than sharded. This is why series "
    "'deepseek_kv_vs_tp_ep1' is byte-for-byte flat across tp=1..32 -- "
    "confirmed empirically below, not just by formula inspection.",
    "deepseek_kv_vs_ep_tp1: with tp=1, groups = pp*ep*adp = 1*ep*1 = ep, so "
    "per-chip KV bytes is exactly (kv_bytes_per_token_fp8 * ctx * batch) / ep "
    "-- halves on every doubling of ep, confirmed for ep=1,2,4,...,64.",
    "validate_deployment [ops.py:80-92] only checks ep>1 requires a MoE model "
    "(deepseek-v3 qualifies) and tp,pp,ep,adp>=1; it does NOT check that ep "
    "divides n_experts=256 evenly, so ep=64 (256/64=4 experts/chip) runs "
    "without error or warning even though real deployments would want ep to "
    "divide n_experts.",
    "weight_bytes_per_chip [simulate.py:19-52] is the function that actually "
    "answers 'how many weight bytes land on one chip' -- NOT "
    "model.weight_bytes(dtype) [workload.py:243-244], which is total_params "
    "* dtype.bytes for the WHOLE unsharded model (a single-chip figure only "
    "if tp=pp=ep=adp=1, and even then it double counts nothing but shards "
    "nothing either -- it's the 'if you tried to fit the whole model on one "
    "chip' number, included here as whole_model_weight_bytes_fp4_GB_unsharded "
    "for contrast).",
    "weight_bytes_per_chip's MoE branch [simulate.py:38-52]: embedding_params "
    "and attn_params divide by tp*pp only (replicated across ep groups); "
    "shared_expert_params (the always-on shared FFN) ALSO divides by tp*pp "
    "only, i.e. it is replicated across ep, not sharded by it "
    "[simulate.py:48, comment at simulate.py:38-39: 'attention + shared "
    "expert replicated across ep groups']; the n_dense_layers prefix (3 "
    "layers modelled as plain dense FFN) likewise divides by tp*pp only "
    "[simulate.py:50]. Only the routed EXPERT BANK (n_experts * "
    "expert_params) divides by tp*ep*pp [simulate.py:49] -- this is the part "
    "that actually shrinks per-chip weight bytes as ep grows.",
    "Consequence: per-chip weight bytes at tp=1 has a FLOOR as ep -> infinity "
    "(embedding + all-layer attention + shared-expert + dense-prefix FFN, "
    "none of which shard with ep) -- computed here as "
    "weights_fp4_floor_GB_as_ep_to_infinity. The expert bank is 256 experts * "
    "~44M params/expert * 58 MoE layers, i.e. most of the 671B total, so the "
    "curve still drops steeply through the ep range tested, but it will not "
    "reach zero.",
    "Why ep>=2 is mandatory at tp=1: series_ep_weights[ep=1] (all 671B "
    "params on one chip, packed at fp4) vastly exceeds the B300 chip's 288 "
    "GB HBM capacity (gb300_chip_hbm_capacity_GB) -- see "
    "ep1_weights_exceed_gb300_capacity and "
    "smallest_ep_that_fits_weights_alone_in_gb300 (weights alone, ignoring "
    "KV/activations) for the crossover point.",
    "kv_bytes_per_token(FP8) [workload.py:246-255] for deepseek-v3 = "
    "n_layers * mla.latent_dim * 1 byte = 61 * 576 = 35136, matching the "
    "value the task expected.",
    "llama-3.1-70b contrast series (GQA, n_kv_heads=8): "
    "_kv_heads_per_chip = n_kv_heads / min(tp, n_kv_heads) [ops.py:95-98] "
    "shards up to tp=8, then plateaus (extra tp ranks beyond 8 replicate the "
    "already-owned KV heads instead of sharding further) -- "
    "llama70_plateaus_at_tp = n_kv_heads = 8.",
    "All KV figures use kv_dtype=FP8 (1 byte/element) and weight figures use "
    "weight_dtype=FP4 (0.5 bytes/element) per DType.bytes mapping "
    "[hardware.py:29-52]. GB = 1e9 bytes throughout (matches "
    "inferencesim.units.GB), NOT GiB.",
    "Deployment defaults pp=1, adp=1 unless stated; ep=1/tp=1 deepseek-v3 "
    "weights-alone would not fit a real GB300 chip (671B params even at fp4 "
    "= ~336 GB > 288 GB HBM) -- kv_cache_bytes_per_chip and "
    "weight_bytes_per_chip are pure accounting functions with no capacity "
    "check of their own (that check lives in simulate.py's MemoryUsage.fits, "
    "simulate.py:84-85), so computing the ep=1/tp=1 point is fine for "
    "charting even though it is not a deployable configuration.",
]

path = (
    "/private/tmp/claude-501/-Users-eric-development-inferencesim2/"
    "aa8b7baa-3a34-46c9-86fc-a58b5f7d2892/scratchpad/ds_kv.json"
)
with open(path, "w") as f:
    json.dump(out, f, indent=2)

print("wrote", path)
print(json.dumps(out["meta"], indent=2))
print(json.dumps(out["headline"], indent=2))
