"""Assemble the DeepSeek-V3 extraction JSONs into one compact DATA object."""
import json
from pathlib import Path

S = Path(__file__).parent
out = {}

# ---- 1. roofline ----
r = json.loads((S / "ds_roofline.json").read_text())
out["roofline"] = {
    "bw": r["chip"]["effective_dram_bandwidth_bytes_per_s"],
    "roofs": {d: v["peak_flops_per_s"] for d, v in r["chip"]["roofline_by_dtype"].items()},
    "ridges": {d: v["ridge_point_flops_per_byte"] for d, v in r["chip"]["roofline_by_dtype"].items()},
    "ops": [
        {
            "name": o["name"], "phase": o["phase"], "cat": o["category"],
            "dtype": o["dtype"], "x": o["intensity_flops_per_byte"],
            "flops": o["flops_per_chip"],
            "bytes": o.get("bytes_per_chip", o.get("dram_bytes", 0)) or (o.get("dram_read_bytes", 0) + o.get("dram_write_bytes", 0)),
            "bound": o["bound"], "t_us": o["time_s"] * 1e6, "count": o.get("count", 1),
        }
        for o in r["ops"]
    ],
}

# ---- 2. ep waterfall ----
w = json.loads((S / "ds_ep_waterfall.json").read_text())
out["ep"] = {
    "rows": [
        {
            "ep": row["ep"],
            "cats_ms": row["cats_ms"],
            "tpot_ms": row["tpot_ms"], "overlap_ms": row["overlap_ms"],
            "dp": row["dp"], "idle": row["idle"], "usd_per_m": row["usd_per_m"],
        }
        for row in w["rows"]
    ],
    "ep1": {"weights_gb": 335.46, "kv_gb": 46.05, "total_gb": 381.53, "capacity_gb": 288.0},
}

# ---- 3. weights & KV sharding ----
k = json.loads((S / "ds_kv.json").read_text())
sr = k["series"]
out["kvw"] = {
    "weights": [{"ep": p["ep"], "gb": p["per_chip_weight_GB_fp4"], "fits": p["fits_gb300_alone"]}
                for p in sr["deepseek_weights_fp4_vs_ep_tp1"]],
    "cap_gb": k["meta"]["gb300_chip_hbm_capacity_GB"],
    "floor_gb": k["headline"]["weights_fp4_floor_GB_as_ep_to_infinity"],
}
out["kvb"] = {
    "ds_ep": [{"chips": p["ep"], "gb": p["per_chip_kv_GB_batch32"]} for p in sr["deepseek_kv_vs_ep_tp1"]],
    "ds_tp": [{"chips": p["tp"], "gb": p["per_chip_kv_GB_batch32"]} for p in sr["deepseek_kv_vs_tp_ep1"]],
    "llama_tp": [{"chips": p["tp"], "gb": p["per_chip_kv_GB_batch32"]} for p in sr["llama70_kv_vs_tp_contrast"]],
}

# ---- 4. batch economics ----
e = json.loads((S / "ds_econ.json").read_text())
out["econ"] = {
    "points": [
        {
            "b": p["batch"], "usd": p["usd_per_m_output_tokens"],
            "user": p["tokens_per_s_per_user"], "tpot_ms": p["tpot_s"] * 1e3,
            "sys": p["output_tokens_per_s"], "bound": p["decode_bound"],
            "joules": p["joules_per_output_token"],
            "warn": bool(p.get("warnings")),
        }
        for p in e["points"]
    ]
}

# ---- 5. moe amortization (reuse the artifact-1 extraction) ----
m = json.loads((S / "moe_amortization.json").read_text())


def experts(name):
    ee = m["models"][name]["expected_active_experts"]
    return {"n": ee["n_experts"], "topk": ee["top_k"],
            "pts": [{"b": p["batch"], "e": p["expected_active_experts"]} for p in ee["sweep"]]}


def bytes_series(name, field):
    return [{"b": p["batch"], "v": p[field]}
            for p in m["models"][name]["decode_dram_bytes_per_token"]["sweep"]]


out["moe"] = {
    "experts": {"gpt-oss-120b": experts("gpt-oss-120b"), "deepseek-v3": experts("deepseek-v3")},
    "bytes": {
        "gpt-oss-120b": bytes_series("gpt-oss-120b", "moe_bytes_per_token"),
        "deepseek-v3": bytes_series("deepseek-v3", "moe_bytes_per_token"),
        "llama-dense": bytes_series("llama-3.1-70b", "other_weight_bytes_per_token"),
    },
}

compact = json.dumps(out, separators=(",", ":"))
(S / "ds_assembled.json").write_text(compact)
print(f"{len(compact)} bytes")
