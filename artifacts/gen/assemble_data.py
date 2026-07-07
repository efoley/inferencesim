"""Assemble the five extraction JSONs into one compact DATA object for the artifact."""
import json
from pathlib import Path

S = Path(__file__).parent
out = {}

# ---- 1. roofline -----------------------------------------------------------
r = json.loads((S / "roofline_scatter.json").read_text())
out["roofline"] = {
    "bw": r["chip"]["effective_dram_bandwidth_bytes_per_s"],
    "roofs": {d: v["peak_flops_per_s"] for d, v in r["chip"]["roofline_by_dtype"].items()},
    "ridges": {d: v["ridge_point_flops_per_byte"] for d, v in r["chip"]["roofline_by_dtype"].items()},
    "ops": [
        {
            "name": o["name"], "phase": o["phase"], "cat": o["category"],
            "dtype": o["dtype"], "x": o["intensity_flops_per_byte"],
            "flops": o["flops_per_chip"], "bytes": o["bytes_per_chip"],
            "bound": o["bound"], "t_us": o["time_s"] * 1e6, "count": o["count"],
        }
        for o in r["ops"]
    ],
}

# ---- 2. tp waterfall --------------------------------------------------------
t = json.loads((S / "tp_waterfall.json").read_text())
rows = {}
for s in t["series"]:
    if not s["overlap"]:
        rows[s["tp"]] = {
            "tp": s["tp"],
            "cats_ms": {k: v * 1e3 for k, v in s["categories"].items()},
            "tpot_ms": s["tpot_s"] * 1e3,
            "dp": s["dp"], "idle": s["idle_chips"],
            "usd_per_m": s["usd_per_m_output_tokens"],
        }
    else:
        rows[s["tp"]]["overlap_ms"] = s["tpot_s"] * 1e3
for key, tpv in (("tp16", 16), ("tp32", 32)):
    a = t["tp16_tp32_raw"][f"{key}_overlapFalse"]
    b = t["tp16_tp32_raw"][f"{key}_overlapTrue"]
    rows[tpv] = {
        "tp": tpv,
        "cats_ms": {k: v * 1e3 for k, v in a["categories"].items()},
        "tpot_ms": a["tpot_s"] * 1e3, "overlap_ms": b["tpot_s"] * 1e3,
        "dp": a["dp"], "idle": a["idle_chips"],
        "usd_per_m": a["usd_per_m_output_tokens"],
    }
out["tp"] = {"rows": [rows[k] for k in sorted(rows)], "kv_heads": 8}

# ---- 3. kv anatomy ----------------------------------------------------------
k = json.loads((S / "kv_anatomy.json").read_text())
pa = []
for name in ["llama-3.1-8b", "llama-3.1-70b", "gpt-oss-120b", "deepseek-v3"]:
    m = k["panel_a"][name]
    pa.append({
        "model": name, "layers": m["n_layers"],
        "uncapped": m["kv_bytes_per_token_uncapped_fp8"],
        "effective": m["kv_bytes_per_token_effective_at_131072ctx_fp8"],
        "kind": ("MLA" if m["mla"] else ("SWA" if m["swa_window"] else "GQA")),
        "kv_heads": m["n_kv_heads"], "d_head": m["d_head"],
        "swa_window": m["swa_window"], "latent": (m["mla"] or {}).get("latent_dim"),
    })
pb = k["panel_b"]
GB = 1e9
out["kv"] = {
    "a": pa,
    "b": {
        "gqa_tp": [{"chips": p["tp"], "gb": p["per_chip_kv_bytes_batch32"] / GB}
                   for p in pb["series1_tp_sweep_adp1"]],
        "gqa_adp": [{"chips": 8 * p["adp"], "adp": p["adp"], "gb": p["per_chip_kv_bytes_batch32"] / GB}
                    for p in pb["series2_tp8_adp_sweep"]],
        "mla_tp": [{"chips": p["tp"], "gb": p["per_chip_kv_bytes_batch32"] / GB}
                   for p in pb["series3_mla_tp_sweep"]],
    },
}

# ---- 4. batch economics -----------------------------------------------------
e = json.loads((S / "batch_econ.json").read_text())
out["econ"] = {
    "points": [
        {
            "b": p["batch"], "usd": p["usd_per_m_output_tokens"],
            "user": p["tokens_per_s_per_user"], "tpot_ms": p["tpot_s"] * 1e3,
            "sys": p["output_tokens_per_s"], "bound": p["decode_bound"],
            "joules": p["joules_per_output_token"],
        }
        for p in e["points"]
    ]
}

# ---- 5. moe amortization ----------------------------------------------------
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
        "attn_floor": {"gpt-oss-120b": m["models"]["gpt-oss-120b"]["decode_dram_bytes_per_token"]["sweep"][0]["attention_kv_bytes_per_token"],
                        "llama-3.1-70b": m["models"]["llama-3.1-70b"]["decode_dram_bytes_per_token"]["sweep"][0]["attention_kv_bytes_per_token"]},
    },
}

compact = json.dumps(out, separators=(",", ":"))
(S / "assembled_data.json").write_text(compact)
print(f"{len(compact)} bytes")
