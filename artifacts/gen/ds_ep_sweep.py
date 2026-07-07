import json
import sys

from inferencesim.simulate import simulate
from inferencesim.presets import MODELS, HARDWARE
from inferencesim.workload import Scenario, Deployment
from inferencesim.hardware import DType

model = MODELS["deepseek-v3"]
system = HARDWARE["gb300-nvl72"]
scenario = Scenario(batch=256, prompt_len=4096, output_len=1024)

rows = []
ep1_result = None
notes = []

def run(ep):
    dep = Deployment(tp=1, ep=ep, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
    try:
        rep = simulate(system, model, scenario, dep)
        return rep, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

# ---- ep=1 special case ----
rep1, err1 = run(1)
if err1 is not None:
    ep1_result = {"raised": True, "error": err1}
else:
    mem = rep1.memory
    ep1_result = {
        "summary": (
            f"simulate() does NOT raise at ep=1: it computes the phases as if "
            f"the weights fit, and appends a warning instead. 671B-param "
            f"fp4 weights = {mem.weights/1e9:.1f} GB alone (>{mem.capacity/1e9:.0f} GB "
            f"capacity already), plus {mem.kv_cache/1e9:.1f} GB fp8 KV cache "
            f"(batch=256 x 4096-6144 ctx) => {mem.total/1e9:.1f} GB total needed "
            f"vs {mem.capacity/1e9:.0f} GB/chip on B300. tpot_s/ttft_s are still "
            f"returned (0.0532s / 0.0703s) -- they are the analytic roofline time "
            f"as if the bytes could stream, so they should be treated as "
            f"infeasible/undefined, not a real operating point."
        ),
        "raised": False,
        "warnings": rep1.warnings,
        "memory_gb": {
            "weights": mem.weights / 1e9,
            "kv_cache": mem.kv_cache / 1e9,
            "activations": mem.activations / 1e9,
            "total": mem.total / 1e9,
            "capacity": mem.capacity / 1e9,
            "fits": mem.fits,
        },
        "tpot_s": rep1.tpot_s,
        "ttft_s": rep1.ttft_s,
        "dp": rep1.dp,
        "idle_chips": rep1.idle_chips,
    }

for ep in [2, 4, 8, 16, 32, 64]:
    dep = Deployment(tp=1, ep=ep, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
    dep_overlap = Deployment(tp=1, ep=ep, weight_dtype=DType.FP4, kv_dtype=DType.FP8, overlap_comm=True)
    rep = simulate(system, model, scenario, dep)
    rep_overlap = simulate(system, model, scenario, dep_overlap)

    decode = rep.decode
    cats = decode.category_times()
    bounds = decode.category_bounds()
    mem = rep.memory

    row = {
        "ep": ep,
        "cats_ms": {k: v * 1e3 for k, v in cats.items()},
        "category_bounds": bounds,
        "tpot_ms": rep.tpot_s * 1e3,
        "overlap_ms": rep_overlap.tpot_s * 1e3,
        "ttft_ms": rep.ttft_s * 1e3,
        "dp": rep.dp,
        "idle": rep.idle_chips,
        "output_tokens_per_s": rep.output_tokens_per_s,
        "usd_per_m": rep.usd_per_m_output_tokens,
        "memory_gb": {
            "weights": mem.weights / 1e9,
            "kv": mem.kv_cache / 1e9,
            "activations": mem.activations / 1e9,
            "total": mem.total / 1e9,
            "capacity": mem.capacity / 1e9,
            "fits": mem.fits,
        },
        "warnings": rep.warnings,
    }
    rows.append(row)

# ---- sanity checks ----
moe_series = [(r["ep"], r["cats_ms"].get("moe", 0.0)) for r in rows]
attn_series = [(r["ep"], r["cats_ms"].get("attention", 0.0)) for r in rows]
comm_series = [(r["ep"], r["cats_ms"].get("comm", 0.0)) for r in rows]
tpot_series = [(r["ep"], r["tpot_ms"]) for r in rows]

notes.append(f"moe (expert) ms by ep: {moe_series}")
notes.append(f"attention ms by ep: {attn_series}")
notes.append(f"comm ms by ep: {comm_series}")
notes.append(f"tpot ms by ep: {tpot_series}")

moe_monotonic_decreasing = all(moe_series[i][1] >= moe_series[i+1][1] for i in range(len(moe_series)-1))
attn_monotonic_decreasing = all(attn_series[i][1] >= attn_series[i+1][1] for i in range(len(attn_series)-1))
comm_monotonic_increasing = all(comm_series[i][1] <= comm_series[i+1][1] for i in range(len(comm_series)-1))
min_tpot = min(tpot_series, key=lambda t: t[1])

notes.append(f"moe monotonic decreasing with ep: {moe_monotonic_decreasing}")
notes.append(f"attention monotonic decreasing with ep: {attn_monotonic_decreasing}")
notes.append(f"comm monotonic increasing with ep: {comm_monotonic_increasing}")
notes.append(f"tpot minimum at ep={min_tpot[0]} ({min_tpot[1]:.4f} ms)")

comm_share = [(r["ep"], r["cats_ms"].get("comm", 0.0) / r["tpot_ms"]) for r in rows]
notes.append(f"comm SHARE of tpot by ep (comm falls in absolute ms but its share "
              f"of the (shrinking) total rises -- it decays slower than moe/attention "
              f"because of a fixed per-op latency floor): {comm_share}")

usd_series = [(r["ep"], r["usd_per_m"]) for r in rows]
dp_series = [(r["ep"], r["dp"], r["idle"]) for r in rows]
notes.append(f"usd_per_m_output_tokens by ep (rises steeply -- lower TPOT per "
              f"replica but far fewer replicas dp fit on the 72-chip rack, and "
              f"once ep doesn't divide 72 evenly chips go idle): {usd_series}")
notes.append(f"(ep, dp, idle_chips): {dp_series}")

out = {
    "rows": rows,
    "ep1_result": ep1_result,
    "notes": notes,
}

out_path = "/private/tmp/claude-501/-Users-eric-development-inferencesim2/aa8b7baa-3a34-46c9-86fc-a58b5f7d2892/scratchpad/ds_ep_waterfall.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
