"""Sweep batch size for deepseek-v3 (DEP8: tp=1, ep=8, fp4 weights, fp8 kv)
on GB300_NVL72 and record batch-size economics: $/M output tokens and
tok/s/user vs batch, plus decode bottleneck and memory pressure.
Read-only against the inferencesim package."""

import json

from inferencesim.hardware import DType
from inferencesim.presets import DEEPSEEK_V3, GB300_NVL72
from inferencesim.simulate import simulate
from inferencesim.workload import Deployment, Scenario

BATCHES = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]

dep = Deployment(tp=1, ep=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8)

points = []
notes = []
last_bound = None

for B in BATCHES:
    scen = Scenario(batch=B, prompt_len=4096, output_len=1024)
    report = simulate(GB300_NVL72, DEEPSEEK_V3, scen, dep)

    mem = report.memory
    decode = report.decode
    # overall decode-phase bottleneck: compare aggregate busy time by resource
    busy = {
        "compute": decode.compute_busy,
        "memory": decode.mem_busy,
        "comm": decode.comm_busy,
    }
    decode_bound = max(busy, key=busy.get)

    point = {
        "batch": B,
        "tpot_s": report.tpot_s,
        "tokens_per_s_per_user": 1.0 / report.tpot_s if report.tpot_s > 0 else None,
        "ttft_s": report.ttft_s,
        "output_tokens_per_s": report.output_tokens_per_s,
        "usd_per_m_output_tokens": report.usd_per_m_output_tokens,
        "capex_share": report.capex_share,
        "joules_per_output_token": report.joules_per_output_token,
        "system_power_w": report.system_power_w,
        "decode_bound": decode_bound,
        "decode_bound_busy_s": busy,
        "warnings": list(report.warnings),
        "memory": {
            "weights_bytes": mem.weights,
            "kv_cache_bytes": mem.kv_cache,
            "activations_bytes": mem.activations,
            "capacity_bytes": mem.capacity,
            "total_bytes": mem.total,
            "fits": mem.fits,
        },
    }
    points.append(point)

    if decode_bound != last_bound:
        notes.append(
            f"decode bound flips to {decode_bound} at batch={B} "
            f"(compute={busy['compute']:.3e}s mem={busy['memory']:.3e}s comm={busy['comm']:.3e}s)"
        )
        last_bound = decode_bound

    if not mem.fits:
        notes.append(
            f"batch={B} INFEASIBLE: memory needed "
            f"{mem.total/1e9:.1f} GB > {mem.capacity/1e9:.1f} GB/chip capacity "
            f"-- stopping sweep here"
        )
        break

feasible = [p for p in points if p["memory"]["fits"]]
largest_feasible = feasible[-1]["batch"] if feasible else None
notes.insert(0, f"largest feasible batch in sweep: {largest_feasible}")

starved = [p["batch"] for p in points if p["warnings"]]
if starved:
    notes.append(
        f"batch < pp*ep*adp=8 (expert/attention groups starved) at B={starved} "
        "-- decode still computed but the ep=8 expert array is under-filled"
    )

bounds_seen = sorted({p["decode_bound"] for p in points})
if bounds_seen == ["memory"]:
    max_comm_over_mem = max(
        p["decode_bound_busy_s"]["comm"] / p["decode_bound_busy_s"]["memory"] for p in points
    )
    notes.append(
        "decode is memory-bound at EVERY batch in this sweep (never comm-bound): "
        "tp=1 means no tensor-parallel allreduce, leaving only the small ep=8 "
        f"MoE dispatch/combine all-to-all as the comm term; comm/memory busy peaks "
        f"at {max_comm_over_mem:.2f}x (B={points[-1]['batch']}), well under 1 -- "
        "contrast with llama-3.1-70b tp=8 DEP, which is comm-bound (tp allreduce) "
        "at low batch"
    )

# marginal doubling ratios usd(B)/usd(2B) -- how much cost-per-token improves
# each time batch doubles (MoE expert-pool amortisation signature)
by_batch = {p["batch"]: p["usd_per_m_output_tokens"] for p in points}
ratio_notes = []
for B in BATCHES:
    if B in by_batch and (2 * B) in by_batch:
        r = by_batch[B] / by_batch[2 * B]
        ratio_notes.append(f"usd({B})/usd({2*B}) = {r:.3f}")
notes.append("marginal doubling ratios: " + "; ".join(ratio_notes))

out = {"points": points, "notes": notes}

out_path = "/private/tmp/claude-501/-Users-eric-development-inferencesim2/aa8b7baa-3a34-46c9-86fc-a58b5f7d2892/scratchpad/ds_econ.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print("wrote", out_path)
print("largest feasible batch:", largest_feasible)
for p in points:
    print(
        f"B={p['batch']:>4} fits={p['memory']['fits']!s:5} "
        f"tpot_ms={p['tpot_s']*1e3:8.3f} tok/s/user={p['tokens_per_s_per_user']:7.2f} "
        f"$/Mtok={p['usd_per_m_output_tokens']:8.3f} bound={p['decode_bound']:8} "
        f"kv_GB={p['memory']['kv_cache_bytes']/1e9:7.2f} "
        f"w_GB={p['memory']['weights_bytes']/1e9:6.2f} "
        f"tot_GB={p['memory']['total_bytes']/1e9:7.2f}"
    )
