"""Sweep batch size for llama-3.1-70b on GB300_NVL72 and record batch-size
economics: $/M output tokens and tok/s/user vs batch, plus decode bottleneck
and memory pressure. Read-only against the inferencesim package."""

import json

from inferencesim.hardware import DType
from inferencesim.presets import GB300_NVL72, LLAMA_3_1_70B
from inferencesim.simulate import simulate
from inferencesim.workload import Deployment, Scenario

BATCHES = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512]

dep = Deployment(tp=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8)

points = []
notes = []
last_bound = None

for B in BATCHES:
    scen = Scenario(batch=B, prompt_len=4096, output_len=1024)
    report = simulate(GB300_NVL72, LLAMA_3_1_70B, scen, dep)

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

out = {"points": points, "notes": notes}

out_path = "/private/tmp/claude-501/-Users-eric-development-inferencesim2/aa8b7baa-3a34-46c9-86fc-a58b5f7d2892/scratchpad/batch_econ.json"
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
