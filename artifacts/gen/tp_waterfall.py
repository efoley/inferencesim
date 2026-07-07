import json
import dataclasses

from inferencesim.presets import GB300_NVL72, LLAMA_3_1_70B
from inferencesim.workload import Deployment, Scenario
from inferencesim.hardware import DType
from inferencesim.simulate import simulate

system = GB300_NVL72
model = LLAMA_3_1_70B
scenario = Scenario(batch=64, prompt_len=4096, output_len=1024)

notes = []
series = []

def run(tp, overlap_comm):
    dep = Deployment(tp=tp, weight_dtype=DType.FP4, kv_dtype=DType.FP8, overlap_comm=overlap_comm)
    report = simulate(system, model, scenario, dep)
    cats = report.decode.category_times()
    bounds = report.decode.category_bounds()
    total = sum(cats.values())
    entry = {
        "tp": tp,
        "overlap": overlap_comm,
        "tpot_s": report.tpot_s,
        "ttft_s": report.ttft_s,
        "categories": cats,
        "category_share": {k: v / total for k, v in cats.items()} if total else {},
        "bounds": bounds,
        "decode_total_time_unoverlapped": total,
        "output_tokens_per_s": report.output_tokens_per_s,
        "usd_per_m_output_tokens": report.usd_per_m_output_tokens,
        "dp": report.dp,
        "idle_chips": report.idle_chips,
        "memory": {
            "weights_gb": report.memory.weights / 1e9,
            "kv_cache_gb": report.memory.kv_cache / 1e9,
            "activations_gb": report.memory.activations / 1e9,
            "total_gb": report.memory.total / 1e9,
            "capacity_gb": report.memory.capacity / 1e9,
            "fits": report.memory.fits,
        },
        "warnings": report.warnings,
    }
    return entry

for tp in [1, 2, 4, 8]:
    for overlap in [False, True]:
        entry = run(tp, overlap)
        series.append(entry)
        print(f"tp={tp} overlap={overlap}: tpot={entry['tpot_s']*1000:.4f}ms "
              f"cats={ {k: round(v*1e6,2) for k,v in entry['categories'].items()} } us "
              f"mem_fits={entry['memory']['fits']} warnings={entry['warnings']}")

# tp=16, tp=32 attempts
tp_extreme_results = {}
for tp in [16, 32]:
    for overlap in [False, True]:
        key = f"tp{tp}_overlap{overlap}"
        try:
            entry = run(tp, overlap)
            tp_extreme_results[key] = {"status": "ok", **entry}
            print(f"tp={tp} overlap={overlap}: OK tpot={entry['tpot_s']*1000:.4f}ms "
                  f"cats={entry['categories']} warnings={entry['warnings']}")
        except Exception as e:
            tp_extreme_results[key] = {"status": "error", "error_type": type(e).__name__, "message": str(e)}
            print(f"tp={tp} overlap={overlap}: ERROR {type(e).__name__}: {e}")

# Also check largest feasible batch at tp=1 if batch=64 doesn't fit (sanity check requested)
entry_tp1 = [e for e in series if e["tp"] == 1 and e["overlap"] == False][0]
fallback_series = None
if not entry_tp1["memory"]["fits"]:
    notes.append("batch=64 at tp=1 does NOT fit in memory; searching for largest feasible batch")
    # binary search largest feasible batch at tp=1
    lo, hi = 1, 64
    best = None
    for b in range(1, 65):
        dep = Deployment(tp=1, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
        scen_b = Scenario(batch=b, prompt_len=4096, output_len=1024)
        rep = simulate(system, model, scen_b, dep)
        if rep.memory.fits:
            best = b
        else:
            break
    notes.append(f"largest feasible batch at tp=1: {best}")
    if best:
        fallback_series = []
        for tp in [1, 2, 4, 8]:
            dep = Deployment(tp=tp, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
            scen_b = Scenario(batch=best, prompt_len=4096, output_len=1024)
            rep = simulate(system, model, scen_b, dep)
            cats = rep.decode.category_times()
            fallback_series.append({
                "tp": tp, "batch": best, "tpot_s": rep.tpot_s,
                "categories": cats,
                "memory_fits": rep.memory.fits,
            })
else:
    notes.append("batch=64 at tp=1 fits in memory; no fallback series needed")

# Sanity check numbers
tp1 = [e for e in series if e["tp"] == 1 and e["overlap"] == False][0]
tp2 = [e for e in series if e["tp"] == 2 and e["overlap"] == False][0]
tp4 = [e for e in series if e["tp"] == 4 and e["overlap"] == False][0]
tp8 = [e for e in series if e["tp"] == 8 and e["overlap"] == False][0]

notes.append(f"tp=1 comm time: {tp1['categories'].get('comm', 0)*1e6:.4f} us (expect ~0)")
notes.append(f"linear time tp1->tp2->tp4->tp8: "
             f"{tp1['categories'].get('linear',0)*1e6:.3f}, "
             f"{tp2['categories'].get('linear',0)*1e6:.3f}, "
             f"{tp4['categories'].get('linear',0)*1e6:.3f}, "
             f"{tp8['categories'].get('linear',0)*1e6:.3f} us")
notes.append(f"comm time tp1->tp2->tp4->tp8: "
             f"{tp1['categories'].get('comm',0)*1e6:.3f}, "
             f"{tp2['categories'].get('comm',0)*1e6:.3f}, "
             f"{tp4['categories'].get('comm',0)*1e6:.3f}, "
             f"{tp8['categories'].get('comm',0)*1e6:.3f} us")

# KV heads per chip check (llama-70b has 8 kv heads)
from inferencesim.ops import _kv_heads_per_chip
for tp in [1, 2, 4, 8, 16, 32]:
    kvh = _kv_heads_per_chip(model, tp)
    notes.append(f"tp={tp}: kv_heads_per_chip={kvh} (n_kv_heads={model.n_kv_heads})")

tp16_summary = (
    f"tp=16 RUNS (does not raise): llama-3.1-70b has 8 KV heads, so "
    f"_kv_heads_per_chip(tp=16)=min(16,8)=8 -> kv_heads_per_chip stays at 1 (same as "
    f"tp=8, KV heads replicated beyond the 8-way shard). Attention op's memory time "
    f"(KV-cache read) therefore PLATEAUS at the tp=8 value "
    f"({tp8['categories'].get('attention',0)*1e6:.2f} us) instead of continuing to "
    f"halve, while comm (ring allreduce) keeps growing -- dominated by per-hop link "
    f"latency (steps=2*(tp-1) hops * 1us NVLink5 latency * 160 allreduce "
    f"instances/decode step), not bandwidth. Net effect: tpot_s WORSENS to "
    f"{tp_extreme_results['tp16_overlapFalse']['tpot_s']*1000:.4f} ms (no-overlap), "
    f"UP from tp=8's {tp8['tpot_s']*1000:.4f} ms -- TP scaling regresses past tp=8 "
    f"for this model/system. Also raises a warning: 8 chips idle (72 not divisible "
    f"by 16)."
)
tp32_summary = (
    f"tp=32 RUNS (does not raise): same KV-head plateau as tp=16 (kv_heads_per_chip="
    f"1). comm grows further (steps=2*31=62 hops) and tpot_s WORSENS further to "
    f"{tp_extreme_results['tp32_overlapFalse']['tpot_s']*1000:.4f} ms (no-overlap) -- "
    f"now even worse than tp=1's {tp1['tpot_s']*1000:.4f} ms. 8 chips idle (72 not "
    f"divisible by 32)."
)
notes.append(tp16_summary)
notes.append(tp32_summary)
notes.append(
    "IMPORTANT for chart: category_times() (the 'categories' dict) is identical "
    "between the overlap=False and overlap=True rows for a given tp -- overlap_comm "
    "only changes how the phase's wall-clock duration is derived from those same "
    "per-category times (sum vs max(comm, rest)), not the underlying op costs. "
    "So: use the overlap=False series for the additive stacked-bar waterfall "
    "(categories sum exactly to tpot_s); show the overlap=True tpot_s as a separate "
    "marker/line (e.g. a dashed 'with comm/compute overlap' total) since its bar "
    "segments would NOT sum to that shorter total."
)
notes.append(
    "Modelling caveat: the roofline engine always costs ALLREDUCE via the ring "
    "closed form (2*(tp-1) steps) regardless of the system's actual topology "
    "(GB300 NVL72 is an NVSwitch ALL_TO_ALL fabric, not a ring) -- so the comm "
    "growth at high tp shown here is a conservative/pessimistic upper bound driven "
    "by per-hop latency accumulation, not a hard physical limit of NVSwitch."
)

out = {
    "system": "GB300_NVL72",
    "model": "llama-3.1-70b",
    "scenario": {"batch": 64, "prompt_len": 4096, "output_len": 1024},
    "deployment_base": {"weight_dtype": "fp4", "kv_dtype": "fp8"},
    "series": series,
    "tp16_result": tp16_summary,
    "tp32_result": tp32_summary,
    "tp16_tp32_raw": tp_extreme_results,
    "fallback_series_if_tp1_infeasible": fallback_series,
    "notes": notes,
}

path = "/private/tmp/claude-501/-Users-eric-development-inferencesim2/aa8b7baa-3a34-46c9-86fc-a58b5f7d2892/scratchpad/tp_waterfall.json"
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print("wrote", path)
