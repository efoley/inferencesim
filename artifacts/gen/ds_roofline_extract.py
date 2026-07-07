"""Extract per-op arithmetic-intensity / roofline data for a chart.

Setup: deepseek-v3 (671B MoE + MLA) on GB300_NVL72, reference deployment
tp=1, ep=8 (TRT-LLM "DEP8" mapping), weight_dtype=fp4, kv_dtype=fp8,
batch=256, prompt_len=4096, output_len=1024.
"""
import json

from inferencesim.presets import MODELS, HARDWARE
from inferencesim.workload import Deployment, Scenario
from inferencesim.ops import prefill_ops, decode_ops
from inferencesim.engine import RooflineEngine, CommContext
from inferencesim.hardware import DType
from inferencesim.simulate import simulate

model = MODELS["deepseek-v3"]
system = HARDWARE["gb300-nvl72"]
chip = system.node.chip

dep = Deployment(tp=1, ep=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
scenario = Scenario(batch=256, prompt_len=4096, output_len=1024)

print("mean_context:", scenario.mean_context)
print("act_dtype (default):", dep.act_dtype)

# ---- memory feasibility check (via simulate()) -------------------------
report = simulate(system, model, scenario, dep)
mem = report.memory
feasibility = {
    "deployment_used": {"tp": dep.tp, "pp": dep.pp, "ep": dep.ep, "adp": dep.adp},
    "weights_bytes_per_chip": mem.weights,
    "kv_cache_bytes_per_chip": mem.kv_cache,
    "activations_bytes_per_chip": mem.activations,
    "total_bytes_per_chip": mem.total,
    "capacity_bytes_per_chip": mem.capacity,
    "fits": mem.fits,
    "warnings": report.warnings,
    "dp_replicas": report.dp,
    "idle_chips": report.idle_chips,
    "ttft_s": report.ttft_s,
    "tpot_s": report.tpot_s,
}
print("Memory feasibility:", json.dumps(feasibility, indent=2))
assert mem.fits, "deployment infeasible -- would need to fall back to ep=16"

pf_ops = prefill_ops(model, scenario.prompt_len, dep)
dc_ops = decode_ops(model, dep, batch=scenario.batch, ctx=scenario.mean_context)

engine = RooflineEngine()
comm = CommContext.for_deployment(system, dep)

def record(op, phase):
    bytes_moved = op.dram_read + op.dram_write
    intensity = op.flops / bytes_moved if bytes_moved > 0 else None
    timing = engine.time_op(op, chip, comm)
    return {
        "name": op.name,
        "category": op.category,
        "phase": phase,
        "dtype": op.dtype.value,
        "count": op.count,
        "flops_per_chip": op.flops,
        "dram_read_bytes": op.dram_read,
        "dram_write_bytes": op.dram_write,
        "bytes_per_chip": bytes_moved,
        "intensity_flops_per_byte": intensity,
        "time_s": timing.time,
        "compute_time_s": timing.compute_time,
        "mem_time_s": timing.mem_time,
        "bound": timing.bound,
    }

ops_out = []
for op in pf_ops:
    if op.flops > 0:
        ops_out.append(record(op, "prefill"))
for op in dc_ops:
    if op.flops > 0:
        ops_out.append(record(op, "decode"))

# also record skipped comm/zero-flops ops for completeness/debugging
skipped = []
for op, phase in [(o, "prefill") for o in pf_ops if o.flops <= 0] + \
                  [(o, "decode") for o in dc_ops if o.flops <= 0]:
    timing = engine.time_op(op, chip, comm)
    skipped.append({"name": op.name, "category": op.category, "phase": phase,
                     "kind": op.kind.value, "comm_bytes": op.comm_bytes,
                     "dram_read": op.dram_read, "dram_write": op.dram_write,
                     "time_s": timing.time, "bound": timing.bound})

# ---- chip roofline parameters -----------------------------------------
mem_bw = chip.effective_dram_bandwidth
dtypes_present = sorted({op.dtype for op in pf_ops + dc_ops if op.flops > 0}, key=lambda d: d.value)

roofline = {}
for dt in dtypes_present:
    peak = chip.compute.flops(dt)
    ridge = peak / mem_bw
    roofline[dt.value] = {
        "peak_flops_per_s": peak,
        "ridge_point_flops_per_byte": ridge,
    }

chip_info = {
    "name": chip.name,
    "effective_dram_bandwidth_bytes_per_s": mem_bw,
    "dram_bandwidth_bytes_per_s": chip.dram.bandwidth,
    "roofline_by_dtype": roofline,
}

out = {
    "meta": {
        "model": model.name,
        "hardware": system.name,
        "deployment": {
            "tp": dep.tp, "pp": dep.pp, "ep": dep.ep, "adp": dep.adp,
            "weight_dtype": dep.weight_dtype.value,
            "kv_dtype": dep.kv_dtype.value,
            "act_dtype": dep.act_dtype.value,
        },
        "scenario": {
            "batch": scenario.batch,
            "prompt_len": scenario.prompt_len,
            "output_len": scenario.output_len,
            "mean_context": scenario.mean_context,
        },
        "memory_feasibility": feasibility,
        "notes": [
            "DeepSeek-V3: 671B total / 37B active MoE, MLA attention, "
            "n_experts=256, top_k=8, n_dense_layers=3 (first_k_dense_replace).",
            "Reference mapping tp=1, ep=8 mirrors TRT-LLM's DEP8: attention "
            "(MLA) is replicated per ep group (batch-sharded 1/ep), while "
            "expert weights shard over the full tp*ep=8-chip array.",
            "GB300_NVL72 has 72 chips; replica_chips = tp*pp*ep*adp = 8, so "
            "dp=9 data-parallel replicas fit with 0 idle chips.",
        ],
    },
    "chip": chip_info,
    "ops": ops_out,
    "skipped": skipped,
}

path = "/private/tmp/claude-501/-Users-eric-development-inferencesim2/aa8b7baa-3a34-46c9-86fc-a58b5f7d2892/scratchpad/ds_roofline.json"
with open(path, "w") as f:
    json.dump(out, f, indent=2)

print("Wrote", path)
print()
print("Chip roofline:")
for dt, v in roofline.items():
    print(f"  {dt}: peak={v['peak_flops_per_s']:.4e} FLOP/s, ridge={v['ridge_point_flops_per_byte']:.3f} FLOP/byte")
print(f"  mem_bw = {mem_bw:.4e} B/s")
print()
print("Ops summary:")
for o in ops_out:
    intensity = o['intensity_flops_per_byte']
    print(f"  [{o['phase']:7s}] {o['name']:14s} cat={o['category']:10s} dtype={o['dtype']:5s} "
          f"count={o['count']:3d} I={intensity:12.3f} time_s={o['time_s']:.6e} bound={o['bound']:8s}")
print()
print("Skipped (comm / zero-flops) ops:")
for s in skipped:
    print(f"  [{s['phase']:7s}] {s['name']:14s} kind={s['kind']:10s} comm_bytes={s['comm_bytes']:.3e} time_s={s['time_s']:.6e} bound={s['bound']}")
