"""Extract per-op arithmetic-intensity / roofline data for a chart.

Setup: llama-3.1-70b on GB300_NVL72, tp=8, weight_dtype=fp4, kv_dtype=fp8,
batch=64, prompt_len=4096, output_len=1024.
"""
import json

from inferencesim.presets import MODELS, HARDWARE
from inferencesim.workload import Deployment, Scenario
from inferencesim.ops import prefill_ops, decode_ops
from inferencesim.engine import RooflineEngine, CommContext
from inferencesim.hardware import DType

model = MODELS["llama-3.1-70b"]
system = HARDWARE["gb300-nvl72"]
chip = system.node.chip

dep = Deployment(tp=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8)
scenario = Scenario(batch=64, prompt_len=4096, output_len=1024)

print("mean_context:", scenario.mean_context)
print("act_dtype (default):", dep.act_dtype)

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
    skipped.append({"name": op.name, "category": op.category, "phase": phase,
                     "kind": op.kind.value, "comm_bytes": op.comm_bytes,
                     "dram_read": op.dram_read, "dram_write": op.dram_write})

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
    },
    "chip": chip_info,
    "ops": ops_out,
    "skipped_zero_flops_or_comm_ops": skipped,
}

path = "/private/tmp/claude-501/-Users-eric-development-inferencesim2/aa8b7baa-3a34-46c9-86fc-a58b5f7d2892/scratchpad/roofline_scatter.json"
with open(path, "w") as f:
    json.dump(out, f, indent=2)

print("Wrote", path)
print()
print("Chip roofline:")
for dt, v in roofline.items():
    print(f"  {dt}: peak={v['peak_flops_per_s']:.3e} FLOP/s, ridge={v['ridge_point_flops_per_byte']:.3f} FLOP/byte")
print(f"  mem_bw = {mem_bw:.3e} B/s")
print()
print("Ops summary:")
for o in ops_out:
    print(f"  [{o['phase']:7s}] {o['name']:12s} cat={o['category']:10s} dtype={o['dtype']:5s} "
          f"I={o['intensity_flops_per_byte']:.2f} bound={o['bound']:8s} count={o['count']}")
print()
print("Skipped (comm / zero-flops) ops:")
for s in skipped:
    print(f"  [{s['phase']:7s}] {s['name']:12s} kind={s['kind']}")
