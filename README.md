# inferencesim

A simulator for the **throughput, latency, power, and cost** of LLM inference
factories — from a single dev box to a rack-scale deployment.

Hardware is described bottom-up from fine-grained, chip-level building blocks
so that very different machines share one vocabulary: an NVIDIA GB300 NVL72
rack, a pair of DGX Sparks, and a Tenstorrent QuietBox are all just
compositions of compute pools, memory levels, and links.

## Quick start

```bash
pip install -e .

inferencesim list

inferencesim run --hardware gb300-nvl72 --model llama-3.1-70b \
    --tp 8 --batch 64 --prompt 4096 --output 1024 \
    --weight-dtype fp4 --kv-dtype fp8 --overlap-comm

# sweep batch sizes
inferencesim run --hardware tt-quietbox --model gpt-oss-120b \
    --tp 4 --batch 1,8,32 --weight-dtype fp8
```

Or from Python:

```python
from inferencesim import Deployment, Scenario, simulate, format_report
from inferencesim.presets import GB300_NVL72, LLAMA_3_1_70B
from inferencesim.hardware import DType

report = simulate(
    GB300_NVL72,
    LLAMA_3_1_70B,
    Scenario(batch=64, prompt_len=4096, output_len=1024),
    Deployment(tp=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8),
)
print(format_report(report))
```

## Architecture

```
hardware.py   Compute / Memory / Link / Chip / Node / System
workload.py   ModelSpec (dense + MoE, GQA), Scenario, Deployment
ops.py        lowering: (model, scenario, deployment) -> per-chip Op list
engine.py     Engine interface + RooflineEngine ("speed of light")
simulate.py   orchestration, memory feasibility, power & cost models
presets.py    built-in chips/systems/models (editable spec sheets)
```

### Hardware: fine-grained building blocks

A `Chip` is a `Compute` pool fed from a DRAM through an ordered **on-chip
data path**. For Tenstorrent's Blackhole this mirrors the Metalium block
diagram:

```
GDDR6 --> NoC --> Tensix L1 SRAM --> matrix engine
```

Each stage is an explicit component with its own bandwidth, capacity, and
power. At speed of light, the effective DRAM streaming bandwidth is the
minimum over the path; a discrete-event engine can later attach queues and
contention to the very same stages. GPUs are currently modeled with an empty
path (HBM feeds the SMs directly) — add L2 or SMEM stages if you want them.

Chips compose upward: a `Node` is N chips on an intra-node interconnect
(NVLink, 800GbE); a `System` is M nodes on a network. Collectives pick the
slowest link their group crosses.

### Workload and lowering

`ModelSpec` covers dense and MoE decoder-only transformers with GQA
(parameter counts are validated against published totals in the tests). The
lowering in `ops.py` turns a serving scenario into per-chip `Op` records —
FLOPs, DRAM bytes, collective bytes — for the two phases:

- **prefill** (one request, compute-bound at speed of light), and
- **decode** (a batch of sequences, dominated by streaming weights + KV).

MoE decode accounts for the *expected number of distinct experts* activated
by a batch, which is what actually determines DRAM traffic.

### Engines

`RooflineEngine` is the "speed of light" model: each op runs at the peak
rate of its bottleneck resource (`t = max(flops/peak, bytes/bw)`), collectives
use bandwidth-optimal ring formulas, and there are no kernel or scheduling
overheads. Results are optimistic bounds — real systems achieve some fraction
of them. TP collectives are serialized by default; pass `--overlap-comm` to
assume perfect overlap (reality is in between).

### Outputs

- **Latency**: TTFT (prefill) and TPOT (decode at mean context), with a
  per-category breakdown (linear / attention / moe / comm / head) and the
  bottleneck (compute-, memory-, or comm-bound) for each.
- **Throughput**: steady-state continuous batching (each request costs one
  exclusive prefill plus shared decode steps), plus the decode-only ceiling.
- **Power**: idle + per-component dynamic power scaled by that component's
  busy fraction → watts and joules per output token.
- **Cost**: amortized capex + electricity (PUE-adjusted) → $/M tokens.

## Accuracy disclaimers

- Preset spec numbers (FLOPs, bandwidths, prices, power splits) are
  best-effort approximations from public material — treat them as editable
  spec sheets (`dataclasses.replace(...)`), not ground truth.
- Roofline numbers are upper bounds: no kernel launch overhead, perfect
  tiling, perfect load balance, bandwidth at 100% efficiency.

## Roadmap

- **Discrete-event engine**: consume the same `Op` stream with explicit
  dependencies and contention on each `Chip` stage (DRAM channels, NoC hops,
  SRAM banks, links) — the block-diagram hardware model is designed for this.
- Pipeline and expert parallelism; multi-rack topologies (rail-optimized
  Ethernet/IB); prefill/decode disaggregation.
- Efficiency factors calibrated against measured benchmarks (MLPerf,
  vendor numbers) to bracket roofline optimism.
- Richer attention variants (MLA, sliding window) and speculative decoding.
