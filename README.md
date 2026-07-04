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

# pipeline + expert parallelism
inferencesim run --hardware gb300-nvl72 --model gpt-oss-120b \
    --tp 4 --ep 8 --batch 128 --weight-dtype fp4 --kv-dtype fp8
inferencesim run --hardware tt-quietbox --model llama-3.1-70b \
    --tp 2 --pp 2 --batch 32 --weight-dtype fp8 --kv-dtype fp8

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
graph.py        hierarchical hardware graph: nodes (constraints) + edges
                (interconnects), arbitrary nesting, JSON in/out
bridge.py       spec-sheet <-> graph conversion; aggregates any graph into
                the view the analytic engine consumes
hardware.py     Compute / Memory / Link / Chip / Node / System (spec-sheet layer)
workload.py     ModelSpec (dense + MoE, GQA), Scenario, Deployment
ops.py          lowering: (model, scenario, deployment) -> per-chip Op list
engine.py       Engine interface + RooflineEngine ("speed of light")
simulate.py     orchestration, memory feasibility, power & cost models
presets.py      built-in chips/systems/models (editable spec sheets)
presets_fine.py hand-built fine-grained graphs (per-core Blackhole, ...)
```

### Hardware: a hierarchical graph

The structural source of truth is a **graph**: nodes carry constraints
(a memory node: capacity + bandwidth; a compute node: which dtypes at what
FLOP/s; a switch: aggregate bandwidth), edges are interconnects with
bandwidth + latency — a NoC hop, NVLink, or the ConnectX-7 between two DGX
Sparks are all just edges at different levels. A node's `count` means "this
many identical instances" (140 Tensix cores) without materialising them.

Groups scale without clutter: `count` on a node means N identical
instances, and edges between counted groups declare a wiring `pattern`
(`interleave` — one link per instance, covering one-to-one and star — or
`all`). `Graph.expand()` materialises the instances (`sram[0]`…`sram[34]`)
and the concrete links the pattern implies; selectors (`sram[3]`,
`sram[0:8]`) let you hand-wire irregular topologies or mark harvested
units. Aggregate bandwidth queries use max-flow (`Graph.max_flow`), which
credits parallel routes and gives identical answers on grouped and
expanded forms (tested).

Nesting sets the abstraction level: a **composite** node contains an inner
graph and exposes ports. The same QuietBox can be modelled with lumped
chips or per-core:

```
inferencesim graph --hardware tt-quietbox        # GDDR6 -> NoC -> SRAM -> compute (lumped)
inferencesim graph --hardware tt-quietbox-fine   # 8 GDDR6 banks, 140 Tensix cores
inferencesim graph --hardware dgx-h100-fine      # 5 HBM stacks, L2 crossbar, 132 SMs
inferencesim graph --hardware tt-quietbox --json > machine.json   # edit it...
inferencesim run --graph machine.json --model llama-3.1-8b        # ...and simulate it
```

Both levels aggregate to identical roofline results (tested); the fine level
exists so a discrete-event engine can put queues and contention on the real
structure, and so an external editor (the planned UI) has real blocks to
draw. Graphs serialise to versioned JSON — that file is the UI's document
format.

The spec-sheet layer (`Chip`/`Node`/`System`) remains as a convenient way to
write presets; `bridge.py` converts it to graphs and aggregates arbitrary
(convention-following) graphs back into the homogeneous view the analytic
engine needs. Collectives pick the slowest link their group crosses.

### Workload and lowering

`ModelSpec` covers dense and MoE decoder-only transformers with GQA
(parameter counts are validated against published totals in the tests). The
lowering in `ops.py` turns a serving scenario into per-chip `Op` records —
FLOPs, DRAM bytes, collective bytes — for the two phases:

- **prefill** (one request, compute-bound at speed of light), and
- **decode** (a batch of sequences, dominated by streaming weights + KV).

MoE decode accounts for the *expected number of distinct experts* activated
by a batch, which is what actually determines DRAM traffic.

### Parallelism

A replica occupies `tp * pp * ep` chips; the remaining chips form
data-parallel (DP) replicas.

- **TP (tensor parallel)**: every matrix sharded `tp` ways; 2 ring
  allreduces per layer. Cuts TPOT (each chip streams 1/tp of the weights)
  at the cost of per-layer collective latency.
- **PP (pipeline parallel)**: layers split into `pp` balanced stages;
  decode runs `pp` microbatches round-robin, so TPOT is the pipeline round
  time plus P2P hops. Per-chip memory drops ~1/pp — PP buys *capacity*
  (bigger batches, or fitting at all), not faster weight streaming.
- **EP (expert parallel, MoE only)**: attention runs data-parallel across
  `ep` groups while expert weights shard over the whole `tp*ep` array;
  the FFN allreduce is replaced by dispatch/combine all-to-alls, and the
  full batch (not just one group's share) amortizes expert weight reads.

### Engines

`RooflineEngine` is the "speed of light" model: each op runs at the peak
rate of its bottleneck resource (`t = max(flops/peak, bytes/bw)`), collectives
use bandwidth-optimal ring formulas, and there are no kernel or scheduling
overheads. Results are optimistic bounds — real systems achieve some fraction
of them. TP collectives are serialized by default; pass `--overlap-comm` to
assume perfect overlap (reality is in between).

`DESEngine` (`--engine des`) is a discrete-event simulation: one task per
(round, microbatch, pipeline stage, layer) with explicit dependencies,
scheduled against FIFO resources (each stage's execution unit, its
collective fabric, its outbound p2p link). Per-task service times reuse
the roofline math, so differences between the engines are pure
scheduling/contention. What's emergent rather than assumed: pipeline
microbatch overlap (TPOT is the *measured* steady-state round period),
the real cost of unbalanced stages (`n_layers % pp != 0` — the analytic
engine only warns), LM-head/hop overlap with other stages' work, and
serial fill/drain for single-request prefill. The report adds a
per-resource utilisation line per phase (which stage unit / collective
fabric / hop was the bottleneck), and `--trace out.json` writes a
Chrome/Perfetto timeline of the run. The scheduler core (`sched.py`)
supports k-server pools and bandwidth-shared links.

**Graph mode** refines this further when the hardware source is a graph
(`--graph FILE` or a fine-grained preset such as `tt-quietbox-fine`) and
`--engine des`: instead of costing a COMPUTE op with the roofline
`max(flops/peak, bytes/bw)`, the engine lowers it to a *tile task graph*
over the expanded chip (`graphdes.py`). Each tile streams from a DRAM bank,
hop by hop along the NoC (a shared processor-sharing resource) into a core's
SRAM, computes its FLOPs on that core's matrix engine, then writes back —
one `sched.py` task per bandwidth-constrained node/edge. Tiles round-robin
over banks and cores (the same interleave `expand()` wires), so what the
roofline averages away becomes *emergent*: DRAM-bank and NoC contention, and
compute/DRAM overlap from double buffering (`--tile-fill` sets the SRAM tile
size; `1/tile-fill` buffers per core). Byte count sizes the memory tiles but
does not partition compute — a compute-bearing op runs at least one tile per
core, so its FLOPs spread over the whole pool (roofline-consistent), while
memory tiling only makes tiles smaller. It is a strict refinement: with a
single bank, unconstrained links and infinite SRAM it collapses to the
lumped result, and it is never optimistic. The per-phase utilisation line
and `--trace` gain the chip resources (`chip:gddr6-bank[3]`, per-op
`op:ffn/…` tracks). Collectives stay closed-form; modelling an op too serial
to fill the chip (few-head attention) needs op structure `ops.py` doesn't
carry yet.

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

- **DES on the expanded chip graph**: contention on DRAM banks, NoC hops
  and per-core SRAM (the `expand()`ed graph is built for this);
  heterogeneous chips; prefill/decode interference in one simulation.
- **Graph editor UI** over the JSON graph format (nodes with constraints,
  edges with latencies, nesting for abstraction levels).
- Multi-rack topologies (rail-optimized Ethernet/IB); prefill/decode
  disaggregation; MoE expert load imbalance.
- Efficiency factors calibrated against measured benchmarks (MLPerf,
  vendor numbers) to bracket roofline optimism.
- Richer attention variants (MLA, sliding window) and speculative decoding.
