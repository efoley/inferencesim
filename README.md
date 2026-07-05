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

Per-instance heterogeneity rides the same selectors: a `derate` on a node
(or `Graph.derate_instances('tensix-fpu[132:140]', 0.0)` on a group) scales
that instance's rate-like figures — effective FLOP/s and bandwidth — with
`0.0` meaning a disabled instance that exists physically but does no work.
So a harvested 132-of-140-core die, a throttled core, or a dead DRAM bank is
one line: the aggregates and the discrete-event engine both see the live
fraction (a disabled unit leaves every aggregate including capacity; a
derated-but-live one keeps its capacity).

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

`ModelSpec` covers dense and MoE decoder-only transformers with a choice of
attention: plain **GQA**, **sliding-window** (`swa_window` + `swa_every`, e.g.
gpt-oss's 128-token window on alternating layers), or **multi-head latent
attention** (**MLA**, the DeepSeek-V2/V3 compressed-KV cache — also how
attention-DP serving is done in practice). Windowed layers cap their KV cache
and decode reads at the window; MLA caches only the shared compressed latent
(`kv_lora_rank + qk_rope_head_dim` per token per layer), ~2 orders of magnitude
smaller than an MHA cache and replicated (not sharded) under TP. Parameter
counts are validated against published totals in the tests — including the
`deepseek-v3` preset (671B total / 37B active, MoE + MLA). The lowering in
`ops.py` turns a serving scenario into per-chip `Op` records — FLOPs, DRAM
bytes, collective bytes — for the two phases:

- **prefill** (one request, compute-bound at speed of light), and
- **decode** (a batch of sequences, dominated by streaming weights + KV).

MoE decode accounts for the *expected number of distinct experts* activated
by a batch, which is what actually determines DRAM traffic.

### Parallelism

A replica occupies `tp * pp * ep * adp` chips; the remaining chips form
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
  **This is exactly TRT-LLM's `DEPn` for MoE**: `DEPn ≡ tp=1, ep=n` here —
  attention weights replicated across the `n` groups, batch and KV sharded by
  `ep`, experts sharded over `ep`, dispatch/combine all-to-alls, and no FFN
  allreduce (the tp=1 attention allreduce is zero-cost). So a benchmark point
  labelled `DEP4` is simulated as `--tp 1 --ep 4` (validated in
  `tests/test_dep.py`); what stays unmodeled is EPLB redundant-expert load
  balancing and mixed ADP+TP MoE attention.
- **ADP (attention data-parallel, dense only)**: the DeepSeek-V3-style
  "DP attention + TP FFN" pattern, and TRT-LLM's dense `DEPn`. Attention runs
  data-parallel across `adp` groups (each `tp`-sharded, handling `batch/adp`
  sequences); its qkv/out weights are sharded `tp` ways and **replicated**
  across the groups, so **per-chip attention weight bytes are unchanged by
  `adp` while per-chip KV divides by `adp`** — that KV cut is the point. The
  dense FFN instead shards over the **whole `tp*adp` array** (per-chip FFN
  weight bytes = `ffn_params/(tp*adp)`, better weight streaming than `tp`
  alone). Because attention leaves each token sequence-sharded but the FFN is
  TP over the whole array, the FFN allreduce is replaced by a **sequence
  allgather before the FFN** (assemble the full-batch hidden state) and a
  **reduce-scatter after** it. Each is exactly *half* a ring allreduce over
  `g = tp*adp` — `(g-1)/g · payload/bw + (g-1)·lat` with `payload` the full
  batch's `B×d_model` hidden state — so `adp` trades one `tp`-group FFN
  allreduce for an allgather + reduce-scatter over the larger `tp*adp` group.
  At `adp = 1` everything is bit-identical to plain TP. MoE attention-DP is
  what `ep` already provides, so `adp` is rejected for MoE models.

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
serial fill/drain for single-request prefill. Collectives are expanded to
their actual per-step transfers over the declared fabric topology
(`collectives.py`): a ring allreduce becomes 2(g-1) barrier-separated steps
on per-member directional links, and a MoE all-to-all becomes g-1 per-member
messages on a switched (all-to-all) fabric or shortest-way store-and-forward
routing on a ring — so ring vs all-to-all fabrics genuinely differ, and
collectives contend with pipeline hops (which ride the boundary chip's link)
instead of on a lumped fabric. Link resources carry only bandwidth occupancy
(`bytes/bw`); propagation latency rides the dependency chain, never a link —
so each collective reproduces its closed form exactly in isolation and
diverges only under genuine bandwidth contention. The report adds a
per-resource utilisation line per phase (which stage unit / link was the
bottleneck), and `--trace out.json` writes a Chrome/Perfetto timeline of the
run. The scheduler core (`sched.py`) supports k-server pools and
bandwidth-shared links.

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
`op:ffn/…` tracks). Collectives are expanded per-step over the fabric
topology in both modes (see above); modelling an op too serial to fill the
chip (few-head attention) needs op structure `ops.py` doesn't carry yet.

### Serving simulation

`inferencesim run` reports steady-state averages. `inferencesim serve` instead
runs a **request-level, iteration-granular** event loop for one replica, so the
numbers carry queueing and prefill/decode interference rather than assuming them
away:

```
inferencesim serve --hardware gb300-nvl72 --model llama-3.1-70b --tp 8 \
    --rate 88 --requests 200 --max-batch 64 --prompt 4096 --output 1024 \
    --weight-dtype fp4 --kv-dtype fp8
# mixed-length trace ("time prompt output" per line) + chunked prefill
inferencesim serve --hardware gb300-nvl72 --model llama-3.1-70b --tp 8 \
    --arrivals chat.txt --prefill-chunk 512 --max-batch 64
# Poisson with sampled lengths and the reserve KV policy
inferencesim serve --hardware gb300-nvl72 --model gpt-oss-120b --tp 4 --ep 8 \
    --rate 20 --prompt-dist lognormal:2048:0.5 --output-dist uniform:64:1024 \
    --kv-policy reserve
```

Requests arrive by a seeded Poisson process (`--rate`, a whole-system rate
divided by DP for one replica) or an explicit trace (`--arrivals`). Prompt and
output lengths are per request: fixed from the scenario, given per-line in the
trace (`time [prompt output]`), or sampled from a `--prompt-dist`/`--output-dist`
(`uniform:LO:HI` or `lognormal:MEDIAN:SIGMA`, off the same seeded RNG).

The loop interleaves iterations under continuous batching. Admission and prefill
have knobs that mirror vLLM:

- **KV policy** (`--kv-policy`). `on_demand` (default) admits a request against
  only its prompt KV (up to a `--kv-watermark` fraction of the budget), grows KV
  a token at a time, and **preempts** the newest decoder (recompute) when a step
  would overflow — the victim's KV is freed, it returns to the front of the
  queue, and on re-admission its prefill recomputes prompt + tokens-so-far before
  decoding resumes (total tokens stay correct, the recompute shows up as a big
  inter-token gap for the victim). `reserve` admits only if the full
  prompt+output KV fits, so it never preempts but packs fewer requests.
- **Prefill** (`--prefill-chunk`). Default is *exclusive*: a waiting request's
  whole prompt is one iteration that freezes the decode batch (a big inter-token
  gap). `--prefill-chunk K` instead mixes a K-token prefill chunk into each
  decode iteration (Sarathi-style): the long-prefill stall collapses into small
  per-iteration bumps, at the cost of a longer TTFT for the prefilling request
  (each chunk re-streams weights and re-reads its growing KV). A 32k-prompt
  request landing in chat traffic on a GB300 replica: exclusive drives the
  batch's inter-token p99 to ~380 ms; `--prefill-chunk 512` collapses it to
  ~10 ms while the 32k request's TTFT roughly doubles.

Each decode iteration emits one token for every running request, its attention
recost at the batch's actual Σ per-request context (per-token KV growth).
Because pp=1 an iteration is a serial op chain — its duration a closed-form sum,
so there is no scheduler here; the interesting dynamics are all *between*
iterations. The report gives offered vs achieved throughput, TTFT p50/p95/p99 +
mean (queueing included), per-request TPOT, inter-token-gap p99 (the
interference metric), batch occupancy, peak KV, preemption count, and
prompt/output length percentiles when lengths are mixed. **Scope**: one replica,
`pp == 1`; a single request prefills at a time; preemption is recompute-only (no
KV swap). Task-level pp>1 serving is future work (see `DES_todo.md` §4). The
per-iteration cost accepts a `DESEngine`, so graph-refined chip costs flow into
serving numbers.

#### Prefill/decode disaggregation (`--disagg`)

`inferencesim serve --disagg` partitions the chips into two pools — a **prefill
pool** and a **decode pool** — and streams the KV cache between them, the
DistServe / NVIDIA Dynamo *disaggregated serving* architecture. A waiting
request runs its whole prompt on any free prefill replica (exclusive, one
request per prefill replica — prefill replicas never batch decode, which is the
point); on completion the KV cache transfers to the least-loaded decode replica
with headroom, and decode runs there as pure decode iterations, **never stalled
by a prefill**. That is the architectural win: the inter-token-gap spike the
aggregated loop pays when a prefill preempts a decoding batch simply does not
arise, and it costs no TTFT (unlike chunked prefill, which caps the stall by
paying more TTFT). First token lands at prefill completion + KV transfer, so
TTFT includes the transfer delay.

```bash
inferencesim serve --hardware gb300-nvl72 --model llama-3.1-70b --disagg \
    --prefill-tp 8 --prefill-replicas 3 --decode-tp 8 --decode-replicas 6 \
    --rate 58 --requests 2000 --prompt 4096 --output 1024 \
    --weight-dtype fp4 --kv-dtype fp8
```

The pools each carry their own `--{prefill,decode}-tp/-ep/-adp` (adp/ep compose
on either side); the chips are partitioned as
`n_p·prefill.replica_chips + n_d·decode.replica_chips ≤ total_chips` (idle chips
reported). Everything else — the arrival process, **mixed request lengths**
(`--prompt-dist`/`--output-dist` or a `time prompt output` trace), `--max-batch`,
and the **`--kv-policy`** — comes from the same `ServeConfig` the aggregated loop
uses, so the whole polished admission surface applies unchanged. The KV transfer
costs `kv_bytes(context) / bw + latency`, where the link resolves from the system
with `link_for_group` semantics — the node interconnect if both pools fit one
node, else the network (`--transfer-bw` / `--transfer-latency` override). The
report gains per-pool utilisation and transfer stats; TTFT now includes the
transfer. The sweet spot balances the pools against the workload's prefill-time
share (a 4k/1k llama-70b point spends ~⅓ of its time in prefill → ~3 of 9 GB300
replicas prefill): undersize the prefill pool and TTFT queues while decode idles;
undersize decode and it saturates.

Under `--kv-policy on_demand` (the default), a decode replica that would overflow
its KV budget preempts its newest decoder — and because decode replicas have no
prefill hardware, the victim returns to the **front of the prefill pool** to
recompute prompt + generated tokens and then **re-transfers**, so a preemption
honestly costs a re-prefill *and* a re-transfer (reported as `n_preemptions`).
`--kv-policy reserve` admits only full-footprint requests and never preempts.
**v1 simplifications**: **no transfer-link contention** between concurrent
transfers (each pays its own `bytes/bw + lat`); **chunked prefill is N/A** with
`--disagg` (exclusive prefill replicas make it moot — the combination is
rejected); prefill is one whole prompt on one replica (no context-parallel
prefill). See `DES_todo.md` §4.

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
- Roofline numbers are upper bounds: perfect tiling, perfect load balance,
  bandwidth at 100% efficiency, bandwidth-optimal collectives, no kernel
  launch overhead. That "speed of light" (`--efficiency sol`) is the default.
  To derate toward measured reality, pass `--efficiency typical`, which scales
  peak compute/memory/collective bandwidth and adds a per-op launch overhead
  (an `Efficiency`); the same knobs are available piecemeal (`--eff-compute`,
  `--eff-memory`, `--eff-collective`, `--op-overhead-s`). The `typical` profile
  is **fitted (coarse)** against measured anchors — `inferencesim calibrate`
  scores the simulator against them and the fit is documented in
  `CALIBRATION.md`.
- **Per-vendor derating (`--efficiency auto`).** One global `typical` cannot
  serve both vendors: the Tenstorrent tt-metal stack reaches a lower effective
  memory bandwidth than NVIDIA (~0.40 vs 0.57), so a QuietBox decode point reads
  ~1.4× optimistic under the global fit. `--efficiency auto` picks the
  vendor-appropriate profile per hardware key — `typical-tt` for Tenstorrent
  (`tt-*`), `typical-nv` (== `typical`) otherwise — bringing that anchor to ~1.0×
  while leaving every NVIDIA number unchanged. `typical-nv` / `typical-tt` can
  also be named explicitly; the default stays `sol` everywhere (`auto` is opt-in).
  Works on `run`, `serve`, and `calibrate`; see `CALIBRATION.md` §8.1.

  ```bash
  inferencesim run --hardware gb300-nvl72 --model llama-3.1-70b \
      --tp 8 --batch 64 --weight-dtype fp4 --kv-dtype fp8 --efficiency typical
  # per-vendor: Tenstorrent decode derated to match tt-metal (~0.40 memory)
  inferencesim run --hardware tt-quietbox-2 --model qwen3-32b --tp 4 \
      --batch 32 --prompt 558 --output 128 --kv-dtype fp8 --efficiency auto
  ```

## Roadmap

- **DES on the expanded chip graph**: contention on DRAM banks, NoC hops
  and per-core SRAM (the `expand()`ed graph is built for this);
  heterogeneous chips. (Prefill/decode interference now lands in the
  request-level serving loop, `inferencesim serve`; extending it to chunked
  prefill and pp>1 is next — see `DES_todo.md` §4.)
- **Graph editor UI** over the JSON graph format (nodes with constraints,
  edges with latencies, nesting for abstraction levels).
- Multi-rack topologies (rail-optimized Ethernet/IB); MoE expert load
  imbalance. (**Prefill/decode disaggregation landed**: `serve --disagg`, a
  prefill pool + decode pool with KV streamed between them — DistServe/Dynamo;
  see the Serving subsection. Remaining: transfer-link contention,
  chunked+disagg, context-parallel prefill.)
- Explicit attention-DP + expert-parallel (TRT-LLM `DEPn`) deployments —
  *landed*: dense attention-DP is `--adp n` (DP attention + TP FFN, KV cut by
  `adp`, FFN streamed over `tp*adp`); MoE `DEPn ≡ tp=1, ep=n` (validated, see
  the Parallelism section and `tests/test_dep.py`). Remaining: EPLB
  redundant-expert load balancing and mixed ADP+TP MoE attention.
- **Efficiency factors calibrated against measured benchmarks** (MLPerf,
  vendor numbers) to bracket roofline optimism — *mechanism landed, coarse fit
  landed, per-vendor profiles landed, refinement ongoing*: `Efficiency`,
  `--efficiency` (incl. `auto`), `inferencesim calibrate`, and the
  `calibration.py` anchor harness are in; the cross-vendor `typical` is fitted
  against 12 sourced anchors, and per-vendor `typical-nv` / `typical-tt`
  (Tenstorrent memory 0.40) are selected by `--efficiency auto`
  (`CALIBRATION.md` §8.1). Next: confirm the `[VERIFY]` anchors, refit the tt
  `compute`/`collective` knobs once a compute-bound tt anchor exists, and score
  offline throughput through the interleaving `serve` path.
- Richer attention variants (MLA, sliding window) — *landed*: sliding-window
  attention (`swa_window`/`swa_every`; the `gpt-oss-120b` preset now models its
  128-token alternating window, cutting KV and decode-attention traffic on half
  its layers) and MLA (the `deepseek-v3` preset, 671B/37B-active MoE + MLA, with
  the compressed latent replicated under TP and batch-sharded under EP). Next:
  speculative decoding.
