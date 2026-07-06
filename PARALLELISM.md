# Parallelism: how a model is spread across chips

The conceptual map behind `Deployment(tp, pp, ep, adp)` — what each strategy
shards, which real deployments use it, where it makes sense, and exactly how
`inferencesim` models it. The ground truth for every "the simulator does X"
claim here is the code: `ops.py` (the lowering — comm patterns and sharding),
`simulate.py` (`weight_bytes_per_chip`, `kv_cache_bytes_per_chip`), and
`engine.py` (the collective closed forms). Companion to `CALIBRATION.md` (which
brackets roofline optimism); this file is the parallelism reference.

A replica occupies `tp * pp * ep * adp` chips; the leftover chips form
data-parallel (DP) replicas (`dp = total_chips // replica_chips`, remainder
idle). `ep` (MoE) and `adp` (dense) are mutually exclusive by model type, so the
per-replica chip array is `tp × pp × ep` for a MoE model or `tp × pp × adp` for a
dense one.

## 1. The map

| strategy | what shards | what replicates | per-layer comm | what it buys | what it costs | typical fabric |
|---|---|---|---|---|---|---|
| **TP** tensor | every weight matrix, `tp` ways; KV up to `n_kv_heads` ways | activations (summed each layer) | 2 ring all-reduces | lower TPOT — each chip streams `1/tp` of the weights; fits a model too big for one chip | 2 collective latencies/layer; KV sharding caps at `n_kv_heads` | fast in-node all-to-all (NVLink/NVSwitch); latency-sensitive |
| **PP** pipeline | layers, into `pp` balanced stages | — (each stage owns distinct layers) | `pp` P2P hops per decode round | per-chip memory `~1/pp` → bigger batch, or fit at all | balance sensitivity; no weight-streaming speedup; `serve` unsupported | any link, incl. slow cross-node — only P2P hops |
| **DP** data | nothing (whole replica copied) | the entire replica | none between replicas | linear throughput at fixed per-request latency | full model + KV per replica; no single-stream or fit help | none between replicas |
| **EP** expert (MoE) | expert bank over `tp*ep`; batch + KV over `ep` | attention & shared-expert weights across `ep` groups | 1 attention all-reduce + dispatch/combine all-to-all | expert-weight sharding + KV batch-shard; full batch amortizes expert reads | 2 all-to-alls/layer (fabric-bound); EPLB unmodeled | high-bandwidth all-to-all; a ring pays multi-hop |
| **ADP** attn-DP (dense) | FFN over `tp*adp`; batch + KV over `adp` | attention weights across `adp` groups | 1 attention all-reduce + allgather + reduce-scatter | per-chip KV `/adp` past the `n_kv_heads` TP wall; FFN streamed over `tp*adp` | gather/scatter/layer; dense-only; decode-only benefit | fast in-node all-to-all (like TP) |

The knobs compose multiplicatively (§3). Read the summary table as "at fixed
hardware, which axis do I grow for this bottleneck" — §4 turns that into a
decision guide.

## 2. Per-strategy

### TP — tensor parallel (`--tp`)

**Mechanics.** Every weight GEMM is sharded `tp` ways: `_linear` divides both the
FLOPs and the weight-byte read by `tp` (`flops = 2·tokens·params/tp`,
`dram_read = params/tp·wbytes + …`). Each chip therefore streams `1/tp` of the
weights per step — the decode (TPOT) win. The price is two ring all-reduces per
layer (`_allreduces_per_layer` returns 2 for a plain dense/`ep=1`/`adp=1` layer):
one after attention, one after the FFN, each summing the `B×d_model` partial
activation across the `tp` group. The roofline cost of one is the
bandwidth-optimal ring form `ring_allreduce_time` = `2(g-1)/g · payload/bw +
2(g-1)·lat` — the `2(g-1)·lat` latency term is why TP is latency-sensitive and
wants a fast switched fabric.

**Where it makes sense / where it doesn't.** TP is the go-to for cutting
single-stream latency and for fitting a model that overflows one chip's DRAM. It
needs a fast all-to-all interconnect: the roofline engine refuses `tp>1` with no
`tp_link` (`RooflineEngine.run_phase`), and `tp_link = link_for_group(tp)`
resolves to the node interconnect only while the group fits one node — a TP group
spanning the network eats the slow link on every all-reduce. **Model-shape cap:**
KV sharding is `n_kv_heads / min(tp, n_kv_heads)` (`_kv_heads_per_chip`), so once
`tp ≥ n_kv_heads` the KV cache stops sharding — `tp=16` and `tp=8` hold the *same*
per-chip KV for an 8-KV-head model. And for **MLA** the KV latent is replicated
across the `tp` group (`kv_cache_bytes_per_chip` divides only by `pp*ep*adp`, not
`tp`), so TP does nothing for MLA KV — batch-sharding (`ep`/`adp`) is the only
lever there.

**Real use.** Llama-70B-class runs TP1 (FP4) / TP2 (FP8) per replica in current
TRT-LLM throughput tables ([perf-overview](https://nvidia.github.io/TensorRT-LLM/latest/developer-guide/perf-overview.html)),
with latency-bound serving pushing TP to a whole node (TP within a node, PP
across nodes — [vLLM Llama 3.1](https://blog.vllm.ai/2024/07/23/llama31.html)).

**Knobs + example.** `--tp N`. Llama-3.1-70B, TP8 on a GB300 rack:

```
$ inferencesim run --hardware gb300-nvl72 --model llama-3.1-70b \
    --tp 8 --batch 64 --prompt 4096 --output 1024 --weight-dtype fp4 --kv-dtype fp8
Parallelism  : TP=8  PP=1  EP=1  DP=9  (8 chips/replica)
Memory/chip  : weights 4.41 GB + kv 6.71 GB + act 4.19 MB = 11.1 GB / 288 GB
TPOT         : 3.9 ms @ mean ctx 4608  -> 256.4 tok/s per request
  breakdown  : comm 66% (comm-bound), attention 19% (memory-bound), linear 15% (memory-bound)
```

The 70B model shards to 4.41 GB of weights/chip; decode is comm-bound (the two
all-reduces/layer over NVLink), which is exactly TP's tradeoff.

### PP — pipeline parallel (`--pp`)

**Mechanics.** The `n_layers` split into `pp` balanced stages; decode runs `pp`
microbatches round-robin through the pipeline. Because stages are balanced,
`decode_ops` lowers one round as the *whole-model op list at the microbatch size*
(`b_tok = B/pp`) plus `pp` P2P hops — the steady-state round period is what a
request feels as TPOT. Prefill of a single request walks the stages sequentially
(`prefill_ops` adds `pp-1` hops). A hop is a P2P op over
`p2p_link = link_for_group(replica_chips)` carrying a `tokens·d_model/tp`
activation slice — cheap and latency-tolerant.

Per-chip weight bytes drop `~1/pp` (`weight_bytes_per_chip` divides the layer
terms by `tp*pp`). But each chip streams its `1/pp` of the layers once per
microbatch, `pp` microbatches per round, so **weight bytes streamed per token are
unchanged** — PP buys *capacity*, not streaming speed. Raise `batch` into the
freed memory to convert that capacity into throughput.

**Where it makes sense / where it doesn't.** PP is the answer when the model +
KV won't fit even after TP, or when replicas must span nodes joined by a slow
link (only small P2P hops cross stage boundaries — no per-layer all-reduce over
the network). It doesn't cut latency and is sensitive to `n_layers % pp` (the
analytic engine only *warns*; the DES engine measures the real bubble cost of
unbalanced stages). `serve` / `serve_disagg` reject `pp>1` (task-level pipeline
fill/drain is future work — `DES_todo.md` §4).

**Real use.** Llama-3.1-405B runs TP8×PP2 across two 8-GPU nodes when they lack
InfiniBand (TP16 with it) — [vLLM Llama 3.1](https://blog.vllm.ai/2024/07/23/llama31.html),
[Llama 3 Herd, arXiv:2407.21783](https://arxiv.org/abs/2407.21783). TRT-LLM also
composes PP with DEP on multi-card workstation parts (e.g. Qwen3-235B
`DEP2,PP2`, [perf-overview](https://nvidia.github.io/TensorRT-LLM/latest/developer-guide/perf-overview.html)).

**Knobs + example.** `--pp N`. Llama-3.1-70B on a 4-chip QuietBox, TP2×PP2 — PP
is what makes the 70B *fit* (18.2 GB weights/chip after the `tp*pp=4` split):

```
$ inferencesim run --hardware tt-quietbox --model llama-3.1-70b \
    --tp 2 --pp 2 --batch 32 --weight-dtype fp8 --kv-dtype fp8
Parallelism  : TP=2  PP=2  EP=1  DP=1  (4 chips/replica)
Memory/chip  : weights 18.2 GB + kv 3.36 GB + act 1.05 MB = 21.5 GB / 32 GB
TPOT         : 75.9 ms @ mean ctx 2304  -> 13.2 tok/s per request
```

### DP — data parallel (implicit remainder)

**Mechanics.** There is no `--dp` flag: `dp = total_chips // replica_chips`
(`simulate`), and any remainder is reported idle. Each replica is a complete,
independent copy of the model under its `tp/pp/ep/adp` mapping — no cross-replica
collective during a step. A whole-system arrival rate is split `/dp` per replica;
whole-system throughput is the per-replica figure `× dp` (`serve` divides `--rate`
by `dp`; `simulate` multiplies `requests_per_s` and `decode_only_tokens_per_s` by
`dp`).

**Where it makes sense / where it doesn't.** DP is the throughput multiplier at
fixed hardware: independent replicas add offered-load capacity with zero
inter-replica communication. It does nothing for single-stream latency and cannot
fit a model that overflows one replica (every replica holds the whole model + its
own KV). Grow the *sharding* array (`tp`, `ep`, `adp`, `pp`) to fit or to speed a
stream; grow `dp` (or `batch`) to serve more of them.

**Knobs + example.** DP falls out of `replica_chips`. On the 72-chip GB300 rack,
`--tp 8` leaves `dp=9` (72/8) with no idle chips; `--tp 16` leaves `dp=4` with
`72 − 4·16 = 8` idle:

```
$ inferencesim run --hardware gb300-nvl72 --model llama-3.1-70b --tp 16 --batch 256 …
Parallelism  : TP=16  PP=1  EP=1  DP=4  (16 chips/replica)  (8 chips idle)
```

### EP — expert parallel, MoE only (`--ep`)

**Mechanics.** For a MoE model, attention runs **data-parallel across the `ep`
groups** (each group is `tp`-sharded and handles `B/ep` sequences,
`b_att = B/(pp*ep*adp)`), while the **expert bank shards over the whole `tp*ep`
array** (`expert_shard = tp*ep` in `_ffn_ops`). The FFN all-reduce is *replaced*
by two all-to-alls — `moe_dispatch` and `moe_combine` — that shuffle each token
to its experts' owners and back (`_allreduces_per_layer` drops to 1, the
attention all-reduce only). Expert weight streaming is charged on the **expected
number of distinct experts** the *full batch* touches
(`expected_active_experts(tokens_total)`, `tokens_total = B/pp` across all
groups): more tokens touch more distinct experts but sub-linearly (saturating at
`n_experts`), so a larger batch amortizes each expert's DRAM read over more
tokens. Per-chip KV divides by `ep` (batch-sharded) — the second EP win.

`DEPn ≡ tp=1, ep=n` in this simulator (validated in `tests/test_dep.py`):
attention replicated across `n` groups, batch+KV sharded by `ep`, experts over
`ep`, dispatch/combine all-to-alls, and a zero-cost `tp=1` attention all-reduce.
A benchmark point labelled `DEP4` is simulated as `--tp 1 --ep 4`.

**Where it makes sense / where it doesn't.** EP is how large MoE models are
served: it shards the huge expert bank across many chips, batch-shards KV, and
feeds each expert a big enough token batch to be compute-efficient. It demands a
high-bandwidth all-to-all fabric — the dispatch/combine cost is
`link_for_group(tp*ep)` bandwidth. On a **ring** fabric the DES engine
(`collectives.py`) routes each all-to-all as shortest-way store-and-forward
(multi-hop), so a ring is genuinely worse for EP than a switched fabric; the
roofline engine costs it as a single `payload/bw + lat` and does not see the
hops. What stays unmodeled: EPLB redundant-expert load balancing and mixed ADP+TP
MoE attention (`--adp` is rejected for MoE — §3).

**Expert-load skew (`MoEConfig.skew` / `--moe-skew`).** By default routing is
uniform, but real MoE traffic is skewed — a few experts are much more popular. A
single Zipf knob `pop_e ∝ 1/(e+1)^skew` (0 = uniform, the bit-identical anchor)
places experts on the `tp*ep` array in contiguous blocks, so a positive skew
concentrates load onto the low-numbered members. The lowering then paces
`moe_routed` by the *hottest* member (its `hot_member_factor` more activation
work and its block's `expected_active_on_member` expert-weight streaming — the
roofline of an unbalanced layer, so TPOT rises with skew on both engines), and
under `--engine des` the switched all-to-all is store-and-forward with per-member
ingress ports: the hot owners **incast** (every sender dispatches a large message
to them, and they egress the most on combine), so their `.l{i}.in` ingress
utilisation tops the `resource_util` report — the observable of the imbalance.
This is the *unmitigated* skew; EPLB / redundant-expert placement is future work.

**Real use.** gpt-oss-120B: DEP2 (B200) / DEP4 (H100); Qwen3-235B-A22B: DEP4
(B200/GB200/H200) / DEP8 (H100); DeepSeek-R1/V3: DEP4 (FP4) / DEP8 (FP8) — all
[TRT-LLM perf-overview](https://nvidia.github.io/TensorRT-LLM/latest/developer-guide/perf-overview.html).
Mixtral-8x7B is commonly TP2×EP8 on 16 GPUs ([TRT-LLM expert-parallelism](https://nvidia.github.io/TensorRT-LLM/advanced/expert-parallelism.html),
[NVIDIA Mixtral blog](https://developer.nvidia.com/blog/achieving-high-mixtral-8x7b-performance-with-nvidia-h100-tensor-core-gpus-and-tensorrt-llm/)).
DeepSeek's own production serving is large-scale EP with DP-attention and PD
disaggregation — prefill EP32/DP32, decode EP144 (or EP320)/DP144
([DeepSeek-V3 report, arXiv:2412.19437](https://arxiv.org/abs/2412.19437);
[DeepSeek inference-system overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md);
[LMSYS large-scale EP](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)).

**Knobs + example.** `--ep N` (MoE only). gpt-oss-120B as DEP4 (`--tp 1 --ep 4`)
on GB300:

```
$ inferencesim run --hardware gb300-nvl72 --model gpt-oss-120b \
    --tp 1 --ep 4 --batch 128 --weight-dtype fp4 --kv-dtype fp8
Parallelism  : TP=1  PP=1  EP=4  DP=18  (4 chips/replica)
Memory/chip  : weights 15.4 GB + kv 1.59 GB + act 737 kB = 17 GB / 288 GB
TPOT         : 2.18 ms @ mean ctx 2304  -> 459.3 tok/s per request
  breakdown  : moe 81% (memory-bound), attention 8% (memory-bound), comm 6% (comm-bound)
```

Decode is dominated by streaming the (expected-active) experts — the MoE weight
read, not comm — which is the regime EP is built for.

### ADP — attention data parallel, dense only (`--adp`)

**Mechanics.** The DeepSeek-V3-style "**DP attention + TP FFN**" pattern (TRT-LLM's
dense DEPn). Attention runs data-parallel across `adp` groups (each `tp`-sharded,
handling `B/adp` sequences); its qkv/out weights are sharded `tp` ways and
**replicated** across the groups — so **per-chip attention weight bytes are
unchanged by `adp`, while per-chip KV divides by `adp`** (`kv_cache_bytes_per_chip`
divides by `pp*ep*adp`). That KV cut is the point. The dense FFN instead shards
over the **whole `tp*adp` array** (`shard = tp*adp` in `_ffn_ops`;
`ffn_params/(tp*adp)` streamed per chip — better than `tp` alone). Because
attention leaves each token sequence-sharded but the FFN is TP over the whole
array, the FFN all-reduce is replaced by a **sequence allgather before the FFN**
(assemble the full-batch hidden state) and a **reduce-scatter after** it (both
`HALFRING` ops). Each is exactly *half* a ring all-reduce over `g = tp*adp` —
`ring_gather_time` = `(g-1)/g · payload/bw + (g-1)·lat` — so `adp` trades one
`tp`-group FFN all-reduce for a gather+scatter over the larger `tp*adp` group. At
`adp=1` everything is bit-identical to plain TP.

**Where it makes sense / where it doesn't.** ADP exists to cut per-chip KV once
TP can't: TP's KV sharding caps at `n_kv_heads`, and MLA replicates the latent
across TP, so past that wall batch-sharding is the only KV lever. It is a
**decode** win — the KV division and the `tp*adp` FFN streaming both pay off in
memory-bound decode; prefill is compute-bound and single-request, so ADP only
adds gather/scatter comm there. ADP is **dense-only**: MoE attention-DP is exactly
what `ep` provides, so `--adp` on a MoE model is rejected (`validate_deployment`).

**Real use.** DeepSeek-V3's DP-attention + TP/EP-FFN serving
([arXiv:2412.19437](https://arxiv.org/abs/2412.19437), MLA from
[DeepSeek-V2, arXiv:2405.04434](https://arxiv.org/abs/2405.04434)); TRT-LLM's dense
DEPn ([perf-overview](https://nvidia.github.io/TensorRT-LLM/latest/developer-guide/perf-overview.html)).

**Knobs + example.** `--adp N` (dense only). Same 16 chips, past the `n_kv_heads=8`
wall — reallocating a factor of 2 from TP to ADP halves per-chip KV (48.3 → 24.2
GB) while keeping the FFN streamed over all 16:

```
$ inferencesim run --hardware gb300-nvl72 --model llama-3.1-70b \
    --tp 16 --batch 256 --prompt 8192 --output 1024 --weight-dtype fp4 --kv-dtype fp8
Parallelism  : TP=16  PP=1  EP=1  DP=4  (16 chips/replica)  (8 chips idle)
Memory/chip  : weights 2.2 GB + kv 48.3 GB + act 16.8 MB = 50.5 GB / 288 GB

$ inferencesim run --hardware gb300-nvl72 --model llama-3.1-70b \
    --tp 8 --adp 2 --batch 256 --prompt 8192 --output 1024 --weight-dtype fp4 --kv-dtype fp8
Parallelism  : TP=8  PP=1  EP=1  ADP=2  DP=4  (16 chips/replica)  (8 chips idle)
Memory/chip  : weights 2.65 GB + kv 24.2 GB + act 8.39 MB = 26.8 GB / 288 GB
```

Weights rise slightly (2.2 → 2.65 GB — attention now sharded `/8` not `/16` but
replicated `×2`); KV halves — the batch-shard TP couldn't deliver past 8 KV heads.
The MLA analogue: deepseek-v3 (MoE, so `ep` not `adp`) holds KV 11.5 GB at `--tp 8`
but 1.44 GB (= 11.5/8) at `--tp 1 --ep 8` — TP does nothing for the replicated MLA
latent; only the `ep` batch-shard cuts it.

## 3. Composition

The four axes multiply into the per-replica chip count, and DP is the remainder:

```
replica_chips = tp × pp × ep × adp
dp            = total_chips // replica_chips      (leftover chips idle)
```

**The constraint set** (`validate_deployment` + `simulate` warnings):

| rule | where | consequence |
|---|---|---|
| `tp, pp, ep, adp ≥ 1` | `validate_deployment` | error otherwise |
| `ep > 1` needs a MoE model | `validate_deployment` | `ep` on a dense model → error |
| `adp > 1` needs a dense model | `validate_deployment` | `adp` on a MoE model → error (use `ep`) |
| `replica_chips ≤ total_chips` | `simulate` | error otherwise |
| `total_chips % replica_chips` | `simulate` | warns; the remainder is idle |
| `n_layers % pp` | `simulate` | warns (stages assumed balanced anyway; DES measures the bubble) |
| `batch < pp*ep*adp` | `simulate` | warns — groups/microbatches starved |
| weights + KV + act ≤ DRAM/chip | `simulate` | warns "does not fit" |

Because `ep` (MoE) and `adp` (dense) never coexist, the FFN-array group is `tp*ep`
or `tp*adp` — `CommContext` folds both into one `a2a` group size, and its fabric
is `link_for_group(tp*ep*adp)`.

**Worked example.** gpt-oss-120B on the 72-chip GB300 rack, `--tp 2 --ep 4`:
`replica_chips = 2·1·4·1 = 8`, `dp = 72 // 8 = 9`, 0 idle. The expert bank shards
over `tp*ep = 8`; attention is DP across the 4 `ep` groups (each `tp=2`-sharded);
9 such replicas run independently.

```
$ inferencesim run --hardware gb300-nvl72 --model gpt-oss-120b \
    --tp 2 --ep 4 --batch 128 --weight-dtype fp4 --kv-dtype fp8
Parallelism  : TP=2  PP=1  EP=4  DP=9  (8 chips/replica)
Memory/chip  : weights 7.69 GB + kv 793 MB + act 737 kB = 8.49 GB / 288 GB
```

## 4. Decision guide — which knob for which bottleneck

Read the report's `TPOT`/`TTFT` breakdown line (which category, and
compute/memory/comm-bound), then:

| bottleneck | grow | why |
|---|---|---|
| decode weight-streaming-bound (TPOT memory-bound on `linear`/`moe`) | `tp` (dense); `tp×adp` or `ep` (past the `n_kv_heads` / MLA wall) | each chip streams a smaller weight share |
| KV-capacity-bound (won't fit, or few requests fit) | `adp`/`ep` (batch-shard KV), or `pp` (fewer layers/chip) | cut per-chip KV footprint |
| latency-sensitive single stream | `tp` on a fast fabric — nothing else | TP is the only axis that cuts one request's per-token time; DP/EP/ADP need a batch, PP adds hops |
| throughput at fixed hardware | `dp` replicas, or a bigger `batch` | independent load capacity, or more decode amortization |
| MoE (large expert bank) | `ep` | shard the bank + batch-shard KV + amortize expert reads over the full batch |
| model/KV won't fit one chip | `tp`, then `pp` | TP first (also speeds decode); PP when TP is exhausted or must cross a slow link |
| long prompts / prefill interference | chunked prefill or disaggregation | not a parallelism axis — see the README serving section (`serve --prefill-chunk` / `serve --disagg`) |

**Fabric caveat.** `tp` and `ep`/`adp` all-reduce or all-to-all every layer, so
they want a fast switched fabric; a **ring** (e.g. TT-QuietBox) makes the DES
engine route MoE all-to-alls as multi-hop store-and-forward, so EP is
disproportionately penalized on a ring vs a switched NVSwitch domain (the roofline
engine only sees the per-hop link bandwidth, not the hop count — see the README
Engines section and `DES_todo.md` §2). `pp` only ships P2P hops and tolerates a
slow cross-node link, which is why it is the axis that spans nodes.

## 5. What real deployments run

One row per (model class, strategy, source). MLPerf/anchor rows are detailed in
`CALIBRATION.md` §6; this table is the deployment-shape map.

| model class | parallelism | maps to | source |
|---|---|---|---|
| Llama-3.3/3.1-70B | TP1 (FP4) / TP2 (FP8) per replica; TP→node for latency | `--tp 1/2/…8` | [TRT-LLM perf-overview](https://nvidia.github.io/TensorRT-LLM/latest/developer-guide/perf-overview.html); [vLLM](https://blog.vllm.ai/2024/07/23/llama31.html) |
| Llama-3.1-405B | TP8×PP2 (2 nodes, no IB) or TP16 (IB) | `--tp 8 --pp 2` | [vLLM Llama 3.1](https://blog.vllm.ai/2024/07/23/llama31.html); [Llama 3 Herd](https://arxiv.org/abs/2407.21783) |
| Mixtral-8x7B (MoE) | TP2×EP8 (16 GPUs); TP+EP hybrid | `--tp 2 --ep 8` | [TRT-LLM EP](https://nvidia.github.io/TensorRT-LLM/advanced/expert-parallelism.html); [NVIDIA Mixtral](https://developer.nvidia.com/blog/achieving-high-mixtral-8x7b-performance-with-nvidia-h100-tensor-core-gpus-and-tensorrt-llm/) |
| gpt-oss-120B (MoE) | DEP2 (B200) / DEP4 (H100) | `--tp 1 --ep 2/4` | [TRT-LLM perf-overview](https://nvidia.github.io/TensorRT-LLM/latest/developer-guide/perf-overview.html); `CALIBRATION.md` §6.3 |
| Qwen3-235B-A22B (MoE) | DEP4 (B200/GB200/H200) / DEP8 (H100); DEP2,PP2 (RTX) | `--tp 1 --ep 4/8` (+`--pp`) | [TRT-LLM perf-overview](https://nvidia.github.io/TensorRT-LLM/latest/developer-guide/perf-overview.html) |
| DeepSeek-V3/R1 (MoE+MLA) | DEP4 (FP4) / DEP8 (FP8); native: DP-attn + EP32/EP144-320, PD-disagg | `--tp 1 --ep n` | [TRT-LLM](https://nvidia.github.io/TensorRT-LLM/latest/developer-guide/perf-overview.html); [V3 report](https://arxiv.org/abs/2412.19437); [DeepSeek inference system](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) |
| 70B / gpt-oss / 405B, rack-scale | MLPerf GB300 / H100 rack configs | (see anchors) | `CALIBRATION.md` §6.2–6.3 (MLPerf, TRT-LLM) |

DEP → `tp=1, ep=n` (MoE) or `--adp n` (dense). TRT-LLM's per-GPU throughput
tables report `tps/gpu`; the `× GPUs` system-total conversions live per-anchor in
`CALIBRATION.md`.

## 6. Not yet modeled (the honest boundary)

- **Context / sequence parallelism** — ring attention, DeepSeek context-parallel
  prefill: a very long prompt is not split across a replica's chips. `serve_disagg`
  prefills one whole prompt on one replica (`DES_todo.md` §4, "context-parallel
  prefill").
- **EPLB / redundant experts** — the `MoEConfig.skew` knob models the
  *unmitigated* hot-expert imbalance (skewed streaming + all-to-all incast onto
  hot owners, §EP above); DEP's redundant-expert load balancing that rebalances
  it is not modeled (`CALIBRATION.md` §6.3, §8.1; README Parallelism).
- **Mixed ADP+TP MoE attention** — real DEP composes TP inside the DP-attention
  groups; here MoE attention-DP is exactly `ep` (`CALIBRATION.md` §8.1).
- **KV-transfer / collective contention** — concurrent disagg KV transfers each
  pay their own `bytes/bw + lat` with no shared-link occupancy (`DES_todo.md` §4).
  (A MoE all-to-all's ingress incast onto hot experts *is* now modeled under
  `MoEConfig.skew` — §EP above, `DES_todo.md` §2.)
- **Task-level pp>1 serving** — `serve`/`serve_disagg` are pp=1 (pipeline
  fill/drain across stages is future work — `DES_todo.md` §4).

See also: README "Parallelism" (the short version), README "Serving simulation"
(chunked prefill, disaggregation), `CALIBRATION.md` (efficiency/anchors).
