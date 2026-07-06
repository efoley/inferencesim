# Calibration: bracketing roofline optimism with measured anchors

The human-readable half of the calibration mechanism in
`inferencesim/efficiency.py` and `inferencesim/calibration.py`. It records the
*measured* benchmark numbers the simulator is scored against, the caveats that
make such comparisons honest, the published derating ranges, and the transparent
fit that produced `PROFILES["typical"]`.

**Status (2026-07-05):** mechanism landed; `calibration.ANCHORS` is populated
with 12 sourced anchors; the cross-vendor `PROFILES["typical"]` is **fitted
(coarse)** from them (§8); **per-vendor `typical-nv` / `typical-tt` profiles
landed** (§8.1), selected by `--efficiency auto`. Every measured figure carries a
source, a retrieval/publication date, and a provenance tag:

- **VERIFIED** — an MLCommons-reviewed MLPerf submission.
- **VENDOR** — vendor-published (NVIDIA/Tenstorrent docs, blogs). *VENDOR-CI*
  = a vendor CI-measured value with a tolerance band (tt-metal).
- **INDEPENDENT** — third-party measurement (LMSYS, SemiAnalysis, reviewers).
- **[VERIFY]** — a specific cell not yet confirmed against a primary source;
  encoded but flagged in `notes`.

Retrieval date is **2026-07-05** unless a publication date is given.

---

## 1. What the knobs mean

The roofline engine computes *upper bounds*: each op at its bottleneck peak
(`t = max(flops/peak, bytes/bw)`), bandwidth-optimal collectives, no launch
overhead. An `Efficiency` derates them:

| knob | scales | roofline term | fitted `typical` | bracket (§5) |
|---|---|---|---|---|
| `compute` | peak FLOP/s (MFU) | `flops/(peak*compute)` | **0.58** | prefill ~0.30-0.58 |
| `memory` | peak bandwidth (MBU) | `bytes/(bw*memory)` | **0.57** | decode ~0.55-0.70 |
| `collective` | link bw for collectives | occupancy term only, **never** latency | **0.85** | 0.70-0.95 large-msg |
| `op_overhead_s` | fixed s per launched op | `+ op_overhead_s` | **1.5 us** | 1-10 us |

`Efficiency()` (all 1.0, zero overhead) is the identity — the `sol` profile and
the default. The `collective` knob scales only the *bandwidth occupancy* term;
the engine models each collective's propagation latency separately, so the
small-message penalty that dominates TP decode all-reduces is captured
structurally — fit `collective` to *large-message* bus-bw efficiency.

---

## 2. The optimism ratio

`calibrate` scores each anchor with a metric-direction-agnostic **optimism
ratio** (`>= 1` == simulator optimistic):

    throughput / rate  : simulated / measured   (a faster sim over-reads)
    latency (tpot/ttft): measured / simulated   (a faster sim under-reads time)

At `sol` every ratio must be `>= 1` (a roofline bound cannot be beaten). A fitted
profile derates until the ratios **bracket 1**.

---

## 3. Global caveats — read before trusting a number

1. **Per-user vs per-GPU vs system-total.** `inferencesim run` reports
   whole-system `output_tokens_per_s`, a `decode_only_tokens_per_s` ceiling, and
   per-request `1/TPOT`. Anchors store `measured` **already normalised to the
   sim's whole-system quantity** (per-GPU × GPUs, per-rack as-is); the raw number
   + conversion is in each anchor's `notes`. Source-specific framing:
   - **TensorRT-LLM's own metric changed by version.** `perf-overview.md` at
     commit **0c9430e** (~v1.1.0rc, Sep 2025) reports **total across the TP
     group**; **main / v1.2.0** report **per-GPU** (tps/gpu). A number without
     its commit/tag is ambiguous. (Reconciled per-row in §6.) Also: **llama-3.1-8B
     was dropped from the current doc**, so 0c9430e is its last primary NVIDIA
     source.
   - **MLPerf NVL72 results are rack totals** (÷72 for per-GPU).
   - **Tenstorrent tt-metal is per-user at batch 1** (latency-optimal); vendor
     press "tokens per second" is aggregate (§6.5).
   - **SemiAnalysis InferenceMAX is per-GPU at a fixed per-user SLA** (a point on
     the throughput/interactivity Pareto — the most simulator-friendly framing).

2. **Prefill vs decode.** TTFT is prefill (compute-bound); TPOT/ITL is decode
   (memory-bound). The `Anchor.metric` (`output_tok_s | decode_ceil_tok_s |
   tpot_ms | ttft_ms | req_per_s`) and `regime` (`decode | prefill | mixed`)
   pin the regime. **Offline / max-sustained throughput** (MLPerf Offline) is
   scored against `decode_ceil_tok_s`, not `output_tok_s`: see caveat 6.

3. **Model-name drift is systemic.** Our `llama-3.1-70b` is benchmarked as
   **Llama-2-70B** (MLPerf) and **Llama-3.3-70B** (NVIDIA's current tables). All
   three share the 80-layer / d_model 8192 / 64-query / 8-KV-head / d_ff 28672
   SwiGLU architecture (confirmed vs our preset). Llama-3.3 is arch-identical to
   3.1; Llama-2 differs mainly in vocab (32k vs 128k) and context. Prefer 3.3
   rows; label Llama-2 a proxy. Similarly `qwen3-32b` has **no** NVIDIA-primary/
   MLPerf number; the "Qwen3.5-27B 1M tok/s on B200" figures circulating are
   *misattributed* to Qwen3-32B — do not use (§7 caution list).

4. **dtype labels are not literal.** GPT-OSS is natively MXFP4; a Hopper "FP8"
   column may be loose. Tenstorrent `bfp4 MLP / bfp8 attention` block-float is
   **not** comparable to NVIDIA FP4/FP8. Confirm dtype from the run command.

5. **MLPerf tables are dynamic.** MLCommons republishes each round; pin round +
   submitter + system + division and re-verify against the results file.

6. **Our analytic throughput serialises prefill.** `simulate()`'s
   `output_tokens_per_s` charges each request one **exclusive** prefill
   (`ttft + O·tpot/B`). Real continuous batching overlaps prefills with other
   requests' decode, so for **offline max-throughput with heavy prefill** our
   `output_tok_s` *under-reads* (e.g. the 70B MLPerf point reads 0.33× on
   `output_tok_s`). The honest roofline upper bound for offline max-sustained
   throughput is the **decode ceiling** `decode_only_tokens_per_s`
   (`decode_ceil_tok_s`), which reality cannot exceed — used for MLPerf Offline
   anchors. (The `serve` loop *does* model the overlap; that is future work for
   the analytic path.)

---

## 4. Preset ↔ real-hardware mapping

| preset | real machine | anchoring notes |
|---|---|---|
| `gb300-nvl72` | GB300 NVL72 (Grace-Blackwell **Ultra**) | **direct MLPerf data exists** (v5.1 debut, v6.0) — not proxy-only. Rack-scale disaggregated runs are coarse for our model (unknown parallelism, disagg unmodeled). |
| `dgx-h100` | DGX H100, 8× H100 SXM 80GB | best-covered: TRT-LLM (exact & proxy models), MLPerf, NIM. |
| `h100` | single H100 SXM 80GB | clean 8B (fp8) checks; 70B fp8 does **not** fit at tp=1 (needs tp≥2). |
| `dgx-spark` / `-x2` | DGX Spark (GB10), 128 GB LPDDR5x ~273 GB/s | reviewer token/s (LMSYS, NVIDIA, Ollama); huge software-epoch sensitivity. x2 RDMA link measured 189.85/200 Gbps (~95%) — a `_SPARK_CX7` link-efficiency datum. |
| `tt-quietbox` / `-2` | TT-QuietBox (gen-1, 4× Blackhole p150) / QB2 (`p300x2` = 4× Blackhole, 120-core) | QB2 = tt-metal `p300x2`/`bh_quietbox_2` (in-file mapping). gen-1 has **no** absolute tokens/s in tt-metal (bh_loudbox/bh_galaxy are TODO); only a % -of-peak review. |

---

## 5. Published derating ranges (bounds on the fit)

| quantity (phase) | achieved range | conditions | tag | source |
|---|---|---|---|---|
| **MFU — training** | PaLM 540B **46.2%** | fwd+bwd, large dense | academic | [PaLM, arXiv:2204.02311](https://arxiv.org/abs/2204.02311) |
| MFU — training | **52-57%** | Megatron 1T, A100 bf16 | academic | [Megatron-LM, arXiv:2104.04473](https://arxiv.org/abs/2104.04473) |
| MFU — training | **38-43%** bf16 | Llama-3 405B pretrain | academic | [Llama 3 Herd, arXiv:2407.21783](https://arxiv.org/abs/2407.21783) |
| MFU — **inference prefill** | **~30-50%** (LOW confidence, blog-only) | H100/A100 | secondary | practitioner blogs; no strong primary |
| **MBU — decode** | **60%** (2× H100 BS1), **55%** (4× A100 BS1); band **~55-70%** | 7B, 16-bit, single req | vendor | [Databricks, LLM Inference Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) (Oct 2023) |
| MBU — decode, high-BW | **~27%** batch-1 on H100 vs **~81%** on L4 — *achieved fraction falls as peak BW rises* | batch-1 | academic | [arXiv:2605.30571](https://arxiv.org/abs/2605.30571) (May 2026) |
| **NCCL allreduce busbw** | **~70-80%** of line rate without SHARP (H100 ~370/450 GB/s); **90%+** with NVLS/SHARP (~475-480) | 8× H100 NVLink4/NVSwitch3, large msg | vendor+indep | [nccl-tests PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) |
| RCCL (AMD) | **NOT VERIFIED** — assume ≈NCCL until measured | — | — | — |
| **kernel launch** | CUDA Graphs recover **1.259×** (≈20.6% of a batch-1 decode step) on H100 | batch-1 decode | academic | [arXiv:2605.30571](https://arxiv.org/abs/2605.30571) |
| INT4 weight-only speedup | **~2×** at memory-bound batch (vs 4× theoretical) | decode | independent | SqueezeBits vLLM-vs-TRT-LLM |

**Flagged gap:** no primary source cleanly isolates FP8/FP4-*prefill* MFU;
prefill `compute` is fitted from anchors (§8), cross-checked against the 0.30-0.58
band. MFU is **not** HFU — calibrate to MFU (excludes rematerialisation). FP8
"MFU" reads low only because the peak-FLOPs denominator doubles.

---

## 6. Candidate measured anchors

### 6.1 Single H100 × Llama-3.1-8B (FP8) — cleanest, exact model

TRT-LLM `perf-overview.md` @**0c9430e** (~v1.1.0rc, Sep 2025; last primary source
for 3.1-8B — later dropped), `trtllm-bench` max load, **total == per-GPU at TP1**.
[Source](https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/0c9430e/docs/source/performance/perf-overview.md) · VENDOR:

| ISL/OSL | 128/128 | 1000/1000 | 500/2000 | 2048/128 | 20000/2000 |
|---|---|---|---|---|---|
| tok/s | 26,401.48 | **14,991.62** | 17,571.01 | **3,275.55** | 1,340.69 |

H200 1000/1000: 17,162.49. Latency cross-check — NIM 1.8.0 TP1 FP8: BS1 1000/1000
→ 220.1 tok/s/user, TTFT 19.03 ms, **ITL 4.53 ms**. *Encoded:* `1k1k` (mixed),
`2k128` (prefill/compute probe).

### 6.2 DGX H100 (8×) × Llama-3.1-70B

TRT-LLM **current/main** (per-GPU metric), **Llama-3.3-70B** proxy (arch-identical),
FP8, [source](https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/developer-guide/perf-overview.md) · VENDOR — 70B FP8 does **not** fit at TP1, so TP2:

| TP2 FP8, ISL/OSL | 1000/1000 | 8192/1024 |
|---|---|---|
| per-GPU tok/s | 2,209 | 398 |
| ×8 (dp=4×tp=2), system | **17,672** | **3,184** |

Cross-check: 0c9430e (total metric) gives 4,181.06 @1000/1000 = 2,090/GPU —
consistent across doc versions. MLPerf v4.1 Offline Llama-**2**-70B on 8× H100 is
widely cited **~24,525 system tok/s** ([MLCommons](https://mlcommons.org/benchmarks/inference-datacenter/),
[VERIFY], proxy). *Encoded:* `tp2-1k1k` (mixed), `tp2-8k1k` (prefill probe),
`mlperf41-offline` (decode-ceiling, mixed).

### 6.3 GB300 NVL72 — **direct MLPerf** (rack-total, coarse for our model)

| workload | dtype | metric | value | tag | source |
|---|---|---|---|---|---|
| **gpt-oss-120b**, MLPerf **v6.0** (Apr 2026) Offline | MXFP4 | system/rack (72 GPU) | **1,046,150** (Server 1,096,770; Interactive 677,199) | VERIFIED | [NVIDIA](https://developer.nvidia.com/blog/nvidia-platform-delivers-lowest-token-cost-enabled-by-extreme-co-design/), [StorageReview](https://www.storagereview.com/news/nvidia-sets-mlperf-inference-v6-0-records-with-blackwell-ultra-platform) |
| **llama-3.1-8b**, MLPerf **v5.1** (Sep 2025 debut) Offline, Dynamo disagg | NVFP4 w / FP8 KV | per-GPU | **18,370** (Server 16,099; Interactive 15,284; ×72 ≈ 1.32M) | VENDOR/NVIDIA | [NVIDIA](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) |
| llama-3.1-405b (bounds big-dense) | — | per-GPU | Offline 224→271, Server 170→259 (v5.1→v6.0) | VERIFIED | same |
| Lambda GB300 tray (4 GPU), gpt-oss-120b v6.0 | MXFP4 | 4-GPU total | Offline 60.2k, Server 53.4k | VERIFIED | [Lambda](https://lambda.ai/blog/lambdas-mlperf-inference-v6.0-hardware-leap-software-maturity-research-breakthrough) |
| GB200 NVL72 llama-2-70b MLPerf v5.0 | FP4 | system | 869,203 (≈12,072/GPU) | **NVIDIA-labelled UNVERIFIED** | v5.0 blog |

*Encoded (coarse cross-checks, excluded from the fit):* `gb300-gptoss-mlperf60`
(MXFP4, ep8, disagg/parallelism unknown), `gb300-llama8b-mlperf51` (NVFP4,
disaggregated → the *serving architecture* is now expressible via
`serve --disagg` (a prefill pool + decode pool with KV streamed between them),
but the anchor stays **excluded from the fit**: the disagg architecture is a
scheduling/placement change, not a kernel-efficiency one, and the MLPerf run's
pool split / parallelism / ISL–OSL are unpublished, so it cannot pin an
`Efficiency` knob. To *reproduce a disagg comparison* on this rack (not a fit),
run e.g. `inferencesim serve --hardware gb300-nvl72 --model llama-3.1-8b
--disagg --prefill-tp 1 --prefill-replicas K --decode-tp 1 --decode-replicas J
...` and read the per-pool utilisation / TTFT-incl-transfer against the
aggregated `serve` numbers; the anchor remains a coarse throughput cross-check
under the roofline `simulate` path.

**DEP4 (attention-DP + expert-parallel) on DGX H100.** TRT-LLM benchmarks
gpt-oss-120b in a `DEPn` layout — attention data-parallel, experts sharded over
the same ranks. `DEPn` maps **exactly** to `tp=1, ep=n` in this simulator
(attention replicated, batch/KV sharded by `ep`, experts over `ep`,
dispatch/combine all-to-alls, zero-cost tp=1 attention allreduce — validated in
`tests/test_dep.py` and documented in the README Parallelism section). Encoded
as `dgxh100-gptoss-trtllm-dep4-1k1k`: MXFP4, 1000/1000, **4,685 tok/s/GPU** ×8
GPUs (dp=2 × ep4) = **37,480 system-total**, sol **1.74×** / typical **1.02×**
(a clean mixed cross-check; **the preset now models gpt-oss's 128-token
sliding-window on alternating layers, which raised both ratios from 1.64×/0.95×**).
**[VERIFY]** the per-GPU figure against a pinned
TRT-LLM commit; DEP4 additionally uses EPLB redundant-expert load balancing that
`tp=1, ep=4` does not model.

### 6.4 DGX Spark (GB10) × Llama-3.1-8B — bandwidth-bound decode

| stack | dtype | batch | decode tok/s/user | prefill tok/s | tag | source |
|---|---|---|---|---|---|---|
| TRT-LLM (NVIDIA) | NVFP4 | BS1, 2048/128 | 38.65 | 10,256.9 | VENDOR | [NVIDIA](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/) (Oct 24 2025) |
| SGLang | FP8 | BS1 | **20.5** | 7,991 | **INDEPENDENT** | [LMSYS](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/) (Oct 13 2025) |
| SGLang | FP8 | BS32 | 368 total (11.5/user) | — | INDEPENDENT | same |
| Ollama q4_K_M | q4 | BS1 | 38 (8B), 4.42 (70B) | — | INDEPENDENT | [Ollama](https://ollama.com/blog) |

The NVIDIA 38.65 (NVFP4 ~4.5 GB) vs LMSYS 20.5 (FP8 ~8 GB) gap is exactly the
weight-bytes difference a 273 GB/s device predicts. 70B q4 (~40 GB / 273 GB/s ⇒
~6.8 tok/s ceiling; measured 2.7-4.4 = 40-65% MBU). **Software epoch matters
hugely** (gpt-oss-120b 14.5→59 tok/s in ~6 months) — date-tag anchors.
*Encoded:* `spark-...-decode` (BS1 FP8, primary **memory** probe, tp=1 pure),
`spark-...-b32`.

### 6.5 TT-QuietBox 2 (4× Blackhole) × Qwen3-32B / others

tt-metal `models/model_targets.yaml` (main, ~2026-07-03; `p300x2` == `bh_quietbox_2`
== our `tt-quietbox-2`, in-file mapping), BFP8, per-user decode. **Provenance:
nominally a targets file, but inline comments cite specific measured CI runs**
("Measured on workflow run 26785408151"; "6/6 recent scheduled runs ~680-703
decode t/s") — VENDOR-CI (softer than MLPerf, harder than marketing).
[Source](https://raw.githubusercontent.com/tenstorrent/tt-metal/main/models/model_targets.yaml):

| model | batch/seq | TTFT | tok/s/user | total |
|---|---|---|---|---|
| **qwen3-32b** | 32 / 686 | 87 ms | 21.6 | **691** |
| gpt-oss-120b | 1 / 128 | 893 ms | 24.46 | — |
| llama-3.1-8b (DP4) | 4 | 19 ms | 20.88 | 83.5 |
| llama-3.1-8b (bh_p150, single) | 32 / 712 | 75 ms | 20.3 | ~650 |

`llama-3.1-70b` on QB2: **accuracy row only, no measured tok/s** in tt-metal. The
vendor "476.5 tokens per second" launch claim (Mar 2026) is **aggregate**: physics
(4×~512 GB/s ÷ ~70 GB BFP8 ⇒ ~29 tok/s/user ceiling) ⇒ ≈32 users × ~14.9 — soft
anchor only. gen-1 QuietBox: **no absolute tok/s anywhere**; [The Register review](https://www.theregister.com/on-prem/2025/11/27/blackhole-quietbox-tenstorrents-ai-workstation-reviewed/2113269)
(Nov 27 2025, BFP8, 2048/128) gives single-p150 8B decode ≈ **50% of theoretical
peak**, 70B across 4 cards ≈ **41%**, scaling +36% (1→2)/+27% (2→4) — directly
usable as efficiency-factor anchors *without* absolutes. Known issue
[tt-metal#28102](https://github.com/tenstorrent/tt-metal): Qwen3-32B hangs on
~3k-token prompts (long-context QB2 not reproducible). *Encoded:* `qb2-qwen32b-b32`
(mixed) and `qb2-qwen32b-decode` (tpot; tp=4 over the 50 GB/s Warp400 ring, so
collective-contaminated — cross-check, not the memory driver).

### 6.6 GPT-OSS-120B on H100/H200 (context; not encoded)

NVIDIA published **no Hopper** gpt-oss number. Best independent:
[SemiAnalysis InferenceMAX v1](https://inferencex.semianalysis.com/compare/gptoss-120b-h100-vs-h200)
(Oct 2025), 1K/1K @ 117 tok/s/user: H100 **2,621.5**/GPU, H200 **2,812.2**/GPU.
vLLM 0.10.1.1 ([Simplismart](https://simplismart.ai/blog/deploy-gpt-oss-120b-h100-vllm)):
1× H100 5,233.56 tok/s (TPOT 16.70 ms), 2× H100 12,690.75. These single/dual-GPU
dense-ish points are not encoded; the encoded DGX-H100 gpt-oss anchor is the
**DEP4** rack point (`dgxh100-gptoss-trtllm-dep4-1k1k`, §6.3), plus the GB300
MLPerf rack point. `qwen3-32b` best secondary: GPUStack 1× H100 vLLM BF16
ShareGPT ~2,352.82; TRT-LLM FP8 2000/100 → 5,902.

### 6.7 Blackhole NoC topology (for the per-router mesh preset)

Facts backing `presets_fine.blackhole_p150_mesh` / `tt-quietbox-mesh` (the NoC
modelled as its real router grid, not one lumped switch):

| fact | value | tag | source |
|---|---|---|---|
| NoC grid | **12 rows × 17 columns** (204 tiles) | VERIFIED | [tt-npe Blackhole impl (DeepWiki)](https://deepwiki.com/tenstorrent/tt-npe/5.3-blackhole-implementation); [Blackhole arch guide](https://anuraagw.me/blog/blackhole-architecture) |
| Tensix cores | **140** (interior positions; some cards firmware-cut to 120) | VERIFIED count | [docs.tenstorrent.com](https://docs.tenstorrent.com/aibs/blackhole/specifications.html); [Tom's Hardware](https://www.tomshardware.com/tech-industry/semiconductors/jim-kellers-tenstorrent-is-downgrading-blackhole-p150-cards-from-140-to-120-tensor-cores-via-firmware-update-will-ship-cards-with-120-tensor-cores-going-forward-company-claims-existing-users-should-expect-1-2-percent-performance-drop) |
| GDDR6 controllers | **8**, in **columns 0 and 9** | VERIFIED cols | tt-npe (DRAM at cols 0/9); arch guide (col 0/9, ~7–8 controllers) |
| NoCs | **two**: NOC0 East-then-South (**X-then-Y = column-first / row-first**), NOC1 North-then-West; both **tori** (wrap-around) | VERIFIED | tt-npe; arch guide (NOC0 reads, NOC1 writes) |
| per-link BW | **~60.9 bytes/cycle** (~82 GB/s/dir at 1.35 GHz), per NoC | VERIFIED | tt-npe (vs Wormhole 30 B/cycle) |
| GDDR6 | 32 GB, **512 GB/s** aggregate | VERIFIED | docs.tenstorrent.com |

**Modelling choices (best-effort, flagged in the preset docstring):** (a) exact
Tensix/bank *row* positions aren't public — the preset uses a documented
convention (Tensix rows 1–10 × cols {1–7,10–16} = 140; banks at cols 0/9, rows
{1,4,7,10}); (b) a plain **mesh, not the torus** (wrap links omitted → a
future refinement); (c) **one NoC plane** carrying reads+writes (real HW splits
NOC0/NOC1, so our model is *conservative* on read/write contention); (d)
per-link bandwidth **derived from the published 3.2 TB/s aggregate** as the mesh
min-bisection (12·B_link = 3.2 TB/s → **266.7 GB/s**), not the ~82 GB/s/wire
spec, so aggregation reproduces the lumped chip *exactly*. The XY routing
matches NOC0's dimension order; that is the one routing fact we lean on.

---

## 7. Caution list — do NOT encode as measurements

- **"Qwen3.5-27B 1M tok/s on B200"** figures are *misattributed* to Qwen3-32B.
- **GB200 "1.5M tok/s" gpt-oss** vendor claim (Aug 2025,
  [NVIDIA](https://developer.nvidia.com/blog/delivering-1-5-m-tps-inference-on-nvidia-gb300-nvl72/)):
  vendor-only, never independently reproduced.
- **GB200 NVL72 llama-2-70b 869,203** (MLPerf v5.0): NVIDIA-labelled *unverified*
  submission.
- Any **GB200→GB300 uplift** (vendor ~1.27×) is interpolation, not measurement.

---

## 8. The encoded anchor set + the fit

`calibration.ANCHORS` holds 12 anchors. Scored under `sol` (must be optimistic,
`>= 1`), the cross-vendor global `typical`, and `auto` (each anchor under its
**per-vendor** profile — tt-* → `typical-tt`, else `typical-nv`; §8.1):

| anchor | regime | metric | measured | sol | typical | auto | role |
|---|---|---|---|---|---|---|---|
| h100-llama8b-trtllm-1k1k | mixed | output_tok_s | 14,991.6 | 1.44 | 0.81 | 0.81 | cross-check |
| h100-llama8b-trtllm-2k128 | prefill | output_tok_s | 3,275.6 | 1.76 | 1.00 | 1.00 | **compute** |
| dgxh100-llama70b-trtllm-tp2-1k1k | mixed | output_tok_s | 17,672 | 1.34 | 0.77 | 0.77 | cross-check |
| dgxh100-llama70b-trtllm-tp2-8k1k | prefill | output_tok_s | 3,184 | 1.66 | 0.97 | 0.97 | **compute** |
| dgxh100-llama70b-mlperf41-offline | mixed | decode_ceil_tok_s | 24,525 | 1.09 | 0.73 | 0.73 | cross-check |
| dgxh100-gptoss-trtllm-dep4-1k1k | mixed | output_tok_s | 37,480 | 1.74 | 1.02 | 1.02 | cross-check (DEP4 mapping) |
| gb300-gptoss-mlperf60-offline | mixed | output_tok_s | 1,046,150 | 1.07 | 0.59 | 0.59 | coarse (excl.) |
| gb300-llama8b-mlperf51-offline | mixed | output_tok_s | 1,322,640 | 3.39 | 1.79 | 1.79 | coarse (excl.) |
| qb2-qwen32b-ttmetal-b32 | mixed | output_tok_s | 691 | 1.57 | 0.96 | **0.72** | cross-check (tt) |
| qb2-qwen32b-ttmetal-decode | decode | tpot_ms | 46.3 | 2.42 | 1.41 | **1.02** | **memory (tt)** |
| spark-llama8b-lmsys-decode | decode | tpot_ms | 48.78 | 1.76 | 1.00 | 1.00 | **memory (nv)** |
| spark-llama8b-lmsys-b32 | mixed | output_tok_s | 368 | 1.16 | 0.66 | 0.66 | cross-check |

**sol:** median **1.61×**, range [1.07×, 3.39×] — every anchor optimistic (the
invariant holds; a test enforces it). **typical (global):** median **0.97×**,
range [0.59×, 1.79×] — brackets 1. **auto (per-vendor):** median **0.89×**, range
[0.59×, 1.79×].

**Both gpt-oss anchors moved when the preset was corrected to model its 128-token
sliding window on alternating layers** (§4, §6.3): the windowed layers cut decode
KV traffic on half the model, so simulated throughput rose and the optimism ratios
climbed — `dgxh100-gptoss-trtllm-dep4-1k1k` **sol 1.64× → 1.74×, typical/auto
0.95× → 1.02×**, and `gb300-gptoss-mlperf60-offline` **sol 1.01× → 1.07×,
typical/auto 0.56× → 0.59×**. Profiles were **not** refitted: the fit is driven by
the single-node llama/spark prefill+decode probes (`2k128`, `tp2-8k1k`,
`spark-decode`), which are unchanged, and both gpt-oss rows are cross-checks /
coarse (excluded), so the knobs stay put. Under `auto` every NVIDIA row is **identical to `typical`**
(`typical-nv` == `typical`); the two Tenstorrent rows move: the decode/memory
anchor `qb2-qwen32b-ttmetal-decode` goes **1.41× → 1.02×** (the residual the
per-vendor profile exists to fix — see §8.1), while the mixed cross-check
`qb2-qwen32b-ttmetal-b32` drops **0.96× → 0.72×** (it now *under*-reads: the harder
memory derating compounds on a throughput metric, and our exclusive-prefill
`output_tok_s` already under-reads the measured *pure-decode* aggregate 691 =
21.6 tok/s/user × 32 — caveat 6). The clean tt memory probe is the decode anchor;
the mixed one stays a soft cross-check.

**The fit (simple + transparent):**
- **memory = 0.57** = 1 / 1.76, the clean tp=1 batch-1 decode probe
  `spark-...-decode` (no collective; pure weight streaming). Inside the Databricks
  0.55-0.70 MBU band. (qb2-decode is *not* used — its tp=4 Warp400-ring collective
  contaminates it.)
- **compute = 0.58** = 1 / median(1.76, 1.66), the prefill-bound probes
  `2k128` and `tp2-8k1k`. Top of the 0.30-0.58 inference-prefill MFU band.
- **collective = 0.85** — literature (NCCL allreduce without SHARP), not
  anchor-fitted (only qb2 exercises a slow ring; low sensitivity elsewhere).
- **op_overhead_s = 1.5 µs** — CUDA-Graphs recover ~20% of a batch-1 decode step
  ([arXiv:2605.30571](https://arxiv.org/abs/2605.30571)); effective per-op launch
  under batching.

**Cross-check against the derating brackets (§5):** memory 0.57 ∈ [0.55, 0.70] ✓;
compute 0.58 ∈ [0.30, 0.58] ✓ (top edge); collective 0.85 ∈ [0.70, 0.95] ✓;
overhead 1.5 µs ∈ [1, 10] ✓. The clean single-node probes (`2k128`, `tp2-8k1k`,
`spark-decode`) land at ~1.0. **Residuals that don't bracket, and why:**
- `qb2-qwen32b-decode` 1.41× under the *global* `typical` — Tenstorrent effective
  memory is lower (implied ~0.57/1.41 ≈ 0.40, matching The Register's 41-50%-of-
  peak). **This is now fixed by the per-vendor `typical-tt` profile (§8.1): the
  anchor brackets 1 at 1.02× under `auto`.** The global `typical` keeps the
  residual by design (it is not refitted).
- `gb300-gptoss` 0.59× and `gb300-llama8b` 1.79× — rack-scale MLPerf runs whose
  Dynamo *disaggregated* serving architecture our aggregated `simulate` path
  cannot express (unknown pool split / parallelism / ISL–OSL). The architecture
  itself is now expressible (`serve --disagg`, §Serving in the README), so a
  disagg comparison run can be reproduced, but these stay **excluded from the
  fit**: disaggregation is a placement/scheduling change, not kernel efficiency,
  so it cannot pin an `Efficiency` knob. Coarse cross-checks only.
- balanced `output_tok_s` cross-checks (0.66-0.96×) sit just under 1: derating
  compute *and* memory compounds on mixed-regime throughput. Acceptable spread for
  a bracketing profile.

Reproduce: `inferencesim calibrate --efficiency sol`, `--efficiency typical`, and
`--efficiency auto` (per-vendor).

### 8.1 Per-vendor fit (`typical-nv`, `typical-tt`)

One global knob set cannot serve both vendors: the residual above shows the
Tenstorrent tt-metal stack reaching a markedly lower effective memory bandwidth
than NVIDIA. So `efficiency.PROFILES` adds two vendor profiles, and
`profile_for(hardware_key, "auto")` (CLI `--efficiency auto`) routes each hardware
key to its vendor's — tt-* keys (plus the explicit fine-preset set, e.g.
`blackhole-p150-fine`) → `typical-tt`, everything else → `typical-nv`. **New
presets whose key does not start with `tt-` must be added to the mapping**
(`efficiency._TT_KEYS`).

**`typical-nv`** is an intentional **alias of the global `typical`** (compute
0.58, memory 0.57, collective 0.85, 1.5 µs): the global fit was driven by the
NVIDIA anchors, so NVIDIA hardware keeps those knobs. Kept as a separate entry so
the two can diverge as anchors grow.

**`typical-tt`** re-fits **only `memory`**; the other three knobs are **kept** from
the global fit for lack of Tenstorrent-specific evidence (we do not invent
numbers):

- **memory = 0.40** — the *only* re-fitted knob. Derived from the clean tt
  decode/memory probe `qb2-qwen32b-ttmetal-decode`: its residual under the global
  `typical` (memory 0.57) is 1.41×, so the effective memory is `0.57 / 1.41 ≈
  0.40`. Independently corroborated by [The Register's gen-1 QuietBox
  review](https://www.theregister.com/on-prem/2025/11/27/blackhole-quietbox-tenstorrents-ai-workstation-reviewed/2113269)
  (Nov 2025), which measured **41–50% of theoretical peak** on Blackhole; the
  anchor-derived 0.40 sits just under that band. Both are cited; we take the
  anchor-derived value. At memory 0.40 the decode anchor reads **1.02×** (brackets
  1; a test enforces `0.9 ≤ ratio ≤ 1.1`).
- **compute = 0.58** — **kept** from the global fit. Tenstorrent prefill evidence
  is insufficient to re-fit it: the only tt prefill datum, the `qb2-qwen32b` TTFT
  anchor (87 ms), is **memory-bound in our model** (weight streaming dominates at
  prompt 558 / batch 32), so the compute knob barely moves it — sweeping compute
  1.0→0.30 changes TTFT by <1 ms — and it cannot isolate MFU. The Register's
  prefill-side numbers are thin. So we keep 0.58 rather than invent a tt value.
  (Consequence: the tt TTFT still reads ~1.16× optimistic under `auto` — an honest
  open residual, not a fitted point.)
- **collective = 0.85** — **kept**. No Tenstorrent collective measurement exists;
  the Warp400 ring's bus-bandwidth efficiency is unmeasured.
- **op_overhead_s = 1.5 µs** — **kept**. No tt-specific launch-overhead
  measurement.

Reproduce: `inferencesim calibrate --efficiency auto`; or a single point,
`inferencesim run --hardware tt-quietbox-2 --model qwen3-32b --tp 4 --batch 32
--prompt 558 --output 128 --weight-dtype fp8 --kv-dtype fp8 --efficiency auto`
(TPOT 45.5 ms vs the measured 46.3 ms; `--efficiency typical` reads 32.8 ms, `sol`
19.1 ms).

---

## 9. Refresh / re-verification checklist

Refit when anchors change (`inferencesim calibrate --efficiency sol`, then fit per
§8). Before encoding or trusting an anchor:

- [ ] TRT-LLM: re-read at the pinned commit/tag; confirm total (0c9430e) vs
      per-GPU (main/v1.2.0); transcribe exact cells; resolve FP8/MXFP4 labels.
- [ ] MLPerf: pin round + submitter + system + division; read the results file,
      not the interactive table; label Llama-2-70B a proxy.
- [ ] GB300 rack anchors: unknown parallelism/disagg → keep coarse, excluded from
      the fit; treat any GB200→GB300 uplift as interpolation. The disaggregated
      *architecture* is now expressible (`serve --disagg`) for a comparison run,
      but that does not make the anchor a fit driver (architecture ≠ efficiency).
- [ ] Spark / Tenstorrent: record per-user vs per-GPU vs total, stack version +
      date (Spark epoch drift is large); Tenstorrent BFP8 ≠ NVIDIA FP8.
- [ ] Every anchor: set `Anchor.metric` to the normalised sim quantity; bake
      per-GPU/per-rack conversions into `measured` with the raw value in `notes`.
- [x] **Per-vendor `Efficiency` landed** (§8.1): `typical-nv` (== global `typical`)
      and `typical-tt` (memory re-fitted to **0.40** from the qb2 decode residual /
      The Register 41–50%-of-peak; other knobs kept for lack of tt evidence), with
      `profile_for(..., "auto")` / `--efficiency auto` routing tt-* keys to
      `typical-tt`. Remaining: refit tt `compute`/`collective`/`op_overhead` once a
      compute-bound tt prefill anchor and a tt collective measurement exist.
- [x] Explicit **attention-DP + expert-parallel (TRT-LLM DEPn)** deployments
      landed: dense attention-DP is `Deployment.adp` (DP attention + TP FFN), and
      MoE `DEPn ≡ tp=1, ep=n` is documented/validated (README Parallelism,
      `tests/test_dep.py`); the `dgxh100-gptoss-trtllm-dep4-1k1k` anchor encodes a
      DEP4 point (sol 1.64×). Remaining honest deltas: **[VERIFY]** the DEP4
      per-GPU figure against a pinned TRT-LLM commit, and model **EPLB
      redundant-expert load balancing** + **mixed ADP+TP MoE attention** (TRT-LLM
      does both; `tp=1, ep=n` does neither).

_All retrieval dates: 2026-07-05. **[VERIFY]** cells and any GB200→GB300 uplift
are unconfirmed/interpolated and flagged in-anchor `notes`._
