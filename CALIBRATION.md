# Calibration: bracketing roofline optimism with measured anchors

This is the human-readable half of the calibration mechanism in
`inferencesim/efficiency.py` and `inferencesim/calibration.py`. It records the
*measured* benchmark numbers the simulator should be scored against, the
caveats that make such comparisons honest, and the published derating ranges
that justify — and will eventually replace — the provisional `typical` profile.

**Status (2026-07-05):** the mechanism has landed; the fitted numbers have not.
`calibration.ANCHORS` is intentionally empty and `PROFILES["typical"]` ships
provisional placeholders (compute 0.80, memory 0.80, collective 0.75, op
overhead 5 us). This document is the working set from which the anchors and the
fitted profile will be drawn. Every measured figure carries a source, a
retrieval date, and a provenance label:

- **VERIFIED** — an MLCommons-reviewed MLPerf submission.
- **VENDOR** — a vendor-published claim (NVIDIA/Tenstorrent docs, blogs).
- **INDEPENDENT** — a third-party measurement (LMSYS, SemiAnalysis, etc.).
- **[VERIFY]** — not yet confirmed against a primary source; must not enter
  `ANCHORS` until checked.

All retrieval dates below are **2026-07-05** unless a publication date is given.

---

## 1. What the knobs mean

The roofline engine computes *upper bounds*: each op runs at the peak rate of
its bottleneck resource (`t = max(flops/peak, bytes/bw)`), collectives use the
bandwidth-optimal ring / all-to-all closed forms, and there is no kernel-launch
overhead. Real systems reach a fraction of each. An `Efficiency` derates them:

| knob | scales | roofline term it multiplies | well-supported range (§4) |
|---|---|---|---|
| `compute` | peak FLOP/s (MFU ceiling) | `flops / (peak * compute)` | prefill ~0.30-0.45; training up to ~0.52 |
| `memory` | peak DRAM / on-chip / path bandwidth (MBU ceiling) | `bytes / (bw * memory)` | decode ~0.55-0.70 (0.80 is an optimistic ceiling) |
| `collective` | link bandwidth for collectives (bus-bw efficiency) | occupancy term only, **never** the latency term | 0.80-0.95 large-message; far lower for small decode all-reduces |
| `op_overhead_s` | fixed seconds per launched op instance | `+ op_overhead_s` per op | ~1-10 us/kernel |

The identity `Efficiency()` (all 1.0, zero overhead) leaves every number
bit-identical — it is the `sol` ("speed of light") profile and the default
everywhere. `calibrate` scores anchors: `sim/measured >= 1` means the simulator
is optimistic (as a bound should be); a *fitted* profile brings the ratios to
**bracket 1** (measured reality lands inside the simulator's range).

Note the `collective` knob scales only the **bandwidth occupancy** term. The
engine already models each collective's propagation latency separately (on the
dependency chain, never on the link), so the small-message latency penalty that
dominates TP decode all-reduces is captured structurally — `collective` should
be fitted against *large-message* bus-bw efficiency, not the tiny-message floor.

---

## 2. Global caveats — read before trusting any number

Most apparent disagreements between a simulator and a benchmark are unit
mismatches, not model error. Normalise all of these before comparing.

1. **Per-user vs per-GPU vs system-total throughput.** Three different numbers
   for the same run: *per-user* = `1/TPOT` for one request; *per-GPU* = system
   total / GPUs in the replica; *system-total* = summed across the replica.
   `inferencesim run` reports **whole-system** `output_tokens_per_s` and a
   `decode_only_tokens_per_s` ceiling, plus per-request `1/TPOT`. Record which a
   source uses and convert to whole-system before scoring. Landmines below:
   - **TensorRT-LLM's own metric changed between versions.** `perf-overview.md`
     at **v0.13.0** reports *total across the TP group*; at **v0.21.0+** it
     reports *per-GPU* ("tps/gpu"). A number copied without its tag is
     ambiguous. (Both are reconciled per-row in §5.)
   - **MLPerf NVL72 rack results are totals** (÷72 for per-GPU).
   - **Tenstorrent PERF.md is per-user at batch 1** (latency-optimal).
   - **SemiAnalysis InferenceMAX is per-GPU at a fixed per-user SLA** — the
     most simulator-friendly framing (a point on the throughput/interactivity
     Pareto curve, not a single raw run).

2. **Prefill vs decode, and which metric.** TTFT is prefill (compute-bound);
   TPOT/ITL is decode (memory-bound). A bare "tokens/s" headline folds both
   under continuous batching at some concurrency. Prefer sources reporting
   ISL/OSL, concurrency, and separate TTFT/TPOT so an anchor pins one regime;
   the `Anchor.metric` field (`output_tok_s | tpot_ms | ttft_ms | req_per_s`)
   forces the choice. Peak "max-throughput/offline" numbers have **no latency
   SLA** — they are the regime the roofline+efficiency model targets; use
   latency-bounded serving numbers (NIM, InferenceMAX SLA points) separately.

3. **Model-name drift is systemic.** Our `llama-3.1-70b` is benchmarked under
   two substitute names: **MLPerf uses Llama-2-70B**; NVIDIA's *current* tables
   use **Llama-3.3-70B**. All three share the 80-layer / d_model 8192 / 64
   query / 8 KV head / d_ff 28672 SwiGLU architecture (confirmed against our
   preset); Llama-3.3-70B is arch-identical to 3.1 (only the instruction tune
   differs), and Llama-2-70B differs mainly in vocab (32k vs 128k) and context.
   Prefer 3.3-70B rows (no proxy penalty); label Llama-2-70B a proxy.

4. **dtype labels are not always literal.** GPT-OSS is natively MXFP4; a Hopper
   "FP8" column may be loose. Blackwell FP4 (NVFP4/MXFP4) vs Hopper FP8 changes
   both the compute peak and the weight bytes streamed, so a mislabelled dtype
   silently breaks a decode anchor. Tenstorrent's `bfp4 MLP / bfp8 attention`
   block-float formats are **not** directly comparable to NVIDIA FP4/FP8.
   Confirm dtype from the run command, not the column header.

5. **MLPerf result tables are dynamic.** MLCommons republishes each round
   (v5.0, v5.1, ...); entries are added/withdrawn and the interactive tables
   re-render. Any MLPerf figure here is a **snapshot** — pin round, submitter,
   system, and division, and re-verify against the round's results file before
   encoding. Do not cite the live interactive table without the round.

6. **The simulator reports steady-state averages** (one exclusive prefill +
   shared decode steps); no scheduler quirks, chunked prefill, or speculative
   decoding. Anchor to *saturated / max-throughput* points. For queueing/TTFT
   percentiles use `inferencesim serve`, not these anchors.

---

## 3. Preset ↔ real-hardware mapping

| preset | real machine | anchoring notes |
|---|---|---|
| `gb300-nvl72` | GB300 NVL72 (Grace-Blackwell **Ultra**) | scarce clean data; GB200/B200 (Blackwell) are the closest **proxy** — GB300 runs ~+27% (vendor), so proxy anchors under-read it. |
| `dgx-h100` | DGX H100, 8x H100 SXM 80GB | best-covered: TRT-LLM v0.13.0 (exact model), NIM, MLPerf. |
| `h100` | single H100 SXM 80GB | good for 8B (fp8) and gpt-oss-120b (MXFP4) single-GPU checks; 70B fp8 does **not** fit at tp=1 (see §5.3). |
| `dgx-spark` / `-x2` | DGX Spark (GB10), 128 GB LPDDR5x ~273 GB/s | reviewer token/s only (LMSYS, NVIDIA). |
| `tt-quietbox` / `-2` | Tenstorrent QuietBox (Wormhole n300 "T3K" / Blackhole) | vendor `tt-metal` PERF.md; confirm board and chip count vs the preset before anchoring. |

---

## 4. Published derating ranges (what `typical` should be fitted toward)

| quantity (phase) | achieved range | conditions | label | source |
|---|---|---|---|---|
| **MFU — training** | PaLM 540B **46.2%** (HFU 57.8%) | large dense, fwd+bwd | academic | [PaLM, arXiv:2204.02311](https://arxiv.org/abs/2204.02311) |
| MFU — training | **52%** peak | 1T GPT, 3072x A100, bf16 | academic | [Megatron-LM, arXiv:2104.04473](https://arxiv.org/abs/2104.04473) |
| MFU — training | **38-43%** bf16 | Llama-3 405B pretrain (8-16K H100) | academic | [Llama 3 Herd, arXiv:2407.21783 §4.1](https://arxiv.org/abs/2407.21783) |
| MFU — training sweep | **~40-55%** bf16; **FP8 ~14-30%** (2x peak denominator) | MPT 125M-70B | vendor | [MosaicML llm-foundry](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/benchmarking/README.md) |
| MFU — inference **prefill** | **~30-45%** (LOW confidence, blog-only) | H100/A100, large prompt | independent | [inferenceengineering.tech](https://inferenceengineering.tech/learn/gpu-inference/) |
| **MBU — decode** | **60%** (2x H100, BS1); **55%** (4x A100, BS1); band **~55-70%**; **80% is an optimistic ceiling with no primary source** | 7B, 16-bit, single request | vendor | [Databricks, LLM Inference Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) (Oct 2023) |
| **NCCL all-reduce busbw** (in-node NVLink) | **~80-95%+** at large messages (~370-480 GB/s vs ~450 peak; NVLS/SHARP tops it); ~250 GB/s untuned | 8x H100, NVLink4/NVSwitch3 | vendor + independent | [nccl-tests PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md), [issue #212](https://github.com/NVIDIA/nccl-tests/issues/212) |
| NCCL health threshold | **>85%** healthy; **<70%** misconfig | tuned multi-node IB | independent | [Spheron NCCL tuning](https://www.spheron.network/blog/nccl-tuning-multi-gpu-llm-training-2026/) |
| **RCCL (AMD MI300X)** | **NOT VERIFIED** — no published busbw/MBU found; assume ≈NCCL until measured | — | — | (qualitative only: AMD ROCm blog) |

**What this implies for `typical` (0.80 / 0.80 / 0.75 / 5 us).**

- `memory = 0.80` sits **above** the well-supported decode MBU band (55-70%);
  the fit will likely **lower** it (~0.60-0.70). This is the most important
  correction the anchors will make — decode TPOT is where the sim is most
  optimistic.
- `compute = 0.80` is far above measured **prefill** MFU (~0.30-0.45). Because
  decode is memory-bound, `compute` mostly moves **TTFT**; a single scalar
  cannot express "prefill MFU 0.4 but decode isn't compute-bound," so expect
  either a low fitted `compute` (~0.40-0.50, accepting it only bites prefill)
  or a future **per-phase efficiency** split. Flag for the fit.
- `collective = 0.75` is defensible for large-message TP allreduce; combined
  with the separately-modelled latency term it should bracket decode collectives
  reasonably. Low sensitivity except at high TP.
- MFU is **not HFU** — calibrate to MFU (excludes rematerialisation). FP8 "MFU"
  reads low only because peak FLOPs doubles; don't confuse that with inefficiency.
- Collective efficiency is **message-size dependent**: keep it a large-message
  number and let the latency model carry the small-message penalty (§1).

---

## 5. Candidate measured anchors

Condensed from the full research pass; only rows without **[VERIFY]** are
eligible for `ANCHORS`. Values are transcribed from the cited primary sources.

### 5.1 DGX H100 (8x H100 SXM) x Llama-3.1-70B — best-sourced 70B set

TensorRT-LLM `perf-overview.md` **v0.13.0** (last tag with the exact 3.1-70B
model), `trtllm-bench throughput`, offline max load, in-flight batching.
**Metric: total output tok/s across the 8-GPU TP group.**
[Source](https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/v0.13.0/docs/source/performance/perf-overview.md) (Oct 2024, VENDOR):

| parallelism / dtype | ISL/OSL | total tok/s (8 GPU) | ≈ per-GPU |
|---|---|---|---|
| TP8, FP8 (FP8 KV) | 128/128 | **15,711.60** | 1,964 |
| TP8, FP8 | 1000/1000 | 11,121.10 | 1,390 |
| TP8, FP8 | 500/2000 | 10,790.85 | 1,349 |
| TP8, FP8 | 2048/2048 | 8,527.52 | 1,066 |
| TP8, FP8 | 2048/128 (prefill-heavy) | 1,964.49 | 246 |
| TP8, FP16 | 128/128 | 10,643.75 | 1,330 |

Latency-bounded cross-check — NIM 1.3.0, TP4 FP8 (NIM publishes 70B at TP4 only),
[NIM perf docs](https://docs.nvidia.com/nim/benchmarking/llm/latest/performance.html)
(VENDOR): concurrency 1 → 69.98 tok/s/user, TTFT 59.9 ms, **ITL 14.24 ms**;
concurrency 250, 1000/1000 → 5,497 tok/s total (4 GPU), TTFT 2,633 ms, ITL 41.9 ms.

MLPerf v4.1 Llama-2-70B on 8x H100 offline is widely cited at **~24,525 tok/s**
but was **[VERIFY]** (not confirmed from the MLCommons v4.1 results CSV,
submission 4.1-0043) and is a Llama-2 proxy.

*Trust:* the v0.13.0 TP8 FP8 rows are the primary 70B anchors — exact model,
verified line-by-line, metric defined in-file. They are peak offline (no SLA).

### 5.2 Single H100 x Llama-3.1-8B (fp8) — clean decode/prefill sweep

TensorRT-LLM **v0.21.0** (`nvidia/Llama-3.1-8B-Instruct-FP8`), **per-GPU =
total at TP1**,
[source](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.21.0/docs/source/performance/perf-overview.md)
(Aug 2025, VENDOR):

| ISL/OSL | 128/128 | 1000/1000 | 128/2048 | 2048/2048 | 2048/128 | 20000/2000 |
|---|---|---|---|---|---|---|
| tok/s | 26,401.48 | 14,991.62 | 21,413.21 | 9,462.43 | 3,275.55 | 1,340.69 |

Corroborated by v0.13.0 (27,147 tok/s @ 128/128). Single-stream latency — NIM
1.8.0 TP1 FP8 (VENDOR): concurrency 1, 1000/1000 → 220.1 tok/s/user, TTFT 19.03
ms, **ITL 4.53 ms**; concurrency 250, 200/200 → 12,964.7 tok/s total.

*Trust:* strong. Two TRT-LLM versions agree on the ceiling; NIM gives the clean
BS-1 decode point (ITL ~4.5 ms) — an excellent `memory`-fitting anchor.

### 5.3 Single H100 x Llama-3.1-70B (fp8) — **infeasible at tp=1**

FP8 weights ≈ 70 GB of 80 GB; after context + activations only a few GB remain
for KV, so NVIDIA benchmarks 70B FP8 on H100 at **tp≥2 only**. Record this pair
as **tp=2 minimum** in the sim. Proxy — TRT-LLM v0.21.0 **Llama-3.3-70B-FP8**,
**tp=2**: 128/128 → 6,092.28 tok/s/GPU (≈12,184 total); 1000/1000 → 4,181.06
tok/s/GPU; 2048/128 → 723.40 tok/s/GPU. (VENDOR)

### 5.4 GB300 NVL72 x Llama-3.1-70B — **no clean pair; proxy required**

| system / stack | dtype | workload | metric | value | label | source |
|---|---|---|---|---|---|---|
| GB200 NVL72, MLPerf v5.1 (Llama-**2**-70B) | FP4 | OpenOrca offline | total/rack | **865,000 tok/s** (≈12,014/GPU) | **VERIFIED** | [Azure ND GB300 v6 blog](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/breaking-the-million-token-barrier-the-technical-achievement-of-azure-nd-gb300-v/4466080) |
| GB300 NVL72 (Llama-**2**-70B) | NVFP4 | OpenOrca offline | total/rack | 1,100,948 (≈15,290/GPU) | VENDOR (unverified) | Azure/[Signal65](https://signal65.com/research/ai/azure-gb300-performance/) |
| GB200, TRT-LLM v0.21 (Llama-**3.3**-70B) | FP4 | 128/128 | per-GPU (=total, TP1) | 11,100.97 | VENDOR | [TRT-LLM perf-overview](https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html) |
| B200, TRT-LLM v0.21 (Llama-3.3-70B) | FP4 | 128/128 | per-GPU | 10,613.84 | VENDOR | same |

*Trust:* the only MLCommons-**verified** 70B-rack datum is GB200 **865k
tok/s/rack** (Llama-2 proxy, variable-length OpenOrca — not fixed ISL/OSL). For
the model's compute/memory profile, the TRT-LLM Llama-3.3-70B FP4 rows are the
best proxy but exist only for GB200/B200. A defensible `gb300-nvl72` point is
**GB200-verified × ~1.27** (vendor GB300/GB200 uplift), *labelled interpolation*.

### 5.5 DGX Spark (GB10) x Llama-3.1-8B — decode is bandwidth-bound

| stack | dtype | batch | decode tok/s/user | prefill tok/s | label | source |
|---|---|---|---|---|---|---|
| TRT-LLM (NVIDIA playbook) | NVFP4 | BS1, 2048/128 | **38.65** | 10,256.9 | VENDOR | [NVIDIA blog](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/) (Oct 24, 2025) |
| SGLang | FP8 | BS1 | **20.5** | 7,991 | **INDEPENDENT** | [LMSYS DGX Spark review](https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/) (Oct 13, 2025) |
| SGLang | FP8 | BS32 | 368 total (11.5/user) | 7,949 | INDEPENDENT | same |

*Trust:* LMSYS (independent, documented, shows BS scaling). The NVIDIA 38.65 vs
LMSYS 20.5 gap is the NVFP4 (~4.5 GB) vs FP8 (~8 GB) weight difference — exactly
what a ~273 GB/s bandwidth-bound device predicts, and a good `memory` check.

### 5.6 Tenstorrent QuietBox x Llama-3.1-70B / 8B

tt-metal `models/tt_transformers/PERF.md`
([raw](https://raw.githubusercontent.com/tenstorrent/tt-metal/main/models/tt_transformers/PERF.md),
VENDOR), `bfp4 MLP / bfp8 attention`, BS1, gen 200, **per-user decode**.
**T3K = 8x Wormhole n300 = QuietBox/LoudBox**; P150 = single Blackhole:

| model | system | tok/s/user | TTFT |
|---|---|---|---|
| Llama-3.1-8B | T3K (8x Wormhole) | 64.3 | 53 ms |
| Llama-3.1-70B | T3K (8x Wormhole) | 16.6 | 164 ms |
| Llama-3.1-8B | P150 (1x Blackhole) | 33.6 | — |

QuietBox-2 (4x Blackhole), Llama-3.1-70B: 476.5 tok/s **total** (batch/OSL
unspecified) — VENDOR claim, [launch coverage](https://www.design-reuse.com/news/202530525-tenstorrent-unveils-tt-quietbox-2-the-first-risc-v-ai-workstation-with-a-fully-open-source-stack-to-deliver-teraflop-class-inference/) (May 2026).

*Trust:* PERF.md is the strongest source (in-repo, per-SKU, explicit method) but
is BS-1 per-user (latency-optimal, not throughput). **Confirm the preset's board
and chip count** (Wormhole T3K vs Blackhole) before anchoring, and note bfp4/bfp8
≠ NVIDIA FP4/FP8.

### 5.7 H100 / H200 x gpt-oss-120b (MoE, MXFP4)

NVIDIA published **no Hopper** gpt-oss-120b tok/s; the best are independent.

| stack | system | workload | metric | value | label | source |
|---|---|---|---|---|---|---|
| InferenceMAX v1 (best of TRT-LLM/vLLM/SGLang) | 1x H100, MXFP4 | 1K/1K | per-GPU @ 117 tok/s/user | **2,621.5** | **INDEPENDENT** | [SemiAnalysis InferenceMAX](https://inferencex.semianalysis.com/compare/gptoss-120b-h100-vs-h200) (Oct 2025) |
| InferenceMAX v1 | 1x H200, MXFP4 | 1K/1K | per-GPU @ 117 tok/s/user | 2,812.2 | INDEPENDENT | same |
| vLLM 0.10.1.1 | 1x H100, MXFP4 | 1024/512, conc 32 | total (1 GPU) | 5,233.56 (TPOT 16.70 ms) | INDEPENDENT | [Simplismart](https://simplismart.ai/blog/deploy-gpt-oss-120b-h100-vllm) (Oct 2025) |
| vLLM 0.10.1.1 | 2x H100 TP2, MXFP4 | 1024/512, conc 32 | total (2 GPU) | 12,690.75 (TPOT 25.75 ms) | INDEPENDENT | same |

*Trust:* InferenceMAX (independent, open, fixed 1K/1K, best-of-three-engines,
per-GPU-vs-per-user Pareto) — decimals are "interpolated from real benchmark
data," so treat as points on a measured curve. Simplismart/GPUStack totals are
real but a *different* quantity (offline aggregate, heavy batching) — use as
offline upper bounds. NVIDIA's own numbers are Blackwell-only.

---

## 6. Recommended anchor set (fill `calibration.ANCHORS` from these)

Concrete encodings that map to our presets. **Units must be normalised to the
sim's whole-system metric at encode time**; the notes say how. Batch is a free
knob for `max-throughput` rows — pick a batch that saturates the replica (or
compare against `decode_only_tokens_per_s`). Still gated by the §7 checklist.

| slug | preset / model | deployment | scenario | metric | measured | source / label |
|---|---|---|---|---|---|---|
| `dgx-h100-llama70b-tp8-fp8-2k2k` | `dgx-h100` / `llama-3.1-70b` | tp8, fp8/fp8kv | 2048/2048, saturating batch | `output_tok_s` (system total, 8 GPU) | **8,527.52** | TRT-LLM v0.13.0 · VENDOR |
| `dgx-h100-llama70b-tp8-fp8-1k1k` | `dgx-h100` / `llama-3.1-70b` | tp8, fp8/fp8kv | 1000/1000, saturating batch | `output_tok_s` | 11,121.10 | TRT-LLM v0.13.0 · VENDOR |
| `h100-llama8b-tp1-fp8-1k1k` | `h100` / `llama-3.1-8b` | tp1, fp8 | 1000/1000, saturating batch | `output_tok_s` (= per-GPU at tp1) | 14,991.62 | TRT-LLM v0.21.0 · VENDOR |
| `h100-llama8b-tp1-fp8-bs1-decode` | `h100` / `llama-3.1-8b` | tp1, fp8 | 1000/1000, **batch 1** | `tpot_ms` | 4.53 | NIM 1.8.0 ITL · VENDOR |
| `h100-llama70b-tp2-fp8-1k1k` (proxy) | `h100`→needs 2 chips → use `dgx-h100` tp2 / `llama-3.1-70b` | tp2, fp8 | 1000/1000 | `output_tok_s` (2 GPU total) | ≈8,362 (2×4,181) | TRT-LLM v0.21.0, **Llama-3.3** proxy · VENDOR |
| `dgx-spark-llama8b-bs1-decode` | `dgx-spark` / `llama-3.1-8b` | tp1, fp8 | 2048/128, **batch 1** | `tpot_ms` | 48.8 (= 1/20.5 s) | LMSYS SGLang · INDEPENDENT |
| `tt-quietbox-llama70b-bs1-decode` | `tt-quietbox` / `llama-3.1-70b` | tp8 (confirm board) | BS1, gen 200 | `tpot_ms` | 60.2 (= 1/16.6 s) | tt-metal PERF.md · VENDOR |
| `h100-gptoss120b-tp1-mxfp4-offline` | `h100` / `gpt-oss-120b` | tp1, fp4 weights | 1024/512, conc 32 | `output_tok_s` (1 GPU total) | 5,233.56 | Simplismart vLLM · INDEPENDENT |
| `gb300-llama70b-nvl72-fp4-offline` (proxy) | `gb300-nvl72` / `llama-3.1-70b` | tp/dp per rack | OpenOrca offline | `output_tok_s` (rack total) | 865,000 × ~1.27 uplift | GB200 MLPerf v5.1 · **VERIFIED** + interpolation |

Regime coverage: `2k2k`/`1k1k` offline rows pin `memory` (decode-heavy) and
`compute` (the `2048/128` prefill-heavy row is the sharpest `compute` probe);
the `bs1-decode` rows pin `memory` cleanly at batch 1 with no batching
confound; the high-TP 70B rows expose `collective` sensitivity.

---

## 7. Fitting procedure

1. Encode confirmed rows as `calibration.Anchor` (slug = row name; record
   `source` URL and retrieval date in `notes`; normalise units to the sim
   metric per §2/§6).
2. `inferencesim calibrate --efficiency sol` — every ratio should be `>= 1`. A
   `sol` ratio `< 1` means the **preset spec sheet** or the anchor's units are
   wrong (no efficiency in (0,1] can raise a sub-1 ratio) — fix that first.
3. Fit so ratios bracket 1: decode / long-OSL / BS1 anchors pin `memory`;
   prefill / long-ISL anchors pin `compute`; high-TP anchors pin `collective`;
   a flat residual across regimes pins `op_overhead_s`. A 2-3 anchor
   least-squares on `log(sim/measured)` suffices — this is a bracketing
   profile, not a per-op curve. Expect `memory ≈ 0.60-0.70` and `compute`
   pulled down toward prefill MFU (§4); consider a per-phase split if one
   scalar can't bracket both prefill and decode anchors.
4. Replace `PROFILES["typical"]` with the fitted values and delete the
   provisional-placeholder TODO in `efficiency.py`.

---

## 8. Re-verification checklist (run before encoding `ANCHORS`)

- [ ] TRT-LLM: re-read `perf-overview.md` at the pinned tag; confirm whether the
      metric is *total* (v0.13.0) or *per-GPU* (v0.21+); transcribe exact cells;
      resolve FP8/MXFP4 dtype labels (caveats 1, 4).
- [ ] MLPerf: pin round + submitter + system + division; read the results file,
      not the interactive table; label Llama-2-70B a proxy (caveats 3, 5).
- [ ] GB300: treat any GB200×uplift as *interpolation*, not measurement.
- [ ] Spark / Tenstorrent: record per-user vs per-GPU vs total, stack version,
      board (caveat 1); Tenstorrent bfp4/bfp8 ≠ NVIDIA FP4/FP8 (caveat 4).
- [ ] gpt-oss-120b: prefer InferenceMAX per-GPU@SLA points; note NVIDIA has no
      Hopper number.
- [ ] Every anchor: set `Anchor.metric` to the normalised sim quantity; record
      `source` (URL) + retrieval date.

_All retrieval dates: 2026-07-05. Numbers marked **[VERIFY]**, and any
GB200→GB300 uplift, are unconfirmed/interpolated and must not enter
`calibration.ANCHORS` as measurements without a primary-source check._
