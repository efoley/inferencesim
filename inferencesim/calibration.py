"""Calibration anchors and the harness that scores the simulator against them.

An `Anchor` pins a *measured* number -- a throughput or latency a real stack
achieved on a real machine, from a citable source (MLPerf, a vendor blog, a
reproducible benchmark) -- to the (hardware, model, deployment, scenario) the
simulator can reproduce.  `run_anchor` simulates that point under a given
`Efficiency` and reports an *optimism ratio*; `calibrate_report` tabulates a
whole anchor set.

Why this exists: roofline numbers are *upper bounds* (perfect tiling, 100%
bandwidth, bandwidth-optimal collectives, no launch overhead), so at the `sol`
efficiency the simulator should be optimistic against every anchor.  We define
the **optimism ratio** so that `>= 1` always means "simulator optimistic",
regardless of metric direction:

    throughput / rate metric : simulated / measured   (higher sim = optimistic)
    latency metric           : measured / simulated   (lower sim = optimistic)

A well-fitted `Efficiency` derates the peaks until those ratios come down to
**bracket 1** -- measured reality lands inside the simulator's range.

The measured research -- candidate rows, sources, retrieval dates, unit
conversions, and the fit that produced `PROFILES["typical"]` -- lives in the
top-level CALIBRATION.md.  Every anchor below carries a `source` URL and its
per-row caveats in `notes`; read CALIBRATION.md before trusting or editing one.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from .efficiency import Efficiency
from .engine import RooflineEngine
from .hardware import DType, System
from .presets import HARDWARE, MODELS
from .presets_fine import GRAPH_PRESETS
from .simulate import simulate
from .workload import Deployment, Scenario

# metric keys an Anchor may pin.  Latency metrics invert the optimism ratio.
#   output_tok_s     : steady-state continuous-batching output throughput
#                      (one exclusive prefill + shared decode) -- our analytic
#                      throughput; appropriate for fixed-shape max-load points.
#   decode_ceil_tok_s: the decode-only throughput ceiling (ignores prefill) --
#                      the true roofline upper bound for *offline* max-sustained
#                      throughput, where prefills overlap decode in reality but
#                      our exclusive-prefill output_tok_s serialises them.
METRICS = ("output_tok_s", "decode_ceil_tok_s", "tpot_ms", "ttft_ms", "req_per_s")
_LATENCY_METRICS = frozenset({"tpot_ms", "ttft_ms"})
# which knob an anchor primarily probes (drives the transparent fit).
REGIMES = ("decode", "prefill", "mixed")


@dataclass(frozen=True)
class Anchor:
    """One measured benchmark point the simulator should reproduce.

    hardware_key : a HARDWARE or GRAPH_PRESETS key (see `inferencesim list`).
    model_key    : a MODELS key.
    tp/pp/ep, weight_dtype/kv_dtype/act_dtype : the deployment.
    batch/prompt/output : the scenario (batch = concurrent sequences/replica,
                   prompt = input tokens, output = generated tokens).
    metric       : one of METRICS -- which reported number `measured` is.
    measured     : the measured value, ALREADY normalised to the simulator's
                   quantity (whole-system tok/s for throughput; ms for latency).
                   Per-GPU / per-rack conversions are baked in here and spelled
                   out in `notes` (see CALIBRATION.md caveat 1).
    regime       : the knob this anchor primarily constrains -- "decode"
                   (memory), "prefill" (compute), or "mixed" (cross-check).
    source       : a URL / citation for `measured`.
    notes        : stack version, retrieval date, unit conversion, caveats.
    """

    name: str
    hardware_key: str
    model_key: str
    metric: str
    measured: float
    source: str
    tp: int = 1
    pp: int = 1
    ep: int = 1
    weight_dtype: DType = DType.FP8
    kv_dtype: DType = DType.BF16
    act_dtype: DType = DType.BF16
    batch: int = 1
    prompt: int = 2048
    output: int = 512
    regime: str = "mixed"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(
                f"Anchor.metric must be one of {METRICS}, got {self.metric!r}"
            )
        if self.regime not in REGIMES:
            raise ValueError(
                f"Anchor.regime must be one of {REGIMES}, got {self.regime!r}"
            )

    @property
    def deployment(self) -> Deployment:
        return Deployment(
            tp=self.tp, pp=self.pp, ep=self.ep,
            weight_dtype=self.weight_dtype, kv_dtype=self.kv_dtype,
            act_dtype=self.act_dtype,
        )

    @property
    def scenario(self) -> Scenario:
        return Scenario(batch=self.batch, prompt_len=self.prompt, output_len=self.output)


# =============================================================================
# The anchor set.
#
# Encoded from CALIBRATION.md's "Recommended anchor set".  Measured values are
# normalised to the simulator's whole-system metric (per-GPU x GPUs, per-rack
# as-is); the raw number and the conversion are in each `notes`.  Provenance
# tags: VERIFIED (MLCommons-reviewed), VENDOR (vendor-published), INDEPENDENT
# (third-party), and [VERIFY] where a specific cell still needs a primary-source
# check.  Read CALIBRATION.md before editing.
# =============================================================================

ANCHORS: list[Anchor] = [
    # ---- single H100 x Llama-3.1-8B (FP8): cleanest, exact model ------------
    Anchor(
        name="h100-llama8b-trtllm-1k1k",
        hardware_key="h100", model_key="llama-3.1-8b",
        tp=1, weight_dtype=DType.FP8, kv_dtype=DType.FP8,
        batch=256, prompt=1000, output=1000,
        metric="output_tok_s", measured=14991.62, regime="mixed",
        source="https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/0c9430e/docs/source/performance/perf-overview.md",
        notes="TRT-LLM perf-overview @0c9430e (~v1.1.0rc, Sep 2025), trtllm-bench "
              "max load; metric is TOTAL across the TP group == per-GPU at TP1 == "
              "system-total (1 GPU). VENDOR. llama-3.1-8B was later dropped from the "
              "doc, so 0c9430e is its last primary NVIDIA source. batch=256 is a "
              "max-load modelling choice. Retrieved 2026-07-05.",
    ),
    Anchor(
        name="h100-llama8b-trtllm-2k128",
        hardware_key="h100", model_key="llama-3.1-8b",
        tp=1, weight_dtype=DType.FP8, kv_dtype=DType.FP8,
        batch=256, prompt=2048, output=128,
        metric="output_tok_s", measured=3275.55, regime="prefill",
        source="https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/0c9430e/docs/source/performance/perf-overview.md",
        notes="Same source/date; ISL/OSL 2048/128 is prefill-bound (compute probe). "
              "VENDOR.",
    ),
    # ---- DGX H100 (8x H100) x Llama-3.1-70B ---------------------------------
    Anchor(
        name="dgxh100-llama70b-trtllm-tp2-1k1k",
        hardware_key="dgx-h100", model_key="llama-3.1-70b",
        tp=2, weight_dtype=DType.FP8, kv_dtype=DType.FP8,
        batch=128, prompt=1000, output=1000,
        metric="output_tok_s", measured=17672.0, regime="mixed",
        source="https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/developer-guide/perf-overview.md",
        notes="TRT-LLM current/main (per-GPU metric), Llama-3.3-70B PROXY "
              "(arch-identical to 3.1-70B), TP2 FP8, 1000/1000: 2209 tok/s/GPU -> "
              "x8 GPUs (dp=4 x tp=2) = 17672 system-total. VENDOR. 70B FP8 does not "
              "fit at TP1, hence TP2. Retrieved 2026-07-05.",
    ),
    Anchor(
        name="dgxh100-llama70b-trtllm-tp2-8k1k",
        hardware_key="dgx-h100", model_key="llama-3.1-70b",
        tp=2, weight_dtype=DType.FP8, kv_dtype=DType.FP8,
        batch=64, prompt=8192, output=1024,
        metric="output_tok_s", measured=3184.0, regime="prefill",
        source="https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/developer-guide/perf-overview.md",
        notes="Same source; ISL/OSL 8192/1024 per-GPU 398 -> x8 = 3184 system-total. "
              "Long-ISL prefill-bound (compute probe). Llama-3.3 proxy, VENDOR.",
    ),
    Anchor(
        name="dgxh100-llama70b-mlperf41-offline",
        hardware_key="dgx-h100", model_key="llama-3.1-70b",
        tp=8, weight_dtype=DType.FP8, kv_dtype=DType.FP8,
        batch=256, prompt=1024, output=256,
        metric="decode_ceil_tok_s", measured=24525.0, regime="mixed",
        source="https://mlcommons.org/benchmarks/inference-datacenter/",
        notes="MLPerf Inference v4.1 Offline, Llama-2-70B PROXY (arch-identical to "
              "3.1-70B bar 32k vs 128k vocab), 8x H100 system-total. Scored against "
              "the DECODE CEILING: Offline is max-sustained throughput, where "
              "prefills overlap decode -- our exclusive-prefill output_tok_s "
              "serialises them and reads a spurious 0.33x here (a known analytic-"
              "throughput limitation; the serve loop models the overlap). [VERIFY] "
              "against the v4.1 results file (widely cited ~24,525; submission "
              "4.1-0043); OpenOrca ISL/OSL is variable. Dynamic table -- re-verify.",
    ),
    # ---- GB300 NVL72: direct MLPerf, rack-total, COARSE regime --------------
    Anchor(
        name="gb300-gptoss-mlperf60-offline",
        hardware_key="gb300-nvl72", model_key="gpt-oss-120b",
        tp=1, ep=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8,
        batch=256, prompt=1024, output=1024,
        metric="output_tok_s", measured=1046150.0, regime="mixed",
        source="https://developer.nvidia.com/blog/nvidia-platform-delivers-lowest-token-cost-enabled-by-extreme-co-design/",
        notes="MLPerf Inference v6.0 (Apr 2026) Offline, GB300 NVL72 (72 GPU) "
              "system-total, gpt-oss-120b MXFP4. COARSE: MLPerf parallelism, disagg, "
              "and ISL/OSL not published; tp1/ep8 (dp=9) fills the rack as a "
              "placeholder. Cross-check, NOT a fit driver. VERIFIED (MLCommons).",
    ),
    Anchor(
        name="gb300-llama8b-mlperf51-offline",
        hardware_key="gb300-nvl72", model_key="llama-3.1-8b",
        tp=1, weight_dtype=DType.FP4, kv_dtype=DType.FP8,
        batch=256, prompt=1024, output=256,
        metric="output_tok_s", measured=1322640.0, regime="mixed",
        source="https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/",
        notes="MLPerf v5.1 (Sep 2025 debut) Offline, GB300 NVL72, Dynamo "
              "DISAGGREGATED (unmodeled here), NVFP4 weights / FP8 KV. Per-GPU 18370 "
              "-> x72 = 1.32M system-total. COARSE cross-check. NVIDIA-reported.",
    ),
    # ---- TT-QuietBox 2 (4x Blackhole) x Qwen3-32B ---------------------------
    Anchor(
        name="qb2-qwen32b-ttmetal-b32",
        hardware_key="tt-quietbox-2", model_key="qwen3-32b",
        tp=4, weight_dtype=DType.FP8, kv_dtype=DType.FP8,
        batch=32, prompt=558, output=128,
        metric="output_tok_s", measured=691.0, regime="mixed",
        source="https://raw.githubusercontent.com/tenstorrent/tt-metal/main/models/model_targets.yaml",
        notes="tt-metal model_targets.yaml (main, ~2026-07-03); p300x2 == "
              "bh_quietbox_2 == our tt-quietbox-2 (in-file mapping). batch 32, seq "
              "686. Measured CI (run 26785408151; ~680-703 decode t/s over 6 runs) "
              "with a tolerance band. System-total 691 tok/s. BFP8 != NVIDIA FP8 "
              "exactly (caveat 4). VENDOR-CI.",
    ),
    Anchor(
        name="qb2-qwen32b-ttmetal-decode",
        hardware_key="tt-quietbox-2", model_key="qwen3-32b",
        tp=4, weight_dtype=DType.FP8, kv_dtype=DType.FP8,
        batch=32, prompt=558, output=128,
        metric="tpot_ms", measured=46.3, regime="decode",
        source="https://raw.githubusercontent.com/tenstorrent/tt-metal/main/models/model_targets.yaml",
        notes="Same run; 21.6 tok/s/user -> TPOT 46.3 ms. Decode probe. BFP8, "
              "tp=4 over the 50 GB/s Warp400 ring (collective-heavy). VENDOR-CI.",
    ),
    # ---- DGX Spark (GB10) x Llama-3.1-8B ------------------------------------
    Anchor(
        name="spark-llama8b-lmsys-decode",
        hardware_key="dgx-spark", model_key="llama-3.1-8b",
        tp=1, weight_dtype=DType.FP8, kv_dtype=DType.FP8,
        batch=1, prompt=1024, output=256,
        metric="tpot_ms", measured=48.78, regime="decode",
        source="https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/",
        notes="LMSYS DGX Spark review (2025-10-13), SGLang FP8, batch-1 decode "
              "20.5 tok/s -> TPOT 48.78 ms. INDEPENDENT. Pure-decode memory probe "
              "(tp=1, no collective; 273 GB/s bandwidth-bound). Software epoch "
              "matters a lot (date-tag).",
    ),
    Anchor(
        name="spark-llama8b-lmsys-b32",
        hardware_key="dgx-spark", model_key="llama-3.1-8b",
        tp=1, weight_dtype=DType.FP8, kv_dtype=DType.FP8,
        batch=32, prompt=2048, output=128,
        metric="output_tok_s", measured=368.0, regime="mixed",
        source="https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/",
        notes="Same review; batch-32 aggregate 368 tok/s (11.5/user). INDEPENDENT.",
    ),
]


# ---- harness ---------------------------------------------------------------


def _system_for(hardware_key: str) -> System:
    """Resolve a HARDWARE or GRAPH_PRESETS key to a spec-sheet System."""
    if hardware_key in HARDWARE:
        return HARDWARE[hardware_key]
    if hardware_key in GRAPH_PRESETS:
        from .bridge import system_from_graph
        return system_from_graph(GRAPH_PRESETS[hardware_key]())
    raise KeyError(
        f"unknown hardware_key {hardware_key!r} "
        f"(known: {sorted(HARDWARE) + sorted(GRAPH_PRESETS)})"
    )


def _metric_value(report, metric: str) -> float:
    if metric == "output_tok_s":
        return report.output_tokens_per_s
    if metric == "decode_ceil_tok_s":
        return report.decode_only_tokens_per_s
    if metric == "tpot_ms":
        return report.tpot_s * 1e3
    if metric == "ttft_ms":
        return report.ttft_s * 1e3
    if metric == "req_per_s":
        return report.requests_per_s
    raise ValueError(f"unknown metric {metric!r}")


def optimism_ratio(simulated: float, measured: float, metric: str) -> float:
    """The metric-direction-agnostic optimism ratio (>= 1 == sim optimistic).

    Throughput/rate: simulated/measured (a faster sim over-reads).  Latency:
    measured/simulated (a faster sim under-reads the time)."""
    if not measured or not simulated:
        return float("nan")
    if metric in _LATENCY_METRICS:
        return measured / simulated
    return simulated / measured


def run_anchor(anchor: Anchor, efficiency: Efficiency) -> tuple[float, float, float]:
    """Simulate `anchor`'s operating point under `efficiency` and return
    (simulated, measured, optimism_ratio).

    Uses the roofline engine derated by `efficiency` -- the calibration target
    is how far the (derated) roofline sits from the measured number.  ratio >= 1
    means still optimistic; a fitted profile brackets 1."""
    system = _system_for(anchor.hardware_key)
    model = MODELS[anchor.model_key]
    report = simulate(
        system, model, anchor.scenario, anchor.deployment,
        engine=RooflineEngine(efficiency),
    )
    simulated = _metric_value(report, anchor.metric)
    return simulated, anchor.measured, optimism_ratio(simulated, anchor.measured,
                                                       anchor.metric)


def calibrate_report(
    efficiency: Efficiency, anchors: list[Anchor] | None = None
) -> str:
    """A plain-text table scoring every anchor under `efficiency`.

    Columns: anchor, regime, metric, measured, simulated, optimism ratio.  The
    optimism ratio is `>= 1` when the simulator is optimistic (metric-direction
    agnostic, see `optimism_ratio`); a fitted profile brings it toward 1."""
    anchors = ANCHORS if anchors is None else anchors
    lines: list[str] = []
    add = lines.append
    add("=" * 90)
    add("inferencesim calibrate")
    add("=" * 90)
    add(f"Efficiency   : compute={efficiency.compute:g}  memory={efficiency.memory:g}"
        f"  collective={efficiency.collective:g}"
        f"  op_overhead={efficiency.op_overhead_s * 1e6:g}us")
    add("-" * 90)
    if not anchors:
        add("No calibration anchors are defined yet.")
        add("The measured-number research is tracked in CALIBRATION.md; "
            "calibration.ANCHORS encodes its recommended set.")
        add("=" * 90)
        return "\n".join(lines)

    add(f"{'anchor':<34}{'regime':<9}{'metric':<12}{'measured':>13}"
        f"{'simulated':>13}{'optimism':>10}")
    add("-" * 90)
    ratios: list[float] = []
    for a in anchors:
        simulated, measured, ratio = run_anchor(a, efficiency)
        ok = ratio == ratio  # not NaN
        ratios.append(ratio) if ok else None
        ratio_s = f"{ratio:.2f}x" if ok else "n/a"
        add(f"{a.name:<34}{a.regime:<9}{a.metric:<12}{measured:>13.4g}"
            f"{simulated:>13.4g}{ratio_s:>10}")
    add("-" * 90)
    if ratios:
        lo, hi = min(ratios), max(ratios)
        add(f"optimism ratio: median {median(ratios):.2f}x, "
            f"range [{lo:.2f}x, {hi:.2f}x]  (>= 1 == optimistic; fit brackets 1)")
    add("=" * 90)
    return "\n".join(lines)
