"""Calibration anchors and the harness that scores the simulator against them.

An `Anchor` pins a *measured* number -- a throughput or latency a real stack
achieved on a real machine, from a citable source (MLPerf, a vendor blog, a
reproducible benchmark) -- to the exact (hardware, model, deployment, scenario)
the simulator can reproduce.  `run_anchor` simulates that point under a given
`Efficiency` and reports the sim/measured ratio; `calibrate_report` tabulates a
whole anchor set.

Why this exists: roofline numbers are *upper bounds* (perfect tiling, 100%
bandwidth, bandwidth-optimal collectives, no launch overhead), so at the `sol`
efficiency the ratio should sit at or above 1 -- the simulator is optimistic by
construction.  A well-fitted `Efficiency` profile derates the peaks until the
ratios *bracket* 1: the measured reality lands inside the simulator's range
rather than always beating it.  Fitting that profile is what the anchors are
for.

The measured research -- candidate rows, sources, retrieval dates, and the
reasoning behind each chosen anchor -- lives in the top-level CALIBRATION.md.
`ANCHORS` below is the machine-readable encoding of its "Recommended anchor
set"; it is intentionally EMPTY until that research lands, so nothing here
asserts an unverified number.
"""

from __future__ import annotations

from dataclasses import dataclass

from .efficiency import Efficiency
from .engine import RooflineEngine
from .hardware import DType, System
from .presets import HARDWARE, MODELS
from .presets_fine import GRAPH_PRESETS
from .simulate import simulate
from .workload import Deployment, Scenario

# metric keys an Anchor may pin, mapped to a (Report -> float) extractor.
METRICS = ("output_tok_s", "tpot_ms", "ttft_ms", "req_per_s")


@dataclass(frozen=True)
class Anchor:
    """One measured benchmark point the simulator should reproduce.

    hardware_key : a HARDWARE or GRAPH_PRESETS key (see `inferencesim list`).
    model_key    : a MODELS key.
    tp/pp/ep, weight_dtype/kv_dtype/act_dtype : the deployment.
    batch/prompt/output : the scenario (batch = concurrent sequences/replica,
                   prompt = input tokens, output = generated tokens).
    metric       : one of METRICS -- which reported number `measured` is.
    measured     : the measured value (tok/s, ms, or req/s per `metric`).
    source       : a URL / citation for `measured` (see CALIBRATION.md).
    notes        : anything a reader needs (stack version, caveats).
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
    notes: str = ""

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(
                f"Anchor.metric must be one of {METRICS}, got {self.metric!r}"
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


# The recommended anchor set -- populated from CALIBRATION.md's final section.
# Left empty on purpose: the measured-number research is a separate task, and
# encoding an unverified figure here would defeat the point.  Each entry, once
# added, mirrors a CALIBRATION.md row by its slug `name`.
#
# Example of the shape (commented out -- the number is illustrative, NOT a
# real measurement, so it must not be enabled):
#
#   ANCHORS = [
#       Anchor(
#           name="gb300-llama70b-tp8-decode",
#           hardware_key="gb300-nvl72", model_key="llama-3.1-70b",
#           tp=8, weight_dtype=DType.FP4, kv_dtype=DType.FP8,
#           batch=64, prompt=4096, output=1024,
#           metric="tpot_ms", measured=0.0,            # <- fill from a source
#           source="https://example.com/benchmark",
#           notes="vLLM x.y, ISL/OSL 4096/1024",
#       ),
#   ]
ANCHORS: list[Anchor] = []


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
    if metric == "tpot_ms":
        return report.tpot_s * 1e3
    if metric == "ttft_ms":
        return report.ttft_s * 1e3
    if metric == "req_per_s":
        return report.requests_per_s
    raise ValueError(f"unknown metric {metric!r}")


def run_anchor(anchor: Anchor, efficiency: Efficiency) -> tuple[float, float, float]:
    """Simulate `anchor`'s operating point under `efficiency` and return
    (simulated, measured, ratio) where ratio = simulated / measured.

    Uses the roofline engine derated by `efficiency` -- the calibration target
    is exactly how far the (derated) roofline sits from the measured number.
    A ratio >= 1 means the simulator is still optimistic; a fitted profile
    brackets 1."""
    system = _system_for(anchor.hardware_key)
    model = MODELS[anchor.model_key]
    report = simulate(
        system, model, anchor.scenario, anchor.deployment,
        engine=RooflineEngine(efficiency),
    )
    simulated = _metric_value(report, anchor.metric)
    ratio = simulated / anchor.measured if anchor.measured else float("nan")
    return simulated, anchor.measured, ratio


def calibrate_report(
    efficiency: Efficiency, anchors: list[Anchor] | None = None
) -> str:
    """A plain-text table scoring every anchor under `efficiency`.

    Columns: anchor, metric, measured, simulated, sim/measured ratio.  With the
    `sol` efficiency the ratios should sit >= 1 (roofline is an upper bound);
    a fitted profile should bring them to bracket 1."""
    anchors = ANCHORS if anchors is None else anchors
    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add("inferencesim calibrate")
    add("=" * 78)
    add(f"Efficiency   : compute={efficiency.compute:g}  memory={efficiency.memory:g}"
        f"  collective={efficiency.collective:g}"
        f"  op_overhead={efficiency.op_overhead_s * 1e6:g}us")
    add("-" * 78)
    if not anchors:
        add("No calibration anchors are defined yet.")
        add("")
        add("The measured-number research (MLPerf / vendor benchmarks) is tracked in")
        add("CALIBRATION.md; calibration.ANCHORS encodes its recommended set once it")
        add("lands.  Until then there is nothing to score -- run again after anchors")
        add("are added, or pass your own list to calibrate_report(...).")
        add("=" * 78)
        return "\n".join(lines)

    header = f"{'anchor':<32} {'metric':<12} {'measured':>12} {'simulated':>12} {'sim/meas':>9}"
    add(header)
    add("-" * 78)
    for a in anchors:
        simulated, measured, ratio = run_anchor(a, efficiency)
        ratio_s = f"{ratio:.2f}x" if ratio == ratio else "n/a"  # NaN check
        add(f"{a.name:<32} {a.metric:<12} {measured:>12.4g} {simulated:>12.4g} "
            f"{ratio_s:>9}")
    add("-" * 78)
    add("ratio >= 1 = simulator optimistic (roofline upper bound); a fitted")
    add("profile should bracket 1.")
    add("=" * 78)
    return "\n".join(lines)
