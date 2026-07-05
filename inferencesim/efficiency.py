"""Efficiency factors: the knobs that bring the roofline back to earth.

The roofline model is a set of *upper bounds* -- perfect tiling, 100% of peak
FLOP/s and DRAM/interconnect bandwidth, bandwidth-optimal collectives, and no
kernel-launch or dispatch overhead.  Real systems achieve some fraction of
each.  An `Efficiency` bundles those fractions (plus a fixed per-op overhead)
so the same derating applies consistently everywhere a cost is computed: the
roofline engine, the discrete-event engine's unit costs and its expanded
collectives, and the chip-graph tile model.

`Efficiency()` (all factors 1.0, zero overhead) is the identity: it must leave
every roofline number bit-identical, so an `Efficiency` never *changes* the
math, it only scales it.  The named `PROFILES` give a "speed of light" bound
(`sol`) and a provisional derated point (`typical`); see the TODO on `typical`.

The measured anchors these are meant to be fitted against live in
`calibration.py` (and the human-readable research writeup in CALIBRATION.md).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Efficiency:
    """Fractions of the roofline peaks a real system is assumed to reach.

    compute     -- fraction of peak FLOP/s actually achieved (an MFU ceiling).
    memory      -- fraction of peak DRAM / on-chip / interconnect-path
                   bandwidth (an MBU ceiling).
    collective  -- fraction of link bandwidth a collective achieves
                   (NCCL-style bus-bandwidth efficiency).  Scales only the
                   *bandwidth* term of a collective; propagation latency is
                   physical flight time and is never derated.
    op_overhead_s -- fixed seconds added per op *instance* (kernel launch /
                   dispatch).  Applies once per launched op, not per tile.

    Every factor must lie in (0, 1]; the overhead must be >= 0.  The default
    is the identity (speed of light).
    """

    compute: float = 1.0
    memory: float = 1.0
    collective: float = 1.0
    op_overhead_s: float = 0.0

    def __post_init__(self) -> None:
        for name in ("compute", "memory", "collective"):
            v = getattr(self, name)
            if not (0.0 < v <= 1.0):
                raise ValueError(
                    f"Efficiency.{name} must be in (0, 1], got {v!r}"
                )
        if self.op_overhead_s < 0.0:
            raise ValueError(
                f"Efficiency.op_overhead_s must be >= 0, got {self.op_overhead_s!r}"
            )


# Named profiles.  `sol` is the identity (leaves the roofline untouched);
# `typical` derates toward measured reality.
#
# `typical` is FITTED against the measured anchors in calibration.py by the
# transparent recipe documented in CALIBRATION.md section 7 (retrieved
# 2026-07-05).  It is a COARSE single global profile over a small, partly-proxy
# anchor set -- refine it as anchors are confirmed/added:
#   compute 0.58    -- 1/median(sol optimism ratios of the prefill-bound anchors
#                      h100-llama8b-2k128 1.76x, dgxh100-70b-tp2-8k1k 1.66x);
#                      top of the ~0.30-0.58 inference-prefill MFU band.
#   memory  0.57    -- 1/1.76, the clean tp=1 batch-1 decode probe
#                      (spark-llama8b-lmsys-decode); inside Databricks' ~0.55-0.70
#                      decode-MBU band.
#   collective 0.85 -- literature, NOT anchor-fitted (NCCL allreduce ~0.70-0.80
#                      of line rate without SHARP, 0.90+ with).
#   op_overhead 1.5us -- CUDA Graphs recover ~20% of a batch-1 decode step
#                      (arXiv:2605.30571); effective per-op launch under batching.
# Fitted ratios bracket 1 (median 0.96x); the clean single-node probes land at
# ~1.0.  Per-vendor profiles (Tenstorrent runs lower -- QuietBox decode residual
# ~1.4x under this global fit) are future work.
PROFILES: dict[str, Efficiency] = {
    "sol": Efficiency(),
    "typical": Efficiency(
        compute=0.58,
        memory=0.57,
        collective=0.85,
        op_overhead_s=1.5e-6,
    ),
}
