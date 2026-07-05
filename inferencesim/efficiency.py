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
# TODO(calibration): the `typical` numbers below are *provisional placeholders*,
# not fitted values.  They await the measured-anchor fit tracked in
# calibration.py / CALIBRATION.md (MLPerf + vendor benchmarks) and WILL be
# updated before this ships.  They are deliberately round, defensible ballparks
# for modern accelerators running well-optimised LLM inference:
#   compute 0.80    -- decode is rarely compute-bound; prefill GEMMs hit
#                      ~0.6-0.85 MFU on tuned stacks.
#   memory  0.80    -- achievable HBM/GDDR streaming bandwidth vs the datasheet
#                      peak (memory-bound decode lives here).
#   collective 0.75 -- NCCL/RCCL bus-bandwidth efficiency for allreduce/a2a.
#   op_overhead 5us -- per-kernel launch/dispatch on the critical path.
PROFILES: dict[str, Efficiency] = {
    "sol": Efficiency(),
    "typical": Efficiency(
        compute=0.80,
        memory=0.80,
        collective=0.75,
        op_overhead_s=5e-6,
    ),
}
