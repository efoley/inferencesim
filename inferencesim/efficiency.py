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
(`sol`), a cross-vendor derated point (`typical`), and per-vendor derated points
(`typical-nv`, `typical-tt`); `profile_for(hardware_key, "auto")` resolves the
vendor-appropriate one.

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


# Named profiles.  `sol` is the identity (leaves the roofline untouched); the
# `typical*` profiles derate toward measured reality.
#
# `typical` is the CROSS-VENDOR global fit -- FITTED against the measured anchors
# in calibration.py by the transparent recipe documented in CALIBRATION.md
# section 8 (retrieved 2026-07-05).  It is a COARSE single profile over a small,
# partly-proxy anchor set (NVIDIA anchors dominate it) -- refine as anchors grow:
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
# Fitted ratios bracket 1 (median 0.96x); the clean single-node probes land at ~1.0.
#
# PER-VENDOR profiles (see CALIBRATION.md section 8 "Per-vendor fit").  One global
# knob set cannot serve both vendors: the Tenstorrent tt-metal stack reaches a
# markedly lower effective memory bandwidth than NVIDIA, so the qb2 decode anchor
# sits at 1.41x under the global `typical`.  `profile_for(..., "auto")` routes each
# hardware key to its vendor profile.
#   typical-nv -- == `typical` today (an intentional alias: the global fit was
#                 driven by the NVIDIA anchors, so NVIDIA hardware keeps those
#                 knobs).  Kept as a separate entry so the two can diverge later.
#   typical-tt -- Tenstorrent.  Only `memory` is re-fitted; the rest are KEPT from
#                 the global fit for lack of Tenstorrent-specific evidence (do NOT
#                 invent numbers):
#     memory 0.40      -- from the qb2-qwen32b-ttmetal-decode residual under
#                         `typical` (0.57 / 1.41 = 0.40), the clean decode/memory
#                         probe.  Cross-checked against The Register's 41-50%-of-
#                         theoretical-peak gen-1 QuietBox review (the anchor-derived
#                         0.40 sits just under that band; both are cited).
#     compute 0.58     -- KEPT: tt prefill evidence is insufficient.  The qb2 TTFT
#                         anchor (87 ms) is memory-bound in our model (the compute
#                         knob barely moves it), so it cannot isolate MFU; The
#                         Register's prefill-side numbers are thin.
#     collective 0.85  -- KEPT: no Tenstorrent collective measurement; the Warp400
#                         ring's bus-bw efficiency is unmeasured.
#     op_overhead 1.5us-- KEPT: no tt-specific launch-overhead measurement.
PROFILES: dict[str, Efficiency] = {
    "sol": Efficiency(),
    "typical": Efficiency(compute=0.58, memory=0.57, collective=0.85,
                          op_overhead_s=1.5e-6),
    # per-vendor; typical-nv holds the global fit's values (an alias that may
    # diverge later), typical-tt re-fits only `memory` (0.57 -> 0.40).  See the
    # comment block above.
    "typical-nv": Efficiency(compute=0.58, memory=0.57, collective=0.85,
                             op_overhead_s=1.5e-6),
    "typical-tt": Efficiency(compute=0.58, memory=0.40, collective=0.85,
                             op_overhead_s=1.5e-6),
}


# --- vendor resolution (drives `profile_for(..., "auto")`) -------------------
#
# "auto" routes Tenstorrent hardware/graph keys to typical-tt and everything else
# to typical-nv.  Most Tenstorrent keys share the "tt-" prefix; the fine-grained
# graph presets whose key is chip-derived (no "tt-" prefix) are listed explicitly.
# ANY NEW Tenstorrent preset whose key does not start with "tt-" MUST be added to
# `_TT_KEYS` (e.g. a future "blackhole-*"/"wormhole-*" graph preset).
_TT_KEYS: frozenset[str] = frozenset({"blackhole-p150-fine", "blackhole-p150-mesh"})


def vendor_profile_name(hardware_key: str) -> str:
    """The vendor-appropriate `typical-*` profile name for a hardware/graph key
    -- exactly what `profile_for(hardware_key, "auto")` resolves to."""
    if hardware_key.startswith("tt-") or hardware_key in _TT_KEYS:
        return "typical-tt"
    return "typical-nv"


def profile_for(hardware_key: str, name: str = "auto") -> Efficiency:
    """Resolve a named efficiency profile for `hardware_key`.

    `name="auto"` picks the vendor-appropriate profile: Tenstorrent hardware
    (tt-* keys, plus the explicit `_TT_KEYS`) -> typical-tt, everything else ->
    typical-nv.  Any other `name` selects `PROFILES[name]` directly, bypassing the
    vendor mapping (so `profile_for("tt-quietbox-2", "sol")` is `sol`).  An unknown
    name raises `ValueError` naming the available profiles.
    """
    resolved = vendor_profile_name(hardware_key) if name == "auto" else name
    try:
        return PROFILES[resolved]
    except KeyError:
        raise ValueError(
            f"unknown efficiency profile {name!r}; "
            f"available: {sorted(PROFILES)} (or 'auto' to vendor-resolve)"
        ) from None
