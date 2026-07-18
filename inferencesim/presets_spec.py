"""Speculative, low-cost inference architectures.

These are NOT shipping products -- they are hypothetical machines built to
answer one question with the simulator: *how far can you get on cheap,
commodity parts instead of HBM?*  The thesis under all of them is memory-cost
arbitrage.  A decode step is bandwidth-bound: it streams the weights (and the
KV cache) from DRAM once per token, so tokens/s tracks aggregate DRAM
bandwidth, not FLOP/s.  HBM buys that bandwidth at a punishing $/(GB/s) and
pJ/bit.  A 256-bit LPDDR5X-8533 interface buys ~273 GB/s at perhaps ~1/10 the
$/GB and a fraction of the energy -- so if you can *aggregate* many such small
LPDDR memory subsystems through a good enough interconnect, you reach HBM-class
aggregate bandwidth on commodity silicon.  (LPDDR is soldered BGA, not a
DIMM/stick; a 256-bit interface is physically several LPDDR packages in
parallel, since one device presents only ~16-32 bits per channel.)

Two design families here:

  * **LPDDR swarms** (`lpddr-swarm-*`): many small serving ASICs, each with a
    single on-package LPDDR5X/6 memory subsystem, wired by a fat *low-latency*
    on-board
    fabric so tensor-parallel collectives are cheap.  Box-to-box uses
    *commodity Ethernet* (400 GbE RoCE) -- deliberately weak, to force the
    "keep TP inside the box, scale out with PP/DP" discipline.

  * **CXL memory disaggregation** (`cxl-*`): compute tiles served from a shared
    pool of cheap CXL-attached DDR5 instead of local HBM.  Capacity and
    bandwidth scale independently; the pool is enormous and cheap but only as
    fast as the CXL links, so these win on $/GB-of-capacity and giant-MoE /
    long-context fit, not on raw bandwidth.

Every number is a best-effort approximation grounded in public JEDEC / CXL
spec points (cited inline); prices and power splits are the softest figures.
Copy any preset with `dataclasses.replace` and edit.
"""

from __future__ import annotations

from .hardware import Chip, Compute, DType, Link, Memory, Node, System, Topology
from .units import GB, GIGA, MB, PETA, TB, TERA, US

# =============================================================================
# Memory building blocks (no HBM anywhere in this file, by design)
# =============================================================================
#
# LPDDR5X (JEDEC JESD209-5): up to 8533 Mbps/pin.  A 256-bit (32-byte) bus at
# 8533 Mbps gives 8533e6 * 32 = 273 GB/s -- exactly the figure NVIDIA's GB10 /
# DGX Spark hits over 256-bit LPDDR5x, so we reuse it as a proven anchor.  A
# 273 GB/s LPDDR5X subsystem draws only ~15-25 W (a few pJ/bit), vs ~250 W for
# an 8 TB/s HBM3e stack.
_LPDDR5X_BW = 273 * GIGA

# LPDDR6 (JEDEC JESD209-6, published 2025): reorganised into 24-bit channels
# (two 12-bit sub-channels) with a launch per-pin rate around 14.4 Gbps.  At a
# 256-bit-equivalent width that is 14400e6 * 32 = 460.8 GB/s, with better
# energy/bit than LPDDR5X.  We take the launch rate, not the roadmap ceiling.
_LPDDR6_BW = 460 * GIGA

# =============================================================================
# Chips
# =============================================================================

# ---- Small LPDDR5X serving tile ---------------------------------------------
# A hypothetical minimal inference ASIC: modest matrix engines fed by one
# 256-bit LPDDR5X-8533 memory subsystem (273 GB/s, the GB10 figure; physically
# several soldered LPDDR packages in parallel, not a DIMM).  Compute is sized at
# ~470 FLOP/byte (fp8) -- in the
# same memory-leaning ballpark as a serving GPU (H100 ~590, GB10 ~915) rather
# than a compute-heavy Tenstorrent part -- so the tile is balanced for
# batched decode, not wasted on FLOP it can't feed.  On-chip NoC and SRAM are
# sized well above LPDDR so the DRAM stack stays the min-cut (as it should).
LPDDR5X_TILE = Chip(
    name="LPDDR5X tile",
    compute=Compute(
        name="serving matrix engines",
        peak_flops={
            DType.FP4: 256 * TERA,
            DType.FP8: 128 * TERA,
            DType.BF16: 64 * TERA,
            DType.FP16: 64 * TERA,
        },
        power_w=45.0,
    ),
    dram=Memory("LPDDR5X (32 GB)", capacity_bytes=32 * GB, bandwidth=_LPDDR5X_BW,
                power_w=22.0, latency_s=0.12 * US),
    on_chip_path=(
        Link("tile NoC", bandwidth=2 * TB, latency_s=0.1 * US, power_w=4.0),
        Memory("tile SRAM (64 MB)", capacity_bytes=64 * MB, bandwidth=8 * TB,
               power_w=6.0),
    ),
    idle_power_w=8.0,
)

# ---- Small LPDDR6 serving tile (next-gen memory) ----------------------------
LPDDR6_TILE = Chip(
    name="LPDDR6 tile",
    compute=Compute(
        name="serving matrix engines (gen2)",
        peak_flops={
            DType.FP4: 384 * TERA,
            DType.FP8: 192 * TERA,
            DType.BF16: 96 * TERA,
            DType.FP16: 96 * TERA,
        },
        power_w=60.0,
    ),
    dram=Memory("LPDDR6 (48 GB)", capacity_bytes=48 * GB, bandwidth=_LPDDR6_BW,
                power_w=25.0, latency_s=0.1 * US),
    on_chip_path=(
        Link("tile NoC", bandwidth=3 * TB, latency_s=0.1 * US, power_w=5.0),
        Memory("tile SRAM (96 MB)", capacity_bytes=96 * MB, bandwidth=12 * TB,
               power_w=8.0),
    ),
    idle_power_w=10.0,
)

# ---- CXL-pooled compute tile ------------------------------------------------
# A compute die whose *working memory is a shared, disaggregated CXL pool of
# DDR5*, not local HBM.  Modelling note: this simulator streams weights from a
# single `dram`, so we make that `dram` the tile's slice of the CXL pool -- big
# capacity, bandwidth capped at the tile's aggregate CXL link rate (the real
# bottleneck), CXL access latency.  The on-die SRAM in on_chip_path is sized
# far above the CXL rate, so the CXL links are the min-cut by construction (the
# whole point of disaggregation).  A genuine two-tier "small fast local DRAM +
# big slow pool" hierarchy is not expressible in the single-DRAM path model;
# the conservative choice is to serve *everything* from the pool.
#
# Bandwidth: 8x CXL 3.0 x16 links at 64 GT/s (PCIe 6.0 PAM4) = 8 * 64 GB/s =
# 512 GB/s per tile.  Capacity: 256 GB of the tile's slice of pooled DDR5.
# Latency: ~0.3 us, the ~150-400 ns CXL memory adder over local DDR.
_CXL_X16 = 64 * GIGA  # one CXL 3.0 x16 link, one direction
CXL_COMPUTE_TILE = Chip(
    name="CXL compute tile",
    compute=Compute(
        name="CXL-served matrix engines",
        peak_flops={
            DType.FP4: 512 * TERA,
            DType.FP8: 256 * TERA,
            DType.BF16: 128 * TERA,
            DType.FP16: 128 * TERA,
        },
        power_w=90.0,
    ),
    dram=Memory("CXL DDR5 pool slice (256 GB)", capacity_bytes=256 * GB,
                bandwidth=8 * _CXL_X16, power_w=40.0, latency_s=0.3 * US),
    on_chip_path=(
        Memory("on-die SRAM (128 MB)", capacity_bytes=128 * MB, bandwidth=10 * TB,
               power_w=10.0),
    ),
    idle_power_w=20.0,
)

# =============================================================================
# Interconnects
# =============================================================================
#
# The load-bearing bet of the swarms is the *bandwidth* of an on-board fabric:
# because the tiles are small and physically close (one board / backplane), a
# fat short-reach link is affordable, so tensor-parallel allreduces -- which
# would throttle a swarm of wimpy chips on a thin fabric -- stay cheap.  We
# model it as a switched (ALL_TO_ALL) backplane at 200 GB/s/dir and 0.4 us:
# NVLink/QuietBox-class bandwidth (NVLink5 ~900 GB/s, QuietBox ring ~200 GB/s),
# below NVLink's ~1 us end-to-end latency but no longer assuming away the switch
# hop.  The bandwidth is the demanding assumption -- the part a real build must
# earn -- not the latency: at fixed 200 GB/s a decode-TP=32 run only loses ~21%
# TPOT going from 0.4 us all the way to NVLink's 1 us, and stays workable to
# ~2 us (see SPECULATIVE.md "Sensitivity to link latency").  Latency only turns
# into a throughput-killer at the 3-5 us of commodity Ethernet, which is exactly
# why box-to-box (below) stays DP-only.
_SWARM_FABRIC = Link("low-latency backplane (200G/dir)", bandwidth=200 * GIGA,
                     latency_s=0.4 * US, power_w=3.0)
_SWARM_FABRIC6 = Link("low-latency backplane (256G/dir)", bandwidth=256 * GIGA,
                      latency_s=0.4 * US, power_w=3.5)

# Box-to-box is COMMODITY ETHERNET on purpose: 400 GbE (400 Gbit/s = 50 GB/s
# per direction) RoCE, ~5 us switch+NIC latency.  An order of magnitude slower
# and higher-latency than the on-board fabric -- the model must keep TP inside a
# box and scale out with pipeline/data parallelism, which is exactly the point.
_ETH_400G = Link("400 GbE RoCE (per box, per dir)", bandwidth=50 * GIGA,
                 latency_s=5 * US, power_w=8.0)

# =============================================================================
# Systems -- LPDDR swarms
# =============================================================================

_SWARM_N = 64

LPDDR_SWARM_64 = System(
    name="LPDDR5X swarm-64",
    node=Node(
        name="swarm box (64x LPDDR5X tile)",
        chip=LPDDR5X_TILE,
        n_chips=_SWARM_N,
        interconnect=_SWARM_FABRIC,
        topology=Topology.ALL_TO_ALL,  # low-latency switched backplane
        overhead_power_w=1_800.0,  # host, backplane switch, VRMs, fans, PSU loss
        # 64 tiles at a commodity-ASIC ~$1.2k + board/switch/host/chassis ~$68k
        cost_usd=145_000.0,
    ),
    description="64 small LPDDR5X serving tiles on a low-latency switched "
                "backplane, one box.  ~17.5 TB/s aggregate DRAM bandwidth and "
                "2 TB of LPDDR for ~$145k and ~7 kW -- no HBM.",
)

LPDDR6_SWARM_64 = System(
    name="LPDDR6 swarm-64",
    node=Node(
        name="swarm box (64x LPDDR6 tile)",
        chip=LPDDR6_TILE,
        n_chips=_SWARM_N,
        interconnect=_SWARM_FABRIC6,
        topology=Topology.ALL_TO_ALL,
        overhead_power_w=2_200.0,
        cost_usd=175_000.0,  # 64 tiles ~$1.6k + $72k board/switch/host
    ),
    description="64 LPDDR6 tiles, one box: ~29.5 TB/s aggregate DRAM bandwidth "
                "and 3 TB for ~$175k -- next-gen LPDDR closes most of the HBM "
                "bandwidth gap on commodity memory.",
)

LPDDR_SWARM_POD = System(
    name="LPDDR5X swarm-pod (4 boxes)",
    node=Node(
        name="swarm box (64x LPDDR5X tile)",
        chip=LPDDR5X_TILE,
        n_chips=_SWARM_N,
        interconnect=_SWARM_FABRIC,
        topology=Topology.ALL_TO_ALL,
        overhead_power_w=1_800.0,
        cost_usd=145_000.0,
    ),
    n_nodes=4,
    network=_ETH_400G,  # commodity Ethernet spine between boxes
    extra_cost_usd=40_000.0,  # 400 GbE spine switch + optics/cabling
    description="Four LPDDR5X swarm boxes on a commodity 400 GbE RoCE spine: "
                "256 tiles, ~70 TB/s aggregate DRAM bandwidth, 8 TB LPDDR. "
                "Keep TP in-box; scale with PP/DP across the slow Ethernet.",
)

# =============================================================================
# Systems -- CXL memory disaggregation
# =============================================================================

CXL_POOL_NODE = System(
    name="CXL DDR5 pool node",
    node=Node(
        name="CXL box (8x compute tile + pooled DDR5)",
        chip=CXL_COMPUTE_TILE,
        n_chips=8,
        interconnect=_SWARM_FABRIC,  # tiles share the same low-latency fabric
        topology=Topology.ALL_TO_ALL,
        overhead_power_w=1_500.0,  # CXL switch, pool controllers, host, PSU loss
        # 8 compute tiles ~$3k + 2 TB pooled DDR5 (~$3/GB) ~$6k + $35k box
        cost_usd=65_000.0,
    ),
    description="8 compute tiles served from a shared 2 TB CXL DDR5 pool "
                "(~4.1 TB/s aggregate over CXL 3.0).  Capacity and bandwidth "
                "scale independently; wins on $/GB and giant-model fit, not on "
                "raw bandwidth -- the honest disaggregation trade.",
)

CXL_MOE_POD = System(
    name="CXL MoE pod (4 boxes)",
    node=Node(
        name="CXL box (8x compute tile + pooled DDR5)",
        chip=CXL_COMPUTE_TILE,
        n_chips=8,
        interconnect=_SWARM_FABRIC,
        topology=Topology.ALL_TO_ALL,
        overhead_power_w=1_500.0,
        cost_usd=65_000.0,
    ),
    n_nodes=4,
    network=_ETH_400G,
    extra_cost_usd=40_000.0,
    description="Four CXL pool boxes on 400 GbE: 32 compute tiles, an 8 TB "
                "pooled-DDR5 capacity tier that swallows a 671B MoE whole, and "
                "~16 TB/s aggregate CXL bandwidth for the 37B active slice.",
)

# The speculative catalogue, merged into presets.HARDWARE at import.
SPEC_HARDWARE: dict[str, System] = {
    "lpddr-swarm-64": LPDDR_SWARM_64,
    "lpddr6-swarm-64": LPDDR6_SWARM_64,
    "lpddr-swarm-pod": LPDDR_SWARM_POD,
    "cxl-lpddr-pool": CXL_POOL_NODE,
    "cxl-moe-pod": CXL_MOE_POD,
}
