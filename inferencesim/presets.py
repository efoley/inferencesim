"""Built-in hardware systems and model specs.

Numbers come from public spec sheets and press material and are best-effort
APPROXIMATIONS (especially prices, power splits, NoC/SRAM aggregates).
Every preset is a plain frozen dataclass -- copy one with
`dataclasses.replace` and edit any figure you have better data for.
"""

from __future__ import annotations

from .hardware import Chip, Compute, DType, Link, Memory, Node, System, Topology
from .units import GB, GIGA, MB, PETA, TB, TERA, US
from .workload import ModelSpec, MoEConfig

# =============================================================================
# Chips
# =============================================================================

# ---- NVIDIA H100 SXM (well-characterised; useful for validating the sim) ---
H100_SXM = Chip(
    name="H100-SXM",
    compute=Compute(
        name="H100 tensor cores",
        peak_flops={  # dense
            DType.FP8: 1979 * TERA,
            DType.BF16: 989 * TERA,
            DType.FP16: 989 * TERA,
            DType.TF32: 495 * TERA,
            DType.FP32: 67 * TERA,
        },
        power_w=420.0,  # approx split of the 700 W TDP
    ),
    dram=Memory("HBM3", capacity_bytes=80 * GB, bandwidth=3.35 * TB, power_w=130.0),
    idle_power_w=150.0,
)

# ---- NVIDIA B300 / Blackwell Ultra (GB300 NVL72 GPU) -- estimates ----------
B300 = Chip(
    name="B300 (Blackwell Ultra)",
    compute=Compute(
        name="B300 tensor cores",
        peak_flops={  # dense; FP4 per NVIDIA "15 PF dense FP4" claim (approx)
            DType.FP4: 15 * PETA,
            DType.FP8: 4.5 * PETA,
            DType.BF16: 2.25 * PETA,
            DType.FP16: 2.25 * PETA,
        },
        power_w=900.0,  # approx split of ~1.4 kW per GPU
    ),
    dram=Memory("HBM3e", capacity_bytes=288 * GB, bandwidth=8 * TB, power_w=250.0),
    idle_power_w=250.0,
)

# ---- NVIDIA GB10 (DGX Spark) -- estimates -----------------------------------
GB10 = Chip(
    name="GB10 (Grace Blackwell)",
    compute=Compute(
        name="GB10 tensor cores",
        peak_flops={  # "1 PFLOP FP4 sparse" marketing -> ~500 TF dense FP4
            DType.FP4: 500 * TERA,
            DType.FP8: 250 * TERA,
            DType.BF16: 125 * TERA,
        },
        power_w=70.0,
    ),
    dram=Memory("LPDDR5x (unified)", capacity_bytes=128 * GB, bandwidth=273 * GIGA,
                power_w=30.0),
    idle_power_w=40.0,
)

# ---- Tenstorrent Blackhole p150 ---------------------------------------------
# Modelled after the Metalium block diagram: GDDR6 -> NoC -> Tensix-core SRAM
# -> matrix engine.  The NoC and SRAM stages are explicit so a discrete-event
# engine can later add contention; at speed-of-light they simply cap the
# effective DRAM streaming bandwidth (min over the path).
_BH_NOC = Link(
    name="Blackhole NoC (aggregate DRAM->cores)",
    bandwidth=3.2 * TB,  # approx aggregate injection bandwidth; not the bottleneck
    latency_s=0.2 * US,
)
_BH_SRAM = Memory(
    name="Tensix L1 SRAM (140 cores x 1.5 MB)",
    capacity_bytes=210 * MB,
    bandwidth=12 * TB,  # approx: 140 cores * ~64 B/cycle * 1.35 GHz
    power_w=20.0,
)
BLACKHOLE_P150 = Chip(
    name="Blackhole p150c",
    compute=Compute(
        name="Tensix matrix engines (140 cores)",
        peak_flops={  # approx from Tenstorrent material
            DType.FP8: 774 * TERA,
            DType.BF16: 387 * TERA,
            DType.FP16: 387 * TERA,
        },
        power_w=170.0,
    ),
    dram=Memory("GDDR6", capacity_bytes=32 * GB, bandwidth=512 * GIGA, power_w=50.0),
    on_chip_path=(_BH_NOC, _BH_SRAM),
    idle_power_w=60.0,
)

# =============================================================================
# Systems
# =============================================================================

NVLINK5 = Link("NVLink 5 (per GPU, per direction)", bandwidth=900 * GIGA, latency_s=1 * US)
NVLINK4 = Link("NVLink 4 (per GPU, per direction)", bandwidth=450 * GIGA, latency_s=1 * US)

GB300_NVL72 = System(
    name="GB300 NVL72",
    node=Node(
        name="NVL72 rack",
        chip=B300,
        n_chips=72,
        interconnect=NVLINK5,
        topology=Topology.ALL_TO_ALL,  # NVSwitch
        overhead_power_w=20_000.0,  # 36x Grace, NVSwitch trays, fans (approx)
        cost_usd=3_500_000.0,  # street-price estimate
    ),
    description="One NVL72 rack: 72x Blackwell Ultra on an NVSwitch domain.",
)

DGX_H100 = System(
    name="DGX H100",
    node=Node(
        name="DGX H100",
        chip=H100_SXM,
        n_chips=8,
        interconnect=NVLINK4,
        overhead_power_w=2_500.0,
        cost_usd=300_000.0,  # approx
    ),
    description="One DGX H100 server (8x H100 SXM, NVLink4).",
)

H100_SINGLE = System(
    name="H100 (single)",
    node=Node(name="1x H100", chip=H100_SXM, n_chips=1, cost_usd=30_000.0,
              overhead_power_w=300.0),
    description="A single H100 SXM for sanity checks.",
)

DGX_SPARK = System(
    name="DGX Spark",
    node=Node(name="DGX Spark", chip=GB10, n_chips=1, overhead_power_w=60.0,
              cost_usd=3_999.0),
    description="One DGX Spark (GB10, 128 GB unified LPDDR5x).",
)

_SPARK_CX7 = Link("ConnectX-7 200GbE", bandwidth=25 * GIGA, latency_s=3 * US)

DGX_SPARK_X2 = System(
    name="DGX Spark x2",
    node=Node(name="DGX Spark", chip=GB10, n_chips=1, overhead_power_w=60.0,
              cost_usd=3_999.0),
    n_nodes=2,
    network=_SPARK_CX7,
    description="Two DGX Sparks back-to-back over ConnectX-7 200 GbE.",
)

# Gen-1 Blackhole QuietBox, $12,000: 4x p150c, EPYC 8124P (125 W) host with
# 512 GB DDR5, 1650 W PSU.  Ships with 8x QSFP-DD 800GbE cables = a ring of
# 4 cards with 2 cables per edge -> ~200 GB/s per direction per neighbour.
_QB_ETH = Link("QSFP-DD 800GbE x2 (card-to-card)", bandwidth=200 * GIGA,
               latency_s=2 * US)

TT_QUIETBOX = System(
    name="TT-QuietBox (Blackhole)",
    node=Node(
        name="QuietBox",
        chip=BLACKHOLE_P150,
        n_chips=4,
        interconnect=_QB_ETH,
        topology=Topology.RING,
        overhead_power_w=350.0,  # EPYC host, 512 GB DDR5, fans, PSU losses (approx)
        cost_usd=12_000.0,  # list price
    ),
    description="Tenstorrent QuietBox: 4x Blackhole p150c on an 800GbE ring, "
                "EPYC 8124P host.",
)

# TT-QuietBox 2, part TW-04003 ($9,999, ships Q2 2026).  Official specs:
# 2 liquid-cooled p300c PCIe cards, each with 2 Blackhole ASICs -- 480
# Tensix cores, 720 MB SRAM (180 MB/ASIC = 120 cores x 1.5 MB) total,
# 2,654 TFLOPS BlockFP8, 128 GB GDDR6 @ 16 GT/s.  Ryzen 7 9700X (65 W)
# host, 256 GB DDR5, 1600 W PSU on a standard outlet.
#
# MEMORY BANDWIDTH IS UNRESOLVED.  The spec sheet reads "(1024 GB/sec)",
# but that is likely per-card: the ASICs are the same Blackhole silicon at
# the same 16 GT/s and 32 GB/ASIC as the p150c (512 GB/s), so the per-die
# bus is probably unchanged -> 512 GB/s/ASIC, 1024/card, 2048/box.  Set
# _QB2_DRAM_BW below to flip the whole model between the two hypotheses.
_QB2_DRAM_BW = 512 * GIGA  # per ASIC; use 256*GIGA for the "1024 GB/s box" reading

BLACKHOLE_QB2 = Chip(
    name="Blackhole (QB2, 120-core)",
    compute=Compute(
        name="Tensix matrix engines (120 cores)",
        peak_flops={
            DType.FP8: 663.5 * TERA,  # 2654 TFLOPS BlockFP8 / 4 ASICs
            DType.BF16: 331.7 * TERA,
            DType.FP16: 331.7 * TERA,
        },
        power_w=150.0,
    ),
    dram=Memory("GDDR6", capacity_bytes=32 * GB, bandwidth=_QB2_DRAM_BW, power_w=50.0),
    on_chip_path=(
        Link(name="Blackhole NoC (aggregate DRAM->cores)", bandwidth=3.2 * TB,
             latency_s=0.2 * US),
        Memory(name="Tensix L1 SRAM (120 cores x 1.5 MB)", capacity_bytes=180 * MB,
               bandwidth=10.4 * TB, power_w=18.0),
    ),
    idle_power_w=55.0,
)

# The 4 ASICs form a ring of homogeneous Warp400 links (Samtec ARP6 copper,
# direct -- no PCIe/switch/QSFP).  Each Warp400 is 400 Gbit/s per direction
# = 50 GB/s.  On-card (die-to-die) and card-to-card edges are the same link
# type, so the ring is uniform at 50 GB/s per direction.
_QB2_WARP400 = Link(name="Warp400 (400G/dir)", bandwidth=50 * GIGA, latency_s=0.5 * US)
_QB2_CARD_LINK = Link(name="Warp400 card-to-card (400G/dir)", bandwidth=50 * GIGA,
                      latency_s=1 * US)

TT_QUIETBOX_2 = System(
    name="TT-QuietBox 2",
    node=Node(
        name="p300c dual-Blackhole card",
        chip=BLACKHOLE_QB2,
        n_chips=2,
        interconnect=_QB2_WARP400,
        topology=Topology.RING,
        overhead_power_w=100.0,  # half the Ryzen host / pump / PSU share
        cost_usd=0.0,  # priced at the system level
    ),
    n_nodes=2,
    network=_QB2_CARD_LINK,
    extra_cost_usd=9_999.0,  # box list price
    description="Tenstorrent TT-QuietBox 2 (TW-04003): 2 p300c cards x 2 Blackhole "
                "(120-core), ring of Warp400 links, 720 MB SRAM total.",
)

HARDWARE: dict[str, System] = {
    "gb300-nvl72": GB300_NVL72,
    "dgx-h100": DGX_H100,
    "h100": H100_SINGLE,
    "dgx-spark": DGX_SPARK,
    "dgx-spark-x2": DGX_SPARK_X2,
    "tt-quietbox": TT_QUIETBOX,
    "tt-quietbox-2": TT_QUIETBOX_2,
}

# =============================================================================
# Models
# =============================================================================

LLAMA_3_1_8B = ModelSpec(
    name="llama-3.1-8b",
    n_layers=32, d_model=4096, n_heads=32, n_kv_heads=8, d_head=128,
    d_ff=14336, vocab_size=128256,
)

LLAMA_3_1_70B = ModelSpec(
    name="llama-3.1-70b",
    n_layers=80, d_model=8192, n_heads=64, n_kv_heads=8, d_head=128,
    d_ff=28672, vocab_size=128256,
)

QWEN3_32B = ModelSpec(
    name="qwen3-32b",
    n_layers=64, d_model=5120, n_heads=64, n_kv_heads=8, d_head=128,
    d_ff=25600, vocab_size=151936,
)

GPT_OSS_120B = ModelSpec(
    name="gpt-oss-120b",
    n_layers=36, d_model=2880, n_heads=64, n_kv_heads=8, d_head=64,
    d_ff=2880, vocab_size=201088,
    moe=MoEConfig(n_experts=128, top_k=4, d_ff_expert=2880),
)

MODELS: dict[str, ModelSpec] = {
    "llama-3.1-8b": LLAMA_3_1_8B,
    "llama-3.1-70b": LLAMA_3_1_70B,
    "qwen3-32b": QWEN3_32B,
    "gpt-oss-120b": GPT_OSS_120B,
}
