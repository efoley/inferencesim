"""Hardware building blocks.

Machines are described bottom-up from fine-grained components so that very
different systems (an NVL72 rack, a pair of DGX Sparks, a Tenstorrent
QuietBox) share one vocabulary:

    Compute -- a pool of math engines with per-dtype peak FLOP/s
    Memory  -- one storage level (GDDR/HBM/LPDDR/SRAM): capacity + bandwidth
    Link    -- an interconnect (NoC, NVLink, PCIe, Ethernet): bandwidth + latency
    Chip    -- Compute fed from a DRAM through an ordered on-chip data path.
               For Tensix this mirrors the Metalium block diagram:
                   GDDR6 -> NoC -> core SRAM -> matrix engine
               and the effective streaming bandwidth is the min over the path.
    Node    -- n_chips Chips joined by an intra-node Link (NVLink/Ethernet)
    System  -- n_nodes Nodes joined by a network Link

Every component optionally carries a dynamic power figure (watts at full
utilisation) so the engine can estimate average power from per-component
utilisation, plus static/idle power at the chip and node level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Union


class DType(str, Enum):
    FP4 = "fp4"
    INT4 = "int4"
    FP6 = "fp6"
    FP8 = "fp8"
    INT8 = "int8"
    FP16 = "fp16"
    BF16 = "bf16"
    TF32 = "tf32"
    FP32 = "fp32"

    @property
    def bytes(self) -> float:
        return _DTYPE_BYTES[self]


_DTYPE_BYTES: dict[DType, float] = {
    DType.FP4: 0.5,
    DType.INT4: 0.5,
    DType.FP6: 0.75,
    DType.FP8: 1.0,
    DType.INT8: 1.0,
    DType.FP16: 2.0,
    DType.BF16: 2.0,
    DType.TF32: 4.0,
    DType.FP32: 4.0,
}

# Narrowest-to-widest order used to resolve a compute rate when a chip has no
# native support for the requested dtype (e.g. FP4 weights on an H100 run
# through the FP8/BF16 pipes after dequantisation).
_WIDENING_ORDER: list[DType] = [
    DType.FP4,
    DType.INT4,
    DType.FP6,
    DType.FP8,
    DType.INT8,
    DType.FP16,
    DType.BF16,
    DType.TF32,
    DType.FP32,
]


@dataclass(frozen=True)
class Compute:
    """A pool of math units (tensor cores, Tensix FPUs, ...).

    peak_flops maps dtype -> dense peak FLOP/s for the whole pool.
    """

    name: str
    peak_flops: Mapping[DType, float]
    power_w: float = 0.0  # dynamic power at full utilisation

    def flops(self, dtype: DType) -> float:
        """Peak FLOP/s for `dtype`, widening to the nearest supported dtype."""
        start = _WIDENING_ORDER.index(dtype)
        for candidate in _WIDENING_ORDER[start:]:
            if candidate in self.peak_flops:
                return self.peak_flops[candidate]
        raise ValueError(f"{self.name}: no compute rate for {dtype} or any wider dtype")


@dataclass(frozen=True)
class Memory:
    """One level of storage: DRAM/HBM/LPDDR or on-chip SRAM."""

    name: str
    capacity_bytes: float
    bandwidth: float  # bytes/s, aggregate read+write
    power_w: float = 0.0  # dynamic power at full utilisation
    latency_s: float = 0.0


@dataclass(frozen=True)
class Link:
    """An interconnect stage: NoC, NVLink, PCIe, Ethernet.

    bandwidth is bytes/s per direction per endpoint (the number that enters
    ring-collective math).
    """

    name: str
    bandwidth: float
    latency_s: float = 0.0
    power_w: float = 0.0


Stage = Union[Memory, Link]


@dataclass(frozen=True)
class Chip:
    """A processor package: compute fed from DRAM through an on-chip path.

    on_chip_path lists the stages data crosses between DRAM and the math
    units, in order (e.g. NoC link, then core SRAM).  The effective DRAM
    streaming bandwidth is the minimum over DRAM and every path stage, which
    is the speed-of-light view of a block diagram like Tensix's
    DRAM -> NoC -> SRAM -> compute.
    """

    name: str
    compute: Compute
    dram: Memory
    on_chip_path: tuple[Stage, ...] = ()
    idle_power_w: float = 0.0

    @property
    def effective_dram_bandwidth(self) -> float:
        return min([self.dram.bandwidth] + [s.bandwidth for s in self.on_chip_path])

    @property
    def max_power_w(self) -> float:
        return (
            self.idle_power_w
            + self.compute.power_w
            + self.dram.power_w
            + sum(s.power_w for s in self.on_chip_path)
        )


class Topology(str, Enum):
    ALL_TO_ALL = "all-to-all"  # switched (NVSwitch) or full mesh
    RING = "ring"
    MESH_2D = "mesh-2d"


@dataclass(frozen=True)
class Node:
    """A server/board: n_chips identical chips plus an intra-node interconnect."""

    name: str
    chip: Chip
    n_chips: int
    interconnect: Link | None = None  # required when n_chips > 1
    topology: Topology = Topology.ALL_TO_ALL
    overhead_power_w: float = 0.0  # host CPU, fans, PSU losses, NICs...
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.n_chips > 1 and self.interconnect is None:
            raise ValueError(f"{self.name}: multi-chip node needs an interconnect")


@dataclass(frozen=True)
class System:
    """The whole machine: n_nodes identical nodes plus a network."""

    name: str
    node: Node
    n_nodes: int = 1
    network: Link | None = None  # required when n_nodes > 1
    extra_cost_usd: float = 0.0  # switches, cabling, ...
    description: str = ""

    def __post_init__(self) -> None:
        if self.n_nodes > 1 and self.network is None:
            raise ValueError(f"{self.name}: multi-node system needs a network link")

    @property
    def total_chips(self) -> int:
        return self.n_nodes * self.node.n_chips

    @property
    def cost_usd(self) -> float:
        return self.n_nodes * self.node.cost_usd + self.extra_cost_usd

    def link_for_group(self, group_size: int) -> Link | None:
        """Slowest link crossed by a communication group of `group_size` chips.

        Groups that fit inside one node use the intra-node interconnect;
        larger groups are bottlenecked by the network.
        """
        if group_size <= 1:
            return None
        if group_size > self.total_chips:
            raise ValueError(
                f"group of {group_size} exceeds {self.total_chips} chips in {self.name}"
            )
        if group_size <= self.node.n_chips:
            return self.node.interconnect
        return self.network
