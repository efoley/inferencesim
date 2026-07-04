"""Hand-built fine-grained hardware graphs.

These demonstrate modelling the same machine at a different abstraction
level than the lumped spec-sheet presets: the Blackhole here is 8 GDDR6
banks feeding a NoC feeding 140 Tensix cores (each with its own L1 SRAM and
matrix engine), following the Metalium block diagram -- instead of one
aggregate DRAM/NoC/SRAM/compute chain.

Aggregated back through bridge.system_from_graph, the fine model produces
the same roofline numbers as the lumped preset (tested), while giving a
discrete-event engine real structure to put contention on.
"""

from __future__ import annotations

from .bridge import swap_chip_model, system_to_graph
from .graph import Edge, Graph, Node, NodeKind
from .hardware import DType
from .presets import BLACKHOLE_P150, TT_QUIETBOX
from .units import GB, MB, TB, TERA, US

_N_CORES = 140
_N_DRAM_BANKS = 8


def blackhole_p150_fine() -> Graph:
    """Blackhole p150 at Tensix-core granularity.

    Totals match the lumped BLACKHOLE_P150 preset exactly (same FLOPs,
    capacities, bandwidths and power split), just distributed over the
    real block structure."""
    chip = BLACKHOLE_P150
    fp8 = chip.compute.peak_flops[DType.FP8]
    fp16 = chip.compute.peak_flops[DType.FP16]

    nodes = [
        Node(
            name="gddr6-bank",
            kind=NodeKind.MEMORY,
            count=_N_DRAM_BANKS,
            capacity_bytes=32 * GB / _N_DRAM_BANKS,
            bandwidth=512 * 1e9 / _N_DRAM_BANKS,
            dynamic_power_w=chip.dram.power_w / _N_DRAM_BANKS,
        ),
        Node(
            name="noc",
            kind=NodeKind.SWITCH,
            bandwidth=3.2 * TB,  # aggregate injection bandwidth (approx)
            latency_s=0.2 * US,
        ),
        Node(
            name="tensix-l1",
            kind=NodeKind.MEMORY,
            count=_N_CORES,
            capacity_bytes=1.5 * MB,
            bandwidth=12 * TB / _N_CORES,  # ~64 B/cycle @ 1.35 GHz per core
            dynamic_power_w=20.0 / _N_CORES,
        ),
        Node(
            name="tensix-fpu",
            kind=NodeKind.COMPUTE,
            count=_N_CORES,
            peak_flops={
                DType.FP8: fp8 / _N_CORES,
                DType.FP16: fp16 / _N_CORES,
                DType.BF16: fp16 / _N_CORES,
            },
            dynamic_power_w=chip.compute.power_w / _N_CORES,
        ),
    ]
    edges = [
        # per-bank memory controllers into the NoC
        Edge(src="gddr6-bank", dst="noc", bandwidth=512 * 1e9 / _N_DRAM_BANKS,
             count=_N_DRAM_BANKS, name="dram controller"),
        # Tensix<->Tensix / Tensix<->DRAM traffic all rides the NoC; each
        # core's injection port:
        Edge(src="noc", dst="tensix-l1", bandwidth=12 * TB / _N_CORES,
             count=_N_CORES, name="noc injection port"),
        # matrix engine reads operands from its local L1
        Edge(src="tensix-l1", dst="tensix-fpu", count=_N_CORES,
             name="l1 to fpu"),
    ]
    return Graph(
        name="blackhole-p150-fine",
        nodes=nodes,
        edges=edges,
        meta={"port": "noc"},
    )


def tt_quietbox_fine() -> Graph:
    """The QuietBox system graph with each p150 modelled per-core: same
    machine, one level deeper.  Outer structure (4 chips on 800GbE, costs,
    host overhead) is inherited from the lumped preset."""
    fine_chip = blackhole_p150_fine()
    return swap_chip_model(system_to_graph(TT_QUIETBOX), fine_chip,
                           port=str(fine_chip.meta["port"]))


GRAPH_PRESETS = {
    "blackhole-p150-fine": blackhole_p150_fine,
    "tt-quietbox-fine": tt_quietbox_fine,
}
