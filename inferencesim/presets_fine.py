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
from .presets import BLACKHOLE_P150, DGX_H100, H100_SXM, TT_QUIETBOX
from .units import GB, KiB, MB, TB, TERA, US

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
        # per-bank memory controllers into the NoC (default INTERLEAVE
        # pattern: one link per bank instance)
        Edge(src="gddr6-bank", dst="noc", bandwidth=512 * 1e9 / _N_DRAM_BANKS,
             name="dram controller"),
        # Tensix<->Tensix / Tensix<->DRAM traffic all rides the NoC; the
        # pattern gives each core its own injection port
        Edge(src="noc", dst="tensix-l1", bandwidth=12 * TB / _N_CORES,
             name="noc injection port"),
        # each matrix engine reads operands from its local L1 (one-to-one)
        Edge(src="tensix-l1", dst="tensix-fpu", name="l1 to fpu"),
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


_H100_N_SM = 132  # SMs on a full H100 SXM
_H100_N_HBM = 5  # HBM3 stacks


def h100_sxm_fine() -> Graph:
    """H100 SXM at SM granularity: 5 HBM3 stacks -> L2 crossbar -> 132 SMs,
    each with its own shared memory and tensor cores.  A different topology
    from the Tenstorrent presets (an all-to-all L2 crossbar, not a NoC ring)
    to exercise the graph-DES elsewhere.

    Every per-instance figure is the lumped H100_SXM scalar divided by the
    instance count, so chip_from_graph aggregates back to the exact same
    roofline chip (FLOPs, capacity, effective bandwidth, power split).  The
    lumped chip has no explicit on-chip path, so the L2 and shared-memory
    bandwidths are sized well above HBM: they never tighten the DRAM->SM
    min-cut, which stays at HBM bandwidth."""
    chip = H100_SXM
    # best-effort approximations (H100 whitepaper ballpark), both >> HBM so
    # the effective DRAM streaming bandwidth stays HBM-bound like the lumped
    # chip: L2 crossbar ~5.5 TB/s, aggregate shared-memory ~33 TB/s.
    l2_bw = 5.5 * TB
    smem_agg_bw = 33 * TB

    nodes = [
        Node(
            name="hbm",
            kind=NodeKind.MEMORY,
            count=_H100_N_HBM,
            capacity_bytes=chip.dram.capacity_bytes / _H100_N_HBM,
            bandwidth=chip.dram.bandwidth / _H100_N_HBM,
            dynamic_power_w=chip.dram.power_w / _H100_N_HBM,
        ),
        Node(
            name="l2",
            kind=NodeKind.SWITCH,
            bandwidth=l2_bw,  # shared crossbar (processor sharing in the DES)
            latency_s=0.1 * US,
        ),
        Node(
            name="sm-smem",
            kind=NodeKind.MEMORY,
            count=_H100_N_SM,
            capacity_bytes=228 * KiB,  # per-SM configurable shared memory
            bandwidth=smem_agg_bw / _H100_N_SM,
        ),
        Node(
            name="sm",
            kind=NodeKind.COMPUTE,
            count=_H100_N_SM,
            peak_flops={d: f / _H100_N_SM for d, f in chip.compute.peak_flops.items()},
            dynamic_power_w=chip.compute.power_w / _H100_N_SM,
        ),
    ]
    edges = [
        # per-stack HBM channels into the L2 (one link per stack)
        Edge(src="hbm", dst="l2", bandwidth=chip.dram.bandwidth / _H100_N_HBM,
             name="hbm channel"),
        # L2 fans out to each SM's shared memory (per-SM injection port)
        Edge(src="l2", dst="sm-smem", bandwidth=smem_agg_bw / _H100_N_SM,
             name="l2 to smem"),
        # each SM reads operands from its local shared memory (one-to-one)
        Edge(src="sm-smem", dst="sm", name="smem to sm"),
    ]
    return Graph(
        name="h100-sxm-fine",
        nodes=nodes,
        edges=edges,
        meta={"port": "l2"},
    )


def dgx_h100_fine() -> Graph:
    """The DGX H100 system graph with each H100 modelled per-SM: same machine
    (8 GPUs on NVLink4, costs, host overhead), one level deeper."""
    fine_chip = h100_sxm_fine()
    return swap_chip_model(system_to_graph(DGX_H100), fine_chip,
                           port=str(fine_chip.meta["port"]))


GRAPH_PRESETS = {
    "blackhole-p150-fine": blackhole_p150_fine,
    "tt-quietbox-fine": tt_quietbox_fine,
    "h100-sxm-fine": h100_sxm_fine,
    "dgx-h100-fine": dgx_h100_fine,
}
