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


# =============================================================================
# Blackhole p150 as a per-router 2D-mesh NoC
# =============================================================================
#
# `blackhole_p150_fine` above lumps the whole physical Network-on-Chip into ONE
# 3.2 TB/s processor-shared switch node.  Real Blackhole is a 2D grid of NoC
# routers: every tile (Tensix, GDDR6 controller, PCIe, Ethernet, ARC, ...) has
# its own router/NIU, and traffic hops router-to-router.  This preset models
# that grid so per-hop NoC contention becomes observable in the graph-DES
# (banks, links and routers each get their own resource) instead of being
# averaged into a single shared switch.
#
# Verified Blackhole NoC facts (cited in the docstring below):
#   * grid = 12 rows x 17 columns = 204 NoC tiles  [VERIFIED]
#   * 140 Tensix compute cores at interior positions; 8 GDDR6 controllers in
#     columns 0 and 9  [VERIFIED cols; row placement best-effort]
#   * two NoCs: NOC0 routes East-then-South (X-then-Y == row-first / XY),
#     NOC1 routes North-then-West; both are tori (wrap-around)  [VERIFIED]
#   * per-link ~60.9 bytes/cycle (~82 GB/s/dir at 1.35 GHz) per NoC  [VERIFIED]
#
# Modelling choices (best-effort, flagged in the docstring):
#   * a plain MESH (no wrap-around links) rather than the real torus -- the
#     two wrap columns/rows are a documented future refinement (they ~halve
#     worst-case hop count and double bisection);
#   * ONE mesh plane carrying both reads and writes (reverse-path write-back
#     shares the same link resource), where real Blackhole has NOC0 for reads
#     and NOC1 for writes -- so this model is *conservative* (read/write share
#     a link here, contend where silicon would not);
#   * per-link bandwidth DERIVED from the published 3.2 TB/s aggregate (below),
#     not from the ~82 GB/s/wire spec, so the mesh reproduces the lumped chip's
#     aggregates EXACTLY.
_BH_MESH_ROWS = 12
_BH_MESH_COLS = 17
# Per-link bandwidth, solved from the lumped aggregate.  See docstring "NoC
# per-link bandwidth" for the full arithmetic; in short: the lumped 3.2 TB/s
# NoC is reproduced as the mesh's MINIMUM BISECTION.  A 12x17 mesh's min
# bisection is a vertical cut crossing R=12 horizontal links (one per row), so
# 12 * B_link = 3.2 TB/s  ->  B_link = 3.2 TB/s / 12 = 266.7 GB/s per link.
_BH_MESH_LINK_BW = 3.2 * TB / _BH_MESH_ROWS


def _bh_mesh_layout() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Deterministic (row, col) grid positions for the 140 Tensix cores and the
    8 GDDR6 banks on the 12x17 grid.

    Convention (VERIFIED where noted, else a documented best-effort):
      * GDDR6 in columns 0 and 9 [VERIFIED]; four banks per column at rows
        {1,4,7,10} [best-effort row placement].
      * Tensix fill the interior: rows 1..10, columns {1..7, 10..16} -- i.e.
        every position that is not a DRAM column (0, 9), the central spine
        (column 8, an undefined/PCIe strip in silicon) or the top/bottom edge
        rows (0, 11).  That is exactly 10 rows x 14 columns = 140 cores, laid
        out row-major (core k at the k-th such position).
    The remaining 56 grid positions (edge/spine tiles: PCIe, Ethernet, ARC,
    L2CPU, ...) are still real NoC routers here -- bare pass-through nodes."""
    tensix_cols = list(range(1, 8)) + list(range(10, 17))  # 14 cols, skip 0/8/9
    tensix = [(r, c) for r in range(1, 11) for c in tensix_cols]  # 140, row-major
    banks = [(r, 0) for r in (1, 4, 7, 10)] + [(r, 9) for r in (1, 4, 7, 10)]
    return tensix, banks


def blackhole_p150_mesh(link_bandwidth: float | None = _BH_MESH_LINK_BW) -> Graph:
    """Blackhole p150 with the NoC modelled as a per-router 2D mesh.

    Same machine and same aggregates as `blackhole_p150_fine` and the lumped
    BLACKHOLE_P150 preset -- 8 GDDR6 banks (512 GB/s), 140 Tensix cores (774 TF
    FP8), 32 GB, identical power split -- but the single 3.2 TB/s NoC switch is
    replaced by a 12x17 grid of `router` SWITCH nodes wired to their four
    neighbours.  Each Tensix's L1 and matrix engine attach to their local
    router; the GDDR6 controllers attach at edge routers.  A tile therefore
    streams DRAM -> edge router -> (XY mesh hops) -> local router -> L1 ->
    matrix engine, store-and-forward per hop, so DRAM-bank, per-link and
    per-router contention all become emergent in the graph-DES.

    Store-and-forward semantics (physical): each hop is a bandwidth-constrained
    stage, so a tile's *latency* grows with hop count (~R+C hops worst case)
    while steady-state *throughput* stays paced by the binding stage (the GDDR6
    banks).  Wormhole-style flit pipelining (a tile streams across hops without
    fully landing at each router) would tighten the fill term; it is a future
    refinement, as is the second NoC plane and the torus wrap links.

    Aggregation is EXACT by construction (`chip_from_graph` reproduces the
    lumped chip):
      * effective_dram_bandwidth == 512e9.  The 8 banks cap at 64 GB/s each ->
        512 GB/s aggregate, which is the DRAM->compute min-cut (as in the fine
        preset), so the mesh must merely be non-binding at 512 GB/s.
      * capacity == 32 GB, FP8 FLOPs == 774 TF, max power identical.

    NoC per-link bandwidth (the "solve"): max_flow over banks->cores is the min
    cut.  The banks give a cut of 8*64 = 512 GB/s.  For the MESH not to bind
    below that, its minimum bisection must exceed 512 GB/s: a 12x17 mesh's min
    bisection is a vertical cut crossing R=12 horizontal links, i.e.
    12 * B_link, so B_link > 512/12 = 42.7 GB/s suffices for EXACT 512e9.  To
    also reproduce the lumped 3.2 TB/s aggregate NoC as the mesh bisection we
    set 12 * B_link = 3.2 TB/s  ->  B_link = 266.7 GB/s (>> 42.7, so the banks
    stay the min cut and effective_dram_bandwidth == 512e9 exactly).  The
    grouped `router` node collapses under `flatten()` (counts are not expanded
    there), so the roofline path sees banks->router->L1->compute == 512 GB/s
    identically; the mesh structure only manifests under `expand()` (the DES).

    Sources: grid dims, dual-NoC XY/YX routing and torus topology from
    Tenstorrent tt-npe (Blackhole implementation) and the community Blackhole
    architecture guide; 512 GB/s GDDR6, 774 TF FP8, 140 cores from the Blackhole
    p150 spec sheet (docs.tenstorrent.com).  See CALIBRATION.md sec 6.

    `link_bandwidth` defaults to the solved 266.7 GB/s; pass None for an
    unconstrained-mesh degenerate (used in tests to collapse to bytes/bank_bw).
    """
    chip = BLACKHOLE_P150
    fp8 = chip.compute.peak_flops[DType.FP8]
    fp16 = chip.compute.peak_flops[DType.FP16]
    R, C = _BH_MESH_ROWS, _BH_MESH_COLS
    tensix, banks = _bh_mesh_layout()
    bank_bw = 512 * 1e9 / _N_DRAM_BANKS
    inject_bw = 12 * TB / _N_CORES  # per-core L1 / NoC injection port

    nodes = [
        Node(
            name="gddr6-bank",
            kind=NodeKind.MEMORY,
            count=_N_DRAM_BANKS,
            capacity_bytes=32 * GB / _N_DRAM_BANKS,
            bandwidth=bank_bw,
            dynamic_power_w=chip.dram.power_w / _N_DRAM_BANKS,
        ),
        Node(
            name="router",
            kind=NodeKind.SWITCH,
            count=R * C,  # one NoC router per grid tile (140 Tensix + banks + edge)
            bandwidth=None,  # ideal crossbar; the mesh LINKS carry the bandwidth
        ),
        Node(
            name="tensix-l1",
            kind=NodeKind.MEMORY,
            count=_N_CORES,
            capacity_bytes=1.5 * MB,
            bandwidth=inject_bw,
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

    edges: list[Edge] = []
    # GDDR6 controllers inject at their edge routers (one 64 GB/s link per bank)
    for b, (r, c) in enumerate(banks):
        edges.append(Edge(src=f"gddr6-bank[{b}]", dst=f"router[{r * C + c}]",
                          bandwidth=bank_bw, name="gddr6 controller"))
    # each Tensix L1 attaches to its local router (per-core injection port)
    for k, (r, c) in enumerate(tensix):
        edges.append(Edge(src=f"tensix-l1[{k}]", dst=f"router[{r * C + c}]",
                          bandwidth=inject_bw, name="noc injection port"))
    # each matrix engine reads operands from its local L1 (one-to-one)
    edges.append(Edge(src="tensix-l1", dst="tensix-fpu", name="l1 to fpu"))
    # mesh neighbour links, wired with selectors (the first irregular-topology
    # workout at scale): one horizontal edge per row (no row wrap-around), and a
    # single strided edge for every vertical link.
    for r in range(R):
        base = r * C
        edges.append(Edge(src=f"router[{base}:{base + C - 1}]",
                          dst=f"router[{base + 1}:{base + C}]",
                          bandwidth=link_bandwidth, name="noc link (row)"))
    edges.append(Edge(src=f"router[0:{(R - 1) * C}]", dst=f"router[{C}:{R * C}]",
                      bandwidth=link_bandwidth, name="noc link (col)"))

    return Graph(
        name="blackhole-p150-mesh",
        nodes=nodes,
        edges=edges,
        meta={"port": "router",
              "mesh": {"rows": R, "cols": C, "router": "router"}},
    )


def tt_quietbox_mesh() -> Graph:
    """The QuietBox system graph with each p150 modelled as a per-router 2D
    mesh NoC: same machine as `tt_quietbox_fine`, one abstraction deeper on the
    NoC.  Outer structure (4 chips on an 800GbE ring, costs, host overhead) is
    inherited from the lumped preset."""
    mesh_chip = blackhole_p150_mesh()
    return swap_chip_model(system_to_graph(TT_QUIETBOX), mesh_chip,
                           port=str(mesh_chip.meta["port"]))


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
    "blackhole-p150-mesh": blackhole_p150_mesh,
    "tt-quietbox-fine": tt_quietbox_fine,
    "tt-quietbox-mesh": tt_quietbox_mesh,
    "h100-sxm-fine": h100_sxm_fine,
    "dgx-h100-fine": dgx_h100_fine,
}
