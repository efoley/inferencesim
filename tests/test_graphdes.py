"""Chip-graph op lowering (inferencesim.graphdes).

Following the DES validation philosophy, every fidelity feature has a
degenerate case that reduces to a closed form: a mem-bound op through one
bank is exactly bytes/bandwidth, a pure-compute op on one core is exactly
flops/rate, banks halve the wall, a shared NoC caps it, and mixed ops
overlap (wall < mem + compute).  The exact-collapse tests use rel=1e-9;
the pipeline fill/drain tests bound the wall within one tile's traversal.
"""

import pytest

from inferencesim.graph import Edge, Graph, Node, NodeKind
from inferencesim.graphdes import ChipModel
from inferencesim.hardware import DType
from inferencesim.ops import Op, OpKind
from inferencesim.presets_fine import blackhole_p150_fine, h100_sxm_fine


# ---- degenerate chips -------------------------------------------------------


def _chip(n_banks: int = 1, n_cores: int = 1, bank_bw: float = 1e11,
          sram_cap: float | None = 1e6, sram_bw: float | None = None,
          noc_bw: float | None = None, core_flops: float = 1e12) -> Graph:
    """bank(s) -> noc -> sram(s) -> core(s).  The bank node carries the DRAM
    bandwidth; noc/sram/edges are unconstrained unless a bandwidth is passed,
    so a single element binds in each test.  sram_cap drives tiling (None
    drops the SRAM entirely -> single tile)."""
    nodes = [
        Node("bank", NodeKind.MEMORY, count=n_banks,
             capacity_bytes=1e12, bandwidth=bank_bw),
        Node("noc", NodeKind.SWITCH, bandwidth=noc_bw),
        Node("core", NodeKind.COMPUTE, count=n_cores,
             peak_flops={DType.FP16: core_flops}),
    ]
    edges = [Edge("bank", "noc"), Edge("noc", "core")]
    if sram_cap is not None:
        nodes.insert(2, Node("sram", NodeKind.MEMORY, count=n_cores,
                             capacity_bytes=sram_cap, bandwidth=sram_bw))
        edges = [Edge("bank", "noc"), Edge("noc", "sram"), Edge("sram", "core")]
    return Graph(name="degenerate", nodes=nodes, edges=edges)


def _mem_op(read: float, write: float = 0.0) -> Op:
    return Op("x", OpKind.COMPUTE, DType.FP16, "linear", 1, 0.0, read, write)


def _mixed_op(read: float, flops: float, write: float = 0.0) -> Op:
    return Op("x", OpKind.COMPUTE, DType.FP16, "linear", 1, flops, read, write)


# ---- exact collapse ---------------------------------------------------------


def test_single_bank_mem_bound_is_bytes_over_bandwidth():
    """Mem-only op, one bank, everything else unconstrained: the tiles all
    serialise on the one bank FIFO, so wall == bytes/bandwidth for ANY tile
    count."""
    B, R = 1e11, 1e7
    m = ChipModel(_chip(bank_bw=B, sram_cap=1e6), tile_fill=0.5)
    s = m.op_wall(_mem_op(R))
    assert s.n_tiles > 1  # 1e7 / (1e6*0.5) = 20 tiles
    assert s.wall == pytest.approx(R / B, rel=1e-9)


def test_single_core_compute_bound_is_flops_over_rate():
    """Compute-only op on a single core: one tile, one core -> flops/rate,
    which equals flops/agg since there is one core."""
    rate, F = 1e12, 5e11
    m = ChipModel(_chip(core_flops=rate), tile_fill=0.5)
    s = m.op_wall(_mixed_op(read=0.0, flops=F))
    assert s.n_tiles == 1  # no reads -> single tile
    assert s.wall == pytest.approx(F / rate, rel=1e-9)


def test_two_banks_halve_a_mem_bound_op():
    """Two banks vs one, mem-bound, tiles a multiple of 2: the wall halves."""
    B, R = 1e11, 1e7  # 20 tiles, even
    one = ChipModel(_chip(1, 1, bank_bw=B), tile_fill=0.5).op_wall(_mem_op(R)).wall
    two = ChipModel(_chip(2, 2, bank_bw=B), tile_fill=0.5).op_wall(_mem_op(R)).wall
    assert two == pytest.approx(one / 2, rel=1e-9)


# ---- emergent overlap and fill/drain bounds ---------------------------------


def test_mixed_op_bounded_by_fill_and_drain():
    """One bank, one core, infinite-bandwidth SRAM: a double-buffered
    two-stage pipeline.  Wall is at least the bottleneck stream and at most
    that plus one tile of the other (fill + drain)."""
    B, R, F, rate = 1e11, 1e7, 8e7, 1e12
    m = ChipModel(_chip(bank_bw=B, core_flops=rate), tile_fill=0.5)
    s = m.op_wall(_mixed_op(R, F))
    mem_t, compute_t = R / B, F / rate
    read_per, compute_per = (R / s.n_tiles) / B, (F / s.n_tiles) / rate
    lo = max(mem_t, compute_t)
    assert lo <= s.wall <= lo + read_per + compute_per + 1e-15


def test_compute_dram_overlap_is_emergent():
    """The same mixed op runs strictly faster than a no-overlap model
    (mem + compute), because reads and math occupy different resources."""
    B, R, F, rate = 1e11, 1e7, 8e7, 1e12
    m = ChipModel(_chip(bank_bw=B, core_flops=rate), tile_fill=0.5)
    s = m.op_wall(_mixed_op(R, F))
    assert s.wall < R / B + F / rate


def test_shared_noc_caps_throughput():
    """A NoC switch with bandwidth NB between banks and cores: when NB is the
    bottleneck, all tiles share it and the wall is bytes/NB up to one tile's
    fill/drain -- the shared resource divides bandwidth among in-flight
    tiles."""
    B, NB, R = 1e11, 4e10, 1e7  # 2 banks give 2e11 > NB, so the NoC binds
    m = ChipModel(_chip(2, 2, bank_bw=B, noc_bw=NB), tile_fill=0.5)
    s = m.op_wall(_mem_op(R))
    lo = R / min(2 * B, NB)
    read_per = R / s.n_tiles
    first_tile = read_per * (1 / B + 1 / NB)  # one tile crossing bank then NoC
    assert lo <= s.wall <= lo + first_tile + 1e-15


# ---- tiling / capacity ------------------------------------------------------


def test_sram_capacity_bounds_tile_size():
    m = ChipModel(_chip(sram_cap=1e6), tile_fill=0.5)
    assert m.tile_bytes == pytest.approx(1e6 * 0.5)
    assert m.op_wall(_mem_op(1e7)).n_tiles > 1     # bigger than SRAM -> tiled
    assert m.op_wall(_mem_op(1e5)).n_tiles == 1    # fits one tile


def test_no_sram_group_is_a_single_tile():
    """A graph with no SRAM adjacent to compute has no capacity constraint:
    tiling is disabled and each op is one tile."""
    m = ChipModel(_chip(sram_cap=None), tile_fill=0.5)
    assert m.sram_capacity is None and m.tile_bytes is None
    assert m.op_wall(_mem_op(1e9)).n_tiles == 1


# ---- fine-chip integration --------------------------------------------------


def test_compute_op_distributes_over_all_cores():
    """A compute-bearing, read-light op spreads over all 140 cores rather
    than piling onto one: one tile per core, so its wall is the roofline
    compute time flops/agg, not 140x it.  Byte count sizes memory tiles; it
    does not partition compute.  (dram_read == 0 keeps the constrained read
    path out of it, so the collapse is exact.)"""
    m = ChipModel(blackhole_p150_fine())
    agg = 774e12  # FP8 peak over all 140 cores
    F = 1e12
    s = m.op_wall(Op("x", OpKind.COMPUTE, DType.FP8, "linear", 1, F, 0.0, 0.0))
    assert s.n_tiles == 140  # one tile per core
    assert s.wall == pytest.approx(F / agg, rel=1e-9)


def test_fine_chip_classifies_groups():
    m = ChipModel(blackhole_p150_fine())
    assert m.dram_base == "gddr6-bank"
    assert len(m.dram_instances) == 8
    assert len(m.compute_instances) == 140
    assert len(m.sram_instances) == 140
    assert all(n.startswith("tensix-fpu[") for n in m.compute_instances)
    assert all(n.startswith("tensix-l1[") for n in m.sram_instances)


def test_h100_fine_chip_classifies_groups():
    """A different topology (5 HBM stacks, an L2 crossbar, 132 SMs) — the
    5-bank/132-core round-robin is deliberately non-divisible."""
    m = ChipModel(h100_sxm_fine(), tile_fill=0.5)
    assert m.dram_base == "hbm"
    assert len(m.dram_instances) == 5
    assert len(m.compute_instances) == 132
    assert len(m.sram_instances) == 132
    assert m.tile_bytes == pytest.approx(228 * 1024 * 0.5)
    assert all(n.startswith("sm[") for n in m.compute_instances)
    assert all(n.startswith("sm-smem[") for n in m.sram_instances)


def test_h100_graph_des_refines_lumped_des():
    """End-to-end on the DGX H100 fine system: the graph-DES runs, TPOT is
    positive and never faster than the lumped stage-DES on the same
    aggregated system, staying within a modest tiling/granularity margin.
    The 5-bank / 132-SM round-robin does not divide evenly — nothing assumes
    it does."""
    from inferencesim.bridge import system_from_graph
    from inferencesim.des import DESEngine
    from inferencesim.presets import LLAMA_3_1_70B
    from inferencesim.presets_fine import dgx_h100_fine
    from inferencesim.simulate import simulate
    from inferencesim.workload import Deployment, Scenario

    system = system_from_graph(dgx_h100_fine())
    scen = Scenario(batch=16, prompt_len=2048, output_len=512)
    dep = Deployment(tp=2, pp=2, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    graph = simulate(system, LLAMA_3_1_70B, scen, dep,
                     engine=DESEngine(chip_graph=h100_sxm_fine()))
    lumped = simulate(system, LLAMA_3_1_70B, scen, dep, engine=DESEngine())
    assert graph.tpot_s > 0
    assert graph.tpot_s >= lumped.tpot_s * (1 - 1e-9)
    assert graph.tpot_s == pytest.approx(lumped.tpot_s, rel=0.25)
