"""Per-router 2D-mesh NoC preset (blackhole_p150_mesh / tt_quietbox_mesh).

Mirrors the -fine equivalence discipline: the mesh reproduces the lumped
BLACKHOLE_P150 chip EXACTLY under aggregation (the banks are the DRAM min-cut,
so the NoC mesh is non-binding), while the graph-DES walks the real 12x17
router grid with deterministic XY (column-then-row) routing, per-hop
store-and-forward, and emergent per-link contention.
"""

import pytest

from inferencesim.bridge import chip_from_graph, system_from_graph
from inferencesim.graph import Edge, Graph, Node, NodeKind
from inferencesim.graphdes import ChipModel, _edge_res
from inferencesim.hardware import DType
from inferencesim.ops import Op, OpKind
from inferencesim.presets import BLACKHOLE_P150, TT_QUIETBOX
from inferencesim.presets_fine import (
    _BH_MESH_COLS,
    _BH_MESH_LINK_BW,
    _BH_MESH_ROWS,
    _bh_mesh_layout,
    blackhole_p150_fine,
    blackhole_p150_mesh,
    tt_quietbox_fine,
    tt_quietbox_mesh,
)


def _memop(read: float, write: float = 0.0) -> Op:
    return Op("x", OpKind.COMPUTE, DType.FP16, "linear", 1, 0.0, read, write)


# ---- aggregation anchor: the mesh IS the lumped chip -------------------------


def test_mesh_aggregates_to_the_lumped_chip():
    """chip_from_graph(mesh) reproduces the lumped BLACKHOLE_P150 exactly: the
    8 banks (512 GB/s) are the DRAM->compute min-cut and the NoC mesh is
    non-binding, so every published aggregate is unchanged."""
    chip = chip_from_graph(blackhole_p150_mesh(),
                           idle_power_w=BLACKHOLE_P150.idle_power_w)
    assert chip.effective_dram_bandwidth == 512e9  # EXACT, not approx
    assert chip.dram.capacity_bytes == pytest.approx(BLACKHOLE_P150.dram.capacity_bytes)
    for d in (DType.FP8, DType.FP16):
        assert chip.compute.flops(d) == pytest.approx(BLACKHOLE_P150.compute.flops(d))
    assert chip.max_power_w == pytest.approx(BLACKHOLE_P150.max_power_w)


def test_mesh_max_flow_invariant_grouped_vs_expanded():
    """max_flow over banks->cores is 512 GB/s on both the grouped and the fully
    expanded mesh (the banks bind in both; the ~380-link mesh is non-binding)."""
    grouped = blackhole_p150_mesh()
    expanded = grouped.expand(deep=True)
    f_grouped = grouped.max_flow("gddr6-bank", "tensix-fpu")
    f_expanded = expanded.max_flow("gddr6-bank", "tensix-fpu")
    assert f_grouped == pytest.approx(512e9)
    assert f_expanded == pytest.approx(f_grouped)


def test_per_link_bandwidth_solves_the_bisection():
    """Documented arithmetic: the lumped 3.2 TB/s NoC is reproduced as the mesh
    minimum bisection.  A 12x17 mesh's min bisection crosses R=12 horizontal
    links, so 12 * B_link = 3.2 TB/s -> B_link = 266.7 GB/s, comfortably above
    the 512/12 = 42.7 GB/s threshold that keeps the banks the min cut."""
    assert _BH_MESH_LINK_BW == pytest.approx(3.2e12 / _BH_MESH_ROWS)
    assert _BH_MESH_ROWS * _BH_MESH_LINK_BW == pytest.approx(3.2e12)  # bisection
    assert _BH_MESH_LINK_BW > 512e9 / _BH_MESH_ROWS  # non-binding at 512 GB/s


# ---- classification + deterministic XY routing ------------------------------


def test_mesh_classifies_groups():
    m = ChipModel(blackhole_p150_mesh())
    assert m.dram_base == "gddr6-bank"
    assert len(m.dram_instances) == 8
    assert len(m.compute_instances) == 140
    assert len(m.sram_instances) == 140
    assert all(n.startswith("tensix-fpu[") for n in m.compute_instances)
    # 12x17 grid -> 204 router instances after expand
    assert sum(1 for n in m.graph.nodes if n.name.startswith("router[")) == 204


def _router_coord_path(m: ChipModel, bank: str, core: str) -> list[tuple[int, int]]:
    """Reconstruct the (row, col) router path a tile traverses, following the
    chain in traversal order (edge resource names are canonically sorted, so we
    track the current node to pick the far endpoint)."""
    C = _BH_MESH_COLS
    cur, seq = bank, []
    for el in m._chain(bank, core):
        if "~" in el.resource:
            a, b = el.resource.split("~")
            nxt = b if a == cur else a
            cur = nxt
            if nxt.startswith("router["):
                i = int(nxt[nxt.index("[") + 1:-1])
                seq.append((i // C, i % C))
    return seq


def test_mesh_routing_is_deterministic_xy():
    """Hand-picked bank->core chains are XY (column-first, then row) paths --
    matching Blackhole's NOC0 (East-then-South) -- for both a down-right and an
    up-left destination (direction-independent, unlike emergent BFS)."""
    m = ChipModel(blackhole_p150_mesh())
    tensix, banks = _bh_mesh_layout()
    C = _BH_MESH_COLS

    # bank[0] @ (1,0) -> core @ (2,3): X to column 3 in row 1, then Y down to row 2
    k = tensix.index((2, 3))
    assert _router_coord_path(m, "gddr6-bank[0]", f"tensix-fpu[{k}]") == [
        (1, 0), (1, 1), (1, 2), (1, 3), (2, 3),
    ]
    # bank[6] @ (7,9) -> core @ (3,3): X left to column 3 in row 7, then Y up
    assert banks[6] == (7, 9)
    k = tensix.index((3, 3))
    assert _router_coord_path(m, "gddr6-bank[6]", f"tensix-fpu[{k}]") == [
        (7, 9), (7, 8), (7, 7), (7, 6), (7, 5), (7, 4), (7, 3),
        (6, 3), (5, 3), (4, 3), (3, 3),
    ]


def test_mesh_chain_elements_are_store_and_forward():
    """The full element list for one tile: bank FIFO, per-hop link FIFOs along
    the XY path, injection port, L1 -- one bandwidth-constrained stage per hop
    (routers are ideal crossbars, contributing no element)."""
    m = ChipModel(blackhole_p150_mesh())
    tensix, _ = _bh_mesh_layout()
    C = _BH_MESH_COLS
    k = tensix.index((2, 3))  # core at (2,3)
    core = f"tensix-fpu[{k}]"
    got = [el.resource for el in m._chain("gddr6-bank[0]", core)]

    def r(rc):  # router name at (row, col)
        return f"router[{rc[0] * C + rc[1]}]"

    coords = [(1, 0), (1, 1), (1, 2), (1, 3), (2, 3)]
    expected = ["gddr6-bank[0]", _edge_res("gddr6-bank[0]", r((1, 0)))]
    for u, v in zip(coords, coords[1:]):
        expected.append(_edge_res(r(u), r(v)))
    expected.append(_edge_res(r((2, 3)), f"tensix-l1[{k}]"))
    expected.append(f"tensix-l1[{k}]")
    assert got == expected


# ---- degenerate meshes (small, full control) --------------------------------


def _mesh_chip(rows, cols, bank_pos, core_pos, link_bw, bank_bw=1e11,
               sram_cap=1e6, sram_bw=None, core_flops=1e12) -> Graph:
    """A small RxC router mesh: bank(s) -> routers -> sram(s) -> core(s).  Bank
    and sram edges are unconstrained (the node figures carry the caps, as in
    tests/test_graphdes._chip); the mesh links carry `link_bw` (None =
    unconstrained).  `mesh` meta drives the XY routing."""
    nodes = [
        Node("bank", NodeKind.MEMORY, count=len(bank_pos),
             capacity_bytes=1e12, bandwidth=bank_bw),
        Node("router", NodeKind.SWITCH, count=rows * cols, bandwidth=None),
        Node("sram", NodeKind.MEMORY, count=len(core_pos),
             capacity_bytes=sram_cap, bandwidth=sram_bw),
        Node("core", NodeKind.COMPUTE, count=len(core_pos),
             peak_flops={DType.FP16: core_flops}),
    ]
    edges = []
    for b, (r, c) in enumerate(bank_pos):
        edges.append(Edge(f"bank[{b}]", f"router[{r * cols + c}]"))
    for k, (r, c) in enumerate(core_pos):
        edges.append(Edge(f"sram[{k}]", f"router[{r * cols + c}]"))
    edges.append(Edge("sram", "core"))
    for r in range(rows):
        if cols > 1:
            base = r * cols
            edges.append(Edge(f"router[{base}:{base + cols - 1}]",
                              f"router[{base + 1}:{base + cols}]", bandwidth=link_bw))
    if rows > 1:
        edges.append(Edge(f"router[0:{(rows - 1) * cols}]",
                          f"router[{cols}:{rows * cols}]", bandwidth=link_bw))
    return Graph("mesh", nodes, edges,
                 meta={"mesh": {"rows": rows, "cols": cols, "router": "router"}})


def test_single_bank_unconstrained_mesh_is_bytes_over_bandwidth():
    """A mem-only op through one bank over a mesh whose links are unconstrained
    collapses to bytes/bank_bw EXACTLY (rel=1e-9) for any tile count -- the
    multi-hop routing adds no bandwidth stage, so only the bank FIFO binds."""
    B, R = 1e11, 1e7
    g = _mesh_chip(3, 3, [(0, 0)], [(2, 2), (2, 1), (1, 2)],
                   link_bw=None, bank_bw=B, sram_bw=None)
    s = ChipModel(g, tile_fill=0.5).op_wall(_memop(R))
    assert s.n_tiles > 1
    assert s.wall == pytest.approx(R / B, rel=1e-9)


def test_unconstrained_link_mesh_matches_fine_within_fill_bound():
    """The full mesh with unconstrained links reduces to the single-switch fine
    preset: both are bank-bound (bytes/512e9), and the walls agree within one
    tile's store-and-forward fill across the (bank, NoC, L1) stages."""
    R = 1e9
    mesh_n = ChipModel(blackhole_p150_mesh(link_bandwidth=None)).op_wall(_memop(R))
    fine = ChipModel(blackhole_p150_fine()).op_wall(_memop(R))
    assert mesh_n.n_tiles == fine.n_tiles
    read_per = R / fine.n_tiles
    # fill bound: one tile fully traversing the deepest chain (bank + NoC + L1)
    fill = read_per * (1 / 64e9 + 1 / (3.2e12) + 1 / (12e12 / 140))
    assert abs(mesh_n.wall - fine.wall) <= fill
    assert mesh_n.wall == pytest.approx(R / 512e9, rel=2e-2)  # bank-bound steady


def test_column_link_contention_raises_the_wall():
    """A single-column mesh funnels every tile through the (0,0)~(1,0) column
    link.  Narrowing that link below the bank bandwidth makes it the bottleneck,
    so the wall rises above the unconstrained (bank-bound) case -- per-hop
    contention the lumped switch averages away."""
    B, R = 1e11, 1e7
    cores = [(r, 0) for r in range(1, 6)]  # 5 cores stacked below the bank
    narrow = _mesh_chip(6, 1, [(0, 0)], cores, link_bw=2e10, bank_bw=B)
    wide = _mesh_chip(6, 1, [(0, 0)], cores, link_bw=None, bank_bw=B)
    w_narrow = ChipModel(narrow, tile_fill=0.5).op_wall(_memop(R)).wall
    w_wide = ChipModel(wide, tile_fill=0.5).op_wall(_memop(R)).wall
    assert w_wide == pytest.approx(R / B, rel=1e-9)      # bank-bound
    assert w_narrow > w_wide                              # the shared link binds
    # link-bound plus store-and-forward drain of the deepest core (5 hops)
    read_per = R / ChipModel(narrow, tile_fill=0.5).op_wall(_memop(R)).n_tiles
    assert R / 2e10 <= w_narrow <= R / 2e10 + 5 * read_per / 2e10


# ---- roofline + system + JSON equivalence -----------------------------------


def test_mesh_quietbox_simulates_like_lumped_under_roofline():
    """tt-quietbox-mesh aggregates and simulates identically to the lumped
    tt-quietbox under the roofline engine (flatten() collapses the router group,
    so the roofline path sees the same banks->compute min-cut)."""
    from inferencesim.presets import LLAMA_3_1_70B
    from inferencesim.simulate import simulate
    from inferencesim.workload import Deployment, Scenario

    mesh = system_from_graph(tt_quietbox_mesh())
    scen = Scenario(batch=32, prompt_len=2048, output_len=512)
    dep = Deployment(tp=2, pp=2, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    a = simulate(TT_QUIETBOX, LLAMA_3_1_70B, scen, dep)
    b = simulate(mesh, LLAMA_3_1_70B, scen, dep)
    assert b.ttft_s == pytest.approx(a.ttft_s, rel=1e-9)
    assert b.tpot_s == pytest.approx(a.tpot_s, rel=1e-9)
    assert b.system_power_w == pytest.approx(a.system_power_w, rel=1e-9)


def test_mesh_presets_json_round_trip():
    from inferencesim.presets_fine import GRAPH_PRESETS

    for key in ("blackhole-p150-mesh", "tt-quietbox-mesh"):
        g = GRAPH_PRESETS[key]()
        assert Graph.from_json(g.to_json()).to_dict() == g.to_dict(), key


def test_mesh_graph_des_refines_fine_des():
    """End-to-end on QuietBox: the per-router mesh runs under the graph-DES and
    is never optimistic against the single-switch fine DES (more serialization
    from per-hop store-and-forward), within a sane granularity margin."""
    from inferencesim.des import DESEngine
    from inferencesim.presets import LLAMA_3_1_70B
    from inferencesim.simulate import simulate
    from inferencesim.workload import Deployment, Scenario

    sys_mesh = system_from_graph(tt_quietbox_mesh())
    sys_fine = system_from_graph(tt_quietbox_fine())
    scen = Scenario(batch=16, prompt_len=2048, output_len=512)
    dep = Deployment(tp=2, pp=2, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    mesh = simulate(sys_mesh, LLAMA_3_1_70B, scen, dep,
                    engine=DESEngine(chip_graph=blackhole_p150_mesh()))
    fine = simulate(sys_fine, LLAMA_3_1_70B, scen, dep,
                    engine=DESEngine(chip_graph=blackhole_p150_fine()))
    assert mesh.tpot_s > 0
    assert mesh.tpot_s >= fine.tpot_s * (1 - 1e-9)   # never optimistic
    assert mesh.tpot_s == pytest.approx(fine.tpot_s, rel=0.5)
