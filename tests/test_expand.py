"""Group expansion, edge patterns, selectors, and max-flow equivalence."""

import pytest

from inferencesim.bridge import chip_from_graph
from inferencesim.graph import Edge, EdgePattern, Graph, Node, NodeKind
from inferencesim.hardware import DType
from inferencesim.presets import BLACKHOLE_P150
from inferencesim.presets_fine import blackhole_p150_fine


def _banks_and_fpus() -> Graph:
    """The awkward case: 35 SRAM banks feeding 80 FPUs."""
    return Graph(
        name="mismatched",
        nodes=[
            Node("sram", NodeKind.MEMORY, count=35, capacity_bytes=1e6,
                 bandwidth=10e9),
            Node("fpu", NodeKind.COMPUTE, count=80,
                 peak_flops={DType.FP16: 1e12}),
        ],
        edges=[Edge("sram", "fpu", bandwidth=10e9)],  # interleave default
    )


def test_expand_materialises_instances():
    g = _banks_and_fpus().expand()
    names = {n.name for n in g.nodes}
    assert "sram[0]" in names and "sram[34]" in names
    assert "fpu[79]" in names
    assert all(n.count == 1 for n in g.nodes)
    # interleave: max(35, 80) = 80 concrete links, fpu[i] -> sram[i % 35]
    assert len(g.edges) == 80
    assert (g.edges[36].src, g.edges[36].dst) == ("sram[1]", "fpu[36]")
    g.validate()


def test_max_flow_invariant_under_expansion():
    grouped = _banks_and_fpus()
    expanded = grouped.expand()
    f_grouped = grouped.max_flow("sram", "fpu")
    f_expanded = expanded.max_flow("sram", "fpu")  # 'sram' names the group
    assert f_grouped == pytest.approx(35 * 10e9)
    assert f_expanded == pytest.approx(f_grouped)


def test_all_pattern_expands_to_product():
    g = Graph(
        name="xbar",
        nodes=[
            Node("a", NodeKind.MEMORY, count=3, capacity_bytes=1, bandwidth=1e9),
            Node("b", NodeKind.COMPUTE, count=2, peak_flops={DType.FP16: 1.0}),
        ],
        edges=[Edge("a", "b", bandwidth=1e9, pattern=EdgePattern.ALL)],
    )
    assert len(g.expand().edges) == 6


def test_selectors_wire_irregular_topologies():
    g = Graph(
        name="affinity",
        nodes=[
            Node("sram", NodeKind.MEMORY, count=35, capacity_bytes=1e6,
                 bandwidth=10e9),
            Node("fpu", NodeKind.COMPUTE, count=80,
                 peak_flops={DType.FP16: 1e12}),
        ],
        edges=[
            Edge("sram[0:8]", "fpu[40]", bandwidth=10e9),  # one FPU, 8 banks
        ],
    )
    g.validate()
    ex = g.expand()
    assert len(ex.edges) == 8
    assert all(e.dst == "fpu[40]" for e in ex.edges)
    assert {e.src for e in ex.edges} == {f"sram[{i}]" for i in range(8)}
    with pytest.raises(ValueError, match="out of range"):
        Graph(name="bad", nodes=g.nodes, edges=[Edge("sram[40]", "fpu[0]")]).validate()


def test_max_flow_credits_parallel_routes():
    g = Graph(
        name="two-routes",
        nodes=[
            Node("a", NodeKind.MEMORY, capacity_bytes=1, bandwidth=1000e9),
            Node("slow", NodeKind.SWITCH, bandwidth=10e9),
            Node("fast", NodeKind.SWITCH, bandwidth=200e9),
            Node("b", NodeKind.COMPUTE, peak_flops={DType.FP16: 1.0}),
        ],
        edges=[
            Edge("a", "slow"), Edge("slow", "b"),
            Edge("a", "fast"), Edge("fast", "b"),
        ],
    )
    # widest_path picks one route (200); max flow uses both (210)
    assert g.widest_path("a", "b").bandwidth == 200e9
    assert g.max_flow("a", "b") == pytest.approx(210e9)


def test_fine_blackhole_expansion_round_trip():
    """Grouped and fully expanded per-core Blackhole aggregate identically."""
    grouped = blackhole_p150_fine()
    expanded = grouped.expand()
    assert sum(1 for n in expanded.nodes if n.name.startswith("tensix-fpu[")) == 140
    a = chip_from_graph(grouped)
    b = chip_from_graph(expanded)
    assert b.effective_dram_bandwidth == pytest.approx(a.effective_dram_bandwidth)
    assert a.effective_dram_bandwidth == pytest.approx(
        BLACKHOLE_P150.effective_dram_bandwidth
    )
    assert b.dram.capacity_bytes == pytest.approx(a.dram.capacity_bytes)
    assert b.compute.flops(DType.FP8) == pytest.approx(a.compute.flops(DType.FP8))
    assert b.max_power_w == pytest.approx(a.max_power_w)


def test_expand_json_round_trip():
    from inferencesim.graph import Graph as G
    ex = _banks_and_fpus().expand()
    assert G.from_json(ex.to_json()).to_dict() == ex.to_dict()


# ---- per-instance heterogeneity (derate) ------------------------------------


def test_derate_is_transparent_by_default():
    """derate 1.0 everywhere aggregates exactly like an underived graph."""
    g = _banks_and_fpus()
    assert g.node("sram").effective_count == 35
    assert g.node("sram").enabled_count == 35
    assert g.node("fpu").agg_flops[DType.FP16] == pytest.approx(80 * 1e12)
    assert g.node("sram").agg_bandwidth == pytest.approx(35 * 10e9)


def test_harvested_die_aggregates_to_the_live_fraction():
    """A 132-of-140-core Blackhole aggregates to 132/140 of the FLOPs, with
    DRAM bandwidth and capacity untouched (compute-only harvesting)."""
    g = blackhole_p150_fine()
    g.derate_instances("tensix-fpu[132:140]", 0.0)
    g.validate()
    fpu = g.node("tensix-fpu")
    assert fpu.effective_count == 132
    assert fpu.enabled_count == 132
    base = chip_from_graph(blackhole_p150_fine())
    chip = chip_from_graph(g)
    assert chip.compute.flops(DType.FP8) == pytest.approx(
        base.compute.flops(DType.FP8) * 132 / 140
    )
    # harvesting cores leaves the DRAM path alone
    assert chip.effective_dram_bandwidth == pytest.approx(base.effective_dram_bandwidth)
    assert chip.dram.capacity_bytes == pytest.approx(base.dram.capacity_bytes)


def test_disabled_bank_drops_from_capacity_and_bandwidth():
    """A dead bank leaves *every* aggregate: 7/8 of the DRAM bandwidth and
    7/8 of the capacity (a disabled instance is excluded entirely, unlike a
    derated-but-live one which keeps full capacity)."""
    g = blackhole_p150_fine()
    g.derate_instances("gddr6-bank[7]", 0.0)
    g.validate()
    bank = g.node("gddr6-bank")
    assert bank.effective_count == 7
    assert bank.enabled_count == 7
    base = chip_from_graph(blackhole_p150_fine())
    chip = chip_from_graph(g)
    assert chip.effective_dram_bandwidth == pytest.approx(
        base.effective_dram_bandwidth * 7 / 8
    )
    assert chip.dram.capacity_bytes == pytest.approx(base.dram.capacity_bytes * 7 / 8)


def test_derated_bank_keeps_capacity_but_scales_bandwidth():
    """A half-derated (but live) bank keeps full capacity; only its rate-like
    bandwidth is scaled -- distinguishing the derate rule from disabling."""
    g = blackhole_p150_fine()
    g.derate_instances("gddr6-bank[7]", 0.5)
    base = chip_from_graph(blackhole_p150_fine())
    chip = chip_from_graph(g)
    assert chip.effective_dram_bandwidth == pytest.approx(
        base.effective_dram_bandwidth * 7.5 / 8
    )
    assert chip.dram.capacity_bytes == pytest.approx(base.dram.capacity_bytes)


def test_derate_instances_bakes_into_expanded_instances():
    """expand() turns instance_derates into each instance's plain derate."""
    g = blackhole_p150_fine()
    g.derate_instances("tensix-fpu[138:140]", 0.5)
    ex = g.expand()
    by_name = {n.name: n for n in ex.nodes}
    assert by_name["tensix-fpu[0]"].derate == 1.0
    assert by_name["tensix-fpu[138]"].derate == 0.5
    assert by_name["tensix-fpu[139]"].derate == 0.5
    assert all(not n.instance_derates for n in ex.nodes)


def test_max_flow_invariant_under_expansion_with_derates():
    """Grouped and expanded forms agree on max flow even with a dead bank."""
    grouped = blackhole_p150_fine()
    grouped.derate_instances("gddr6-bank[3]", 0.0)
    expanded = grouped.expand()
    f_grouped = grouped.max_flow("gddr6-bank", "tensix-fpu")
    f_expanded = expanded.max_flow("gddr6-bank", "tensix-fpu")
    assert f_expanded == pytest.approx(f_grouped)
    assert f_grouped == pytest.approx(512e9 * 7 / 8)


def test_derate_selector_convenience_and_bounds():
    """Out-of-range selectors raise; derate on a count-1 (or literal instance)
    node sets the node-level derate; ranges/values are validated."""
    g = _banks_and_fpus()
    with pytest.raises(ValueError, match="out of range"):
        g.derate_instances("fpu[80]", 0.0)
    with pytest.raises(ValueError, match="not in"):
        g.derate_instances("fpu[0]", 1.5)
    # count-1 node: node-level derate
    single = Graph(
        name="one",
        nodes=[
            Node("bank", NodeKind.MEMORY, capacity_bytes=1e6, bandwidth=1e11),
            Node("core", NodeKind.COMPUTE, peak_flops={DType.FP16: 1e12}),
        ],
        edges=[Edge("bank", "core", bandwidth=1e11)],
    )
    single.derate_instances("core", 0.5)
    assert single.node("core").derate == 0.5
    # a literal expanded instance name sets that instance's derate
    ex = _banks_and_fpus().expand()
    ex.derate_instances("fpu[3]", 0.0)
    assert ex.node("fpu[3]").derate == 0.0


def test_derate_json_round_trip_is_exact():
    g = blackhole_p150_fine()
    g.derate_instances("tensix-fpu[132:140]", 0.0)
    g.derate_instances("gddr6-bank[7]", 0.5)
    g.node("noc").derate = 0.9  # a node-level derate too
    assert Graph.from_json(g.to_json()).to_dict() == g.to_dict()
    # instance_derates keys survive as ints, not strings
    rebuilt = Graph.from_json(g.to_json())
    assert set(rebuilt.node("tensix-fpu").instance_derates) == set(range(132, 140))
