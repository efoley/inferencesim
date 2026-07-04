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
