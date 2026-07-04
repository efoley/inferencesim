"""Graph IR, bridge round-trips, and abstraction-level equivalence."""

import pytest

from inferencesim.bridge import (
    chip_from_graph,
    chip_to_graph,
    system_from_graph,
    system_to_graph,
)
from inferencesim.graph import Edge, Graph, Node, NodeKind
from inferencesim.hardware import DType
from inferencesim.presets import (
    BLACKHOLE_P150,
    HARDWARE,
    LLAMA_3_1_8B,
    LLAMA_3_1_70B,
    TT_QUIETBOX,
)
from inferencesim.presets_fine import blackhole_p150_fine, tt_quietbox_fine
from inferencesim.simulate import simulate
from inferencesim.workload import Deployment, Scenario


# ---- core graph mechanics ---------------------------------------------------


def _toy_chip() -> Graph:
    return Graph(
        name="toy",
        nodes=[
            Node("dram", NodeKind.MEMORY, capacity_bytes=1e9, bandwidth=100e9),
            Node("noc", NodeKind.SWITCH, bandwidth=50e9, latency_s=1e-6),
            Node("sram", NodeKind.MEMORY, capacity_bytes=1e6, bandwidth=500e9),
            Node("fpu", NodeKind.COMPUTE, peak_flops={DType.FP16: 1e12}),
        ],
        edges=[
            Edge("dram", "noc"),
            Edge("noc", "sram"),
            Edge("sram", "fpu"),
        ],
    )


def test_widest_path_min_over_nodes_and_edges():
    g = _toy_chip()
    p = g.widest_path("dram", "fpu")
    assert p.bandwidth == 50e9  # the NoC is the bottleneck
    assert p.nodes == ("dram", "noc", "sram", "fpu")
    assert p.latency_s == pytest.approx(1e-6)


def test_widest_path_prefers_wider_route():
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
    assert g.widest_path("a", "b").bandwidth == 200e9


def test_grouped_edge_capacity_from_pattern():
    g = Graph(
        name="banked",
        nodes=[
            Node("bank", NodeKind.MEMORY, count=8, capacity_bytes=4e9, bandwidth=64e9),
            Node("fpu", NodeKind.COMPUTE, peak_flops={DType.FP16: 1.0}),
        ],
        # INTERLEAVE default: one 64 GB/s link per bank instance
        edges=[Edge("bank", "fpu", bandwidth=64e9)],
    )
    p = g.widest_path("bank", "fpu")
    assert p.bandwidth == 8 * 64e9
    # edge count still means parallel links per pair
    g.edges[0].count = 2
    assert g.widest_path("bank", "fpu").bandwidth == pytest.approx(
        min(2 * 8 * 64e9, 8 * 64e9)
    )  # node cap now binds


def test_validation_catches_mistakes():
    g = _toy_chip()
    g.nodes.append(Node("dram", NodeKind.MEMORY, capacity_bytes=1))
    with pytest.raises(ValueError, match="duplicate"):
        g.validate()

    g2 = _toy_chip()
    g2.edges.append(Edge("dram", "nothere"))
    with pytest.raises(ValueError, match="does not exist"):
        g2.validate()

    g3 = _toy_chip()
    g3.nodes[3] = Node("fpu", NodeKind.COMPUTE)  # no dtypes
    with pytest.raises(ValueError, match="peak_flops"):
        g3.validate()


def test_flatten_rewires_ports():
    inner = _toy_chip()
    g = Graph(
        name="sys",
        nodes=[
            Node("chip", NodeKind.COMPOSITE, count=2, role="chip",
                 inner=inner, ports=("noc",)),
            Node("fabric", NodeKind.SWITCH),
        ],
        edges=[Edge("chip", "fabric", bandwidth=25e9)],
    )
    flat = g.flatten()
    names = {n.name for n in flat.nodes}
    assert "chip/dram" in names and "chip/fpu" in names
    rewired = [e for e in flat.edges if e.dst == "fabric"]
    assert rewired and rewired[0].src == "chip/noc"


def test_json_round_trip():
    from inferencesim.presets_fine import GRAPH_PRESETS

    for key, system in HARDWARE.items():
        g = system_to_graph(system)
        g2 = Graph.from_json(g.to_json())
        assert g2.to_dict() == g.to_dict(), key
    for key, factory in GRAPH_PRESETS.items():
        g = factory()
        assert Graph.from_json(g.to_json()).to_dict() == g.to_dict(), key


# ---- bridge round-trips -----------------------------------------------------


def test_chip_round_trip_preserves_roofline_quantities():
    g = chip_to_graph(BLACKHOLE_P150)
    chip = chip_from_graph(g, idle_power_w=BLACKHOLE_P150.idle_power_w)
    assert chip.effective_dram_bandwidth == BLACKHOLE_P150.effective_dram_bandwidth
    assert chip.dram.capacity_bytes == BLACKHOLE_P150.dram.capacity_bytes
    for d in (DType.FP8, DType.FP16):
        assert chip.compute.flops(d) == BLACKHOLE_P150.compute.flops(d)
    assert chip.max_power_w == pytest.approx(BLACKHOLE_P150.max_power_w)


@pytest.mark.parametrize("key", list(HARDWARE))
def test_system_round_trip_simulates_identically(key):
    system = HARDWARE[key]
    rebuilt = system_from_graph(system_to_graph(system))
    model = LLAMA_3_1_8B
    scen = Scenario(batch=4, prompt_len=1024, output_len=256)
    dep = Deployment(tp=1, weight_dtype=DType.FP8)
    a = simulate(system, model, scen, dep)
    b = simulate(rebuilt, model, scen, dep)
    assert b.ttft_s == pytest.approx(a.ttft_s, rel=1e-12)
    assert b.tpot_s == pytest.approx(a.tpot_s, rel=1e-12)
    assert b.system_power_w == pytest.approx(a.system_power_w, rel=1e-12)
    assert b.usd_per_m_output_tokens == pytest.approx(a.usd_per_m_output_tokens, rel=1e-12)


# ---- abstraction levels -----------------------------------------------------


def test_fine_blackhole_matches_lumped_aggregates():
    chip = chip_from_graph(blackhole_p150_fine())
    assert chip.effective_dram_bandwidth == pytest.approx(
        BLACKHOLE_P150.effective_dram_bandwidth
    )
    assert chip.dram.capacity_bytes == pytest.approx(BLACKHOLE_P150.dram.capacity_bytes)
    assert chip.compute.flops(DType.FP8) == pytest.approx(
        BLACKHOLE_P150.compute.flops(DType.FP8)
    )


def test_fine_quietbox_simulates_like_lumped_preset():
    """Same machine, modelled per-core vs lumped: identical roofline results."""
    fine = system_from_graph(tt_quietbox_fine())
    scen = Scenario(batch=32, prompt_len=2048, output_len=512)
    dep = Deployment(tp=2, pp=2, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    a = simulate(TT_QUIETBOX, LLAMA_3_1_70B, scen, dep)
    b = simulate(fine, LLAMA_3_1_70B, scen, dep)
    assert b.ttft_s == pytest.approx(a.ttft_s, rel=1e-9)
    assert b.tpot_s == pytest.approx(a.tpot_s, rel=1e-9)
    assert b.system_power_w == pytest.approx(a.system_power_w, rel=1e-9)


def test_fine_h100_matches_lumped_aggregates():
    """Per-SM H100 built from lumped scalars aggregates back to the exact
    lumped chip (a different topology: L2 crossbar, not a NoC)."""
    from inferencesim.presets import H100_SXM
    from inferencesim.presets_fine import h100_sxm_fine

    chip = chip_from_graph(h100_sxm_fine(), idle_power_w=H100_SXM.idle_power_w)
    assert chip.effective_dram_bandwidth == pytest.approx(
        H100_SXM.effective_dram_bandwidth
    )
    assert chip.dram.capacity_bytes == pytest.approx(H100_SXM.dram.capacity_bytes)
    for d in (DType.FP8, DType.FP16):
        assert chip.compute.flops(d) == pytest.approx(H100_SXM.compute.flops(d))
    assert chip.max_power_w == pytest.approx(H100_SXM.max_power_w)


def test_fine_dgx_h100_simulates_like_lumped_preset():
    """Same DGX H100, modelled per-SM vs lumped: identical roofline results."""
    from inferencesim.presets import DGX_H100
    from inferencesim.presets_fine import dgx_h100_fine

    fine = system_from_graph(dgx_h100_fine())
    scen = Scenario(batch=32, prompt_len=2048, output_len=512)
    dep = Deployment(tp=2, pp=2, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    a = simulate(DGX_H100, LLAMA_3_1_70B, scen, dep)
    b = simulate(fine, LLAMA_3_1_70B, scen, dep)
    assert b.ttft_s == pytest.approx(a.ttft_s, rel=1e-9)
    assert b.tpot_s == pytest.approx(a.tpot_s, rel=1e-9)
    assert b.system_power_w == pytest.approx(a.system_power_w, rel=1e-9)


def test_heterogeneous_chips_rejected():
    inner = _toy_chip()
    g = Graph(
        name="hetero",
        nodes=[
            Node("chip-a", NodeKind.COMPOSITE, role="chip", inner=inner, ports=("noc",)),
            Node("chip-b", NodeKind.COMPOSITE, role="chip", inner=inner, ports=("noc",)),
            Node("fabric", NodeKind.SWITCH),
        ],
        edges=[
            Edge("chip-a", "fabric", bandwidth=25e9),
            Edge("chip-b", "fabric", bandwidth=25e9),
        ],
    )
    with pytest.raises(ValueError, match="homogeneous"):
        system_from_graph(g)


def test_bare_chip_graph_becomes_single_chip_system():
    system = system_from_graph(blackhole_p150_fine())
    assert system.total_chips == 1
    assert system.node.chip.dram.capacity_bytes == pytest.approx(32e9)
