import pytest

from inferencesim.engine import ring_allreduce_time
from inferencesim.hardware import Chip, Compute, DType, Link, Memory, Node, System
from inferencesim.presets import BLACKHOLE_P150, DGX_SPARK_X2, GB300_NVL72


def test_effective_bandwidth_is_min_over_path():
    # Blackhole: GDDR6 512 GB/s is the narrowest stage of DRAM->NoC->SRAM
    assert BLACKHOLE_P150.effective_dram_bandwidth == BLACKHOLE_P150.dram.bandwidth

    slow_noc = Link("noc", bandwidth=100e9)
    chip = Chip(
        name="t",
        compute=Compute("c", {DType.FP16: 1e12}),
        dram=Memory("d", 1e9, 500e9),
        on_chip_path=(slow_noc,),
    )
    assert chip.effective_dram_bandwidth == 100e9


def test_dtype_fallback_widens():
    c = Compute("c", {DType.FP8: 100.0, DType.BF16: 50.0})
    assert c.flops(DType.FP4) == 100.0  # no FP4 -> runs at FP8 rate
    assert c.flops(DType.FP8) == 100.0
    assert c.flops(DType.FP16) == 50.0
    with pytest.raises(ValueError):
        c.flops(DType.FP32)


def test_ring_allreduce():
    link = Link("l", bandwidth=100e9, latency_s=1e-6)
    assert ring_allreduce_time(1e9, 1, link) == 0.0
    # 4 ranks: 2*(3/4)*N/bw + 6*lat
    t = ring_allreduce_time(1e9, 4, link)
    assert t == pytest.approx(2 * 0.75 * 1e9 / 100e9 + 6e-6)


def test_group_link_selection():
    # inside a node -> interconnect; across nodes -> network
    assert GB300_NVL72.link_for_group(8) is GB300_NVL72.node.interconnect
    assert DGX_SPARK_X2.link_for_group(2) is DGX_SPARK_X2.network
    with pytest.raises(ValueError):
        GB300_NVL72.link_for_group(73)


def test_multichip_node_requires_interconnect():
    chip = Chip("t", Compute("c", {DType.FP16: 1e12}), Memory("d", 1e9, 1e9))
    with pytest.raises(ValueError):
        Node(name="bad", chip=chip, n_chips=2)
    with pytest.raises(ValueError):
        System(name="bad", node=Node("n", chip, 1), n_nodes=2)
