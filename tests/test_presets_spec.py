"""Speculative low-cost architectures: registration, design invariants, runs."""

from __future__ import annotations

import pytest

from inferencesim.hardware import DType, Topology
from inferencesim.presets import HARDWARE
from inferencesim.presets_spec import (
    CXL_COMPUTE_TILE,
    LPDDR5X_TILE,
    LPDDR6_TILE,
    LPDDR_SWARM_64,
    LPDDR_SWARM_POD,
    SPEC_HARDWARE,
)
from inferencesim.presets import DEEPSEEK_V3, LLAMA_3_1_70B
from inferencesim.simulate import simulate
from inferencesim.units import GIGA, TB
from inferencesim.workload import Deployment, Scenario

SPEC_TILES = [LPDDR5X_TILE, LPDDR6_TILE, CXL_COMPUTE_TILE]


def test_all_registered_in_catalogue():
    for key, system in SPEC_HARDWARE.items():
        assert HARDWARE[key] is system


def test_no_hbm_anywhere():
    # The defining constraint of this file: not a byte of HBM.
    for tile in SPEC_TILES:
        names = [tile.dram.name] + [s.name for s in tile.on_chip_path]
        assert not any("hbm" in n.lower() for n in names), tile.name


def test_effective_bandwidth_is_the_dram_stack():
    # On-chip NoC/SRAM must be sized above DRAM so the memory stack (LPDDR or
    # the CXL links) stays the min-cut -- otherwise the tile is mis-modelled.
    assert LPDDR5X_TILE.effective_dram_bandwidth == LPDDR5X_TILE.dram.bandwidth
    assert LPDDR6_TILE.effective_dram_bandwidth == LPDDR6_TILE.dram.bandwidth
    assert CXL_COMPUTE_TILE.effective_dram_bandwidth == CXL_COMPUTE_TILE.dram.bandwidth


def test_lpddr5x_matches_proven_gb10_anchor():
    # 256-bit LPDDR5X-8533 -> 273 GB/s, the figure GB10/DGX Spark actually hits.
    assert LPDDR5X_TILE.dram.bandwidth == 273 * GIGA


def test_swarm_aggregate_bandwidth():
    # 64 tiles each at their LPDDR bandwidth -> the aggregate the whole bet rests
    # on.  ~17.5 TB/s from commodity LPDDR5X, no HBM.
    n = LPDDR_SWARM_64.node.n_chips
    agg = n * LPDDR5X_TILE.effective_dram_bandwidth
    assert agg == pytest.approx(64 * 273 * GIGA)
    assert agg > 17 * TB


def test_swarm_pod_uses_commodity_ethernet_between_boxes():
    pod = LPDDR_SWARM_POD
    # In-box collectives ride the fat low-latency fabric; crossing boxes drops to
    # commodity 400 GbE (50 GB/s/dir) -- an order of magnitude slower.
    in_box = pod.link_for_group(pod.node.n_chips)
    cross_box = pod.link_for_group(pod.total_chips)
    assert in_box is pod.node.interconnect
    assert cross_box is pod.network
    assert cross_box.bandwidth == 50 * GIGA
    assert cross_box.bandwidth < in_box.bandwidth / 3
    assert pod.total_chips == 256


def test_all_spec_systems_construct_and_report_positive_chips():
    for system in SPEC_HARDWARE.values():
        assert system.total_chips >= 1
        assert system.cost_usd > 0


def test_swarm_runs_and_produces_throughput():
    dep = Deployment(tp=32, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    scen = Scenario(batch=64, prompt_len=2048, output_len=256)
    r = simulate(LPDDR_SWARM_64, LLAMA_3_1_70B, scen, dep)
    assert r.output_tokens_per_s > 0
    assert r.tpot_s > 0
    assert r.usd_per_m_output_tokens > 0


def test_cxl_pool_fits_giant_moe_in_pooled_capacity():
    # The disaggregation win: a 671B-total MoE fits with room to spare because
    # the pooled DDR5 slice is huge (256 GB/tile), while only the 37B active
    # slice is streamed at CXL bandwidth.
    dep = Deployment(tp=8, ep=4, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    scen = Scenario(batch=64, prompt_len=2048, output_len=256)
    r = simulate(HARDWARE["cxl-moe-pod"], DEEPSEEK_V3, scen, dep)
    assert r.output_tokens_per_s > 0
    # Comfortably fits: weights+kv+act well under the 256 GB pooled slice.
    assert r.memory.fits
    assert r.memory.total < 256 * GIGA


def test_swarm_fabric_is_low_latency_switched():
    assert LPDDR_SWARM_64.node.topology is Topology.ALL_TO_ALL
    assert LPDDR_SWARM_64.node.interconnect.latency_s <= 0.5e-6
