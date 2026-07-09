"""The replay document contract (inferencesim/replay.py).

These lock the shape build_replay() promises the viewer (and any future native
front-end): a versioned document whose every task resource resolves to a real
graph element, with a synthesised stage level always present and chip levels
only in graph mode.
"""

import re
from dataclasses import replace

import pytest

from inferencesim.bridge import chip_graph_of, system_from_graph
from inferencesim.des import DESEngine
from inferencesim.hardware import DType
from inferencesim.presets import GB300_NVL72, GPT_OSS_120B, LLAMA_3_1_70B
from inferencesim.presets_fine import blackhole_p150_mesh, tt_quietbox_fine
from inferencesim.replay import FORMAT, TASK_KINDS, build_replay
from inferencesim.simulate import simulate
from inferencesim.workload import Deployment, Scenario


# ---- fixtures ---------------------------------------------------------------


@pytest.fixture(scope="module")
def lumped_replay():
    dep = Deployment(tp=2, pp=2, weight_dtype=DType.FP8)
    scen = Scenario(batch=16, prompt_len=512, output_len=64)
    engine = DESEngine(decode_rounds=6)  # pin measurement for test speed
    simulate(GB300_NVL72, LLAMA_3_1_70B, scen, dep, engine=engine)
    return build_replay(engine, GB300_NVL72, LLAMA_3_1_70B, scen, dep, None)


@pytest.fixture(scope="module")
def graph_replay():
    hw = tt_quietbox_fine()
    system = system_from_graph(hw)
    dep = Deployment(tp=2, pp=2, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    scen = Scenario(batch=8, prompt_len=512, output_len=64)
    engine = DESEngine(decode_rounds=6, chip_graph=chip_graph_of(hw))
    simulate(system, LLAMA_3_1_70B, scen, dep, engine=engine)
    return build_replay(engine, system, LLAMA_3_1_70B, scen, dep, hw)


@pytest.fixture(scope="module")
def moe_skew_replay():
    """A skewed MoE run: the dispatch/combine all-to-all incasts onto the hot
    owner's ingress port, so the stage level carries `s{s}.l{i}.in` resources
    the dense fixtures never exercise."""
    model = replace(GPT_OSS_120B, moe=replace(GPT_OSS_120B.moe, skew=0.6))
    dep = Deployment(tp=1, ep=8, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    scen = Scenario(batch=4, prompt_len=256, output_len=32)
    engine = DESEngine(decode_rounds=6)
    simulate(GB300_NVL72, model, scen, dep, engine=engine)
    return build_replay(engine, GB300_NVL72, model, scen, dep, None)


@pytest.fixture(scope="module")
def cp_replay():
    """A context-parallel prefill run (adp > 1): each layer pays a `cp_kv_ring`
    circulating the K/V blocks over the cp = adp groups."""
    dep = Deployment(tp=2, adp=4, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    scen = Scenario(batch=4, prompt_len=2048, output_len=32)
    engine = DESEngine(decode_rounds=6)
    simulate(GB300_NVL72, LLAMA_3_1_70B, scen, dep, engine=engine)
    return build_replay(engine, GB300_NVL72, LLAMA_3_1_70B, scen, dep, None)


@pytest.fixture(scope="module")
def ring_replay():
    """A RING interconnect (Tenstorrent node): the member level wires each
    stage's members into a cw/ccw ring instead of through a central switch hub."""
    hw = tt_quietbox_fine()
    system = system_from_graph(hw)
    dep = Deployment(tp=4, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    scen = Scenario(batch=4, prompt_len=256, output_len=32)
    engine = DESEngine(decode_rounds=6, chip_graph=chip_graph_of(hw))
    simulate(system, LLAMA_3_1_70B, scen, dep, engine=engine)
    return build_replay(engine, system, LLAMA_3_1_70B, scen, dep, hw)


@pytest.fixture(scope="module")
def mesh_replay():
    """The per-router 2-D mesh chip preset (12x17 router grid): a dense chip
    level whose generic layout must stay geometrically valid."""
    hw = blackhole_p150_mesh()
    system = system_from_graph(hw)
    dep = Deployment(tp=1, weight_dtype=DType.FP8, kv_dtype=DType.FP8)
    scen = Scenario(batch=4, prompt_len=256, output_len=32)
    engine = DESEngine(decode_rounds=6, chip_graph=chip_graph_of(hw))
    simulate(system, LLAMA_3_1_70B, scen, dep, engine=engine)
    return build_replay(engine, system, LLAMA_3_1_70B, scen, dep, hw)


# ---- helpers ----------------------------------------------------------------


def _node_index(graph):
    """name -> node dict, walking nested composites."""
    idx = {}

    def walk(g):
        for n in g["nodes"]:
            idx[n["name"]] = n
            if n.get("inner"):
                walk(n["inner"])

    walk(graph)
    return idx


def _tracks(level):
    """Yield (phase, op_or_None, track) for every task track of a level."""
    if level["kind"] == "chip":
        for phase, ops in level["phases"].items():
            for op, track in ops.items():
                yield phase, op, track
    else:
        for phase, track in level["phases"].items():
            yield phase, None, track


def _levels_by_kind(doc, kind):
    return [lvl for lvl in doc["levels"] if lvl["kind"] == kind]


# ---- format / meta ----------------------------------------------------------


def test_format_version_present(lumped_replay):
    assert lumped_replay["format"] == FORMAT == "inferencesim-replay-v1"
    meta = lumped_replay["meta"]
    for key in ("model", "hardware", "deployment", "scenario"):
        assert isinstance(meta[key], str) and meta[key]
    assert meta["task_kinds"] == TASK_KINDS


def test_meta_carries_header_numbers(lumped_replay):
    meta = lumped_replay["meta"]
    assert meta["tp"] == 2 and meta["pp"] == 2
    assert meta["batch"] == 16 and meta["prompt_len"] == 512
    assert meta["total_chips"] == GB300_NVL72.total_chips


# ---- resource_map resolves --------------------------------------------------


@pytest.mark.parametrize("fixture", ["lumped_replay", "graph_replay",
                                     "moe_skew_replay", "cp_replay", "ring_replay",
                                     "mesh_replay"])
def test_every_task_resource_resolves_to_an_existing_element(fixture, request):
    doc = request.getfixturevalue(fixture)
    for level in doc["levels"]:
        nodes = _node_index(level["graph"])
        n_edges = len(level["graph"]["edges"])
        rmap = level["resource_map"]
        for _phase, _op, track in _tracks(level):
            for row in track["rows"]:
                res = track["res"][row[0]]
                assert res in rmap, f"{res} not mapped in level {level['id']}"
                ref = rmap[res]
                if ref["kind"] == "node":
                    assert ref["id"] in nodes, f"node {ref['id']} missing"
                    if "instance" in ref:
                        node = nodes[ref["id"]]
                        assert 0 <= ref["instance"] < node.get("count", 1)
                else:
                    assert ref["kind"] == "edge"
                    assert 0 <= ref["id"] < n_edges


@pytest.mark.parametrize("fixture", ["lumped_replay", "graph_replay",
                                     "moe_skew_replay", "cp_replay", "ring_replay",
                                     "mesh_replay"])
def test_task_times_within_makespan(fixture, request):
    doc = request.getfixturevalue(fixture)
    for level in doc["levels"]:
        for _phase, _op, track in _tracks(level):
            ms = track["makespan"]
            for row in track["rows"]:
                start, end = row[1], row[2]
                assert 0.0 <= start <= end <= ms + 1e-9
                assert 0 <= row[3] < len(TASK_KINDS)


# ---- new comm families: all-to-all ingress + context-parallel ring ----------


def _stage_level(doc):
    return _levels_by_kind(doc, "stage")[0]


def test_ingress_ports_map_to_their_member_and_read_as_collective(moe_skew_replay):
    """A skewed all-to-all lands each message on the receiver's `.in` ingress
    port; those must map to their own fabric member (not member 0) and read as
    collective traffic, never compute -- the `_LINK_RE`/`_classify_stage` `.in`
    extension.  Without it the catch-all would fold every ingress onto member 0
    and mislabel it compute."""
    level = _stage_level(moe_skew_replay)
    rmap = level["resource_map"]
    collective = TASK_KINDS.index("collective")
    seen_ingress = 0
    for _phase, _op, track in _tracks(level):
        for row in track["rows"]:
            res = track["res"][row[0]]
            m = re.match(r"^s\d+\.l(\d+)\.in$", res)
            if not m:
                continue
            seen_ingress += 1
            ref = rmap[res]
            assert ref["kind"] == "node" and ref["id"].endswith(".fabric")
            assert ref["instance"] == int(m.group(1))  # its own member, not 0
            assert row[3] == collective
    assert seen_ingress > 0, "skewed MoE run emitted no ingress-port tasks"


def test_context_parallel_ring_tasks_present_and_collective(cp_replay):
    """CP prefill circulates the K/V blocks as a `cp_kv_ring`; its link
    occupancy rides the member egress ports and must read as collective (the
    ring barriers stay sync)."""
    level = _stage_level(cp_replay)
    collective = TASK_KINDS.index("collective")
    sync = TASK_KINDS.index("sync")
    ring_kinds = set()
    for _phase, _op, track in _tracks(level):
        for row in track["rows"]:
            if "cp_ring" in track["labels"][row[4]]:
                ring_kinds.add(row[3])
    assert collective in ring_kinds, "no cp_ring link occupancy classified collective"
    assert ring_kinds <= {collective, sync}, f"unexpected cp_ring kinds {ring_kinds}"


# ---- stage level always synthesised -----------------------------------------


def test_stage_level_synthesised_for_a_lumped_run(lumped_replay):
    stage_levels = _levels_by_kind(lumped_replay, "stage")
    assert len(stage_levels) == 1
    graph = stage_levels[0]["graph"]
    assert graph["format"] == "inferencesim-graph-v1"  # standard graph format
    # one composite box per pipeline stage, each nesting a compute unit
    composites = [n for n in graph["nodes"] if n["kind"] == "composite"]
    assert {n["name"] for n in composites} == {"stage0", "stage1"}
    for c in composites:
        inner = {n["name"]: n for n in c["inner"]["nodes"]}
        assert any(n["kind"] == "compute" for n in inner.values())
        assert any(n["role"] == "link" for n in inner.values())


def test_stage_level_present_in_graph_mode_too(graph_replay):
    assert len(_levels_by_kind(graph_replay, "stage")) == 1


# ---- member level: per-member fabric, emitted iff > 1 member ----------------


def _member_level(doc):
    return _levels_by_kind(doc, "member")[0]


def test_member_level_emitted_when_stage_has_multiple_members(lumped_replay):
    """tp*ep*adp > 1 -> a member level lays each stage's members out as separate
    nodes (a pooled compute unit + one node per member)."""
    members = _levels_by_kind(lumped_replay, "member")
    assert len(members) == 1
    lvl = members[0]
    assert lvl["kind"] == "member" and lvl["title"] == "Stage members"
    assert lvl["graph"]["format"] == "inferencesim-graph-v1"
    composites = [n for n in lvl["graph"]["nodes"] if n["kind"] == "composite"]
    assert {n["name"] for n in composites} == {"stage0", "stage1"}
    for c in composites:
        inner = {n["name"]: n for n in c["inner"]["nodes"]}
        assert any(n["role"] == "compute" for n in inner.values())  # pooled u{s}
        member_nodes = [n for n in inner.values() if n["role"] == "link"]
        assert len(member_nodes) == 2  # tp=2 members, individually laid out


def test_member_level_ordered_between_stage_and_chip(graph_replay):
    """The member level sits between the stage and chip levels."""
    kinds = [lvl["kind"] for lvl in graph_replay["levels"]]
    assert kinds == ["stage", "member", "chip"]


def test_member_level_absent_for_single_member(mesh_replay):
    """tp=ep=adp=1 -> a single member adds nothing, so no member level."""
    assert _levels_by_kind(mesh_replay, "member") == []


def test_member_links_map_to_their_own_member_node(moe_skew_replay):
    """Every `s{s}.l{i}.{out,in,cw,ccw}` resolves to member i's OWN node (a
    distinct graph node, not an instance of a shared fabric) and `u{s}` to the
    pooled compute node.  This is what makes per-member incast spatially legible."""
    lvl = _member_level(moe_skew_replay)
    nodes = _node_index(lvl["graph"])
    rmap = lvl["resource_map"]
    link_re = re.compile(r"^s(\d+)\.l(\d+)\.(out|in|cw|ccw)$")
    seen_link = seen_unit = 0
    for _phase, _op, track in _tracks(lvl):
        for row in track["rows"]:
            res = track["res"][row[0]]
            m = link_re.match(res)
            if m:
                seen_link += 1
                assert rmap[res] == {"kind": "node",
                                     "id": f"s{m.group(1)}.m{m.group(2)}"}
                assert rmap[res]["id"] in nodes
            elif re.match(r"^u\d+$", res):
                seen_unit += 1
                assert rmap[res] == {"kind": "node", "id": res}
    assert seen_link > 0 and seen_unit > 0


def test_member_nodes_carry_coordinate_decomposition(moe_skew_replay):
    """Each member node stores its tp/ep/adp decomposition + a short label in
    meta; member 0 (the lowest, most-popular expert block) is flagged hot."""
    lvl = _member_level(moe_skew_replay)
    nodes = _node_index(lvl["graph"])
    m0, m3 = nodes["s0.m0"], nodes["s0.m3"]
    assert (m0["meta"]["member"], m0["meta"]["ep"]) == (0, 0) and m0["meta"]["hot"]
    assert m3["meta"]["ep"] == 3 and m3["meta"]["label"] == "ep3"
    assert not m3["meta"]["hot"]
    assert "pool" in nodes["u0"]["meta"]["desc"].lower()  # honest: compute pooled


def test_member_fabric_is_switched_hub_on_a_switched_fabric(moe_skew_replay):
    """A switched (all-to-all) fabric -> a central hub with member<->hub edges."""
    g = _member_level(moe_skew_replay)["graph"]
    assert g["meta"]["topology"] == "all-to-all"
    inner_names = {n["name"] for c in g["nodes"] if c.get("inner")
                   for n in c["inner"]["nodes"]}
    assert "s0.hub" in inner_names
    assert {"egress", "ingress"} <= {e["name"] for e in g["edges"]}


def test_member_fabric_is_ring_on_a_ring_node(ring_replay):
    """A RING interconnect -> members wired cw/ccw member-to-member, no hub."""
    g = _member_level(ring_replay)["graph"]
    assert g["meta"]["topology"] == "ring"
    inner_names = {n["name"] for c in g["nodes"] if c.get("inner")
                   for n in c["inner"]["nodes"]}
    assert not any(n.endswith(".hub") for n in inner_names)
    member_nodes = [n for n in inner_names if re.match(r"^s\d+\.m\d+$", n)]
    assert len(member_nodes) == 4  # tp=4 members
    assert {"cw", "ccw"} <= {e["name"] for e in g["edges"]}


# ---- chip levels iff graph mode ---------------------------------------------


def test_chip_level_absent_for_lumped(lumped_replay):
    assert _levels_by_kind(lumped_replay, "chip") == []
    assert lumped_replay["meta"]["graph_mode"] is False


def test_chip_level_present_for_graph_mode(graph_replay):
    chips = _levels_by_kind(graph_replay, "chip")
    assert len(chips) == 1
    assert graph_replay["meta"]["graph_mode"] is True
    chip = chips[0]
    # graph is the *expanded* chip: instance node ids like gddr6-bank[3]
    names = _node_index(chip["graph"])
    assert any(re.search(r"\[\d+\]$", n) for n in names)
    # one selectable track per distinct op, present in both phases
    assert chip["ops"] and "ffn" in chip["ops"]
    for phase in ("prefill", "decode"):
        assert set(chip["phases"][phase]) == set(chip["ops"])


# ---- sync tasks flagged -----------------------------------------------------


def test_sync_tasks_flagged_as_fences(lumped_replay):
    sync = TASK_KINDS.index("sync")
    stage = _levels_by_kind(lumped_replay, "stage")[0]
    track = stage["phases"]["decode"]
    sync_rows = [r for r in track["rows"] if r[3] == sync]
    assert sync_rows, "expected barrier/propagation sync tasks in the stage level"
    for r in sync_rows:
        res = track["res"][r[0]]
        assert ".bar" in res or ".prop" in res  # a real sync resource


# ---- decode window cap ------------------------------------------------------


def test_decode_window_documented_in_meta(lumped_replay):
    win = lumped_replay["meta"]["decode_window"]
    assert win["window_rounds"] >= 1
    assert win["warmup_rounds"] >= 1
    assert "note" in win and win["note"]


def test_decode_window_honoured(lumped_replay):
    """Only the steady-state window survives: the kept decode rounds are exactly
    `window_rounds` distinct rounds, all past the warmup fill."""
    win = lumped_replay["meta"]["decode_window"]
    stage = _levels_by_kind(lumped_replay, "stage")[0]
    track = stage["phases"]["decode"]
    rounds = set()
    for r in track["rows"]:
        m = re.match(r"^r(\d+)", track["labels"][r[4]])
        if m:
            rounds.add(int(m.group(1)))
    assert rounds, "decode labels should carry round indices"
    assert min(rounds) >= win["warmup_rounds"]
    assert len(rounds) == win["window_rounds"]


# ---- bytes-per-tile on chip tasks -------------------------------------------


def test_bytes_per_tile_present_on_chip_transfers(graph_replay):
    read = TASK_KINDS.index("read")
    write = TASK_KINDS.index("writeback")
    compute = TASK_KINDS.index("compute")
    chip = _levels_by_kind(graph_replay, "chip")[0]
    track = chip["phases"]["decode"]["ffn"]  # a weight-streaming op
    reads = [r for r in track["rows"] if r[3] == read]
    writes = [r for r in track["rows"] if r[3] == write]
    computes = [r for r in track["rows"] if r[3] == compute]
    assert reads and all(r[5] > 0 for r in reads)      # bytes/tile on reads
    assert writes and all(r[5] > 0 for r in writes)    # ... and writebacks
    assert computes and all(r[5] == 0 for r in computes)  # compute carries none


def test_downsample_cap_flag_is_honest(graph_replay):
    """A capped track reports the pre-cap count and keeps within the cap."""
    for level in graph_replay["levels"]:
        for _phase, _op, track in _tracks(level):
            if track["capped"]:
                assert len(track["rows"]) <= track["n"]
            else:
                assert len(track["rows"]) == track["n"]
