"""The replay document contract (inferencesim/replay.py).

These lock the shape build_replay() promises the viewer (and any future native
front-end): a versioned document whose every task resource resolves to a real
graph element, with a synthesised stage level always present and chip levels
only in graph mode.
"""

import re

import pytest

from inferencesim.bridge import chip_graph_of, system_from_graph
from inferencesim.des import DESEngine
from inferencesim.hardware import DType
from inferencesim.presets import GB300_NVL72, LLAMA_3_1_70B
from inferencesim.presets_fine import tt_quietbox_fine
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


@pytest.mark.parametrize("fixture", ["lumped_replay", "graph_replay"])
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


@pytest.mark.parametrize("fixture", ["lumped_replay", "graph_replay"])
def test_task_times_within_makespan(fixture, request):
    doc = request.getfixturevalue(fixture)
    for level in doc["levels"]:
        for _phase, _op, track in _tracks(level):
            ms = track["makespan"]
            for row in track["rows"]:
                start, end = row[1], row[2]
                assert 0.0 <= start <= end <= ms + 1e-9
                assert 0 <= row[3] < len(TASK_KINDS)


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
