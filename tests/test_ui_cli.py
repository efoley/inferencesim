"""`inferencesim ui` -- the CLI that renders the self-contained HTML viewer,
and the packaging that ships the template."""

import importlib.resources
import json
import re

import pytest

from inferencesim.cli import main


def _extract_replay(html: str) -> dict:
    """Pull the embedded replay JSON back out of a generated viewer file."""
    m = re.search(
        r'<script type="application/json" id="replay">(.*?)</script>',
        html, re.DOTALL,
    )
    assert m, "no embedded replay JSON block found"
    return json.loads(m.group(1))


def _run_ui(tmp_path, hardware, extra=()):
    out = tmp_path / "viewer.html"
    rc = main([
        "ui", "--hardware", hardware, "--model", "llama-3.1-70b",
        "--tp", "2", "--pp", "2", "--batch", "8",
        "--prompt", "256", "--output", "32",
        "--weight-dtype", "fp8", "--kv-dtype", "fp8",
        "--decode-rounds", "6",  # pin the measurement run for test speed
        "--no-open", "-o", str(out), *extra,
    ])
    assert rc == 0
    return out


def test_ui_writes_parseable_file_for_lumped_preset(tmp_path):
    out = _run_ui(tmp_path, "gb300-nvl72")
    html = out.read_text()
    assert "inferencesim-replay-v1" in html          # format marker in the file
    assert "<canvas" in html                          # it is the viewer scene
    doc = _extract_replay(html)                        # JSON block parses
    assert doc["format"] == "inferencesim-replay-v1"
    assert doc["meta"]["graph_mode"] is False
    assert [l["kind"] for l in doc["levels"]] == ["stage"]  # no chip level


def test_ui_writes_parseable_file_for_graph_preset(tmp_path):
    out = _run_ui(tmp_path, "tt-quietbox-fine")
    html = out.read_text()
    assert "inferencesim-replay-v1" in html
    doc = _extract_replay(html)
    assert doc["meta"]["graph_mode"] is True
    kinds = [l["kind"] for l in doc["levels"]]
    assert "stage" in kinds and "chip" in kinds       # chip level only here


def test_ui_no_open_does_not_launch_a_browser(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("inferencesim.cli.webbrowser.open", lambda *a, **k: calls.append(a))
    _run_ui(tmp_path, "gb300-nvl72")
    assert calls == []


def test_ui_opens_browser_without_no_open(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("inferencesim.cli.webbrowser.open", lambda *a, **k: calls.append(a))
    out = tmp_path / "v.html"
    rc = main([
        "ui", "--hardware", "gb300-nvl72", "--model", "llama-3.1-70b",
        "--tp", "1", "--batch", "4", "--prompt", "128", "--output", "16",
        "--decode-rounds", "6", "-o", str(out),
    ])
    assert rc == 0
    assert len(calls) == 1
    assert str(out) in calls[0][0]  # opened the file we wrote (a file:// URI)


def test_ui_rejects_unknown_model(tmp_path, capsys):
    rc = main(["ui", "--hardware", "gb300-nvl72", "--model", "nope", "--no-open",
               "-o", str(tmp_path / "v.html")])
    assert rc == 2


def test_ui_accepts_moe_skew_and_no_cp_prefill(tmp_path):
    """The ui subcommand mirrors run's --moe-skew (MoE hot-expert incast) and
    --no-cp-prefill plumbing into the same Deployment/ModelSpec the run uses."""
    out = tmp_path / "v.html"
    rc = main([
        "ui", "--hardware", "gb300-nvl72", "--model", "gpt-oss-120b",
        "--ep", "8", "--moe-skew", "0.5", "--no-cp-prefill",
        "--batch", "4", "--prompt", "128", "--output", "16",
        "--decode-rounds", "6", "--no-open", "-o", str(out),
    ])
    assert rc == 0
    doc = _extract_replay(out.read_text())
    assert doc["meta"]["ep"] == 8


def test_ui_moe_skew_rejected_on_dense_model(tmp_path):
    """--moe-skew is MoE-only, exactly as on `run`/`serve`."""
    with pytest.raises(SystemExit) as ei:
        main(["ui", "--hardware", "gb300-nvl72", "--model", "llama-3.1-70b",
              "--moe-skew", "0.5", "--no-open", "-o", str(tmp_path / "v.html")])
    assert ei.value.code == 2


# ---- packaging: the template ships and is loadable --------------------------


def test_viewer_template_ships_in_the_package():
    res = importlib.resources.files("inferencesim").joinpath("viewer.html")
    text = res.read_text(encoding="utf-8")
    assert "/*__REPLAY_JSON__*/" in text   # injection marker intact
    assert "<canvas" in text and "requestAnimationFrame" in text
