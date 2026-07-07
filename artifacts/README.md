# artifacts/ — pedagogical chart pages

Self-contained HTML+SVG explainers generated from inferencesim's Python API.
Each page teaches five lessons about LLM inference systems with numbers computed
by the analytic roofline engine at **speed-of-light efficiency** (`sol`, the
default): perfect tiling, 100%-efficient bandwidth, bandwidth-optimal
collectives, zero kernel overhead. The shapes and crossovers are the lesson,
not the absolute values (see `CALIBRATION.md` for how far reality sits below
`sol`, and `--efficiency typical / auto` to derate).

| page | deployment | the five charts |
|---|---|---|
| `llama-3.1-70b-gb300-nvl72.html` | dense 70B, tp=8, fp4/fp8, 4k/1k, batch 64 | roofline (prefill vs decode) · TP scaling waterfall (tp 1→32, the KV-head wall) · KV anatomy (GQA/SWA/MLA + tp-vs-adp sharding) · batch economics ($/M tok vs tok/s/user) · MoE expert amortization |
| `deepseek-v3-gb300-nvl72.html` | 671B MoE+MLA, tp=1 ep=8 ("DEP8"), fp4/fp8, 4k/1k, batch 256 | roofline (with MoE ops) · EP scaling waterfall (ep 1→64; ep=1 doesn't fit) · MLA sharding (ep divides KV, tp cannot) · batch economics at DEP8 · MoE expert amortization |

## How they're built

1. Scripts in `gen/` call the library directly (`prefill_ops` / `decode_ops` /
   `simulate` / `kv_cache_bytes_per_chip` / `expected_active_experts`) and dump
   one JSON per chart. Run them with `uv run python gen/<script>.py` from the
   repo root; each script's docstring states its exact model/hardware/deployment.
2. The per-chart JSONs are condensed into one compact object and embedded in the
   page inside `<script id="data" type="application/json">…</script>`; all
   rendering is inline vanilla JS + SVG. To refresh a page after changing
   presets or engine math, re-run the scripts and replace that one tag's
   contents — the markup and chart code don't need to change unless axis
   domains do.

No external assets, fonts, or network access; pages render in light and dark
themes and every chart carries hover tooltips plus a `Data table` disclosure
with the exact values.

## Honesty notes baked into the pages

- All numbers are optimistic bounds (`sol`); footers say so.
- Allreduces are priced with the ring closed form regardless of fabric topology,
  so high-tp comm growth is a conservative bound on NVL72's switched fabric.
- MoE expert traffic uses expected distinct experts (no routing skew / EPLB).
- Preset spec sheets are best-effort approximations of public material.
