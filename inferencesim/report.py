"""Plain-text rendering of a simulation Report."""

from __future__ import annotations

from .engine import Phase
from .simulate import Report
from .units import fmt_bytes, fmt_si, fmt_time


def _phase_breakdown(phase: Phase) -> str:
    total = phase.total_time
    if total <= 0:
        return "  (empty)"
    times = sorted(phase.category_times().items(), key=lambda kv: -kv[1])
    bounds = phase.category_bounds()
    parts = [
        f"{cat} {100 * t / total:.0f}% ({bounds[cat]}-bound)"
        for cat, t in times
        if t / total >= 0.005
    ]
    return ", ".join(parts)


def format_report(r: Report) -> str:
    s, m, sc, d = r.system, r.model, r.scenario, r.deployment
    chip = s.node.chip
    lines: list[str] = []
    add = lines.append

    add("=" * 72)
    add(f"inferencesim  |  {s.name}  x  {m.name}")
    add("=" * 72)
    add(f"Hardware     : {s.n_nodes} node(s) x {s.node.n_chips} x {chip.name} "
        f"({s.total_chips} chips)")
    add(f"               chip: {fmt_si(chip.compute.flops(d.weight_dtype), 'FLOP/s')} "
        f"@{d.weight_dtype.value}, DRAM {fmt_bytes(chip.dram.capacity_bytes)} "
        f"@ {fmt_si(chip.effective_dram_bandwidth, 'B/s')} effective")
    add(f"Model        : {m.name}  ({m.total_params / 1e9:.1f}B params"
        + (f", {m.active_params / 1e9:.1f}B active" if m.moe else "")
        + f"), weights {d.weight_dtype.value}, kv {d.kv_dtype.value}")
    add(f"Parallelism  : TP={d.tp}  PP={d.pp}  EP={d.ep}  DP={r.dp}"
        f"  ({d.replica_chips} chips/replica)"
        + ("  (comm overlapped)" if d.overlap_comm else "")
        + (f"  ({r.idle_chips} chips idle)" if r.idle_chips else ""))
    add(f"Scenario     : batch/replica={sc.batch}, prompt={sc.prompt_len}, "
        f"output={sc.output_len}")
    mem = r.memory
    add(f"Memory/chip  : weights {fmt_bytes(mem.weights)} + kv {fmt_bytes(mem.kv_cache)} "
        f"+ act {fmt_bytes(mem.activations)} = {fmt_bytes(mem.total)} "
        f"/ {fmt_bytes(mem.capacity)}" + ("" if mem.fits else "  ** DOES NOT FIT **"))
    add("-" * 72)
    add(f"TTFT         : {fmt_time(r.ttft_s)}  (prefill {sc.prompt_len} tokens, 1 request)")
    add(f"  breakdown  : {_phase_breakdown(r.prefill)}")
    add(f"TPOT         : {fmt_time(r.tpot_s)} @ mean ctx {sc.mean_context:.0f}  "
        f"-> {1.0 / r.tpot_s:.1f} tok/s per request" if r.tpot_s > 0 else "TPOT         : n/a")
    add(f"  breakdown  : {_phase_breakdown(r.decode)}")
    if r.resource_util:
        for pname in ("prefill", "decode"):
            util = r.resource_util.get(pname)
            if util:
                ranked = sorted(util.items(), key=lambda kv: -kv[1])
                body = "  ".join(f"{res} {100 * frac:.0f}%" for res, frac in ranked)
                add(f"  {pname} resource util: {body}")
    add("-" * 72)
    add(f"Throughput   : {fmt_si(r.output_tokens_per_s, 'tok/s')} output "
        f"({fmt_si(r.input_tokens_per_s, 'tok/s')} input, "
        f"{r.requests_per_s:.2f} req/s)")
    add(f"  decode-only ceiling: {fmt_si(r.decode_only_tokens_per_s, 'tok/s')}")
    add(f"Power        : {fmt_si(r.system_power_w, 'W')} avg  ->  "
        f"{r.joules_per_output_token:.2f} J per output token")
    add(f"Cost         : ${r.usd_per_m_output_tokens:.3f} / M output tokens  "
        f"(${r.usd_per_m_total_tokens:.3f} / M total; "
        f"{100 * r.capex_share:.0f}% capex, {100 * (1 - r.capex_share):.0f}% power)")
    for w in r.warnings:
        add(f"WARNING      : {w}")
    add("=" * 72)
    return "\n".join(lines)
