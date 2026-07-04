"""Command-line interface.

    inferencesim list
    inferencesim run --hardware gb300-nvl72 --model llama-3.1-70b \
        --tp 8 --batch 64 --prompt 4096 --output 1024 --weight-dtype fp8
    inferencesim graph --hardware tt-quietbox-fine [--json]
    inferencesim run --graph my-machine.json --model llama-3.1-8b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .bridge import system_from_graph, system_to_graph
from .graph import Graph
from .hardware import DType, System
from .presets import HARDWARE, MODELS
from .presets_fine import GRAPH_PRESETS
from .report import format_report
from .simulate import CostModel, simulate
from .units import fmt_bytes, fmt_si
from .workload import Deployment, Scenario


def _resolve_graph(key: str) -> Graph | None:
    if key in GRAPH_PRESETS:
        return GRAPH_PRESETS[key]()
    if key in HARDWARE:
        return system_to_graph(HARDWARE[key])
    return None


def _resolve_system(args: argparse.Namespace) -> System | None:
    if getattr(args, "graph", None):
        g = Graph.from_json(Path(args.graph).read_text())
        return system_from_graph(g)
    if args.hardware in HARDWARE:
        return HARDWARE[args.hardware]
    if args.hardware in GRAPH_PRESETS:
        return system_from_graph(GRAPH_PRESETS[args.hardware]())
    return None


def _cmd_list(_: argparse.Namespace) -> int:
    print("Hardware presets:")
    for key, sys_ in HARDWARE.items():
        chip = sys_.node.chip
        print(f"  {key:16s} {sys_.total_chips:>3d}x {chip.name:26s} "
              f"DRAM {fmt_bytes(chip.dram.capacity_bytes):>7s}/chip "
              f"@ {fmt_si(chip.effective_dram_bandwidth, 'B/s')}")
    print("\nHardware graph presets (fine-grained):")
    for key in GRAPH_PRESETS:
        print(f"  {key}")
    print("\nModel presets:")
    for key, m in MODELS.items():
        extra = f", {m.active_params / 1e9:.1f}B active" if m.moe else ""
        print(f"  {key:16s} {m.total_params / 1e9:6.1f}B params{extra}")
    return 0


def _cmd_graph(args: argparse.Namespace) -> int:
    g = _resolve_graph(args.hardware)
    if g is None:
        print(f"unknown hardware '{args.hardware}' (try: inferencesim list)",
              file=sys.stderr)
        return 2
    if args.flat:
        g = g.flatten()
    if args.json:
        print(g.to_json())
    else:
        print(g.describe())
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if args.model not in MODELS:
        print(f"unknown model '{args.model}' (try: inferencesim list)", file=sys.stderr)
        return 2
    if not args.graph and not args.hardware:
        print("pass --hardware KEY or --graph FILE", file=sys.stderr)
        return 2
    system = _resolve_system(args)
    if system is None:
        print(f"unknown hardware '{args.hardware}' (try: inferencesim list)",
              file=sys.stderr)
        return 2
    model = MODELS[args.model]
    dep = Deployment(
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        weight_dtype=DType(args.weight_dtype),
        kv_dtype=DType(args.kv_dtype),
        act_dtype=DType(args.act_dtype),
        overlap_comm=args.overlap_comm,
    )
    cost = CostModel(
        amortization_years=args.amortization_years,
        electricity_usd_per_kwh=args.kwh_price,
        pue=args.pue,
    )

    batches = [int(b) for b in args.batch.split(",")]
    for i, batch in enumerate(batches):
        scen = Scenario(batch=batch, prompt_len=args.prompt, output_len=args.output)
        report = simulate(system, model, scen, dep, cost)
        if i:
            print()
        print(format_report(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="inferencesim",
                                description="LLM inference factory simulator")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list hardware and model presets").set_defaults(fn=_cmd_list)

    gr = sub.add_parser("graph", help="show a hardware description as a graph")
    gr.add_argument("--hardware", required=True,
                    help="hardware preset or graph preset key")
    gr.add_argument("--json", action="store_true", help="emit JSON instead of a tree")
    gr.add_argument("--flat", action="store_true", help="flatten nesting first")
    gr.set_defaults(fn=_cmd_graph)

    run = sub.add_parser("run", help="simulate a serving scenario")
    run.add_argument("--hardware", help="hardware preset or graph preset key")
    run.add_argument("--graph", help="path to a hardware graph JSON file")
    run.add_argument("--model", required=True, help="model preset key")
    run.add_argument("--tp", type=int, default=1, help="tensor-parallel degree")
    run.add_argument("--pp", type=int, default=1, help="pipeline-parallel stages")
    run.add_argument("--ep", type=int, default=1,
                     help="expert-parallel groups (MoE models only)")
    run.add_argument("--batch", default="32",
                     help="concurrent sequences per replica (comma list sweeps)")
    run.add_argument("--prompt", type=int, default=2048, help="prompt tokens per request")
    run.add_argument("--output", type=int, default=512, help="output tokens per request")
    run.add_argument("--weight-dtype", default="fp8",
                     choices=[d.value for d in DType])
    run.add_argument("--kv-dtype", default="bf16", choices=[d.value for d in DType])
    run.add_argument("--act-dtype", default="bf16", choices=[d.value for d in DType])
    run.add_argument("--overlap-comm", action="store_true",
                     help="assume TP collectives fully overlap with compute")
    run.add_argument("--amortization-years", type=float, default=4.0)
    run.add_argument("--kwh-price", type=float, default=0.12, help="USD per kWh")
    run.add_argument("--pue", type=float, default=1.25)
    run.set_defaults(fn=_cmd_run)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except BrokenPipeError:
        # output piped into head/less which closed early; not an error
        sys.stderr.close()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
