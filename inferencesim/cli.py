"""Command-line interface.

    inferencesim list
    inferencesim run --hardware gb300-nvl72 --model llama-3.1-70b \
        --tp 8 --batch 64 --prompt 4096 --output 1024 --weight-dtype fp8
"""

from __future__ import annotations

import argparse
import sys

from .hardware import DType
from .presets import HARDWARE, MODELS
from .report import format_report
from .simulate import CostModel, simulate
from .units import fmt_bytes, fmt_si
from .workload import Deployment, Scenario


def _cmd_list(_: argparse.Namespace) -> int:
    print("Hardware presets:")
    for key, sys_ in HARDWARE.items():
        chip = sys_.node.chip
        print(f"  {key:16s} {sys_.total_chips:>3d}x {chip.name:26s} "
              f"DRAM {fmt_bytes(chip.dram.capacity_bytes):>7s}/chip "
              f"@ {fmt_si(chip.effective_dram_bandwidth, 'B/s')}")
    print("\nModel presets:")
    for key, m in MODELS.items():
        extra = f", {m.active_params / 1e9:.1f}B active" if m.moe else ""
        print(f"  {key:16s} {m.total_params / 1e9:6.1f}B params{extra}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if args.hardware not in HARDWARE:
        print(f"unknown hardware '{args.hardware}' (try: inferencesim list)", file=sys.stderr)
        return 2
    if args.model not in MODELS:
        print(f"unknown model '{args.model}' (try: inferencesim list)", file=sys.stderr)
        return 2

    system = HARDWARE[args.hardware]
    model = MODELS[args.model]
    dep = Deployment(
        tp=args.tp,
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

    run = sub.add_parser("run", help="simulate a serving scenario")
    run.add_argument("--hardware", required=True, help="hardware preset key")
    run.add_argument("--model", required=True, help="model preset key")
    run.add_argument("--tp", type=int, default=1, help="tensor-parallel degree")
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
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
