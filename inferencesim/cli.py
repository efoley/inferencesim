"""Command-line interface.

    inferencesim list
    inferencesim run --hardware gb300-nvl72 --model llama-3.1-70b \
        --tp 8 --batch 64 --prompt 4096 --output 1024 --weight-dtype fp8
    inferencesim graph --hardware tt-quietbox-fine [--json]
    inferencesim run --graph my-machine.json --model llama-3.1-8b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bridge import system_from_graph, system_to_graph
from .des import DESEngine
from .engine import RooflineEngine
from .graph import Graph
from .hardware import DType, System
from .presets import HARDWARE, MODELS
from .presets_fine import GRAPH_PRESETS
from .report import format_report
from .sched import chrome_trace
from .serve import ServeConfig, format_serve_report, serve
from .simulate import CostModel, simulate
from .units import fmt_bytes, fmt_si
from .workload import Deployment, Scenario


def _resolve_graph(key: str) -> Graph | None:
    if key in GRAPH_PRESETS:
        return GRAPH_PRESETS[key]()
    if key in HARDWARE:
        return system_to_graph(HARDWARE[key])
    return None


def _resolve_hw_graph(args: argparse.Namespace) -> Graph | None:
    """The hardware graph iff the source is graph-based (--graph FILE or a
    GRAPH_PRESETS key); None for lumped HARDWARE presets, which the DES treats
    exactly as before."""
    if getattr(args, "graph", None):
        return Graph.from_json(Path(args.graph).read_text())
    if args.hardware in GRAPH_PRESETS:
        return GRAPH_PRESETS[args.hardware]()
    return None


def _chip_graph_of(g: Graph) -> Graph:
    """The chip-level model to walk in graph mode: the first role='chip'
    composite's inner graph, or the whole graph if there is no such composite
    (a bare chip graph)."""
    for _path, node in g.walk():
        if node.role == "chip" and node.inner is not None:
            return node.inner
    return g


def _resolve_system(args: argparse.Namespace) -> System | None:
    hw_graph = _resolve_hw_graph(args)
    if hw_graph is not None:
        return system_from_graph(hw_graph)
    if args.hardware in HARDWARE:
        return HARDWARE[args.hardware]
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
    if args.trace and args.engine != "des":
        print("--trace requires --engine des", file=sys.stderr)
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

    if args.engine == "des":
        hw_graph = _resolve_hw_graph(args)
        chip_graph = _chip_graph_of(hw_graph) if hw_graph is not None else None
        if chip_graph is not None:
            print(f"graph mode: costing chip ops on the expanded "
                  f"'{chip_graph.name}' graph (tile-fill {args.tile_fill})",
                  file=sys.stderr)
        engine = DESEngine(decode_rounds=args.decode_rounds,
                           chip_graph=chip_graph, tile_fill=args.tile_fill)
    else:
        engine = RooflineEngine()
    batches = [int(b) for b in args.batch.split(",")]
    for i, batch in enumerate(batches):
        scen = Scenario(batch=batch, prompt_len=args.prompt, output_len=args.output)
        report = simulate(system, model, scen, dep, cost, engine=engine)
        if i:
            print()
        print(format_report(report))
    if args.trace:
        events: list[dict] = []
        pid = 0
        for pname, (tasks, result) in engine.last_runs.items():
            events += chrome_trace(tasks, result, pid_base=pid,
                                   prefix=f"{pname}/")["traceEvents"]
            pid += 1000
        # graph mode: one extra track group per distinct chip-lowered op
        for pname, op_runs in engine.last_op_runs.items():
            for opname, sched in op_runs.items():
                events += chrome_trace(sched.tasks, sched.result, pid_base=pid,
                                       prefix=f"{pname}/op:{opname}/")["traceEvents"]
                pid += 1000
        Path(args.trace).write_text(
            json.dumps({"traceEvents": events, "displayTimeUnit": "ms"}))
        print(f"wrote Chrome trace ({len(events)} events) to {args.trace}",
              file=sys.stderr)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
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
        tp=args.tp, pp=args.pp, ep=args.ep,
        weight_dtype=DType(args.weight_dtype),
        kv_dtype=DType(args.kv_dtype),
        act_dtype=DType(args.act_dtype),
    )
    scen = Scenario(batch=args.max_batch, prompt_len=args.prompt, output_len=args.output)
    if args.arrivals:
        arrivals = [float(x) for x in Path(args.arrivals).read_text().split()]
        cfg = ServeConfig(
            arrivals=arrivals, n_requests=len(arrivals), max_batch=args.max_batch,
            seed=args.seed, prefill_first=not args.decode_first,
        )
    else:
        cfg = ServeConfig(
            arrival_rate=args.rate, n_requests=args.requests, max_batch=args.max_batch,
            seed=args.seed, prefill_first=not args.decode_first,
        )
    report = serve(system, model, scen, dep, cfg)
    print(format_serve_report(report))
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
    run.add_argument("--engine", choices=["roofline", "des"], default="roofline",
                     help="roofline: analytic speed-of-light sums; des: "
                          "discrete-event simulation with resource queues "
                          "(overlap is emergent; --overlap-comm is ignored)")
    run.add_argument("--overlap-comm", action="store_true",
                     help="assume TP collectives fully overlap with compute")
    run.add_argument("--trace", metavar="FILE",
                     help="write a Chrome/Perfetto trace JSON of the run "
                          "(requires --engine des)")
    run.add_argument("--decode-rounds", type=int, default=None,
                     help="pin the DES decode measurement to a fixed round "
                          "count (default: auto-grow until the period "
                          "converges; deep pipelines can take a while)")
    run.add_argument("--tile-fill", type=float, default=0.5,
                     help="graph mode: fraction of per-core SRAM a tile may "
                          "use; a core double-buffers 1/tile-fill tiles")
    run.add_argument("--amortization-years", type=float, default=4.0)
    run.add_argument("--kwh-price", type=float, default=0.12, help="USD per kWh")
    run.add_argument("--pue", type=float, default=1.25)
    run.set_defaults(fn=_cmd_run)

    srv = sub.add_parser("serve", help="request-level continuous-batching serving "
                                       "simulation (one replica, pp=1)")
    srv.add_argument("--hardware", help="hardware preset or graph preset key")
    srv.add_argument("--graph", help="path to a hardware graph JSON file")
    srv.add_argument("--model", required=True, help="model preset key")
    srv.add_argument("--tp", type=int, default=1, help="tensor-parallel degree")
    srv.add_argument("--pp", type=int, default=1, help="pipeline stages (serve requires 1)")
    srv.add_argument("--ep", type=int, default=1,
                     help="expert-parallel groups (MoE models only)")
    srv.add_argument("--rate", type=float, default=5.0,
                     help="whole-system arrival rate (requests/s, Poisson); "
                          "divided by DP for one replica")
    srv.add_argument("--requests", type=int, default=200,
                     help="simulate until this many requests complete")
    srv.add_argument("--max-batch", type=int, default=64,
                     help="continuous-batching slots per replica")
    srv.add_argument("--prompt", type=int, default=2048, help="prompt tokens per request")
    srv.add_argument("--output", type=int, default=512, help="output tokens per request")
    srv.add_argument("--seed", type=int, default=0, help="RNG seed for arrivals")
    srv.add_argument("--decode-first", action="store_true",
                     help="decode has priority over waiting prefills "
                          "(default: prefill-first, vLLM-like)")
    srv.add_argument("--arrivals", metavar="FILE",
                     help="read explicit per-replica arrival times (one float per "
                          "line) instead of a Poisson --rate")
    srv.add_argument("--weight-dtype", default="fp8", choices=[d.value for d in DType])
    srv.add_argument("--kv-dtype", default="bf16", choices=[d.value for d in DType])
    srv.add_argument("--act-dtype", default="bf16", choices=[d.value for d in DType])
    srv.set_defaults(fn=_cmd_serve)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except BrokenPipeError:
        # output piped into head/less which closed early; not an error
        sys.stderr.close()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
