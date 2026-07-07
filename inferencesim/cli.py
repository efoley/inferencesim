"""Command-line interface.

    inferencesim list
    inferencesim run --hardware gb300-nvl72 --model llama-3.1-70b \
        --tp 8 --batch 64 --prompt 4096 --output 1024 --weight-dtype fp8
    inferencesim graph --hardware tt-quietbox-fine [--json]
    inferencesim run --graph my-machine.json --model llama-3.1-8b
"""

from __future__ import annotations

import argparse
import importlib.resources
import json
import sys
import tempfile
import webbrowser
from dataclasses import replace
from pathlib import Path

from .bridge import chip_graph_of, system_from_graph, system_to_graph
from .calibration import calibrate_report
from .des import DESEngine
from .efficiency import PROFILES, Efficiency, profile_for
from .engine import RooflineEngine
from .graph import Graph
from .hardware import DType, System
from .presets import HARDWARE, MODELS
from .presets_fine import GRAPH_PRESETS
from .replay import build_replay
from .report import format_report
from .sched import chrome_trace
from .serve import (
    DisaggConfig,
    LengthDist,
    ServeConfig,
    format_serve_report,
    serve,
    serve_disagg,
)
from .simulate import CostModel, simulate
from .units import fmt_bytes, fmt_si
from .workload import Deployment, Scenario


def _add_efficiency_args(parser: argparse.ArgumentParser) -> None:
    """The shared derating knobs: a named profile plus per-factor overrides."""
    parser.add_argument("--efficiency", choices=[*sorted(PROFILES), "auto"],
                        default=None,
                        help="named efficiency profile (default: sol = speed of "
                             "light, no derating). 'auto' picks the vendor-"
                             "appropriate profile per hardware (tt-* -> typical-tt, "
                             "else typical-nv); 'typical' is the cross-vendor global "
                             "fit; 'typical-nv'/'typical-tt' select a vendor profile "
                             "explicitly. See CALIBRATION.md.")
    parser.add_argument("--eff-compute", type=float, default=None,
                        help="override compute efficiency (fraction of peak FLOP/s)")
    parser.add_argument("--eff-memory", type=float, default=None,
                        help="override memory efficiency (fraction of peak bandwidth)")
    parser.add_argument("--eff-collective", type=float, default=None,
                        help="override collective efficiency (fraction of link bw)")
    parser.add_argument("--op-overhead-s", type=float, default=None,
                        help="override fixed per-op overhead in seconds")


def _override_dict(args: argparse.Namespace) -> dict:
    """The explicitly-passed per-factor overrides, as replace() kwargs."""
    overrides = {}
    if args.eff_compute is not None:
        overrides["compute"] = args.eff_compute
    if args.eff_memory is not None:
        overrides["memory"] = args.eff_memory
    if args.eff_collective is not None:
        overrides["collective"] = args.eff_collective
    if args.op_overhead_s is not None:
        overrides["op_overhead_s"] = args.op_overhead_s
    return overrides


def _efficiency_from_args(args: argparse.Namespace, hardware_key: str = "") -> Efficiency:
    """Build an Efficiency: start from the named profile (default 'sol'; 'auto'
    vendor-resolves against `hardware_key`), then apply any explicitly-passed
    per-factor overrides."""
    if args.efficiency == "auto":
        base = profile_for(hardware_key, "auto")
    elif args.efficiency:
        base = PROFILES[args.efficiency]
    else:
        base = PROFILES["sol"]
    overrides = _override_dict(args)
    return replace(base, **overrides) if overrides else base


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


def _apply_moe_skew(model, skew: float | None):
    """Override a MoE model's expert-load `skew` from the CLI (no-op for None or
    a dense model, with a friendly error if --moe-skew is given for a dense one)."""
    if skew is None:
        return model
    if model.moe is None:
        print(f"--moe-skew is MoE-only; {model.name} is dense", file=sys.stderr)
        raise SystemExit(2)
    return replace(model, moe=replace(model.moe, skew=skew))


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
    model = _apply_moe_skew(MODELS[args.model], args.moe_skew)
    dep = Deployment(
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        adp=args.adp,
        weight_dtype=DType(args.weight_dtype),
        kv_dtype=DType(args.kv_dtype),
        act_dtype=DType(args.act_dtype),
        cp_prefill=args.cp_prefill,
        overlap_comm=args.overlap_comm,
    )
    cost = CostModel(
        amortization_years=args.amortization_years,
        electricity_usd_per_kwh=args.kwh_price,
        pue=args.pue,
    )

    efficiency = _efficiency_from_args(args, getattr(args, "hardware", None) or "")
    if args.engine == "des":
        hw_graph = _resolve_hw_graph(args)
        chip_graph = chip_graph_of(hw_graph) if hw_graph is not None else None
        if chip_graph is not None:
            print(f"graph mode: costing chip ops on the expanded "
                  f"'{chip_graph.name}' graph (tile-fill {args.tile_fill})",
                  file=sys.stderr)
        engine = DESEngine(decode_rounds=args.decode_rounds,
                           chip_graph=chip_graph, tile_fill=args.tile_fill,
                           efficiency=efficiency)
    else:
        engine = RooflineEngine(efficiency)
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


def _render_viewer(replay: dict) -> str:
    """Inject the replay JSON into the packaged viewer template."""
    template = (importlib.resources.files("inferencesim")
                .joinpath("viewer.html").read_text(encoding="utf-8"))
    if "/*__REPLAY_JSON__*/" not in template:
        raise RuntimeError("viewer.html is missing the /*__REPLAY_JSON__*/ marker")
    payload = json.dumps(replay, separators=(",", ":"))
    # defensive: never let a data string terminate the <script> block early.
    payload = payload.replace("</", "<\\/")
    return template.replace("/*__REPLAY_JSON__*/", payload)


def _cmd_ui(args: argparse.Namespace) -> int:
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
    model = _apply_moe_skew(MODELS[args.model], args.moe_skew)
    dep = Deployment(
        tp=args.tp, pp=args.pp, ep=args.ep, adp=args.adp,
        weight_dtype=DType(args.weight_dtype),
        kv_dtype=DType(args.kv_dtype),
        act_dtype=DType(args.act_dtype),
        cp_prefill=args.cp_prefill,
    )
    scen = Scenario(batch=args.batch, prompt_len=args.prompt, output_len=args.output)
    efficiency = _efficiency_from_args(args, getattr(args, "hardware", None) or "")

    hw_graph = _resolve_hw_graph(args)
    chip_graph = chip_graph_of(hw_graph) if hw_graph is not None else None
    if chip_graph is not None:
        print(f"graph mode: chip ops walk the expanded '{chip_graph.name}' graph "
              f"(tile-fill {args.tile_fill})", file=sys.stderr)
    engine = DESEngine(decode_rounds=args.decode_rounds, chip_graph=chip_graph,
                       tile_fill=args.tile_fill, efficiency=efficiency)
    # a full run populates the engine and lets us surface a header sanity number
    simulate(system, model, scen, dep, engine=engine)
    replay = build_replay(engine, system, model, scen, dep, hw_graph)
    html = _render_viewer(replay)

    if args.out:
        out_path = Path(args.out)
    else:
        fd = tempfile.NamedTemporaryFile(
            prefix="inferencesim-", suffix=".html", delete=False)
        fd.close()
        out_path = Path(fd.name)
    out_path.write_text(html, encoding="utf-8")
    n_levels = len(replay["levels"])
    print(f"wrote viewer ({len(html) // 1024} KiB, {n_levels} level(s)) to {out_path}",
          file=sys.stderr)
    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    if args.efficiency == "auto":
        # score each anchor under its own vendor profile (tt anchors under
        # typical-tt, NVIDIA under typical-nv), applying any --eff-* overrides.
        overrides = _override_dict(args)

        def resolve(hardware_key: str) -> Efficiency:
            base = profile_for(hardware_key, "auto")
            return replace(base, **overrides) if overrides else base

        print(calibrate_report(resolve=resolve))
    else:
        print(calibrate_report(_efficiency_from_args(args)))
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
    model = _apply_moe_skew(MODELS[args.model], args.moe_skew)
    dep = Deployment(
        tp=args.tp, pp=args.pp, ep=args.ep, adp=args.adp,
        weight_dtype=DType(args.weight_dtype),
        kv_dtype=DType(args.kv_dtype),
        act_dtype=DType(args.act_dtype),
        cp_prefill=args.cp_prefill,
    )
    scen = Scenario(batch=args.max_batch, prompt_len=args.prompt, output_len=args.output)
    common = dict(
        max_batch=args.max_batch, seed=args.seed,
        prefill_first=not args.decode_first, kv_policy=args.kv_policy,
        kv_watermark=args.kv_watermark, prefill_chunk=args.prefill_chunk,
    )
    if args.arrivals:
        # each line: `time` or `time prompt output`
        arrivals: list[float] = []
        prompt_lens: list[int] = []
        output_lens: list[int] = []
        mixed = False
        for line in Path(args.arrivals).read_text().splitlines():
            cols = line.split()
            if not cols:
                continue
            arrivals.append(float(cols[0]))
            if len(cols) >= 3:
                mixed = True
                prompt_lens.append(int(cols[1]))
                output_lens.append(int(cols[2]))
            else:
                prompt_lens.append(args.prompt)
                output_lens.append(args.output)
        cfg = ServeConfig(
            arrivals=arrivals,
            prompt_lens=prompt_lens if mixed else None,
            output_lens=output_lens if mixed else None,
            **common,
        )
    else:
        cfg = ServeConfig(
            arrival_rate=args.rate, n_requests=args.requests,
            prompt_dist=_parse_dist(args.prompt_dist),
            output_dist=_parse_dist(args.output_dist),
            **common,
        )
    efficiency = _efficiency_from_args(args, args.hardware or "")
    if args.disagg:
        shared = dict(weight_dtype=DType(args.weight_dtype),
                      kv_dtype=DType(args.kv_dtype), act_dtype=DType(args.act_dtype),
                      cp_prefill=args.cp_prefill)
        dcfg = DisaggConfig(
            prefill_deployment=Deployment(tp=args.prefill_tp, ep=args.prefill_ep,
                                          adp=args.prefill_adp, **shared),
            decode_deployment=Deployment(tp=args.decode_tp, ep=args.decode_ep,
                                         adp=args.decode_adp, **shared),
            n_prefill_replicas=args.prefill_replicas,
            n_decode_replicas=args.decode_replicas,
            transfer_bw=args.transfer_bw, transfer_latency=args.transfer_latency,
        )
        try:
            report = serve_disagg(system, model, scen, cfg, dcfg,
                                  engine=RooflineEngine(efficiency))
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2
    else:
        report = serve(system, model, scen, dep, cfg,
                       engine=RooflineEngine(efficiency))
    print(format_serve_report(report))
    return 0


def _parse_dist(spec: str | None) -> LengthDist | None:
    """Parse `uniform:LO:HI` or `lognormal:MEDIAN:SIGMA` into a LengthDist."""
    if not spec:
        return None
    parts = spec.split(":")
    if len(parts) != 3:
        raise SystemExit(f"bad --*-dist {spec!r}: use uniform:LO:HI or lognormal:MED:SIGMA")
    kind, a, b = parts[0], float(parts[1]), float(parts[2])
    if kind == "uniform":
        return LengthDist.uniform(int(a), int(b))
    if kind == "lognormal":
        return LengthDist.lognormal(a, b)
    raise SystemExit(f"unknown distribution {kind!r} (uniform | lognormal)")


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
    run.add_argument("--adp", type=int, default=1,
                     help="attention-data-parallel groups (dense models only): "
                          "DP attention + TP FFN, TRT-LLM DEPn -- cuts per-chip "
                          "KV by adp, streams the FFN over the tp*adp array")
    run.add_argument("--no-cp-prefill", dest="cp_prefill", action="store_false",
                     help="disable context-parallel prefill: run the old single-"
                          "group-per-request prefill (attention on one adp group) "
                          "instead of splitting the prompt across the tp*adp array")
    run.set_defaults(cp_prefill=True)
    run.add_argument("--moe-skew", type=float, default=None,
                     help="MoE expert-load imbalance (Zipf exponent over expert "
                          "popularity; 0 = uniform, larger = hotter experts). "
                          "Paces moe_routed by the hottest tp*ep member and, "
                          "under --engine des, incasts the dispatch/combine "
                          "all-to-all onto the hot owner's ingress port")
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
    _add_efficiency_args(run)
    run.set_defaults(fn=_cmd_run)

    ui = sub.add_parser("ui", help="build a single-file HTML replay viewer of "
                                   "the discrete-event run and open it")
    ui.add_argument("--hardware", help="hardware preset or graph preset key")
    ui.add_argument("--graph", help="path to a hardware graph JSON file")
    ui.add_argument("--model", required=True, help="model preset key")
    ui.add_argument("--tp", type=int, default=1, help="tensor-parallel degree")
    ui.add_argument("--pp", type=int, default=1, help="pipeline-parallel stages")
    ui.add_argument("--ep", type=int, default=1,
                    help="expert-parallel groups (MoE models only)")
    ui.add_argument("--adp", type=int, default=1,
                    help="attention-data-parallel groups (dense models only)")
    ui.add_argument("--no-cp-prefill", dest="cp_prefill", action="store_false",
                    help="disable context-parallel prefill (see `run --no-cp-prefill`)")
    ui.set_defaults(cp_prefill=True)
    ui.add_argument("--moe-skew", type=float, default=None,
                    help="MoE expert-load imbalance (Zipf exponent; 0 = uniform, "
                         "larger = hotter experts). Under the DES the dispatch/"
                         "combine all-to-all incasts onto the hot owner's ingress "
                         "port -- visible in the viewer as ingress-port occupancy")
    ui.add_argument("--batch", type=int, default=32,
                    help="concurrent sequences per replica")
    ui.add_argument("--prompt", type=int, default=2048, help="prompt tokens per request")
    ui.add_argument("--output", type=int, default=512, help="output tokens per request")
    ui.add_argument("--weight-dtype", default="fp8", choices=[d.value for d in DType])
    ui.add_argument("--kv-dtype", default="bf16", choices=[d.value for d in DType])
    ui.add_argument("--act-dtype", default="bf16", choices=[d.value for d in DType])
    ui.add_argument("--tile-fill", type=float, default=0.5,
                    help="graph mode: fraction of per-core SRAM a tile may use")
    ui.add_argument("--decode-rounds", type=int, default=None,
                    help="pin the DES decode measurement to a fixed round count")
    ui.add_argument("-o", "--out", metavar="FILE",
                    help="write the HTML here (default: a temp file)")
    ui.add_argument("--no-open", action="store_true",
                    help="do not open a browser (just write the file)")
    _add_efficiency_args(ui)
    ui.set_defaults(fn=_cmd_ui)

    cal = sub.add_parser("calibrate",
                         help="score the simulator against measured anchors "
                              "under an efficiency profile")
    _add_efficiency_args(cal)
    cal.set_defaults(fn=_cmd_calibrate)

    srv = sub.add_parser("serve", help="request-level continuous-batching serving "
                                       "simulation (one replica, pp=1)")
    srv.add_argument("--hardware", help="hardware preset or graph preset key")
    srv.add_argument("--graph", help="path to a hardware graph JSON file")
    srv.add_argument("--model", required=True, help="model preset key")
    srv.add_argument("--tp", type=int, default=1, help="tensor-parallel degree")
    srv.add_argument("--pp", type=int, default=1, help="pipeline stages (serve requires 1)")
    srv.add_argument("--ep", type=int, default=1,
                     help="expert-parallel groups (MoE models only)")
    srv.add_argument("--adp", type=int, default=1,
                     help="attention-data-parallel groups (dense models only): "
                          "DP attention + TP FFN, TRT-LLM DEPn")
    srv.add_argument("--no-cp-prefill", dest="cp_prefill", action="store_false",
                     help="disable context-parallel prefill (see `run --no-cp-prefill`)")
    srv.set_defaults(cp_prefill=True)
    srv.add_argument("--moe-skew", type=float, default=None,
                     help="MoE expert-load imbalance (Zipf exponent; 0 = uniform). "
                          "Paces moe_routed by the hottest tp*ep member, so serve "
                          "throughput drops as hot experts stream more weight")
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
    srv.add_argument("--kv-policy", choices=["on_demand", "reserve"],
                     default="on_demand",
                     help="on_demand: admit against prompt KV, preempt (recompute) "
                          "on overflow; reserve: admit only if full prompt+output "
                          "KV fits, never preempt")
    srv.add_argument("--kv-watermark", type=float, default=0.95,
                     help="usable fraction of the KV budget (on_demand only)")
    srv.add_argument("--prefill-chunk", type=int, default=None,
                     help="mix this many prefill tokens into each decode iteration "
                          "(Sarathi chunked prefill); default: exclusive prefill")
    srv.add_argument("--prompt-dist", metavar="SPEC",
                     help="Poisson-mode prompt length distribution: "
                          "uniform:LO:HI or lognormal:MEDIAN:SIGMA")
    srv.add_argument("--output-dist", metavar="SPEC",
                     help="Poisson-mode output length distribution (see --prompt-dist)")
    srv.add_argument("--arrivals", metavar="FILE",
                     help="explicit per-replica arrivals, one per line as "
                          "`time` or `time prompt output` (mixed lengths)")
    srv.add_argument("--weight-dtype", default="fp8", choices=[d.value for d in DType])
    srv.add_argument("--kv-dtype", default="bf16", choices=[d.value for d in DType])
    srv.add_argument("--act-dtype", default="bf16", choices=[d.value for d in DType])
    # ---- prefill/decode disaggregation (two chip pools) ----
    srv.add_argument("--disagg", action="store_true",
                     help="prefill/decode disaggregated serving: partition the "
                          "chips into a prefill pool and a decode pool, streaming "
                          "the KV cache between them (DistServe/Dynamo). Uses the "
                          "--prefill-*/--decode-* pool flags plus the shared "
                          "arrival/length/--kv-policy knobs; --tp/--pp/--ep/--adp, "
                          "prefill_first and --prefill-chunk (rejected) do not apply.")
    srv.add_argument("--prefill-tp", type=int, default=1, help="prefill pool TP")
    srv.add_argument("--prefill-ep", type=int, default=1, help="prefill pool EP (MoE)")
    srv.add_argument("--prefill-adp", type=int, default=1, help="prefill pool ADP (dense)")
    srv.add_argument("--prefill-replicas", type=int, default=1,
                     help="number of prefill replicas")
    srv.add_argument("--decode-tp", type=int, default=1, help="decode pool TP")
    srv.add_argument("--decode-ep", type=int, default=1, help="decode pool EP (MoE)")
    srv.add_argument("--decode-adp", type=int, default=1, help="decode pool ADP (dense)")
    srv.add_argument("--decode-replicas", type=int, default=1,
                     help="number of decode replicas")
    srv.add_argument("--transfer-bw", type=float, default=None,
                     help="override the prefill<->decode link bandwidth (bytes/s); "
                          "default resolves the system's node/network link")
    srv.add_argument("--transfer-latency", type=float, default=None,
                     help="override the prefill<->decode link latency (s)")
    _add_efficiency_args(srv)
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
