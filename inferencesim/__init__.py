"""inferencesim: throughput / latency / power / cost simulator for LLM
inference factories, built from chip-level hardware blocks."""

from .hardware import Chip, Compute, DType, Link, Memory, Node, System, Topology
from .workload import Deployment, ModelSpec, MoEConfig, Scenario
from .efficiency import Efficiency, PROFILES, profile_for, vendor_profile_name
from .engine import Engine, RooflineEngine, ring_allreduce_time
from .sched import Resource, ScheduleResult, Task, chrome_trace, schedule
from .graphdes import ChipModel, OpSchedule
from .des import DESEngine
from .simulate import CostModel, MemoryUsage, Report, simulate, weight_bytes_per_chip
from .calibration import ANCHORS, Anchor, calibrate_report, run_anchor
from .report import format_report
from .serve import (
    LengthDist,
    RequestRecord,
    ServeConfig,
    ServeReport,
    chunked_prefill_ttft,
    decode_iteration_time,
    format_serve_report,
    prefill_iteration_time,
    serve,
)
from .graph import Edge as GraphEdge, Graph, Node as GraphNode, NodeKind
from .bridge import (
    chip_from_graph,
    chip_to_graph,
    swap_chip_model,
    system_from_graph,
    system_to_graph,
)
from . import presets, presets_fine

__version__ = "0.2.0"

__all__ = [
    "Chip", "Compute", "DType", "Link", "Memory", "Node", "System", "Topology",
    "Deployment", "ModelSpec", "MoEConfig", "Scenario",
    "Efficiency", "PROFILES", "profile_for", "vendor_profile_name",
    "Anchor", "ANCHORS", "calibrate_report", "run_anchor",
    "Engine", "RooflineEngine", "DESEngine", "ring_allreduce_time",
    "Task", "Resource", "ScheduleResult", "schedule", "chrome_trace",
    "ChipModel", "OpSchedule",
    "CostModel", "MemoryUsage", "Report", "simulate", "weight_bytes_per_chip",
    "format_report",
    "ServeConfig", "ServeReport", "RequestRecord", "LengthDist", "serve",
    "format_serve_report", "decode_iteration_time", "prefill_iteration_time",
    "chunked_prefill_ttft",
    "Graph", "GraphNode", "GraphEdge", "NodeKind",
    "chip_to_graph", "chip_from_graph", "system_to_graph", "system_from_graph",
    "swap_chip_model",
    "presets", "presets_fine",
]
