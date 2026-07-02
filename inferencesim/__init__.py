"""inferencesim: throughput / latency / power / cost simulator for LLM
inference factories, built from chip-level hardware blocks."""

from .hardware import Chip, Compute, DType, Link, Memory, Node, System, Topology
from .workload import Deployment, ModelSpec, MoEConfig, Scenario
from .engine import Engine, RooflineEngine, ring_allreduce_time
from .simulate import CostModel, MemoryUsage, Report, simulate
from .report import format_report
from . import presets

__version__ = "0.1.0"

__all__ = [
    "Chip", "Compute", "DType", "Link", "Memory", "Node", "System", "Topology",
    "Deployment", "ModelSpec", "MoEConfig", "Scenario",
    "Engine", "RooflineEngine", "ring_allreduce_time",
    "CostModel", "MemoryUsage", "Report", "simulate",
    "format_report",
    "presets",
]
