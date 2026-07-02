"""Top-level orchestration: map a workload onto a system and report
latency, throughput, power and cost."""

from __future__ import annotations

from dataclasses import dataclass, field

from .engine import Engine, Phase, RooflineEngine
from .hardware import System
from .ops import decode_step_ops, kv_cache_bytes_per_chip, prefill_ops
from .workload import Deployment, ModelSpec, Scenario


@dataclass(frozen=True)
class CostModel:
    """Turns capex + power into $/token."""

    amortization_years: float = 4.0
    electricity_usd_per_kwh: float = 0.12
    pue: float = 1.25  # datacenter power overhead multiplier

    def capex_usd_per_s(self, system_cost_usd: float) -> float:
        return system_cost_usd / (self.amortization_years * 365.25 * 24 * 3600)

    def power_usd_per_s(self, watts: float) -> float:
        return watts * self.pue / 1000.0 * self.electricity_usd_per_kwh / 3600.0


@dataclass(frozen=True)
class MemoryUsage:
    """Per-chip DRAM usage."""

    weights: float
    kv_cache: float
    activations: float
    capacity: float

    @property
    def total(self) -> float:
        return self.weights + self.kv_cache + self.activations

    @property
    def fits(self) -> bool:
        return self.total <= self.capacity


@dataclass
class Report:
    system: System
    model: ModelSpec
    scenario: Scenario
    deployment: Deployment
    cost_model: CostModel

    dp: int  # data-parallel replicas
    idle_chips: int
    memory: MemoryUsage = None  # type: ignore[assignment]
    prefill: Phase = None  # type: ignore[assignment]
    decode: Phase = None  # type: ignore[assignment]

    # latency
    ttft_s: float = 0.0
    tpot_s: float = 0.0

    # throughput (steady-state continuous batching)
    requests_per_s: float = 0.0  # whole system
    output_tokens_per_s: float = 0.0  # whole system
    input_tokens_per_s: float = 0.0
    decode_only_tokens_per_s: float = 0.0  # ceiling ignoring prefill cost

    # power / energy
    system_power_w: float = 0.0
    joules_per_output_token: float = 0.0

    # cost
    usd_per_m_output_tokens: float = 0.0
    usd_per_m_total_tokens: float = 0.0
    capex_share: float = 0.0  # fraction of $/token that is capex

    warnings: list[str] = field(default_factory=list)


def simulate(
    system: System,
    model: ModelSpec,
    scenario: Scenario,
    deployment: Deployment = Deployment(),
    cost_model: CostModel = CostModel(),
    engine: Engine | None = None,
) -> Report:
    engine = engine or RooflineEngine()
    tp = deployment.tp
    if tp < 1:
        raise ValueError("tp must be >= 1")
    if tp > system.total_chips:
        raise ValueError(f"tp={tp} exceeds {system.total_chips} chips in {system.name}")

    dp = system.total_chips // tp
    idle_chips = system.total_chips - dp * tp
    warnings: list[str] = []
    if idle_chips:
        warnings.append(f"{idle_chips} chip(s) idle: total chips not divisible by tp")

    chip = system.node.chip
    B = scenario.batch

    # ---- memory feasibility ---------------------------------------------
    memory = MemoryUsage(
        weights=model.weight_bytes(deployment.weight_dtype) / tp,
        kv_cache=B * kv_cache_bytes_per_chip(model, scenario.max_context, deployment),
        # rough working-set allowance for activations / scratch
        activations=4 * B * model.d_model * deployment.act_dtype.bytes,
        capacity=chip.dram.capacity_bytes,
    )
    if not memory.fits:
        warnings.append(
            f"does not fit: {memory.total / 1e9:.1f} GB needed vs "
            f"{memory.capacity / 1e9:.1f} GB per chip (raise tp, shrink batch/context, "
            f"or use a smaller dtype)"
        )

    # ---- timed phases -----------------------------------------------------
    prefill = engine.run_phase(
        "prefill", prefill_ops(model, scenario.prompt_len, deployment), system, tp
    )
    decode = engine.run_phase(
        "decode", decode_step_ops(model, scenario, deployment), system, tp
    )
    ttft = prefill.duration(deployment.overlap_comm)
    tpot = decode.duration(deployment.overlap_comm)

    # ---- steady-state throughput (continuous batching) --------------------
    # Per replica, each admitted request costs one exclusive prefill (TTFT)
    # plus output_len decode steps shared with B-1 other requests.
    O = scenario.output_len
    replica_s_per_request = ttft + O * tpot / B
    req_per_s_replica = 1.0 / replica_s_per_request if replica_s_per_request > 0 else 0.0
    requests_per_s = dp * req_per_s_replica
    output_tokens_per_s = requests_per_s * O
    input_tokens_per_s = requests_per_s * scenario.prompt_len
    decode_only_tokens_per_s = dp * B / tpot if tpot > 0 else 0.0

    # ---- power -------------------------------------------------------------
    tp_link = system.link_for_group(tp)
    prefill_frac = ttft / replica_s_per_request if replica_s_per_request > 0 else 0.0
    chip_power = (
        prefill_frac * prefill.chip_avg_power_w(chip, tp_link, ttft)
        + (1 - prefill_frac) * decode.chip_avg_power_w(chip, tp_link, tpot)
    )
    busy_chips = dp * tp
    system_power = (
        system.n_nodes * system.node.overhead_power_w
        + busy_chips * chip_power
        + idle_chips * chip.idle_power_w
    )
    joules_per_token = (
        system_power / output_tokens_per_s if output_tokens_per_s > 0 else 0.0
    )

    # ---- cost ---------------------------------------------------------------
    capex_per_s = cost_model.capex_usd_per_s(system.cost_usd)
    power_per_s = cost_model.power_usd_per_s(system_power)
    usd_per_s = capex_per_s + power_per_s
    usd_per_m_out = (
        usd_per_s / output_tokens_per_s * 1e6 if output_tokens_per_s > 0 else float("inf")
    )
    total_tok_per_s = output_tokens_per_s + input_tokens_per_s
    usd_per_m_total = usd_per_s / total_tok_per_s * 1e6 if total_tok_per_s > 0 else float("inf")

    return Report(
        system=system,
        model=model,
        scenario=scenario,
        deployment=deployment,
        cost_model=cost_model,
        dp=dp,
        idle_chips=idle_chips,
        memory=memory,
        prefill=prefill,
        decode=decode,
        ttft_s=ttft,
        tpot_s=tpot,
        requests_per_s=requests_per_s,
        output_tokens_per_s=output_tokens_per_s,
        input_tokens_per_s=input_tokens_per_s,
        decode_only_tokens_per_s=decode_only_tokens_per_s,
        system_power_w=system_power,
        joules_per_output_token=joules_per_token,
        usd_per_m_output_tokens=usd_per_m_out,
        usd_per_m_total_tokens=usd_per_m_total,
        capex_share=capex_per_s / usd_per_s if usd_per_s > 0 else 0.0,
        warnings=warnings,
    )
