"""Top-level orchestration: map a workload onto a system and report
latency, throughput, power and cost."""

from __future__ import annotations

from dataclasses import dataclass, field

from .engine import Engine, Phase, RooflineEngine
from .hardware import System
from .ops import (
    decode_step_ops,
    kv_cache_bytes_per_chip,
    prefill_ops,
    validate_deployment,
)
from .workload import Deployment, ModelSpec, Scenario


def weight_bytes_per_chip(model: ModelSpec, dep: Deployment) -> float:
    """Per-chip weight footprint under the tp/pp/ep/adp mapping.

    Layers are split across pp stages and tp-sharded.  The FFN sub-array
    additionally spreads over its extra groups -- MoE expert weights over ep,
    the dense FFN over adp (attention-DP + TP FFN) -- while attention (and MoE
    shared-expert) weights are replicated across those groups.  Embedding + LM
    head live on the edge stages, tp-sharded.
    """
    if model.moe is None:
        layer_replicated = model.attn_params  # replicated across adp groups
        layer_ffn = model.ffn_params_total  # dense FFN shards over tp*adp
        ffn_denom = dep.tp * dep.adp * dep.pp
        params = (
            model.embedding_params / dep.tp
            + model.n_layers * layer_replicated / (dep.tp * dep.pp)
            + model.n_layers * layer_ffn / ffn_denom
        )
        return params * dep.weight_dtype.bytes
    # MoE: attention + shared expert replicated across ep groups (tp*pp), the
    # expert bank sharded over the whole tp*ep array.  A DeepSeek-style
    # `n_dense_layers` prefix carries plain dense FFNs (width d_ff) instead of
    # experts -- those shard over tp and replicate across ep, like attention.
    nd = model.moe.n_dense_layers
    moe_layers = model.n_layers - nd
    experts_bank = model.moe.n_experts * model.expert_params
    params = (
        model.embedding_params / dep.tp
        + model.n_layers * model.attn_params / (dep.tp * dep.pp)
        + moe_layers * model.shared_expert_params / (dep.tp * dep.pp)
        + moe_layers * experts_bank / (dep.tp * dep.ep * dep.pp)
        + nd * model._dense_ffn_params / (dep.tp * dep.pp)
    )
    return params * dep.weight_dtype.bytes


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

    # per-phase per-resource utilisation ({phase: {resource: fraction}}),
    # only populated when the engine measures it (discrete-event engine).
    resource_util: dict[str, dict[str, float]] | None = None

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
    validate_deployment(model, deployment)
    replica = deployment.replica_chips
    if replica > system.total_chips:
        raise ValueError(
            f"replica needs tp*pp*ep*adp={replica} chips but {system.name} has "
            f"{system.total_chips}"
        )

    dp = system.total_chips // replica
    idle_chips = system.total_chips - dp * replica
    warnings: list[str] = []
    if idle_chips:
        warnings.append(
            f"{idle_chips} chip(s) idle: total chips not divisible by tp*pp*ep*adp={replica}"
        )
    if deployment.pp > 1 and model.n_layers % deployment.pp:
        warnings.append(
            f"pp={deployment.pp} does not divide {model.n_layers} layers; "
            f"stages assumed balanced anyway"
        )
    groups = deployment.pp * deployment.ep * deployment.adp
    microbatch = scenario.batch / groups
    if microbatch < 1:
        warnings.append(
            f"batch={scenario.batch} < pp*ep*adp={groups}: "
            f"pipeline/expert/attention groups are starved (raise batch)"
        )

    chip = system.node.chip
    B = scenario.batch

    # ---- memory feasibility ---------------------------------------------
    memory = MemoryUsage(
        weights=weight_bytes_per_chip(model, deployment),
        kv_cache=B * kv_cache_bytes_per_chip(model, scenario.max_context, deployment),
        # rough working-set allowance for activations / scratch
        activations=4 * microbatch * model.d_model * deployment.act_dtype.bytes,
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
        "prefill", prefill_ops(model, scenario.prompt_len, deployment), system, deployment
    )
    decode = engine.run_phase(
        "decode", decode_step_ops(model, scenario, deployment), system, deployment
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
    tp_link = system.link_for_group(deployment.tp)
    prefill_frac = ttft / replica_s_per_request if replica_s_per_request > 0 else 0.0
    chip_power = (
        prefill_frac * prefill.chip_avg_power_w(chip, tp_link, ttft)
        + (1 - prefill_frac) * decode.chip_avg_power_w(chip, tp_link, tpot)
    )
    busy_chips = dp * replica
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

    # ---- per-resource utilisation (discrete-event engine only) -------------
    resource_util: dict[str, dict[str, float]] = {}
    for phase in (prefill, decode):
        if phase.resource_busy and phase.resource_span:
            resource_util[phase.name] = {
                r: b / phase.resource_span for r, b in phase.resource_busy.items()
            }

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
        resource_util=resource_util or None,
        warnings=warnings,
    )
