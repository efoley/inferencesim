"""Lowering: (model, scenario, deployment) -> per-chip operation lists.

An Op is a resource-demand record: FLOPs to execute, bytes to move through
the DRAM path, bytes to communicate.  Engines decide what those demands cost
on a given chip.  The roofline engine treats ops as a sequential critical
path; a future discrete-event engine can consume the same ops with explicit
dependencies and resource contention.

All demands are *per chip* (already divided by the tensor-parallel degree)
and *per instance*; `count` says how many identical instances run (e.g. one
per layer).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .hardware import DType
from .workload import Deployment, ModelSpec, Scenario


class OpKind(str, Enum):
    COMPUTE = "compute"  # math + DRAM traffic on one chip
    ALLREDUCE = "allreduce"  # collective across the TP group


@dataclass(frozen=True)
class Op:
    name: str
    kind: OpKind
    dtype: DType  # dtype the math runs in (picks the compute rate)
    category: str  # linear | attention | moe | head | comm
    count: int = 1
    flops: float = 0.0
    dram_read: float = 0.0
    dram_write: float = 0.0
    comm_bytes: float = 0.0  # payload per chip for collectives


def _kv_heads_per_chip(model: ModelSpec, tp: int) -> float:
    """KV heads resident per chip: sharded up to n_kv_heads ways, then
    replicated (standard GQA tensor parallelism)."""
    return model.n_kv_heads / min(tp, model.n_kv_heads)


def kv_cache_bytes_per_chip(
    model: ModelSpec, n_tokens: float, dep: Deployment
) -> float:
    per_layer = 2 * _kv_heads_per_chip(model, dep.tp) * model.d_head * dep.kv_dtype.bytes
    return model.n_layers * per_layer * n_tokens


def _linear(
    name: str,
    tokens: float,
    params: float,
    d_in: float,
    d_out: float,
    dep: Deployment,
    count: int,
    category: str = "linear",
    weight_read_params: float | None = None,
) -> Op:
    """A weight GEMM: [tokens, d_in] x [d_in, d_out], weights sharded /tp.

    weight_read_params overrides how many parameters actually stream from
    DRAM (MoE reads only the activated experts).
    """
    tp = dep.tp
    wread = params if weight_read_params is None else weight_read_params
    return Op(
        name=name,
        kind=OpKind.COMPUTE,
        dtype=dep.weight_dtype,
        category=category,
        count=count,
        flops=2.0 * tokens * params / tp,
        dram_read=wread / tp * dep.weight_dtype.bytes + tokens * d_in * dep.act_dtype.bytes,
        dram_write=tokens * (d_out / tp) * dep.act_dtype.bytes,
    )


def _ffn_ops(model: ModelSpec, tokens: float, dep: Deployment, count: int) -> list[Op]:
    if model.moe is None:
        return [
            _linear(
                "ffn",
                tokens,
                model.ffn_params_total,
                model.d_model,
                model.d_model,
                dep,
                count,
            )
        ]
    moe = model.moe
    ops: list[Op] = []
    active = moe.expected_active_experts(int(tokens))
    routed_read = active * model.expert_params
    routed_compute = moe.top_k * model.expert_params
    ops.append(
        _linear(
            "moe_routed",
            tokens,
            routed_compute,
            model.d_model,
            model.d_model,
            dep,
            count,
            category="moe",
            weight_read_params=routed_read,
        )
    )
    if model.shared_expert_params:
        ops.append(
            _linear(
                "moe_shared",
                tokens,
                model.shared_expert_params,
                model.d_model,
                model.d_model,
                dep,
                count,
                category="moe",
            )
        )
    return ops


def _allreduce(name: str, tokens: float, model: ModelSpec, dep: Deployment, count: int) -> Op:
    return Op(
        name=name,
        kind=OpKind.ALLREDUCE,
        dtype=dep.act_dtype,
        category="comm",
        count=count,
        comm_bytes=tokens * model.d_model * dep.act_dtype.bytes,
    )


def _embed_and_head(model: ModelSpec, tokens_in: float, tokens_out: float, dep: Deployment) -> list[Op]:
    """Embedding lookup for tokens_in positions, LM head for tokens_out."""
    ops = [
        Op(
            name="embed",
            kind=OpKind.COMPUTE,
            dtype=dep.weight_dtype,
            category="head",
            dram_read=tokens_in * model.d_model * dep.weight_dtype.bytes,
            dram_write=tokens_in * model.d_model * dep.act_dtype.bytes,
        ),
        _linear(
            "lm_head",
            tokens_out,
            model.vocab_size * model.d_model,
            model.d_model,
            model.vocab_size,
            dep,
            count=1,
            category="head",
        ),
    ]
    return ops


def decode_step_ops(model: ModelSpec, scen: Scenario, dep: Deployment) -> list[Op]:
    """One decode step: every one of `batch` sequences emits one token,
    at the scenario's mean context length."""
    B = scen.batch
    ctx = scen.mean_context
    tp = dep.tp
    kvh = _kv_heads_per_chip(model, tp)
    heads = model.n_heads / tp
    L = model.n_layers

    ops: list[Op] = [
        _linear("qkv_proj", B, model.attn_qkv_params, model.d_model,
                (model.n_heads + 2 * model.n_kv_heads) * model.d_head, dep, count=L),
        Op(
            name="attention",
            kind=OpKind.COMPUTE,
            dtype=dep.act_dtype,
            category="attention",
            count=L,
            # QK^T and PV: 2 GEMVs of [1, ctx] x [ctx, d_head] per head
            flops=B * heads * 2 * 2.0 * ctx * model.d_head,
            # stream the KV cache; write this step's new K,V
            dram_read=B * ctx * 2 * kvh * model.d_head * dep.kv_dtype.bytes,
            dram_write=B * 2 * kvh * model.d_head * dep.kv_dtype.bytes,
        ),
        _linear("out_proj", B, model.attn_out_params,
                model.n_heads * model.d_head, model.d_model, dep, count=L),
        *_ffn_ops(model, B, dep, count=L),
        # one reduction after attention, one after the FFN (row-parallel)
        _allreduce("allreduce", B, model, dep, count=2 * L),
        *_embed_and_head(model, tokens_in=B, tokens_out=B, dep=dep),
    ]
    return ops


def prefill_ops(model: ModelSpec, n_prompt_tokens: int, dep: Deployment) -> list[Op]:
    """Prefill of one request with n_prompt_tokens (TTFT is measured on a
    single request occupying the replica)."""
    S = n_prompt_tokens
    tp = dep.tp
    kvh = _kv_heads_per_chip(model, tp)
    heads = model.n_heads / tp
    L = model.n_layers

    ops: list[Op] = [
        _linear("qkv_proj", S, model.attn_qkv_params, model.d_model,
                (model.n_heads + 2 * model.n_kv_heads) * model.d_head, dep, count=L),
        Op(
            name="attention",
            kind=OpKind.COMPUTE,
            dtype=dep.act_dtype,
            category="attention",
            count=L,
            # causal self-attention: 2 * (S^2/2) MACs per head for QK^T, same for PV
            flops=heads * 2 * 2.0 * (S * S / 2.0) * model.d_head,
            # speed-of-light flash attention: read K,V once, write them once
            dram_read=S * 2 * kvh * model.d_head * dep.kv_dtype.bytes,
            dram_write=S * 2 * kvh * model.d_head * dep.kv_dtype.bytes,
        ),
        _linear("out_proj", S, model.attn_out_params,
                model.n_heads * model.d_head, model.d_model, dep, count=L),
        *_ffn_ops(model, S, dep, count=L),
        _allreduce("allreduce", S, model, dep, count=2 * L),
        # only the last position needs logits during prefill
        *_embed_and_head(model, tokens_in=S, tokens_out=1, dep=dep),
    ]
    return ops
