"""Lowering: (model, scenario, deployment) -> per-chip operation lists.

An Op is a resource-demand record: FLOPs to execute, bytes to move through
the DRAM path, bytes to communicate.  Engines decide what those demands cost
on a given chip.  The roofline engine treats ops as a sequential critical
path; a future discrete-event engine can consume the same ops with explicit
dependencies and resource contention.

All demands are *per chip* and *per instance*; `count` says how many
identical instances run (e.g. one per layer).

Parallelism mapping (a replica = tp * pp * ep chips):

  TP  every matrix is sharded tp ways; partial results are summed with ring
      allreduces over the tp group.
  PP  layers are split into pp equal stages.  Decode runs pp microbatches
      (each batch/(pp*ep) sequences per attention group) round-robin through
      the pipeline; because stages are balanced, the pipeline round time --
      which is what a request experiences as TPOT -- equals the whole-model
      op list evaluated at the microbatch size, plus pp P2P hops.  Prefill
      of a single request traverses the stages sequentially, so its time is
      the whole-model op list at full sequence length plus pp-1 hops.
  EP  (MoE) attention/dense weights are replicated across ep groups, each
      group running tp-sharded attention on its own share of the batch;
      expert weights are sharded across the full tp*ep array.  Tokens are
      shuffled to their experts' owners with dispatch/combine all-to-alls,
      and the FFN allreduce disappears.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .hardware import DType
from .workload import Deployment, ModelSpec, Scenario


class OpKind(str, Enum):
    COMPUTE = "compute"  # math + DRAM traffic on one chip
    ALLREDUCE = "allreduce"  # ring collective across the tp group
    ALLTOALL = "alltoall"  # MoE dispatch/combine across the tp*ep array
    P2P = "p2p"  # activation hop between adjacent pipeline stages


COMM_KINDS = {OpKind.ALLREDUCE, OpKind.ALLTOALL, OpKind.P2P}


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
    comm_bytes: float = 0.0  # payload per chip for communication ops

    @property
    def is_comm(self) -> bool:
        return self.kind in COMM_KINDS


def validate_deployment(model: ModelSpec, dep: Deployment) -> None:
    if min(dep.tp, dep.pp, dep.ep) < 1:
        raise ValueError("tp, pp and ep must all be >= 1")
    if dep.ep > 1 and model.moe is None:
        raise ValueError(
            f"ep={dep.ep} but {model.name} is dense; expert parallelism needs a MoE model"
        )


def _kv_heads_per_chip(model: ModelSpec, tp: int) -> float:
    """KV heads resident per chip: sharded up to n_kv_heads ways, then
    replicated (standard GQA tensor parallelism)."""
    return model.n_kv_heads / min(tp, model.n_kv_heads)


def kv_cache_bytes_per_chip(model: ModelSpec, n_tokens: float, dep: Deployment) -> float:
    """KV bytes per chip for n_tokens total cached tokens in the replica.

    A chip stores its pipeline stage's layers (1/pp) for its attention
    group's sequences (1/ep), sharded across kv heads (up to n_kv ways).
    """
    per_layer = 2 * _kv_heads_per_chip(model, dep.tp) * model.d_head * dep.kv_dtype.bytes
    return model.n_layers * per_layer * n_tokens / (dep.pp * dep.ep)


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
    shard: int | None = None,
) -> Op:
    """A weight GEMM: [tokens, d_in] x [d_in, d_out], weights sharded
    `shard` ways (default tp).

    weight_read_params overrides how many parameters actually stream from
    DRAM (MoE reads only the activated experts).
    """
    shard = dep.tp if shard is None else shard
    wread = params if weight_read_params is None else weight_read_params
    return Op(
        name=name,
        kind=OpKind.COMPUTE,
        dtype=dep.weight_dtype,
        category=category,
        count=count,
        flops=2.0 * tokens * params / shard,
        dram_read=wread / shard * dep.weight_dtype.bytes + tokens * d_in * dep.act_dtype.bytes,
        dram_write=tokens * (d_out / shard) * dep.act_dtype.bytes,
    )


def _attention(model: ModelSpec, n_seq: float, q_len: float, kv_len: float,
               causal_new: bool, dep: Deployment, count: int) -> Op:
    """Self-attention over the KV cache for n_seq sequences.

    q_len queries attend to kv_len cached tokens each; causal_new halves the
    score work (prefill attends triangularly to its own tokens)."""
    kvh = _kv_heads_per_chip(model, dep.tp)
    heads = model.n_heads / dep.tp
    frac = 0.5 if causal_new else 1.0
    return Op(
        name="attention",
        kind=OpKind.COMPUTE,
        dtype=dep.act_dtype,
        category="attention",
        count=count,
        # QK^T and PV: 2 matmuls of [q_len, kv_len] x [kv_len, d_head] per head
        flops=n_seq * heads * 2 * 2.0 * q_len * kv_len * frac * model.d_head,
        # speed-of-light flash attention: stream K,V once; append new K,V
        dram_read=n_seq * kv_len * 2 * kvh * model.d_head * dep.kv_dtype.bytes,
        dram_write=n_seq * q_len * 2 * kvh * model.d_head * dep.kv_dtype.bytes,
    )


def _ffn_ops(model: ModelSpec, tokens_per_group: float, tokens_total: float,
             dep: Deployment, count: int) -> list[Op]:
    """FFN for one layer.

    tokens_per_group: tokens seen by one attention/ep group (drives the
    dense/shared path and the all-to-all payload per chip).
    tokens_total: tokens across all ep groups (drives expert traffic, since
    expert weights are sharded over the whole tp*ep array).
    """
    if model.moe is None:
        return [
            _linear("ffn", tokens_per_group, model.ffn_params_total,
                    model.d_model, model.d_model, dep, count)
        ]
    moe = model.moe
    ops: list[Op] = []
    expert_shard = dep.tp * dep.ep
    active = moe.expected_active_experts(int(tokens_total))
    ops.append(
        _linear(
            "moe_routed",
            tokens_total,
            moe.top_k * model.expert_params,
            model.d_model,
            model.d_model,
            dep,
            count,
            category="moe",
            weight_read_params=active * model.expert_params,
            shard=expert_shard,
        )
    )
    if dep.ep > 1:
        # dispatch tokens to expert owners and combine the results back
        payload = tokens_per_group * moe.top_k * model.d_model * dep.act_dtype.bytes / dep.tp
        for name in ("moe_dispatch", "moe_combine"):
            ops.append(Op(name=name, kind=OpKind.ALLTOALL, dtype=dep.act_dtype,
                          category="comm", count=count, comm_bytes=payload))
    if model.shared_expert_params:
        ops.append(
            _linear("moe_shared", tokens_per_group, model.shared_expert_params,
                    model.d_model, model.d_model, dep, count, category="moe")
        )
    return ops


def _allreduce(tokens: float, model: ModelSpec, dep: Deployment, count: int) -> Op:
    return Op(
        name="allreduce",
        kind=OpKind.ALLREDUCE,
        dtype=dep.act_dtype,
        category="comm",
        count=count,
        comm_bytes=tokens * model.d_model * dep.act_dtype.bytes,
    )


def _allreduces_per_layer(model: ModelSpec, dep: Deployment) -> int:
    """One reduction after attention and one after the FFN, except that with
    EP the FFN result is combined by the all-to-all instead."""
    return 1 if (model.moe is not None and dep.ep > 1) else 2


def _pipeline_hops(tokens: float, model: ModelSpec, dep: Deployment, count: int) -> list[Op]:
    if dep.pp <= 1 or count <= 0:
        return []
    return [Op(
        name="pp_hop",
        kind=OpKind.P2P,
        dtype=dep.act_dtype,
        category="comm",
        count=count,
        # the sending stage's tp chips each ship a slice of the activations
        comm_bytes=tokens * model.d_model * dep.act_dtype.bytes / dep.tp,
    )]


def _embed_and_head(model: ModelSpec, tokens_in: float, tokens_out: float,
                    dep: Deployment, count: int) -> list[Op]:
    """Embedding lookup for tokens_in positions, LM head for tokens_out."""
    return [
        Op(
            name="embed",
            kind=OpKind.COMPUTE,
            dtype=dep.weight_dtype,
            category="head",
            count=count,
            dram_read=tokens_in * model.d_model * dep.weight_dtype.bytes,
            dram_write=tokens_in * model.d_model * dep.act_dtype.bytes,
        ),
        _linear("lm_head", tokens_out, model.vocab_size * model.d_model,
                model.d_model, model.vocab_size, dep, count=count, category="head"),
    ]


def decode_ops(model: ModelSpec, dep: Deployment, batch: float, ctx: float) -> list[Op]:
    """One decode round for `batch` sequences, each at (mean) context `ctx`.

    Split out of `decode_step_ops` so callers that step the batch or the
    context by hand -- notably the request-level serving simulator, which
    varies the running-batch size and grows the KV per token -- can lower a
    decode iteration at an arbitrary (batch, ctx) without a Scenario.  Only
    the attention op depends on `ctx`; everything else is a function of
    `batch` alone (and, for MoE, of the expected active experts at that batch).

    With PP this is the pipeline round over pp microbatches; because stages
    are balanced it reduces to the whole-model op list at microbatch size
    (each stage streams its weights once per microbatch) plus pp hops.
    """
    validate_deployment(model, dep)
    B = batch
    L = model.n_layers
    b_att = B / (dep.pp * dep.ep)  # sequences per attention group per microbatch
    b_tok = B / dep.pp  # tokens per microbatch across all ep groups

    ops: list[Op] = [
        _linear("qkv_proj", b_att, model.attn_qkv_params, model.d_model,
                (model.n_heads + 2 * model.n_kv_heads) * model.d_head, dep, count=L),
        decode_attention_op(model, dep, batch, ctx),
        _linear("out_proj", b_att, model.attn_out_params,
                model.n_heads * model.d_head, model.d_model, dep, count=L),
        *_ffn_ops(model, tokens_per_group=b_att, tokens_total=b_tok, dep=dep, count=L),
        _allreduce(b_att, model, dep, count=_allreduces_per_layer(model, dep) * L),
        # pp hops per round: pp-1 forward plus the wrap-around to start the
        # next token; embed/head run once per microbatch on the edge stages
        *_pipeline_hops(b_att, model, dep, count=dep.pp),
        *_embed_and_head(model, tokens_in=b_att, tokens_out=b_att, dep=dep, count=dep.pp),
    ]
    return ops


def decode_attention_op(model: ModelSpec, dep: Deployment, batch: float, ctx: float) -> Op:
    """The single self-attention op of one decode iteration: `batch` sequences
    each emit one token attending to `ctx` cached tokens.  Its flops and KV
    bytes are linear in `batch * ctx` (= the total context streamed this step),
    so a serving loop can recost just this op as the KV cache grows while the
    rest of the decode step stays fixed for a given batch size."""
    b_att = batch / (dep.pp * dep.ep)  # sequences per attention group
    return _attention(model, n_seq=b_att, q_len=1, kv_len=ctx, causal_new=False,
                      dep=dep, count=model.n_layers)


def decode_step_ops(model: ModelSpec, scen: Scenario, dep: Deployment) -> list[Op]:
    """One decode round: every one of `batch` sequences emits one token, at
    the scenario's mean context length."""
    return decode_ops(model, dep, scen.batch, scen.mean_context)


def prefill_ops(model: ModelSpec, n_prompt_tokens: int, dep: Deployment) -> list[Op]:
    """Prefill of one request with n_prompt_tokens (TTFT is measured on a
    single request; it occupies one attention group and traverses the pp
    stages sequentially, so the whole-model op list is its critical path)."""
    validate_deployment(model, dep)
    S = n_prompt_tokens
    L = model.n_layers

    ops: list[Op] = [
        _linear("qkv_proj", S, model.attn_qkv_params, model.d_model,
                (model.n_heads + 2 * model.n_kv_heads) * model.d_head, dep, count=L),
        _attention(model, n_seq=1, q_len=S, kv_len=S, causal_new=True,
                   dep=dep, count=L),
        _linear("out_proj", S, model.attn_out_params,
                model.n_heads * model.d_head, model.d_model, dep, count=L),
        *_ffn_ops(model, tokens_per_group=S, tokens_total=S, dep=dep, count=L),
        _allreduce(S, model, dep, count=_allreduces_per_layer(model, dep) * L),
        *_pipeline_hops(S, model, dep, count=dep.pp - 1),
        # only the last position needs logits during prefill
        *_embed_and_head(model, tokens_in=S, tokens_out=1, dep=dep, count=1),
    ]
    return ops


def prefill_chunk_ops(model: ModelSpec, dep: Deployment, chunk: int,
                      prior_context: int, produce_logits: bool) -> list[Op]:
    """One chunked-prefill iteration: `chunk` fresh prompt tokens processed with
    `prior_context` tokens already in the KV cache (Sarathi / vLLM chunked
    prefill).

    The chunk's `chunk` query positions attend to kv_len = prior_context + chunk
    keys.  `causal_new=False` costs that as a full [chunk x kv_len] block: the
    cross-chunk part (chunk x prior_context) is genuinely dense and the
    intra-chunk triangle is over-counted by chunk^2/2 -- negligible for
    chunk << context, and the term that actually makes chunking cost MORE than
    an exclusive prefill lives in two places this lowering already captures: the
    attention op re-reads the whole prior KV every chunk (dram_read grows with
    prior_context), and the per-layer weights are re-streamed once per chunk
    (each chunk is its own op list).  Only the final chunk runs the LM head
    (it emits the request's first token)."""
    validate_deployment(model, dep)
    L = model.n_layers
    kv_len = prior_context + chunk
    edge = _embed_and_head(model, tokens_in=chunk, tokens_out=1, dep=dep, count=1)
    ops: list[Op] = [
        _linear("qkv_proj", chunk, model.attn_qkv_params, model.d_model,
                (model.n_heads + 2 * model.n_kv_heads) * model.d_head, dep, count=L),
        _attention(model, n_seq=1, q_len=chunk, kv_len=kv_len, causal_new=False,
                   dep=dep, count=L),
        _linear("out_proj", chunk, model.attn_out_params,
                model.n_heads * model.d_head, model.d_model, dep, count=L),
        *_ffn_ops(model, tokens_per_group=chunk, tokens_total=chunk, dep=dep, count=L),
        _allreduce(chunk, model, dep, count=_allreduces_per_layer(model, dep) * L),
        *_pipeline_hops(chunk, model, dep, count=dep.pp - 1),
        edge[0],  # embed the chunk's tokens
    ]
    if produce_logits:
        ops.append(edge[1])  # LM head on the last position -> first output token
    return ops
