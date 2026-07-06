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
      (each batch/(pp*ep*adp) sequences per attention group) round-robin through
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
  ADP (dense) attention runs data-parallel across adp groups (each tp-sharded,
      handling batch/adp sequences, attention weights replicated across the
      groups) while the dense FFN weights shard over the full tp*adp array.
      The FFN allreduce is replaced by an allgather that assembles each token's
      full hidden state before the FFN and a reduce-scatter after it -- the
      DeepSeek-V3 "DP attention + TP FFN" pattern, and TRT-LLM's dense DEPn.
      Per-chip KV divides by adp (batch-sharded); the FFN streams 1/(tp*adp) of
      the weights.  adp is dense-only (MoE attention-DP is exactly ep).
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
    # one-pass ring collective across the tp*adp array: the dense attention-DP
    # FFN's allgather (assemble the full-batch hidden state) and reduce-scatter
    # (reduce it back to sequence shards).  Both cost the same -- exactly HALF a
    # ring allreduce (g-1 steps vs 2(g-1)) -- since reduce-scatter is the
    # arithmetic dual of allgather with identical communication volume.
    HALFRING = "halfring"
    P2P = "p2p"  # activation hop between adjacent pipeline stages


COMM_KINDS = {OpKind.ALLREDUCE, OpKind.ALLTOALL, OpKind.HALFRING, OpKind.P2P}


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
    # MoE dispatch/combine only: the per-member routing-popularity vector over
    # the tp*ep array (sum == 1), so the discrete-event engine can size the
    # all-to-all's per-member messages under expert-load skew (incast onto the
    # hot owners).  None everywhere else (including uniform MoE), so non-MoE and
    # roofline paths are untouched -- `comm_bytes` stays the per-chip average.
    member_weights: tuple[float, ...] | None = None

    @property
    def is_comm(self) -> bool:
        return self.kind in COMM_KINDS


def validate_deployment(model: ModelSpec, dep: Deployment) -> None:
    if min(dep.tp, dep.pp, dep.ep, dep.adp) < 1:
        raise ValueError("tp, pp, ep and adp must all be >= 1")
    if dep.ep > 1 and model.moe is None:
        raise ValueError(
            f"ep={dep.ep} but {model.name} is dense; expert parallelism needs a MoE model"
        )
    if dep.adp > 1 and model.moe is not None:
        raise ValueError(
            f"adp={dep.adp} but {model.name} is MoE; MoE attention data-parallelism "
            f"is exactly what expert parallelism provides (use ep={dep.adp}, tp=1) -- "
            f"adp is dense-only (DP attention + TP FFN)"
        )


def _kv_heads_per_chip(model: ModelSpec, tp: int) -> float:
    """KV heads resident per chip: sharded up to n_kv_heads ways, then
    replicated (standard GQA tensor parallelism)."""
    return model.n_kv_heads / min(tp, model.n_kv_heads)


def kv_cache_bytes_per_chip(model: ModelSpec, n_tokens: float, dep: Deployment) -> float:
    """KV bytes per chip for ONE sequence holding `n_tokens` of context (the
    per-sequence footprint; callers multiply by the batch).

    A chip stores its pipeline stage's layers (1/pp) for its attention
    group's sequences (1/ep for MoE, 1/adp for dense attention-DP).  The
    batch-sharding by ep/adp is the point of those patterns: it cuts per-chip
    KV by the group degree.

    Attention variants:
      * GQA: sharded across kv heads (up to n_kv ways); linear in n_tokens.
      * MLA: only the shared compressed latent (kv_lora_rank + qk_rope_head_dim)
        is cached, replicated across the tp group (it is tiny), so per-chip KV
        does NOT divide by tp -- only by pp*ep*adp.
      * SWA: windowed layers cap at `swa_window` tokens (a ring buffer), so the
        footprint is sub-linear once the window saturates.
    """
    groups = dep.pp * dep.ep * dep.adp
    if model.mla is not None:
        # compressed latent, shared across heads AND replicated across tp
        per_token = model.n_layers * model.mla.latent_dim * dep.kv_dtype.bytes
        return per_token * n_tokens / groups
    per_layer = 2 * _kv_heads_per_chip(model, dep.tp) * model.d_head * dep.kv_dtype.bytes
    swa = model.n_swa_layers
    if swa:
        cached = model.n_full_attn_layers * n_tokens + swa * min(n_tokens, model.swa_window)
        return per_layer * cached / groups
    return model.n_layers * per_layer * n_tokens / (dep.pp * dep.ep * dep.adp)


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


def _attention_ops(model: ModelSpec, n_seq: float, q_len: float, kv_len: float,
                   causal_new: bool, dep: Deployment, count: int) -> list[Op]:
    """Self-attention over the KV cache, lowered to one or more Ops.

    * GQA (default): a single `attention` op (count = all layers).
    * MLA: a single `attention` op that streams the shared compressed latent.
    * SWA: the layers split by class, so their heterogeneous costs are honest --
      an `attention` op over the `n_full_attn_layers` dense layers plus an
      `attention_swa` op over the `n_swa_layers` windowed layers (which read and
      score only the last `swa_window` cached tokens).  Both keep category
      `attention`, so the DES/serve cost paths bucket and recost them together.
    """
    if model.mla is not None:
        return [_mla_attention(model, n_seq, q_len, kv_len, causal_new, dep, count)]
    swa = model.n_swa_layers
    if not swa:
        return [_attention(model, n_seq, q_len, kv_len, causal_new, dep, count)]
    ops: list[Op] = []
    full = model.n_full_attn_layers
    if full:
        ops.append(_attention(model, n_seq, q_len, kv_len, causal_new, dep, full))
    ops.append(_swa_attention(model, n_seq, q_len, kv_len, causal_new, dep, swa))
    return ops


def _attention(model: ModelSpec, n_seq: float, q_len: float, kv_len: float,
               causal_new: bool, dep: Deployment, count: int,
               name: str = "attention") -> Op:
    """Full-context GQA self-attention for n_seq sequences.

    q_len queries attend to kv_len cached tokens each; causal_new halves the
    score work (prefill attends triangularly to its own tokens)."""
    kvh = _kv_heads_per_chip(model, dep.tp)
    heads = model.n_heads / dep.tp
    frac = 0.5 if causal_new else 1.0
    return Op(
        name=name,
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


def _swa_attention(model: ModelSpec, n_seq: float, q_len: float, kv_len: float,
                   causal_new: bool, dep: Deployment, count: int) -> Op:
    """Sliding-window GQA self-attention over the windowed layers.

    Decode / chunk (causal_new=False): each query attends only to the last
    W = swa_window cached tokens, so both the score work and the KV DRAM read
    stream min(kv_len, W) tokens instead of kv_len -- the decode win.

    Prefill (causal_new=True, q_len == kv_len == S): the causal score matrix is
    *banded*.  Position i attends to min(i+1, W) keys, so the number of score
    entries is sum_i min(i+1, W) = S^2/2 - max(0, S-W)^2/2, which is the full
    triangle S^2/2 when S <= W and S*W - W^2/2 when S >= W (dropping the +S/2
    diagonal, consistent with the dense op's frac=0.5).  Each K,V is still
    streamed once (every cached token lies in some query's band), so the DRAM
    read matches the dense prefill -- the prefill win is in FLOPs, not bytes."""
    W = model.swa_window
    kvh = _kv_heads_per_chip(model, dep.tp)
    heads = model.n_heads / dep.tp
    if causal_new:
        excess = max(0.0, q_len - W)
        score_pairs = 0.5 * (q_len * q_len - excess * excess)
        kv_tokens = kv_len
    else:
        eff = min(kv_len, W)
        score_pairs = q_len * eff
        kv_tokens = eff
    return Op(
        name="attention_swa",
        kind=OpKind.COMPUTE,
        dtype=dep.act_dtype,
        category="attention",
        count=count,
        flops=n_seq * heads * 2 * 2.0 * score_pairs * model.d_head,
        dram_read=n_seq * kv_tokens * 2 * kvh * model.d_head * dep.kv_dtype.bytes,
        dram_write=n_seq * q_len * 2 * kvh * model.d_head * dep.kv_dtype.bytes,
    )


def _mla_attention(model: ModelSpec, n_seq: float, q_len: float, kv_len: float,
                   causal_new: bool, dep: Deployment, count: int) -> Op:
    """Multi-head latent attention (DeepSeek-V2/V3).

    The KV cache is the shared compressed latent (kv_lora_rank + qk_rope_head_dim
    per token per layer), streamed ONCE per step regardless of head count -- that
    single small read is the MLA memory win.  It is replicated across the tp
    group (tiny), so the DRAM read does not divide by tp; only the FLOPs (which
    shard with the heads) do.

    Decode FLOPs use the absorbed-weight inference form (DeepSeek-V2 paper,
    arXiv:2405.04434, §"absorbing" the up-projections into Q/O): per cached
    position per head, the score q.[c^{KV};k^R] costs (d_c + d_R) MACs and the
    value softmax.c^{KV} costs d_c MACs, i.e. 2*d_c + d_R MACs/position/head.
    Prefill uses the naive decompressed form instead (the absorbed matrices are
    huge for long q, so real stacks decompress during prefill): per (query,key)
    pair per head, QK^T over qk_head_dim + PV over v_head_dim.  Simplifications:
    the small Q/KV down/up projections are folded into qkv_proj (a GEMM), and the
    RoPE application is free at speed-of-light."""
    m = model.mla
    assert m is not None
    heads = model.n_heads / dep.tp
    frac = 0.5 if causal_new else 1.0
    if q_len == 1:  # decode: absorbed weights, score + value against the latent
        per_pos = 2 * m.kv_lora_rank + m.qk_rope_head_dim
    else:  # prefill / chunk: decompressed per-head attention
        per_pos = m.qk_head_dim + m.v_head_dim
    latent = m.latent_dim
    return Op(
        name="attention",
        kind=OpKind.COMPUTE,
        dtype=dep.act_dtype,
        category="attention",
        count=count,
        flops=n_seq * heads * 2.0 * q_len * kv_len * frac * per_pos,
        dram_read=n_seq * kv_len * latent * dep.kv_dtype.bytes,
        dram_write=n_seq * q_len * latent * dep.kv_dtype.bytes,
    )


def _ffn_gather_scatter(model: ModelSpec, tokens_total: float,
                        dep: Deployment, count: int) -> list[Op]:
    """The two ring collectives that bracket a dense attention-DP FFN.

    With adp > 1 the FFN is tensor-parallel over the whole g = tp*adp array,
    but attention-DP leaves each token's hidden state sequence-sharded (one
    B/g-row shard per chip).  So before the FFN an **allgather** assembles the
    full [tokens_total, d_model] batch on every chip (the FFN's column-parallel
    first GEMM needs the whole d_model row), and after it a **reduce-scatter**
    sums the row-parallel output partials back to one B/g shard per chip.

    Payload and closed form (derived from the ring send volumes):
      Let D = tokens_total * d_model * act_bytes be the FULL-batch hidden state
      each chip ends up holding after the allgather.  A ring allgather over g
      chips runs g-1 steps; in each step every chip forwards one D/g-byte shard
      to its neighbour, so per-chip egress is (g-1)*(D/g) and the makespan is

          (g-1)/g * D/bw + (g-1)*lat.

      The reduce-scatter is the arithmetic dual -- same g-1 steps, same D/g per
      step -- so it has the identical closed form.  Together the two are exactly
      one ring allreduce of D (2(g-1) steps), which is what a non-adp TP FFN
      would pay: adp trades that single tp-group allreduce for an allgather +
      reduce-scatter over the larger tp*adp group.

    `comm_bytes` carries D (the per-rank result size); the engine's
    `ring_gather_time` applies the (g-1)/g and (g-1) factors with g = tp*adp.
    """
    if dep.adp <= 1:
        return []
    payload = tokens_total * model.d_model * dep.act_dtype.bytes
    return [
        Op(name=name, kind=OpKind.HALFRING, dtype=dep.act_dtype,
           category="comm", count=count, comm_bytes=payload)
        for name in ("ffn_gather", "ffn_scatter")
    ]


def _ffn_ops(model: ModelSpec, tokens_per_group: float, tokens_total: float,
             dep: Deployment, count: int) -> list[Op]:
    """FFN for one layer.

    tokens_per_group: tokens seen by one attention/ep group (drives the
    dense/shared path and the all-to-all payload per chip).
    tokens_total: tokens across all ep/adp groups (drives expert traffic and
    the dense attention-DP FFN, since those weights are sharded over the whole
    tp*ep / tp*adp array).
    """
    if model.moe is None:
        # dense FFN shards over the whole tp*adp array and processes the full
        # batch (all adp groups); with adp == 1 this is the historical tp-shard
        # over one group's tokens (tokens_total == tokens_per_group).  The
        # gather/scatter bracket vanishes at adp == 1.
        gather_scatter = _ffn_gather_scatter(model, tokens_total, dep, count)
        gather = gather_scatter[:1]  # allgather before the FFN (or nothing)
        scatter = gather_scatter[1:]  # reduce-scatter after (or nothing)
        ffn = _linear("ffn", tokens_total, model.ffn_params_total,
                      model.d_model, model.d_model, dep, count,
                      shard=dep.tp * dep.adp)
        return [*gather, ffn, *scatter]
    moe = model.moe
    ops: list[Op] = []
    expert_shard = dep.tp * dep.ep  # the expert-placement array (a2a group)
    # moe_routed is *per chip = the pacing chip*.  Uniform routing (skew=0) paces
    # by the average member -- the historical smeared per-chip share, kept
    # bit-identical.  Under skew the layer is paced by the HOTTEST member: its
    # activation flops/bytes scale by `hot_member_factor` (it processes
    # `tokens_to_member(hot)` routings) and its weight-byte read is the
    # `expected_active_on_member(hot)` distinct experts of its block.  This is
    # the roofline of an unbalanced layer (max-member); the DES recovers true
    # per-member costs via the dispatch/combine payload vector below.
    if moe.skew > 0.0:
        hot = moe.hot_member_factor(expert_shard)
        active_hot = moe.expected_active_on_member(0, int(tokens_total), expert_shard)
        routed_tokens = tokens_total * hot
        wread = active_hot * expert_shard * model.expert_params
        member_w: tuple[float, ...] | None = tuple(moe.member_popularity(expert_shard))
    else:
        routed_tokens = tokens_total
        wread = moe.expected_active_experts(int(tokens_total)) * model.expert_params
        member_w = None
    ops.append(
        _linear(
            "moe_routed",
            routed_tokens,
            moe.top_k * model.expert_params,
            model.d_model,
            model.d_model,
            dep,
            count,
            category="moe",
            weight_read_params=wread,
            shard=expert_shard,
        )
    )
    if dep.ep > 1:
        # dispatch tokens to expert owners and combine the results back.
        # comm_bytes stays the per-chip *average* payload; member_weights carries
        # the per-member routing-popularity vector so the DES can size the
        # non-uniform per-member messages (dispatch incasts onto the hot owner's
        # ingress, combine egresses the most from it).  None on the uniform path.
        payload = tokens_per_group * moe.top_k * model.d_model * dep.act_dtype.bytes / dep.tp
        for name in ("moe_dispatch", "moe_combine"):
            ops.append(Op(name=name, kind=OpKind.ALLTOALL, dtype=dep.act_dtype,
                          category="comm", count=count, comm_bytes=payload,
                          member_weights=member_w))
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
    """One reduction after attention and one after the FFN, except that the FFN
    reduction is subsumed by another collective -- the MoE dispatch/combine
    all-to-all (ep > 1), or the dense attention-DP gather/reduce-scatter
    (adp > 1) -- leaving only the within-group attention allreduce."""
    if model.moe is not None and dep.ep > 1:
        return 1
    if model.moe is None and dep.adp > 1:
        return 1
    return 2


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
    # sequences per attention group per microbatch (batch-sharded by ep for MoE,
    # by adp for dense attention-DP); tokens per microbatch across all groups
    b_att = B / (dep.pp * dep.ep * dep.adp)
    b_tok = B / dep.pp

    ops: list[Op] = [
        _linear("qkv_proj", b_att, model.attn_qkv_params, model.d_model,
                model.qkv_proj_out_dim, dep, count=L),
        *decode_attention_ops(model, dep, batch, ctx),
        _linear("out_proj", b_att, model.attn_out_params,
                model.out_proj_in_dim, model.d_model, dep, count=L),
        *_ffn_ops(model, tokens_per_group=b_att, tokens_total=b_tok, dep=dep, count=L),
        _allreduce(b_att, model, dep, count=_allreduces_per_layer(model, dep) * L),
        # pp hops per round: pp-1 forward plus the wrap-around to start the
        # next token; embed/head run once per microbatch on the edge stages
        *_pipeline_hops(b_att, model, dep, count=dep.pp),
        *_embed_and_head(model, tokens_in=b_att, tokens_out=b_att, dep=dep, count=dep.pp),
    ]
    return ops


def decode_attention_ops(model: ModelSpec, dep: Deployment, batch: float,
                         ctx: float) -> list[Op]:
    """The self-attention op(s) of one decode iteration: `batch` sequences each
    emit one token attending to `ctx` cached tokens.  Their flops and KV bytes
    are (piecewise) linear in the total context streamed this step, so a serving
    loop can recost just these ops as the KV cache grows while the rest of the
    decode step stays fixed for a given batch size.  Usually one op; SWA models
    return two (full-context + windowed layer classes), MLA one (compressed)."""
    b_att = batch / (dep.pp * dep.ep * dep.adp)  # sequences per attention group
    return _attention_ops(model, n_seq=b_att, q_len=1, kv_len=ctx, causal_new=False,
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
                model.qkv_proj_out_dim, dep, count=L),
        *_attention_ops(model, n_seq=1, q_len=S, kv_len=S, causal_new=True,
                        dep=dep, count=L),
        _linear("out_proj", S, model.attn_out_params,
                model.out_proj_in_dim, model.d_model, dep, count=L),
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
                model.qkv_proj_out_dim, dep, count=L),
        *_attention_ops(model, n_seq=1, q_len=chunk, kv_len=kv_len, causal_new=False,
                        dep=dep, count=L),
        _linear("out_proj", chunk, model.attn_out_params,
                model.out_proj_in_dim, model.d_model, dep, count=L),
        *_ffn_ops(model, tokens_per_group=chunk, tokens_total=chunk, dep=dep, count=L),
        _allreduce(chunk, model, dep, count=_allreduces_per_layer(model, dep) * L),
        *_pipeline_hops(chunk, model, dep, count=dep.pp - 1),
        edge[0],  # embed the chunk's tokens
    ]
    if produce_logits:
        ops.append(edge[1])  # LM head on the last position -> first output token
    return ops
