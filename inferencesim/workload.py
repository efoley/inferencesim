"""Workload description: transformer model specs and serving scenarios."""

from __future__ import annotations

from dataclasses import dataclass

from .hardware import DType


@dataclass(frozen=True)
class MoEConfig:
    """Mixture-of-experts FFN. d_ff_shared is the total width of always-on
    shared expert(s); 0 means none.

    `n_dense_layers` (the DeepSeek `first_k_dense_replace` pattern) makes the
    first k layers plain *dense* FFNs (width `ModelSpec.d_ff`) instead of MoE
    -- they carry far fewer parameters than a full expert bank, so counting
    them dense is what makes the total param count land (DeepSeek-V3's 3 dense
    layers are ~33 B of the 671 B).  It is applied to parameter counting and
    the per-chip weight footprint; the *op lowering* still costs every layer as
    MoE (a documented ~n_dense/n_layers over-count of expert streaming --
    negligible at 3/61 layers)."""

    n_experts: int
    top_k: int
    d_ff_expert: int
    d_ff_shared: int = 0
    n_dense_layers: int = 0

    def expected_active_experts(self, n_tokens: int) -> float:
        """Expected number of *distinct* routed experts touched by n_tokens,
        assuming uniform routing.  Governs how many expert weights must be
        streamed from DRAM in a decode step."""
        if n_tokens <= 0:
            return 0.0
        p_untouched = (1.0 - self.top_k / self.n_experts) ** n_tokens
        return self.n_experts * (1.0 - p_untouched)


@dataclass(frozen=True)
class MLAConfig:
    """Multi-head latent attention (DeepSeek-V2/V3).  Q, K and V are produced
    from low-rank latents; the KV *cache* stores only the shared compressed
    latent `c^{KV}` (dim `kv_lora_rank`) plus one decoupled RoPE key
    (`qk_rope_head_dim`) per token -- shared across all heads, so ~1-2 orders
    of magnitude smaller than an MHA cache.  Names follow DeepSeek-V3's config.

      kv_lora_rank      d_c, the compressed KV latent (512 in V3)
      qk_rope_head_dim  d_R, the decoupled RoPE key/query dim (64)
      qk_nope_head_dim  the non-RoPE per-head q/k dim (128)
      v_head_dim        per-head value dim (128)
      q_lora_rank       optional query down-projection rank (1536; None = q is
                        projected directly from d_model)
    """

    kv_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    q_lora_rank: int | None = None

    @property
    def qk_head_dim(self) -> int:
        """Full per-head query/key width (non-RoPE + RoPE)."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def latent_dim(self) -> int:
        """Per-token, per-layer cached width: compressed KV latent + RoPE key,
        shared across all heads."""
        return self.kv_lora_rank + self.qk_rope_head_dim


@dataclass(frozen=True)
class ModelSpec:
    """A decoder-only transformer (dense or MoE) with GQA, sliding-window, or
    multi-head-latent (MLA) attention.

    Attention variants (mutually exclusive, validated):
      * plain GQA (the default): `n_kv_heads` KV heads.
      * sliding window (`swa_window` + `swa_every`): windowed layers cap their
        KV cache and decode reads at `swa_window` tokens.  `swa_every` selects
        which layers are windowed -- layer `i` is windowed iff
        `i % swa_every == 0`, so `swa_every=1` windows every layer (Mistral)
        and `swa_every=2` windows every other layer (gpt-oss).  `swa_every=0`
        (the default) means no windowing.
      * MLA (`mla`): the DeepSeek latent-attention cache.
    """

    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    d_ff: int  # dense FFN width (also the width of MoE `n_dense_layers`)
    vocab_size: int
    gated_mlp: bool = True  # SwiGLU-style: 3 matrices instead of 2
    moe: MoEConfig | None = None
    tied_embeddings: bool = False
    swa_window: int | None = None  # sliding-window size (tokens); None = dense
    swa_every: int = 0  # window layer i iff i % swa_every == 0 (0 = none)
    mla: MLAConfig | None = None

    def __post_init__(self) -> None:
        if self.mla is not None and self.swa_window is not None:
            raise ValueError(
                f"{self.name}: MLA and sliding-window attention are mutually "
                "exclusive (no known model combines them)"
            )
        if self.swa_window is not None and self.swa_every < 1:
            raise ValueError(
                f"{self.name}: swa_window set but swa_every < 1; set swa_every "
                ">= 1 (1 = every layer, 2 = every other layer)"
            )
        if self.swa_every >= 1 and self.swa_window is None:
            raise ValueError(
                f"{self.name}: swa_every set but no swa_window given"
            )

    # ---- attention-variant layer split ----------------------------------

    @property
    def n_swa_layers(self) -> int:
        """Number of sliding-window layers (0 unless SWA is configured)."""
        if self.swa_window is None or self.swa_every < 1:
            return 0
        return (self.n_layers + self.swa_every - 1) // self.swa_every

    @property
    def n_full_attn_layers(self) -> int:
        """Layers that attend to the full context (non-windowed)."""
        return self.n_layers - self.n_swa_layers

    # ---- per-layer parameter counts -------------------------------------

    @property
    def _ffn_matrices(self) -> int:
        return 3 if self.gated_mlp else 2

    @property
    def _dense_ffn_params(self) -> float:
        """Parameters of one plain dense FFN layer (width d_ff)."""
        return self._ffn_matrices * self.d_model * self.d_ff

    @property
    def attn_qkv_params(self) -> float:
        """Input-side attention projections (everything but the output proj)."""
        if self.mla is not None:
            m = self.mla
            q = (self.d_model * m.q_lora_rank + m.q_lora_rank * self.n_heads * m.qk_head_dim
                 if m.q_lora_rank else self.d_model * self.n_heads * m.qk_head_dim)
            kv_a = self.d_model * m.latent_dim  # down-proj to c^{KV} + RoPE key
            kv_b = m.kv_lora_rank * self.n_heads * (m.qk_nope_head_dim + m.v_head_dim)
            return q + kv_a + kv_b
        q = self.d_model * self.n_heads * self.d_head
        kv = 2 * self.d_model * self.n_kv_heads * self.d_head
        return q + kv

    @property
    def attn_out_params(self) -> float:
        if self.mla is not None:
            return self.n_heads * self.mla.v_head_dim * self.d_model
        return self.n_heads * self.d_head * self.d_model

    @property
    def qkv_proj_out_dim(self) -> float:
        """Width written by the qkv projection (for activation DRAM traffic)."""
        if self.mla is not None:
            return self.n_heads * self.mla.qk_head_dim + self.mla.latent_dim
        return (self.n_heads + 2 * self.n_kv_heads) * self.d_head

    @property
    def out_proj_in_dim(self) -> float:
        """Width read by the output projection."""
        if self.mla is not None:
            return self.n_heads * self.mla.v_head_dim
        return self.n_heads * self.d_head

    @property
    def attn_params(self) -> float:
        return self.attn_qkv_params + self.attn_out_params

    @property
    def expert_params(self) -> float:
        """Parameters of one routed expert (MoE only)."""
        assert self.moe is not None
        return self._ffn_matrices * self.d_model * self.moe.d_ff_expert

    @property
    def shared_expert_params(self) -> float:
        if self.moe is None or self.moe.d_ff_shared == 0:
            return 0.0
        return self._ffn_matrices * self.d_model * self.moe.d_ff_shared

    @property
    def ffn_params_total(self) -> float:
        """All FFN parameters in one layer (all experts for MoE)."""
        if self.moe is None:
            return self._ffn_matrices * self.d_model * self.d_ff
        return self.moe.n_experts * self.expert_params + self.shared_expert_params

    @property
    def ffn_params_active(self) -> float:
        """FFN parameters exercised per token in one layer."""
        if self.moe is None:
            return self.ffn_params_total
        return self.moe.top_k * self.expert_params + self.shared_expert_params

    # ---- whole-model counts ---------------------------------------------

    @property
    def embedding_params(self) -> float:
        n = self.vocab_size * self.d_model
        return n if self.tied_embeddings else 2 * n

    @property
    def _n_dense_ffn_layers(self) -> int:
        """MoE layers that are actually plain dense FFNs (first_k_dense)."""
        return self.moe.n_dense_layers if self.moe is not None else 0

    def _whole_model_params(self, ffn_active: bool) -> float:
        """Sum attention + FFN over all layers + embeddings.  Honours the MoE
        `n_dense_layers` prefix (dense FFN) when set."""
        per_ffn = self.ffn_params_active if ffn_active else self.ffn_params_total
        nd = self._n_dense_ffn_layers
        if nd:
            moe_layers = self.n_layers - nd
            layers = (moe_layers * (self.attn_params + per_ffn)
                      + nd * (self.attn_params + self._dense_ffn_params))
        else:
            layers = self.n_layers * (self.attn_params + per_ffn)
        return layers + self.embedding_params

    @property
    def total_params(self) -> float:
        return self._whole_model_params(ffn_active=False)

    @property
    def active_params(self) -> float:
        return self._whole_model_params(ffn_active=True)

    def weight_bytes(self, dtype: DType) -> float:
        return self.total_params * dtype.bytes

    def kv_bytes_per_token(self, dtype: DType) -> float:
        """Whole-logical-cache bytes for one token (all layers, all KV heads).

        MLA stores only the shared compressed latent; SWA's per-layer cap is a
        per-sequence (context-dependent) effect applied in
        `ops.kv_cache_bytes_per_chip`, so this uncapped per-token figure is the
        growth rate before any window saturates."""
        if self.mla is not None:
            return self.n_layers * self.mla.latent_dim * dtype.bytes
        return self.n_layers * 2 * self.n_kv_heads * self.d_head * dtype.bytes


@dataclass(frozen=True)
class Scenario:
    """One serving operating point (per model replica)."""

    batch: int  # concurrent sequences per replica (continuous batching slots)
    prompt_len: int
    output_len: int

    @property
    def mean_context(self) -> float:
        """Average context length over a request's decode phase."""
        return self.prompt_len + self.output_len / 2.0

    @property
    def max_context(self) -> int:
        return self.prompt_len + self.output_len


@dataclass(frozen=True)
class Deployment:
    """How the model is mapped onto the hardware.

    A replica occupies tp * pp * ep * adp chips:

      tp -- tensor parallelism: every weight matrix sharded tp ways; adds
            2 allreduces per layer (1 for MoE layers when ep > 1, or dense
            layers when adp > 1).
      pp -- pipeline parallelism: layers split into pp stages; decode runs
            pp microbatches of batch/(pp*ep*adp) through the pipeline, so TPOT
            is the pipeline round time and per-chip memory drops ~1/pp.
            Wins come from raising `batch` into the freed memory, not from
            faster weight streaming.
      ep -- expert parallelism (MoE only): attention runs data-parallel
            across ep groups (each tp-sharded, handling batch/ep sequences)
            while expert weights are sharded over the full tp*ep array;
            the FFN allreduce is replaced by dispatch/combine all-to-alls.
      adp -- attention data parallelism (dense only): the DeepSeek-V3-style
            "DP attention + TP FFN" pattern, and TRT-LLM's DEPn for dense.
            Attention runs data-parallel across adp groups (each tp-sharded,
            handling batch/adp sequences, with attention weights replicated
            across groups) so per-chip KV divides by adp; the dense FFN weights
            shard over the whole tp*adp array (better weight streaming), and the
            FFN allreduce is replaced by a sequence-gather before the FFN and a
            reduce-scatter after.  MoE attention-DP is what `ep` already
            provides, so adp is dense-only (validated).
    """

    tp: int = 1
    pp: int = 1
    ep: int = 1
    adp: int = 1
    weight_dtype: DType = DType.FP8
    kv_dtype: DType = DType.BF16
    act_dtype: DType = DType.BF16
    # If True, assume collectives fully overlap with compute/memory work
    # (phase time = max of the two streams instead of their sum).  Real stacks
    # land somewhere between the two settings.
    overlap_comm: bool = False

    @property
    def replica_chips(self) -> int:
        return self.tp * self.pp * self.ep * self.adp
