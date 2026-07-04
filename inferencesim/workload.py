"""Workload description: transformer model specs and serving scenarios."""

from __future__ import annotations

from dataclasses import dataclass

from .hardware import DType


@dataclass(frozen=True)
class MoEConfig:
    """Mixture-of-experts FFN. d_ff_shared is the total width of always-on
    shared expert(s); 0 means none."""

    n_experts: int
    top_k: int
    d_ff_expert: int
    d_ff_shared: int = 0

    def expected_active_experts(self, n_tokens: int) -> float:
        """Expected number of *distinct* routed experts touched by n_tokens,
        assuming uniform routing.  Governs how many expert weights must be
        streamed from DRAM in a decode step."""
        if n_tokens <= 0:
            return 0.0
        p_untouched = (1.0 - self.top_k / self.n_experts) ** n_tokens
        return self.n_experts * (1.0 - p_untouched)


@dataclass(frozen=True)
class ModelSpec:
    """A decoder-only transformer (dense or MoE) with GQA attention."""

    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    d_ff: int  # dense FFN width (ignored for MoE layers)
    vocab_size: int
    gated_mlp: bool = True  # SwiGLU-style: 3 matrices instead of 2
    moe: MoEConfig | None = None
    tied_embeddings: bool = False

    # ---- per-layer parameter counts -------------------------------------

    @property
    def _ffn_matrices(self) -> int:
        return 3 if self.gated_mlp else 2

    @property
    def attn_qkv_params(self) -> float:
        q = self.d_model * self.n_heads * self.d_head
        kv = 2 * self.d_model * self.n_kv_heads * self.d_head
        return q + kv

    @property
    def attn_out_params(self) -> float:
        return self.n_heads * self.d_head * self.d_model

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
    def total_params(self) -> float:
        return self.n_layers * (self.attn_params + self.ffn_params_total) + self.embedding_params

    @property
    def active_params(self) -> float:
        return self.n_layers * (self.attn_params + self.ffn_params_active) + self.embedding_params

    def weight_bytes(self, dtype: DType) -> float:
        return self.total_params * dtype.bytes

    def kv_bytes_per_token(self, dtype: DType) -> float:
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

    A replica occupies tp * pp * ep chips:

      tp -- tensor parallelism: every weight matrix sharded tp ways; adds
            2 allreduces per layer (1 for MoE layers when ep > 1).
      pp -- pipeline parallelism: layers split into pp stages; decode runs
            pp microbatches of batch/(pp*ep) through the pipeline, so TPOT
            is the pipeline round time and per-chip memory drops ~1/pp.
            Wins come from raising `batch` into the freed memory, not from
            faster weight streaming.
      ep -- expert parallelism (MoE only): attention runs data-parallel
            across ep groups (each tp-sharded, handling batch/ep sequences)
            while expert weights are sharded over the full tp*ep array;
            the FFN allreduce is replaced by dispatch/combine all-to-alls.
    """

    tp: int = 1
    pp: int = 1
    ep: int = 1
    weight_dtype: DType = DType.FP8
    kv_dtype: DType = DType.BF16
    act_dtype: DType = DType.BF16
    # If True, assume collectives fully overlap with compute/memory work
    # (phase time = max of the two streams instead of their sum).  Real stacks
    # land somewhere between the two settings.
    overlap_comm: bool = False

    @property
    def replica_chips(self) -> int:
        return self.tp * self.pp * self.ep
