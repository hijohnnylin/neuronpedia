"""Attention mechanism analysis utilities for transformers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from .model_utils import gemma2_rmsnorm

if TYPE_CHECKING:
    from transformers import PreTrainedModel


def compute_qk_affinity_matrix(
    vectors: torch.Tensor,
    layer_idx: int,
    model: PreTrainedModel
) -> torch.Tensor:
    """
    Compute raw QK attention logits for all pairwise vector combinations.

    This computes the attention affinity Q@K^T / √d BEFORE softmax, which reveals
    the raw semantic affinity between query and key directions.

    Args:
        vectors: Tensor [n_vectors, hidden_dim] containing input vectors
        layer_idx: Layer index to analyze
        model: Transformer model (base or instruct)

    Returns:
        affinity_matrix: Tensor [n_vectors, n_vectors] - raw attention logits averaged across heads

    Notes:
        - Applies input layernorm before Q/K projection (as in actual forward pass)
        - Handles Grouped Query Attention (GQA) by repeating K heads
        - Averages across all attention heads for simplified interpretation
        - Returns raw logits (not softmaxed probabilities)
    """
    layer = model.model.layers[layer_idx]

    # 1. Apply input layernorm
    input_ln_weight = layer.input_layernorm.weight
    vectors_normed = gemma2_rmsnorm(vectors, input_ln_weight)

    # 2. Project to Q and K
    q_proj_weight = layer.self_attn.q_proj.weight
    k_proj_weight = layer.self_attn.k_proj.weight

    Q = F.linear(vectors_normed.float(), q_proj_weight.float())  # [n, num_heads * head_dim]
    K = F.linear(vectors_normed.float(), k_proj_weight.float())  # [n, num_kv_heads * head_dim]

    # 3. Reshape to heads
    num_heads = layer.config.num_attention_heads
    num_kv_heads = layer.config.num_key_value_heads
    head_dim = Q.shape[-1] // num_heads

    Q_heads = Q.view(-1, num_heads, head_dim)  # [n, num_heads, head_dim]
    K_heads = K.view(-1, num_kv_heads, head_dim)  # [n, num_kv_heads, head_dim]

    # 4. Handle GQA: repeat K heads if needed
    if num_kv_heads != num_heads:
        repeat_factor = num_heads // num_kv_heads
        K_heads = K_heads.repeat_interleave(repeat_factor, dim=1)  # [n, num_heads, head_dim]

    # 5. Compute pairwise attention logits: [n_queries, n_keys, num_heads]
    scores = torch.einsum('qhd,khd->qkh', Q_heads, K_heads) / (head_dim ** 0.5)

    # 6. Average across heads
    affinity_matrix = scores.mean(dim=-1)  # [n_queries, n_keys]

    return affinity_matrix.to(torch.float32)


def compute_vo_decomposition(
    key_vectors: torch.Tensor,
    probe_vectors: torch.Tensor,
    layer_idx: int,
    model: PreTrainedModel
) -> torch.Tensor:
    """
    Compute semantic decomposition of V→O transformation.

    This reveals what semantic content flows through the attention mechanism
    when attending to a given key vector. This is query-independent - it only
    depends on what is being attended to (the key).

    Args:
        key_vectors: Tensor [n_keys, hidden_dim] - vectors being attended to
        probe_vectors: Tensor [n_probes, hidden_dim] - semantic directions for decomposition
        layer_idx: Layer index to analyze
        model: Transformer model (base or instruct)

    Returns:
        decomposition: Tensor [n_keys, n_probes] - cosine similarity between outputs and probes

    Notes:
        - Projects keys through V, then through O (value-output path)
        - Applies post-attention layernorm
        - Returns cosine similarities to probe vectors (normalized dot products)
        - Handles GQA by repeating V heads to match expected head count
    """
    layer = model.model.layers[layer_idx]

    # 1. Apply input layernorm to keys
    input_ln_weight = layer.input_layernorm.weight
    keys_normed = gemma2_rmsnorm(key_vectors, input_ln_weight)

    # 2. Project through V
    v_proj_weight = layer.self_attn.v_proj.weight
    V = F.linear(keys_normed.float(), v_proj_weight.float())  # [n_keys, num_kv_heads * head_dim]

    num_heads = layer.config.num_attention_heads
    num_kv_heads = layer.config.num_key_value_heads
    head_dim = V.shape[-1] // num_kv_heads

    # 3. Handle GQA: repeat V heads if needed
    if num_kv_heads != num_heads:
        V = V.reshape(-1, num_kv_heads, head_dim)
        repeat_factor = num_heads // num_kv_heads
        V = V.repeat_interleave(repeat_factor, dim=1)  # [n, num_heads, head_dim]
        V = V.reshape(-1, num_heads * head_dim)

    # 4. Project through O
    o_proj_weight = layer.self_attn.o_proj.weight
    O_output = F.linear(V, o_proj_weight.float())  # [n_keys, hidden_dim]

    # 5. Apply post-attention layernorm
    post_attn_ln_weight = layer.post_attention_layernorm.weight
    output_normed = gemma2_rmsnorm(O_output, post_attn_ln_weight.float())

    # 6. Normalize for cosine similarity
    output_normalized = F.normalize(output_normed, dim=-1)  # [n_keys, hidden_dim]
    probes_normalized = F.normalize(probe_vectors.float(), dim=-1)  # [n_probes, hidden_dim]

    # 7. Compute cosine similarity: [n_keys, n_probes]
    decomposition = output_normalized @ probes_normalized.T

    return decomposition.to(torch.float32)


def compute_full_attention_patterns(
    vectors: torch.Tensor,
    layer_idx: int,
    base_model: PreTrainedModel,
    instruct_model: PreTrainedModel
) -> dict[str, torch.Tensor]:
    """
    Compute full attention analysis comparing base and instruct models.

    Computes both QK affinities and VO decompositions for both models,
    plus the deltas (instruct - base).

    Args:
        vectors: Tensor [n_vectors, hidden_dim]
        layer_idx: Layer to analyze
        base_model: Base model
        instruct_model: Instruction-tuned model

    Returns:
        Dict containing:
        - 'qk_base': QK affinity matrix for base model
        - 'qk_instruct': QK affinity matrix for instruct model
        - 'qk_delta': Difference (instruct - base)
        - 'vo_base': VO decomposition for base model
        - 'vo_instruct': VO decomposition for instruct model
        - 'vo_delta': Difference (instruct - base)

    Notes:
        - All computations done with torch.inference_mode() for efficiency
        - Useful for comparing how instruction tuning changes attention patterns
    """
    with torch.inference_mode():
        # QK Affinity
        qk_base = compute_qk_affinity_matrix(vectors, layer_idx, base_model)
        qk_instruct = compute_qk_affinity_matrix(vectors, layer_idx, instruct_model)
        qk_delta = qk_instruct - qk_base

        # VO Decomposition
        vo_base = compute_vo_decomposition(vectors, vectors, layer_idx, base_model)
        vo_instruct = compute_vo_decomposition(vectors, vectors, layer_idx, instruct_model)
        vo_delta = vo_instruct - vo_base

    return {
        'qk_base': qk_base,
        'qk_instruct': qk_instruct,
        'qk_delta': qk_delta,
        'vo_base': vo_base,
        'vo_instruct': vo_instruct,
        'vo_delta': vo_delta
    }


def compute_attention_head_patterns(
    query_vectors: torch.Tensor,
    key_vectors: torch.Tensor,
    layer_idx: int,
    model: PreTrainedModel,
    head_idx: Optional[int] = None
) -> torch.Tensor:
    """
    Compute per-head attention patterns (optionally for a specific head).

    Args:
        query_vectors: Tensor [n_queries, hidden_dim]
        key_vectors: Tensor [n_keys, hidden_dim]
        layer_idx: Layer index
        model: Transformer model
        head_idx: Optional specific head to analyze (None = all heads)

    Returns:
        If head_idx is None: Tensor [n_heads, n_queries, n_keys]
        If head_idx is int: Tensor [n_queries, n_keys]

    Notes:
        - Returns raw attention logits (not softmaxed)
        - Useful for understanding per-head specialization
    """
    layer = model.model.layers[layer_idx]

    # Apply input layernorm
    input_ln_weight = layer.input_layernorm.weight
    queries_normed = gemma2_rmsnorm(query_vectors, input_ln_weight)
    keys_normed = gemma2_rmsnorm(key_vectors, input_ln_weight)

    # Project to Q and K
    q_proj_weight = layer.self_attn.q_proj.weight
    k_proj_weight = layer.self_attn.k_proj.weight

    Q = F.linear(queries_normed.float(), q_proj_weight.float())
    K = F.linear(keys_normed.float(), k_proj_weight.float())

    # Reshape to heads
    num_heads = layer.config.num_attention_heads
    num_kv_heads = layer.config.num_key_value_heads
    head_dim = Q.shape[-1] // num_heads

    Q_heads = Q.view(-1, num_heads, head_dim)  # [n_queries, num_heads, head_dim]
    K_heads = K.view(-1, num_kv_heads, head_dim)  # [n_keys, num_kv_heads, head_dim]

    # Handle GQA
    if num_kv_heads != num_heads:
        repeat_factor = num_heads // num_kv_heads
        K_heads = K_heads.repeat_interleave(repeat_factor, dim=1)

    # Compute attention scores per head
    scores = torch.einsum('qhd,khd->hqk', Q_heads, K_heads) / (head_dim ** 0.5)

    if head_idx is not None:
        return scores[head_idx].to(torch.float32)
    return scores.to(torch.float32)
