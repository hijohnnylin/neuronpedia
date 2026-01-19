"""Model-specific utilities for analyzing transformer weights and activations."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def gemma2_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Apply Gemma2's RMSNorm following HuggingFace implementation.

    Key difference from standard RMSNorm: Gemma2 multiplies by (1.0 + weight)
    instead of just weight. This is because Gemma2 initializes norm weights to zero.

    Args:
        x: Input tensor of shape (..., hidden_dim)
        weight: Norm weight of shape (hidden_dim,) - initialized as zeros in Gemma2
        eps: Epsilon for numerical stability (default: 1e-6)

    Returns:
        Normalized tensor of same shape as x

    References:
        - https://github.com/huggingface/transformers/pull/29402
    """
    # Convert to float32 for numerical stability
    x_float = x.float()
    weight_float = weight.float()

    # Compute RMS norm
    output = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + eps)

    # Gemma2-specific: multiply by (1.0 + weight)
    output = output * (1.0 + weight_float)

    return output.to(x.dtype)


def gelu_approx(x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation with tanh approximation.

    This is the activation function used by Gemma2 MLPs.

    Args:
        x: Input tensor

    Returns:
        Activated tensor with same shape as input
    """
    return 0.5 * x * (
        1.0 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        )
    )


def compute_cosine_distances_batch(
    vectors: torch.Tensor,
    weight_base: torch.Tensor,
    weight_instruct: torch.Tensor,
    ln_weight_base: torch.Tensor,
    ln_weight_instruct: torch.Tensor,
    eps: float = 1e-6
) -> Optional[np.ndarray]:
    """
    Compute cosine distances for a batch of vectors through weight matrices.

    Applies layernorm before projection to match actual forward pass behavior.

    Args:
        vectors: Tensor of shape (n_vectors, hidden_dim)
        weight_base: Base model weight matrix
        weight_instruct: Instruct model weight matrix
        ln_weight_base: Base model layernorm weight
        ln_weight_instruct: Instruct model layernorm weight
        eps: Epsilon for RMSNorm (default: 1e-6)

    Returns:
        Array of cosine distances (1 - cosine_similarity), or None if shapes don't match

    Notes:
        - Returns None for 1D weights or shape mismatches
        - Computes activations through both models and measures their alignment
    """
    if weight_base.dim() == 1:
        return None

    if weight_base.shape[1] != vectors.shape[1]:
        return None

    # Apply layernorm to all vectors in batch
    vectors_normed_base = gemma2_rmsnorm(vectors, ln_weight_base, eps=eps)
    vectors_normed_instruct = gemma2_rmsnorm(vectors, ln_weight_instruct, eps=eps)

    # Batch compute activations
    base_acts = weight_base.float() @ vectors_normed_base.T.float()  # (out_features, n_vectors)
    instruct_acts = weight_instruct.float() @ vectors_normed_instruct.T.float()  # (out_features, n_vectors)

    # Compute norms for each vector
    base_norms = base_acts.norm(dim=0)  # (n_vectors,)
    instruct_norms = instruct_acts.norm(dim=0)  # (n_vectors,)

    # Compute cosine similarities
    dot_products = (base_acts * instruct_acts).sum(dim=0)  # (n_vectors,)
    cosine_sims = dot_products / (base_norms * instruct_norms + 1e-8)

    # Compute cosine distances
    cosine_dists = 1.0 - cosine_sims

    return cosine_dists.cpu().numpy()


def full_mlp_forward_batch(
    x_batch: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    pre_ln_weight: torch.Tensor,
    post_ln_weight: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Run full MLP forward pass on a batch of vectors.

    Implements the complete Gemma2 MLP architecture:
    1. Pre-feedforward layernorm
    2. Gate projection + GELU activation
    3. Up projection
    4. Element-wise multiply (gate * up)
    5. Down projection
    6. Post-feedforward layernorm

    Args:
        x_batch: Input vectors (n_vectors, hidden_dim)
        gate_weight: Gate projection weight matrix
        up_weight: Up projection weight matrix
        down_weight: Down projection weight matrix
        pre_ln_weight: Pre-feedforward layernorm weight
        post_ln_weight: Post-feedforward layernorm weight
        eps: Epsilon for RMSNorm (default: 1e-6)

    Returns:
        Output vectors after full MLP + output layernorm (n_vectors, hidden_dim)

    Notes:
        - All operations are batched for efficiency
        - Uses Gemma2's RMSNorm (multiplies by 1.0 + weight)
        - Uses GELU with tanh approximation
    """
    # 1. Pre-feedforward layernorm (batched)
    x_normed = gemma2_rmsnorm(x_batch, pre_ln_weight, eps=eps)

    # 2. Gate projection + GELU activation (batched)
    gate_out = x_normed.float() @ gate_weight.T.float()  # (n_vectors, intermediate_size)
    gate_act = gelu_approx(gate_out)

    # 3. Up projection (batched)
    up_out = x_normed.float() @ up_weight.T.float()  # (n_vectors, intermediate_size)

    # 4. Element-wise multiply
    mlp_hidden = gate_act * up_out

    # 5. Down projection (batched)
    mlp_out = mlp_hidden @ down_weight.T.float()  # (n_vectors, hidden_dim)

    # 6. Post-feedforward layernorm (batched)
    output = gemma2_rmsnorm(mlp_out, post_ln_weight, eps=eps)

    return output.to(x_batch.dtype)


def extract_layer_weights(
    base_state_dict: dict[str, torch.Tensor],
    instruct_state_dict: dict[str, torch.Tensor],
    layer_num: int,
    weight_types: Optional[list[str]] = None
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract weights from a specific layer for both base and instruct models.

    Args:
        base_state_dict: Base model state dict
        instruct_state_dict: Instruct model state dict
        layer_num: Layer number to extract
        weight_types: List of weight suffixes to extract (default: all common weights)

    Returns:
        Dict mapping weight names to (base_weight, instruct_weight) tuples

    Notes:
        - Default weight types: q/k/v_proj, gate/up/down_proj, input_layernorm, etc.
        - Skips weights that don't exist in both models
    """
    if weight_types is None:
        weight_types = [
            'self_attn.q_proj.weight',
            'self_attn.k_proj.weight',
            'self_attn.v_proj.weight',
            'self_attn.o_proj.weight',
            'mlp.gate_proj.weight',
            'mlp.up_proj.weight',
            'mlp.down_proj.weight',
            'input_layernorm.weight',
            'pre_feedforward_layernorm.weight',
            'post_attention_layernorm.weight',
            'post_feedforward_layernorm.weight'
        ]

    weights = {}
    for weight_type in weight_types:
        key = f"model.layers.{layer_num}.{weight_type}"
        if key in base_state_dict and key in instruct_state_dict:
            weights[weight_type] = (
                base_state_dict[key],
                instruct_state_dict[key]
            )

    return weights


def compute_weight_statistics(
    weight_base: torch.Tensor,
    weight_instruct: torch.Tensor
) -> dict[str, float]:
    """
    Compute statistics comparing two weight matrices.

    Args:
        weight_base: Base model weight
        weight_instruct: Instruct model weight

    Returns:
        Dict with statistics: diff_norm, base_norm, instruct_norm, cosine_sim

    Notes:
        - All norms are Frobenius norms
        - Cosine similarity measures overall alignment of weight spaces
    """
    diff = weight_instruct - weight_base

    stats = {
        'diff_norm': diff.float().norm().item(),
        'base_norm': weight_base.float().norm().item(),
        'instruct_norm': weight_instruct.float().norm().item(),
    }

    # Compute cosine similarity
    base_flat = weight_base.float().flatten()
    instruct_flat = weight_instruct.float().flatten()
    cosine_sim = (base_flat @ instruct_flat) / (
        base_flat.norm() * instruct_flat.norm() + 1e-8
    )
    stats['cosine_sim'] = cosine_sim.item()

    return stats
