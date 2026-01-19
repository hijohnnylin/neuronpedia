"""
Analysis utilities for transformer interpretability.

This package provides tools for analyzing principal components, model weights,
attention patterns, and other aspects of transformer models, with a focus on
understanding how instruction tuning affects model behavior.

Modules:
    pcs: Principal component loading and manipulation
    model_utils: Model-specific utilities (RMSNorm, GELU, weight analysis)
    attention: Attention mechanism analysis (QK affinities, VO decomposition)
    stats: Statistical analysis utilities (z-scores, top interactions)
"""

from .attention import (
    compute_attention_head_patterns,
    compute_full_attention_patterns,
    compute_qk_affinity_matrix,
    compute_vo_decomposition,
)
from .model_utils import (
    compute_cosine_distances_batch,
    compute_weight_statistics,
    extract_layer_weights,
    full_mlp_forward_batch,
    gelu_approx,
    gemma2_rmsnorm,
)
from .pcs import (
    extract_pc_components,
    get_pc_interpretation,
    load_individual_role_vectors,
    load_individual_trait_vectors,
    load_layer_semantic_vectors,
    load_pca_data,
    normalize_vector,
)
from .stats import (
    analyze_pc_pattern,
    compute_layer_statistics,
    compute_z_score_matrices,
    compute_z_score_matrices_semantic,
    compute_z_scores,
    find_layer_peaks,
    get_subset_stats,
    get_top_interactions,
)

__all__ = [
    # PC utilities
    "load_pca_data",
    "normalize_vector",
    "load_layer_semantic_vectors",
    "load_individual_role_vectors",
    "load_individual_trait_vectors",
    "extract_pc_components",
    "get_pc_interpretation",
    # Model utilities
    "gemma2_rmsnorm",
    "gelu_approx",
    "compute_cosine_distances_batch",
    "full_mlp_forward_batch",
    "extract_layer_weights",
    "compute_weight_statistics",
    # Attention analysis
    "compute_qk_affinity_matrix",
    "compute_vo_decomposition",
    "compute_full_attention_patterns",
    "compute_attention_head_patterns",
    # Statistical utilities
    "compute_z_scores",
    "compute_z_score_matrices",
    "compute_z_score_matrices_semantic",
    "get_subset_stats",
    "get_top_interactions",
    "analyze_pc_pattern",
    "compute_layer_statistics",
    "find_layer_peaks",
]
