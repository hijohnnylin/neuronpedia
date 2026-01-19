"""Statistical analysis utilities for transformer interpretability."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_z_scores(values: np.ndarray, random_values: np.ndarray) -> np.ndarray:
    """
    Compute z-scores relative to random baseline.

    Args:
        values: Array of values to normalize
        random_values: Array of random baseline values

    Returns:
        z_scores: (values - mean(random)) / std(random)

    Notes:
        - If std is too small (<1e-10), returns centered values without scaling
        - All values are normalized using the same baseline statistics
    """
    random_mean = random_values.mean()
    random_std = random_values.std()

    if random_std < 1e-10:
        return values - random_mean

    return (values - random_mean) / random_std


def compute_z_score_matrices(
    data: np.ndarray,
    random_idx: list[int]
) -> np.ndarray:
    """
    Compute z-scores for all entries relative to random baseline.

    Normalizes a matrix of data (typically across layers) by computing statistics
    from random-random interactions and applying them to all entries.

    Args:
        data: Array [layers, n, m] containing data to normalize
        random_idx: List of indices corresponding to random vectors

    Returns:
        z_scores: Array [layers, n, m] normalized by random statistics

    Notes:
        - Extracts random-random subset for baseline statistics
        - Excludes diagonal for self-similarity when computing statistics
        - Computes per-layer statistics (each layer normalized independently)
        - Adds small epsilon (1e-10) to avoid division by zero
    """
    # Extract random-random subset for baseline statistics
    random_subset = data[:, random_idx, :][:, :, random_idx]  # [layers, n_random, n_random]

    # Compute mean and std across random samples (excluding diagonal for self-similarity)
    n_random = len(random_idx)
    mask = ~np.eye(n_random, dtype=bool)
    random_values = random_subset[:, mask]  # [layers, n_random*(n_random-1)]

    random_mean = random_values.mean(axis=1, keepdims=True)[:, :, None]  # [layers, 1, 1]
    random_std = random_values.std(axis=1, keepdims=True)[:, :, None]  # [layers, 1, 1]
    random_std = np.maximum(random_std, 1e-10)  # Avoid division by zero

    # Compute z-scores
    z_scores = (data - random_mean) / random_std

    return z_scores


def compute_z_score_matrices_semantic(
    data: np.ndarray,
    pc_idx: list[int],
    semantic_idx: list[int]
) -> np.ndarray:
    """
    Compute z-scores normalized by all semantic vector interactions.

    Alternative normalization that uses semantic (PC + role/trait) vectors
    as the baseline instead of random vectors.

    Args:
        data: Array [layers, n, m]
        pc_idx: Indices of PC vectors
        semantic_idx: Indices of semantic vectors (roles/traits)

    Returns:
        z_scores: Array [layers, n, m] normalized by semantic statistics

    Notes:
        - Combines PC and semantic indices to define "semantic space"
        - Normalizes relative to typical semantic interactions
        - Useful for detecting outliers within the semantic space
        - Excludes diagonal when computing statistics
    """
    # Combine PC and semantic indices
    all_semantic_idx = pc_idx + semantic_idx

    # Extract all semantic interactions
    semantic_subset = data[:, all_semantic_idx, :][:, :, all_semantic_idx]

    # Compute mean and std across all semantic pairs (excluding diagonal)
    n_sem = len(all_semantic_idx)
    mask = ~np.eye(n_sem, dtype=bool)
    semantic_values = semantic_subset[:, mask]  # [layers, n_sem*(n_sem-1)]

    semantic_mean = semantic_values.mean(axis=1, keepdims=True)[:, :, None]  # [layers, 1, 1]
    semantic_std = semantic_values.std(axis=1, keepdims=True)[:, :, None]  # [layers, 1, 1]
    semantic_std = np.maximum(semantic_std, 1e-10)  # Avoid division by zero

    # Compute z-scores
    z_scores = (data - semantic_mean) / semantic_std

    return z_scores


def get_subset_stats(
    matrix: np.ndarray,
    row_indices: list[int],
    col_indices: list[int],
    exclude_diagonal: bool = True
) -> tuple[float, float, float, float]:
    """
    Compute statistics for a subset of a matrix.

    Args:
        matrix: 2D array (or 3D with first dimension as layer)
        row_indices: Row indices to select
        col_indices: Column indices to select
        exclude_diagonal: Whether to exclude diagonal elements (default: True)

    Returns:
        Tuple of (mean, std, min, max) for the selected subset

    Notes:
        - If matrix is 3D, uses first layer (index 0)
        - Diagonal exclusion only applies if row and column indices match
        - Returns flattened statistics across all selected elements
    """
    # Handle 3D matrices (take first layer)
    if matrix.ndim == 3:
        matrix = matrix[0]

    subset = matrix[np.ix_(row_indices, col_indices)]

    if exclude_diagonal and len(row_indices) == len(col_indices):
        if row_indices == col_indices:
            mask = ~np.eye(len(row_indices), dtype=bool)
            values = subset[mask]
        else:
            values = subset.flatten()
    else:
        values = subset.flatten()

    return values.mean(), values.std(), values.min(), values.max()


def get_top_interactions(
    matrix: np.ndarray,
    vector_names: list[str],
    layer_idx: int,
    top_k: int = 10,
    exclude_self: bool = True
) -> pd.DataFrame:
    """
    Get top interactions from a matrix at a specific layer.

    Args:
        matrix: Array [layers, n, n]
        vector_names: List of vector names
        layer_idx: Layer to analyze
        top_k: Number of top interactions to return
        exclude_self: Whether to exclude diagonal (self-interactions)

    Returns:
        DataFrame with columns: rank, query, key, value

    Notes:
        - Returns top_k highest values
        - Optionally excludes diagonal (self-interactions)
        - Useful for finding strongest attention patterns or semantic relationships
    """
    layer_data = matrix[layer_idx]

    # Optionally mask diagonal
    if exclude_self:
        layer_data = layer_data.copy()
        np.fill_diagonal(layer_data, -np.inf)

    # Get top-k indices
    flat_indices = np.argsort(layer_data.flatten())[-top_k:][::-1]
    row_indices, col_indices = np.unravel_index(flat_indices, layer_data.shape)

    # Create DataFrame
    results = []
    for i, (row, col) in enumerate(zip(row_indices, col_indices)):
        results.append({
            'rank': i + 1,
            'query': vector_names[row],
            'key': vector_names[col],
            'value': matrix[layer_idx, row, col]
        })

    return pd.DataFrame(results)


def analyze_pc_pattern(
    matrix: np.ndarray,
    vector_names: list[str],
    pc_name: str,
    layer_idx: int,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Analyze what a specific PC attends to or retrieves.

    Args:
        matrix: Array [layers, n, n] (QK affinity or VO decomposition)
        vector_names: List of vector names
        pc_name: Name of PC to analyze (e.g., 'PC1')
        layer_idx: Layer to analyze
        top_k: Number of top interactions

    Returns:
        DataFrame with columns: rank, target, value

    Notes:
        - Extracts the row corresponding to pc_name
        - Returns top_k highest values (what PC attends to most)
        - Useful for understanding PC-specific patterns
    """
    pc_idx = vector_names.index(pc_name)
    pc_row = matrix[layer_idx, pc_idx, :]  # What PC attends to / retrieves

    # Get top-k
    top_indices = np.argsort(pc_row)[-top_k:][::-1]

    results = []
    for i, idx in enumerate(top_indices):
        results.append({
            'rank': i + 1,
            'target': vector_names[idx],
            'value': pc_row[idx]
        })

    return pd.DataFrame(results)


def compute_layer_statistics(
    data: np.ndarray,
    layer_idx: int,
    vector_groups: dict[str, list[int]]
) -> pd.DataFrame:
    """
    Compute summary statistics across different vector groups for a specific layer.

    Args:
        data: Array [layers, n, m]
        layer_idx: Layer to analyze
        vector_groups: Dict mapping group names to lists of indices

    Returns:
        DataFrame with statistics for each group pair

    Notes:
        - Computes mean, std, min, max for each (group1, group2) pair
        - Excludes diagonal for within-group statistics
        - Useful for comparing attention patterns across semantic categories
    """
    layer_data = data[layer_idx]
    results = []

    for group1_name, group1_indices in vector_groups.items():
        for group2_name, group2_indices in vector_groups.items():
            exclude_diag = (group1_name == group2_name)
            mean, std, min_val, max_val = get_subset_stats(
                layer_data,
                group1_indices,
                group2_indices,
                exclude_diagonal=exclude_diag
            )

            results.append({
                'group1': group1_name,
                'group2': group2_name,
                'mean': mean,
                'std': std,
                'min': min_val,
                'max': max_val
            })

    return pd.DataFrame(results)


def find_layer_peaks(
    data: np.ndarray,
    vector_idx: int,
    target_idx: int
) -> dict[str, int | float]:
    """
    Find the layer with peak value for a specific (vector, target) pair.

    Args:
        data: Array [layers, n, m]
        vector_idx: Index of source vector
        target_idx: Index of target vector

    Returns:
        Dict with keys: 'peak_layer', 'peak_value', 'mean_value'

    Notes:
        - Useful for identifying where specific interactions are strongest
        - Can be applied to QK affinities, VO decompositions, or deltas
    """
    values = data[:, vector_idx, target_idx]

    peak_layer = int(np.argmax(values))
    peak_value = float(values[peak_layer])
    mean_value = float(values.mean())

    return {
        'peak_layer': peak_layer,
        'peak_value': peak_value,
        'mean_value': mean_value
    }
