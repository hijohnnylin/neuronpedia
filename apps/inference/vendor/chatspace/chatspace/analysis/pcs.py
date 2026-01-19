"""Utilities for loading and manipulating principal component vectors from persona subspace data."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def load_pca_data(pca_dir: Path, pattern: str = "*.pt") -> tuple[dict, list[Path]]:
    """
    Load PCA data from a directory.

    Args:
        pca_dir: Directory containing PCA .pt files
        pattern: Glob pattern to match PCA files (default: "*.pt")

    Returns:
        Tuple of (pca_data dict, list of all PCA files found)

    Raises:
        FileNotFoundError: If no PCA files match the pattern

    Notes:
        - PCA files must be loaded with weights_only=False since they contain sklearn objects
        - Only the first matching file is loaded, but all matching files are returned
    """
    pca_files = sorted(pca_dir.glob(pattern))
    if not pca_files:
        raise FileNotFoundError(f"No PCA files found matching {pattern} in {pca_dir}")

    # Load first file
    pca_file = pca_files[0]
    # Note: weights_only=False is needed because PCA files contain sklearn objects
    pca_data = torch.load(pca_file, map_location='cpu', weights_only=False)

    return pca_data, pca_files


def normalize_vector(vec: torch.Tensor) -> torch.Tensor:
    """
    Normalize a vector to unit length.

    Args:
        vec: Input vector (any dtype, any shape)

    Returns:
        Normalized vector in bfloat16
    """
    vec_float = vec.float()
    return (vec_float / (vec_float.norm() + 1e-8)).to(torch.bfloat16)


def load_layer_semantic_vectors(
    pca_dir: Path,
    layer_idx: int,
    label_key: str = 'roles',
    prefix: str = 'role'
) -> OrderedDict[str, torch.Tensor]:
    """
    Load semantic vectors (roles/traits) for a specific layer from PCA data.

    DEPRECATED: This loads from PCA files. Use load_individual_role_vectors() or
    load_individual_trait_vectors() to load actual role/trait vectors.

    Args:
        pca_dir: Directory containing PCA files
        layer_idx: Layer index to extract vectors from
        label_key: Key in PCA data for labels (default: 'roles', can be 'traits')
        prefix: Prefix for vector names in output dict (default: 'role')

    Returns:
        OrderedDict mapping "{prefix}:{vector_type}:{label}" to normalized vectors

    Notes:
        - Automatically finds PCA file matching the layer index
        - Falls back to first file if no exact match
        - Handles multi-layer vectors by extracting the specified layer
        - Returns empty dict if directory or files don't exist
    """
    semantic = OrderedDict()

    if not pca_dir.exists():
        return semantic

    pca_files = sorted(pca_dir.glob('*.pt'))
    if not pca_files:
        return semantic

    # Find file matching target layer
    target_file = None
    for path in pca_files:
        if f"layer{layer_idx}" in path.stem:
            target_file = path
            break

    # Fallback to first file
    if target_file is None:
        target_file = pca_files[0]

    # Load data
    data = torch.load(target_file, map_location='cpu', weights_only=False)
    vectors_dict = data.get('vectors', {})
    labels_dict = data.get(label_key, {})

    # Extract vectors for this layer
    for vector_type, vect_list in vectors_dict.items():
        labels = labels_dict.get(
            vector_type,
            [f"{vector_type}_{i}" for i in range(len(vect_list))]
        )

        for vec, label in zip(vect_list, labels):
            # Handle multi-layer vectors
            vec_layer = vec[layer_idx] if vec.dim() == 2 else vec
            key = f"{prefix}:{vector_type}:{label}"
            semantic[key] = normalize_vector(vec_layer.to(torch.bfloat16))

    return semantic


def load_individual_role_vectors(
    vectors_dir: Path,
    layer_idx: int,
    vector_type: str = 'pos_3',
    max_roles: Optional[int] = None,
    compute_difference: bool = True,
    default_type: str = 'default_1'
) -> OrderedDict[str, torch.Tensor]:
    """
    Load individual role vectors from the vectors directory.

    Args:
        vectors_dir: Directory containing individual role .pt files (e.g., accountant.pt)
        layer_idx: Layer index to extract (0-45 for gemma-2-27b)
        vector_type: Which vector variant to load (default: 'pos_3')
            Options: 'pos_0', 'pos_1', 'pos_2', 'pos_3', 'pos_all'
        max_roles: Maximum number of roles to load (None = all)
        compute_difference: If True, compute role - default difference vectors (default: True)
            This matches production usage: pos_3 - default_1 for discriminative vectors
        default_type: Which default vector to subtract (default: 'default_1')
            Only used if compute_difference=True

    Returns:
        OrderedDict mapping role names to normalized vectors

    Notes:
        - Each role file contains vectors for all layers (shape: [46, 4608])
        - Vector types represent different label strengths (0=weak, 3=strong)
        - By default, computes difference vectors (pos_3 - default_1) for discriminative power
        - Default vectors are loaded from parent_dir/default_vectors.pt
        - Returns normalized vectors for the specified layer
    """
    role_vectors = OrderedDict()

    if not vectors_dir.exists():
        return role_vectors

    # Load default vectors if computing differences
    default_vec = None
    if compute_difference:
        default_path = vectors_dir.parent / 'default_vectors.pt'
        if default_path.exists():
            try:
                default_data = torch.load(default_path, map_location='cpu', weights_only=False)
                default_vec = default_data['activations'][default_type][layer_idx].float()
            except Exception as e:
                print(f"Warning: Failed to load default vectors: {e}")
                print(f"  Will return raw vectors without subtraction")
        else:
            print(f"Warning: default_vectors.pt not found at {default_path}")
            print(f"  Will return raw vectors without subtraction")

    # Get all .pt files
    role_files = sorted(vectors_dir.glob('*.pt'))
    if max_roles is not None:
        role_files = role_files[:max_roles]

    for role_file in role_files:
        role_name = role_file.stem  # e.g., "accountant"
        try:
            data = torch.load(role_file, map_location='cpu', weights_only=False)

            if vector_type not in data:
                continue

            # Extract vector for specified layer
            vec = data[vector_type]  # Shape: [46, 4608]
            vec_layer = vec[layer_idx].float()  # Shape: [4608]

            # Compute difference if requested
            if compute_difference and default_vec is not None:
                vec_layer = vec_layer - default_vec

            # Normalize and store
            role_vectors[role_name] = normalize_vector(vec_layer.to(torch.bfloat16))
        except Exception as e:
            print(f"Warning: Failed to load {role_file.name}: {e}")
            continue

    return role_vectors


def load_individual_trait_vectors(
    vectors_dir: Path,
    layer_idx: int,
    vector_type: str = 'pos_neg_50',
    max_traits: Optional[int] = None
) -> OrderedDict[str, torch.Tensor]:
    """
    Load individual trait vectors from the vectors directory.

    Args:
        vectors_dir: Directory containing individual trait .pt files (e.g., analytical.pt)
        layer_idx: Layer index to extract (0-45 for gemma-2-27b)
        vector_type: Which vector variant to load (default: 'pos_neg_50')
            Options for traits: 'pos_neg', 'pos_neg_50', 'pos_default', 'pos_default_50', 'pos_70', 'pos_40_70'
            Note: Trait files have different keys than role files!
            Production default: 'pos_neg_50' (precomputed contrast vector: 50% pos vs neg)
        max_traits: Maximum number of traits to load (None = all)

    Returns:
        OrderedDict mapping trait names to normalized vectors

    Notes:
        - Trait files have different key structure than role files
        - Role keys: 'pos_0', 'pos_1', 'pos_2', 'pos_3', 'pos_all'
        - Trait keys: 'pos_neg', 'pos_neg_50', 'pos_default', 'pos_default_50', 'pos_70', 'pos_40_70'
        - Default 'pos_neg_50' is a precomputed contrast vector (pos vs neg)
        - This matches production usage for discriminative trait vectors
        - Each trait file contains vectors for all layers
        - Returns normalized vectors for the specified layer
    """
    # Load trait vectors using same logic as roles, but without difference computation
    # (trait contrast vectors are precomputed)
    return load_individual_role_vectors(
        vectors_dir,
        layer_idx,
        vector_type,
        max_traits,
        compute_difference=False  # Traits use precomputed contrast vectors
    )


def extract_pc_components(
    pca_data: dict,
    n_components: int = 3,
    dtype: torch.dtype = torch.bfloat16
) -> tuple[list[torch.Tensor], np.ndarray]:
    """
    Extract PC component vectors from PCA data.

    Args:
        pca_data: PCA data dict loaded from .pt file
        n_components: Number of PC components to extract (default: 3)
        dtype: Target dtype for PC vectors (default: bfloat16)

    Returns:
        Tuple of (list of PC tensors, variance_explained array)

    Notes:
        - PCs are extracted from pca_data['pca'].components_
        - Returns fewer PCs if n_components exceeds available components
    """
    pca = pca_data['pca']
    pc_components = pca.components_

    n_available = pc_components.shape[0]
    n_extract = min(n_components, n_available)

    pcs = [
        torch.tensor(pc_components[i], dtype=dtype)
        for i in range(n_extract)
    ]

    variance_explained = pca_data.get('variance_explained', np.array([]))

    return pcs, variance_explained


def get_pc_interpretation(
    pca_data: dict,
    pc_idx: int = 0,
    top_k: int = 10
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """
    Interpret a PC by finding roles/traits with highest projections.

    Args:
        pca_data: PCA data dict with 'vectors', 'roles'/'traits' keys
        pc_idx: Index of PC to interpret (default: 0 for PC1)
        top_k: Number of top projections to return in each direction

    Returns:
        Tuple of (positive_projections, negative_projections)
        Each is a list of (label, projection_value) tuples

    Notes:
        - Requires PCA data to contain 'vectors' and label keys
        - Projects role/trait vectors onto the specified PC
        - Returns empty lists if data is missing
    """
    if 'vectors' not in pca_data:
        return [], []

    pca = pca_data['pca']
    pc = torch.tensor(pca.components_[pc_idx], dtype=torch.float32)
    pca_layer = pca_data.get('layer', 0)

    # Get role/trait vectors and labels
    role_vectors = pca_data.get('vectors', {})
    role_labels = pca_data.get('roles', {})  # Try 'roles' first
    if not role_labels:
        role_labels = pca_data.get('traits', {})  # Fall back to 'traits'

    projections = {}

    for role_type in role_vectors.keys():
        vectors = role_vectors[role_type]
        labels = role_labels.get(role_type, [])

        for vec, label in zip(vectors, labels):
            # Extract layer if multi-layer vector
            if vec.dim() == 2:
                vec_at_layer = vec[pca_layer]
            else:
                vec_at_layer = vec

            # Project onto PC
            projection = (vec_at_layer.float() @ pc).item()
            projections[label] = projection

    # Sort by projection
    sorted_projs = sorted(projections.items(), key=lambda x: x[1], reverse=True)

    positive = sorted_projs[:top_k]
    negative = sorted_projs[-top_k:][::-1]  # Reverse to show most negative first

    return positive, negative
