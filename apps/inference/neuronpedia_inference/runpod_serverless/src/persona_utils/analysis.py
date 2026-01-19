"""Analysis utilities for persona monitoring."""

import numpy as np
import torch


def pc_projection(mean_acts_per_turn: torch.Tensor, pca_results: dict, n_pcs: int = 1) -> np.ndarray:
    """
    Project activations onto principal components.
    
    Args:
        mean_acts_per_turn: Tensor of shape (num_turns, hidden_size)
        pca_results: Dict with 'pca' and 'scaler'
        n_pcs: Number of principal components to project onto
        
    Returns:
        Array of shape (num_turns, n_pcs) with projection values
    """
    if isinstance(mean_acts_per_turn, list):
        stacked_acts = torch.stack(mean_acts_per_turn)
    else:
        stacked_acts = mean_acts_per_turn
    
    stacked_acts = stacked_acts.float().numpy()
    scaled_acts = pca_results["scaler"].transform(stacked_acts)
    projected_acts = pca_results["pca"].transform(scaled_acts)
    
    return projected_acts[:, :n_pcs]

