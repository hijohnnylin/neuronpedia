"""Persona projection helpers shared by the assistant-axis path.

These are backend-agnostic: they take already-extracted per-turn activations
and project them onto the pre-fitted persona principal components. They live
here (rather than in an endpoint module) so the steer/completion-chat
assistant-axis path can import them without depending on any single endpoint.
"""

from __future__ import annotations

import numpy as np
import torch


def pc_projection(
    mean_acts_per_turn: torch.Tensor | list[torch.Tensor],
    pca_results: dict,
    n_pcs: int = 1,
) -> np.ndarray:
    """Project activations onto principal components.

    Args:
        mean_acts_per_turn: Tensor of shape (num_turns, hidden_size), or a list
            of per-turn tensors.
        pca_results: Dict with 'pca' and 'scaler'.
        n_pcs: Number of principal components to project onto.

    Returns:
        Array of shape (num_turns, n_pcs) with projection values.
    """
    stacked_acts = torch.stack(mean_acts_per_turn) if isinstance(mean_acts_per_turn, list) else mean_acts_per_turn

    stacked_acts = stacked_acts.float().numpy()
    scaled_acts = pca_results["scaler"].transform(stacked_acts)
    projected_acts = pca_results["pca"].transform(scaled_acts)

    return projected_acts[:, :n_pcs]


def _truncate_content(content: str, max_length: int = 120) -> str:
    """Truncate content to a reasonable length for snippets."""
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."
