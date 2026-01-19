"""Helpers for loading persona activation vectors."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch

DEFAULT_PERSONA_ROOT = Path("/workspace/persona-data")
DEFAULT_TARGET_LAYER = 22
_ROLE_POSITIVE_KEYS: Sequence[str] = ("pos_3", "pos", "positive")
# Production default: pos_neg_50 (precomputed contrast vector for 50% pos vs neg)
_TRAIT_POSITIVE_KEYS: Sequence[str] = ("pos_neg_50", "pos_70", "pos", "positive")
_ROLE_DEFAULT_KEY = "default_1"
_ROLE_DEFAULT_CACHE: dict[str, torch.Tensor] | None = None


def _select_block(data: dict[str, torch.Tensor], preferred_keys: Iterable[str]) -> torch.Tensor:
    for key in preferred_keys:
        block = data.get(key)
        if block is not None:
            return block
    return next(iter(data.values()))


def _load_role_defaults(persona_root: Path) -> dict[str, torch.Tensor] | None:
    global _ROLE_DEFAULT_CACHE
    if _ROLE_DEFAULT_CACHE is not None:
        return _ROLE_DEFAULT_CACHE
    default_path = persona_root / "qwen-3-32b/roles_240/default_vectors.pt"
    if not default_path.exists():
        return None
    payload = torch.load(default_path, map_location="cpu")
    activations = payload.get("activations")
    if not isinstance(activations, dict):
        return None
    _ROLE_DEFAULT_CACHE = activations
    return _ROLE_DEFAULT_CACHE


def load_activation_vector(
    dataset: str,
    *,
    persona_root: Path = DEFAULT_PERSONA_ROOT,
    target_layer: int = DEFAULT_TARGET_LAYER,
    role_model_prefix: str = "qwen-3-32b",
    role_contrast_default: bool = True,
) -> torch.Tensor | None:
    """Load a persona activation vector for a trait or role dataset if available.

    Parameters
    ----------
    dataset:
        Fully qualified dataset name such as ``qwen-3-32b__trait__analytical``.
    persona_root:
        Base directory containing persona-subspace artifacts.
    target_layer:
        Residual layer index to select from the activation tensor list.
    role_model_prefix:
        Model name used for role activation vectors; defaults to Qwen-3-32b.
    role_contrast_default:
        When True, subtract the default activation vector for roles to obtain a
        contrastive steering direction (pos_3 - default_1). Default is True to
        match production usage in eval_comprehensive_classifiers.py.
    """

    if "__trait__" in dataset:
        model_prefix, trait = dataset.split("__trait__", 1)
        vec_path = persona_root / f"{model_prefix}/traits_240/vectors/{trait}.pt"
        if not vec_path.exists():
            return None
        data = torch.load(vec_path, map_location="cpu")
        block = _select_block(data, _TRAIT_POSITIVE_KEYS)
        try:
            return block[target_layer].float()
        except (KeyError, IndexError):
            return None

    if "__role__" in dataset:
        _, role = dataset.split("__role__", 1)
        vec_path = persona_root / f"{role_model_prefix}/roles_240/vectors/{role}.pt"
        if not vec_path.exists():
            return None
        role_data = torch.load(vec_path, map_location="cpu")
        pos_block = _select_block(role_data, _ROLE_POSITIVE_KEYS)
        try:
            vector = pos_block[target_layer].float()
        except (KeyError, IndexError):
            return None

        if role_contrast_default:
            defaults = _load_role_defaults(persona_root)
            if not defaults:
                return None
            default_block = defaults.get(_ROLE_DEFAULT_KEY)
            if default_block is None:
                return None
            try:
                vector = vector - default_block[target_layer].float()
            except (KeyError, IndexError):
                return None
        return vector

    return None
