"""Model-specific handling for lens readout.

Both the logit lens and the Jacobian lens decode an (optionally transported)
residual with the model's own final norm + unembedding:

    logits = unembed(final_norm(residual))         # logit lens
    logits = unembed(final_norm(J_bar @ residual)) # jacobian lens

Some model families apply an extra transform to the logits *inside* their
forward pass that the bare ``unembed`` / ``lm_head`` module does not reproduce.
The important one for the models Neuronpedia serves is Gemma-2's *final logit
softcapping* (``logits = cap * tanh(logits / cap)``).

This matters for correctness: the offline Jacobian-lens fit
(``utils/.../jlens/jlens/hf.py``) reproduces softcapping by reading
``text_config.final_logit_softcapping``. To keep the served logits in the exact
same basis as the fitted ``J_bar`` (and to make the logit-lens baseline match the
model's true output), we must reproduce it here too.

Notes on the Gemma family:
    - Gemma-2 uses ``final_logit_softcapping = 30.0``.
    - Gemma-3 dropped softcapping (it uses QK-norm instead), so its config has
      ``final_logit_softcapping = None``.
    - Gemma-4 ("E2B"/"E4B"/...) likewise does not set final-logit softcapping in
      its HF config; if a future revision reintroduces it, the config-driven path
      below picks it up automatically with no code change.

This module is the single place to add other per-model readout quirks later.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Known final-logit softcap values keyed by neuronpedia model id. Used only as a
# fallback when the HuggingFace config cannot be read (offline / unusual loads).
# When the HF config IS available we read it directly so this stays correct for
# models added later without touching this map.
_KNOWN_FINAL_LOGIT_SOFTCAP: dict[str, float] = {
    "gemma-2-2b": 30.0,
    "gemma-2-2b-it": 30.0,
    "gemma-2-9b": 30.0,
    "gemma-2-9b-it": 30.0,
    "gemma-2-27b": 30.0,
}


def _softcap_from_hf_config(hf_config: Any) -> float | None:
    """Read ``final_logit_softcapping`` from an HF config (or its text sub-config)."""
    text_config = hf_config
    if hasattr(hf_config, "get_text_config"):
        try:
            text_config = hf_config.get_text_config()
        except Exception:  # noqa: BLE001 - be permissive about odd configs
            text_config = hf_config
    softcap = getattr(text_config, "final_logit_softcapping", None)
    return float(softcap) if softcap is not None else None


def _resolve_hf_model_id(model_id: str | None) -> str | None:
    """Resolve a HuggingFace model id from a (possibly neuronpedia) model id.

    The id passed in may already be a HF id (``google/gemma-2-9b-it``) or a
    neuronpedia id (``gemma-2-9b-it``). If ``np_model_to_hf.json`` exists at the
    repo root and contains a mapping for the id, use the mapped HF id; otherwise
    fall back to the id itself.
    """
    if model_id is None:
        return None
    try:
        from neuronpedia_inference.endpoints.lens.lens_loader import (
            _load_np_to_hf_mapping,
        )

        mapping = _load_np_to_hf_mapping()
    except Exception:  # noqa: BLE001 - never let mapping lookup break config load
        mapping = None
    if mapping is not None and model_id in mapping:
        return mapping[model_id]
    return model_id


def _try_get_hf_config(model: Any, hf_model_id: str | None) -> Any | None:
    """Best-effort retrieval of the underlying HuggingFace config.

    - nnsight / nnterp (``StandardizedTransformer``) exposes ``model.config``.
    - TransformerLens (``HookedTransformer``) does not keep the HF config, so we
      load it standalone (cheap, weights are not downloaded, and the config is
      already cached because the model is loaded).
    """
    cfg = getattr(model, "config", None)
    if cfg is not None and (
        hasattr(cfg, "get_text_config") or hasattr(cfg, "final_logit_softcapping")
    ):
        return cfg

    resolved_id = _resolve_hf_model_id(hf_model_id)
    if resolved_id is not None:
        try:
            from transformers import AutoConfig

            return AutoConfig.from_pretrained(resolved_id, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load HF config for %s to resolve softcap: %s",
                resolved_id,
                exc,
            )
    return None


def resolve_final_logit_softcap(
    model: Any,
    np_model_id: str | None,
    hf_model_id: str | None,
) -> float | None:
    """Return the model's final-logit softcap value, or ``None`` if it has none.

    Prefers the live HF config (matches what the jlens fit saw exactly) and falls
    back to a small hardcoded map keyed by neuronpedia model id.
    """
    hf_config = _try_get_hf_config(model, hf_model_id)
    if hf_config is not None:
        softcap = _softcap_from_hf_config(hf_config)
        logger.info(
            "Resolved final_logit_softcapping=%s from HF config for %s",
            softcap,
            hf_model_id or np_model_id,
        )
        return softcap

    softcap = _KNOWN_FINAL_LOGIT_SOFTCAP.get(np_model_id or "")
    logger.info(
        "Resolved final_logit_softcapping=%s from fallback map for %s",
        softcap,
        np_model_id,
    )
    return softcap


def apply_final_logit_softcap(
    logits: torch.Tensor, softcap: float | None
) -> torch.Tensor:
    """Apply ``cap * tanh(logits / cap)`` if ``softcap`` is set, else return as-is."""
    if softcap is None:
        return logits
    return softcap * torch.tanh(logits / softcap)
