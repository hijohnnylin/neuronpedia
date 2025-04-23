import asyncio
from functools import wraps
from typing import Any

import einops
import torch
from transformer_lens import HookedTransformer

request_lock = asyncio.Lock()


def with_request_lock():
    def decorator(func):  # type: ignore
        @wraps(func)
        async def wrapper(*args, **kwargs):  # type: ignore
            async with request_lock:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


class Model:
    _instance: HookedTransformer  # type: ignore

    @classmethod
    def get_instance(cls) -> HookedTransformer:
        if cls._instance is None:
            raise ValueError("Model not initialized")
        return cls._instance

    @classmethod
    def set_instance(cls, model: HookedTransformer) -> None:
        cls._instance = model


MODEL = Model()

STR_TO_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _get_safe_dtype(dtype: torch.dtype) -> torch.dtype:
    """
    Convert float16 to float32, leave other dtypes unchanged.
    """
    return torch.float32 if dtype == torch.float16 else dtype


def safe_cast(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """
    Safely cast a tensor to the target dtype, creating a copy if needed.
    Convert float16 to float32, leave other dtypes unchanged.
    """
    safe_dtype = _get_safe_dtype(tensor.dtype)
    if safe_dtype != tensor.dtype or safe_dtype != target_dtype:
        return tensor.to(target_dtype)
    return tensor


def calculate_per_source_dfa(
    model: HookedTransformer,
    encoder: Any,
    v: torch.Tensor,
    attn_weights: torch.Tensor,
    feature_index: int,
    max_value_index: int,
) -> torch.Tensor:
    """
    Core DFA calculation logic that can be shared between different endpoints.

    Args:
        model: The transformer model
        encoder: The SAE or encoder object with W_enc attribute
        v: The value tensor from cache
        attn_weights: The attention pattern tensor from cache
        feature_index: Index of the feature in the encoder
        max_value_index: Index where maximum activation occurs

    Returns:
        Tensor with the raw DFA values (before formatting or casting)
    """
    # Determine the safe dtype for operations
    v_dtype = _get_safe_dtype(v.dtype)
    attn_weights_dtype = _get_safe_dtype(attn_weights.dtype)
    encoder_dtype = _get_safe_dtype(encoder.W_enc.dtype)

    # Use the highest precision dtype
    op_dtype = max(v_dtype, attn_weights_dtype, encoder_dtype, key=lambda x: x.itemsize)

    # Check if the model uses GQA
    use_gqa = (
        hasattr(model.cfg, "n_key_value_heads")
        and model.cfg.n_key_value_heads is not None
        and model.cfg.n_key_value_heads < model.cfg.n_heads
    )

    if use_gqa:
        n_query_heads = attn_weights.shape[1]
        n_kv_heads = v.shape[2]
        expansion_factor = n_query_heads // n_kv_heads
        v = v.repeat_interleave(expansion_factor, dim=2)

    # Cast tensors to operation dtype
    v = safe_cast(v, op_dtype)
    attn_weights = safe_cast(attn_weights, op_dtype)

    v_cat = einops.rearrange(
        v, "batch src_pos n_heads d_head -> batch src_pos (n_heads d_head)"
    )

    attn_weights_bcast = einops.repeat(
        attn_weights,
        "batch n_heads dest_pos src_pos -> batch dest_pos src_pos (n_heads d_head)",
        d_head=model.cfg.d_head,
    )

    decomposed_z_cat = attn_weights_bcast * v_cat.unsqueeze(1)

    # Cast encoder weights to operation dtype
    W_enc = safe_cast(encoder.W_enc[:, feature_index], op_dtype)

    per_src_pos_dfa = einops.einsum(
        decomposed_z_cat,
        W_enc,
        "batch dest_pos src_pos d_model, d_model -> batch dest_pos src_pos",
    )

    return per_src_pos_dfa[torch.arange(1), torch.tensor([max_value_index]), :]


def get_layer_num_from_sae_id(sae_id: str) -> int:
    return int(sae_id.split("-")[0]) if not sae_id.isdigit() else int(sae_id)
