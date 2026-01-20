import asyncio
import logging
import os
import time
from functools import wraps

import torch

# from transformer_lens.model_bridge import TransformerBridge
from nnterp import StandardizedTransformer
from transformer_lens import HookedTransformer
from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig

logger = logging.getLogger(__name__)

request_lock = asyncio.Lock()

# Timeout for acquiring the request lock (seconds). 0 = no timeout
REQUEST_LOCK_TIMEOUT = float(os.environ.get("REQUEST_LOCK_TIMEOUT", "300"))  # 5 min default


def with_request_lock():
    def decorator(func):  # type: ignore
        @wraps(func)
        async def wrapper(*args, **kwargs):  # type: ignore
            wait_start = time.time()
            
            if request_lock.locked():
                logger.warning(f"[LOCK] Request waiting for lock (another request in progress)...")
            
            try:
                if REQUEST_LOCK_TIMEOUT > 0:
                    # Use wait_for with timeout
                    await asyncio.wait_for(request_lock.acquire(), timeout=REQUEST_LOCK_TIMEOUT)
                else:
                    await request_lock.acquire()
                
                wait_time = time.time() - wait_start
                if wait_time > 0.1:  # Only log if waited more than 100ms
                    logger.info(f"[LOCK] Acquired lock after {wait_time:.2f}s wait")
                
                try:
                    return await func(*args, **kwargs)
                finally:
                    request_lock.release()
                    
            except asyncio.TimeoutError:
                wait_time = time.time() - wait_start
                logger.error(f"[LOCK] Timeout waiting for lock after {wait_time:.2f}s")
                raise TimeoutError(f"Request timed out waiting for lock after {wait_time:.2f}s")

        return wrapper

    return decorator


class Model:
    _instance: (
        HookedTransformer | StandardizedTransformer | VLLMSteerModel
    )  # | TransformerBridge  # type: ignore

    @classmethod
    def get_instance(
        cls,
    ) -> HookedTransformer | StandardizedTransformer | VLLMSteerModel:  # | TransformerBridge:
        if cls._instance is None:
            raise ValueError("Model not initialized")
        return cls._instance

    @classmethod
    def set_instance(
        cls,
        model: HookedTransformer | StandardizedTransformer | VLLMSteerModel,  # | TransformerBridge
    ) -> None:
        cls._instance = model


MODEL = Model()

STR_TO_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


TLENS_MODEL_ID_TO_HF_MODEL_ID = {
    "gpt2-small": "openai-community/gpt2",
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-2b-it": "google/gemma-2-2b-it",
}


def replace_tlens_model_id_with_hf_model_id(model_id: str) -> str:
    if model_id in TLENS_MODEL_ID_TO_HF_MODEL_ID:
        return TLENS_MODEL_ID_TO_HF_MODEL_ID[model_id]
    return model_id


def is_nnterp_model(model) -> bool:  # type: ignore
    """Check if model is an nnterp model (StandardizedTransformer or StandardizedVLLM).
    
    Both StandardizedTransformer and StandardizedVLLM inherit from StandardizationMixin
    which provides layers_output and is_vllm attributes.
    """
    return hasattr(model, 'layers_output') and hasattr(model, 'is_vllm')


def get_nnterp_layer_output(model, layer_num: int):  # type: ignore
    """Get layer output for nnterp models, with fallback for models where renaming doesn't work.
    
    For models where nnterp's standardized renaming works, uses model.layers_output[layer_num].
    For models where it doesn't work (like GptOssForCausalLM), falls back to model.layers[layer_num].output.
    
    Note: Some layers return tuples (hidden_states, cache). This function returns the raw output;
    use get_nnterp_layer_hidden_states() if you need just the hidden states tensor.
    """
    # Try the standardized accessor first
    try:
        return model.layers_output[layer_num]
    except (AttributeError, KeyError):
        pass
    
    # Fallback: try direct layer access
    if hasattr(model, 'layers'):
        return model.layers[layer_num].output
    
    # Another fallback for nested model structure
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_num].output
    
    raise AttributeError(f"Cannot access layer {layer_num} output for model {type(model)}")


def extract_hidden_states(output):  # type: ignore
    """Extract hidden states from layer output, handling both tensor and tuple returns.
    
    Some transformer layers return (hidden_states, cache) tuples, others return just the tensor.
    """
    if isinstance(output, tuple):
        return output[0]
    return output
