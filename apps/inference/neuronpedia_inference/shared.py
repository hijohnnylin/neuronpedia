import asyncio
import logging
import os
import time
from functools import wraps
from typing import TYPE_CHECKING

import torch

# from transformer_lens.model_bridge import TransformerBridge
from nnterp import StandardizedTransformer
from transformer_lens import HookedTransformer

# vLLM/chatspace only available on Linux
try:
    from chatspace.generation import VLLMSteerModel

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    if TYPE_CHECKING:
        from chatspace.generation import VLLMSteerModel

logger = logging.getLogger(__name__)

request_lock = asyncio.Lock()

# Timeout for acquiring the request lock (seconds). 0 = no timeout
REQUEST_LOCK_TIMEOUT = float(
    os.environ.get("REQUEST_LOCK_TIMEOUT", "300")
)  # 5 min default


def with_request_lock():
    def decorator(func):  # type: ignore
        @wraps(func)
        async def wrapper(*args, **kwargs):  # type: ignore
            wait_start = time.time()

            if request_lock.locked():
                logger.warning(
                    "[LOCK] Request waiting for lock (another request in progress)..."
                )

            try:
                if REQUEST_LOCK_TIMEOUT > 0:
                    # Use wait_for with timeout
                    await asyncio.wait_for(
                        request_lock.acquire(), timeout=REQUEST_LOCK_TIMEOUT
                    )
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
                raise TimeoutError(
                    f"Request timed out waiting for lock after {wait_time:.2f}s"
                )

        return wrapper

    return decorator


class Model:
    _instance: "HookedTransformer | StandardizedTransformer | VLLMSteerModel"  # type: ignore  # | TransformerBridge

    @classmethod
    def get_instance(
        cls,
    ) -> (
        "HookedTransformer | StandardizedTransformer | VLLMSteerModel"
    ):  # | TransformerBridge:
        if cls._instance is None:
            raise ValueError("Model not initialized")
        return cls._instance

    @classmethod
    def set_instance(
        cls,
        model: "HookedTransformer | StandardizedTransformer | VLLMSteerModel",  # | TransformerBridge
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
    if model_id.startswith("gemma-3-") and "/" not in model_id:
        return f"google/{model_id}"
    return model_id


def resolve_model_id_for_load(
    model_id: str,
    override_model_id: str | None = None,
    custom_hf_model_id: str | None = None,
) -> str:
    return replace_tlens_model_id_with_hf_model_id(
        custom_hf_model_id or override_model_id or model_id
    )
