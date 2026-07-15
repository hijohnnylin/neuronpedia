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


async def acquire_request_lock() -> None:
    """Acquire the global request lock with timeout and logging.

    For streaming endpoints the lock must be released in the generator's
    ``finally`` (Starlette iterates a StreamingResponse body after the handler
    returns, so a decorator-scoped lock would be released before generation
    even runs).  Call this function directly in the handler and release via
    ``request_lock.release()`` when generation completes.

    Raises ``TimeoutError`` if the lock cannot be acquired within
    ``REQUEST_LOCK_TIMEOUT`` seconds.
    """
    wait_start = time.time()

    if request_lock.locked():
        logger.warning(
            "[LOCK] Request waiting for lock (another request in progress)..."
        )

    if REQUEST_LOCK_TIMEOUT > 0:
        await asyncio.wait_for(request_lock.acquire(), timeout=REQUEST_LOCK_TIMEOUT)
    else:
        await request_lock.acquire()

    wait_time = time.time() - wait_start
    if wait_time > 0.1:
        logger.info(f"[LOCK] Acquired lock after {wait_time:.2f}s wait")


def with_request_lock():
    def decorator(func):  # type: ignore
        @wraps(func)
        async def wrapper(*args, **kwargs):  # type: ignore
            try:
                await acquire_request_lock()
                try:
                    return await func(*args, **kwargs)
                finally:
                    request_lock.release()
            except asyncio.TimeoutError:
                logger.error(
                    "[LOCK] Timeout waiting for lock after %.2fs",
                    REQUEST_LOCK_TIMEOUT,
                )
                raise TimeoutError(
                    f"Request timed out waiting for lock after "
                    f"{REQUEST_LOCK_TIMEOUT:.2f}s"
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
    return model_id
