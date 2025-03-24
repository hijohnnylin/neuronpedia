import asyncio
from functools import wraps

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
