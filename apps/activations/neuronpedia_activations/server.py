"""
Example:
curl -X POST "http://localhost:5010/raw" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompts": ["The Eiffel Tower is in", "The Colosseum is in"]
  }'
"""

import asyncio
import logging
from functools import wraps
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from neuronpedia_activations.config import get_config_from_env
from neuronpedia_activations.model_manager import (
    ModelManager,
    RawActivationResult,
    RawBatchActivationResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG = get_config_from_env()


app = FastAPI(title="Neuronpedia Activations Server")
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=6)
model_manager = ModelManager()
request_lock = asyncio.Lock()


def with_request_lock():
    def decorator(func):  # type: ignore
        @wraps(func)
        async def wrapper(*args, **kwargs):  # type: ignore
            async with request_lock:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


@app.middleware("http")
async def check_secret_key(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    if CONFIG.secret is None:
        return await call_next(request)
    secret_key = request.headers.get("X-SECRET-KEY")
    if not secret_key or secret_key != CONFIG.secret:
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid or missing X-SECRET-KEY header"},
        )
    return await call_next(request)


class RawRequest(BaseModel):
    model: str
    prompts: list[str]
    hook_point: Literal["residual_stream"] = "residual_stream"
    type: Literal["final_output_token"] = "final_output_token"


class RawLayerResponse(BaseModel):
    layer: int
    token_indices: list[int]
    values: list[list[float]]


class RawResponse(BaseModel):
    hook_point: str
    type: str
    results: list["RawPromptResponse"]
    dtype: str
    device: str


class RawPromptResponse(BaseModel):
    token_strings: list[str]
    token_ids: list[int]
    activations: list[RawLayerResponse]


def _to_prompt_response(result: RawActivationResult) -> RawPromptResponse:
    return RawPromptResponse(
        token_strings=result.token_strings,
        token_ids=result.token_ids,
        activations=[
            RawLayerResponse(
                layer=layer.layer,
                token_indices=layer.token_indices,
                values=layer.values,
            )
            for layer in result.activations
        ],
    )


def _to_response(result: RawBatchActivationResult) -> RawResponse:
    return RawResponse(
        hook_point=result.hook_point,
        type=result.type,
        results=[_to_prompt_response(item) for item in result.results],
        dtype=result.dtype,
        device=result.device,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/busy")
async def busy() -> dict[str, bool]:
    return {"busy": request_lock.locked()}


@app.on_event("startup")  # pyright: ignore[reportDeprecated]
async def startup_preload_model() -> None:
    if not CONFIG.model_id:
        raise RuntimeError(
            "MODEL_ID is required for eager startup loading. "
            "Set MODEL_ID (or pass --model_id to start.py)."
        )
    logger.info(
        "Eager-loading startup model '%s' (dtype=%s, device=%s)",
        CONFIG.model_id,
        CONFIG.model_dtype if CONFIG.model_dtype else "default",
        CONFIG.device if CONFIG.device else "auto",
    )
    await asyncio.to_thread(
        model_manager.preload_model,
        CONFIG.model_id,
        CONFIG.model_dtype,
        CONFIG.device,
    )
    logger.info("Startup model loaded successfully")


@app.post("/raw")
@with_request_lock()
async def raw(request: RawRequest) -> RawResponse:
    result = model_manager.capture_raw_residual_stream_batch(
        model_id=request.model,
        prompts=request.prompts,
        hook_point=request.hook_point,
        extraction_type=request.type,
        dtype_override=CONFIG.model_dtype,
        device_override=CONFIG.device,
    )
    return _to_response(result)
