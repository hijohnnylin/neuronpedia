"""``GET /health``: one real forward pass, not a liveness ping.

A process can answer HTTP while the model behind it cannot serve: the vLLM engine child
died, an OOM wedged the allocator, a device-side assert poisoned the CUDA context. A ping
says "up" through all of those. This runs one token through the loaded model each time it
is asked, on the path requests take, so a 200 means a request would work right now.

The probe is single-flight: monitors that ask at the same moment share one forward pass.
On the eager backend it runs on the event loop, as every handler does, so it never overlaps
a request's hooks. On vLLM it is one more request in the engine's batch, capped by
``HEALTH_PROBE_TIMEOUT`` seconds so an engine that stopped scheduling reports as such
instead of holding the connection open until the monitor gives up.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import torch
from interp_engine import EagerModel, VLLMModel

from neuronpedia_inference.config import Config
from neuronpedia_inference.schemas import HealthGpu, HealthResponse
from neuronpedia_inference.shared import LoadedModel

logger = logging.getLogger(__name__)

#: Seconds the vLLM probe may wait for its one token before the pod reports unhealthy.
#: Below a monitor's own timeout on purpose, so the answer is a 503 with a reason and not
#: a dropped connection.
PROBE_TIMEOUT_SECONDS = float(os.environ.get("HEALTH_PROBE_TIMEOUT", "20"))

_inflight: asyncio.Task[HealthResponse] | None = None


def gpu_report() -> list[HealthGpu]:
    """Free and total memory per visible CUDA card.

    ``mem_get_info`` is a no-op query that still fails on a poisoned context, so this doubles
    as the CUDA check; the caller reports the error and the post-request probe in
    ``server.CudaHealthMiddleware`` then restarts the process.
    """
    if not torch.cuda.is_available():
        return []
    gpus: list[HealthGpu] = []
    for index in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{index}")
        free, total = torch.cuda.mem_get_info(device)
        gpus.append(
            HealthGpu(
                index=index,
                name=torch.cuda.get_device_name(device),
                free_bytes=int(free),
                total_bytes=int(total),
            )
        )
    return gpus


def _probe_token_ids(model: LoadedModel) -> list[int]:
    """One token: BOS when the tokenizer has one, else the first token of a short word."""
    tokenizer = model.tokenizer
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None:
        return [int(bos)]
    ids = tokenizer.encode("The", add_special_tokens=False)
    return [int(ids[0])] if ids else [0]


class EngineNotReady(Exception):
    """The model object exists but its engine is still being built."""


async def _forward(model: LoadedModel) -> None:
    """Run one token through the model the way a request would."""
    token_ids = _probe_token_ids(model)

    if isinstance(model, VLLMModel):
        engine = model.engine
        if engine is None:
            # ``initialized`` flips before warmup builds the engine, so this is startup, not
            # a fault.
            raise EngineNotReady("vLLM engine is still starting")
        if getattr(engine, "errored", False):
            raise RuntimeError("vLLM engine is dead")
        await asyncio.wait_for(
            model.generate_text(token_ids, max_tokens=1, temperature=0.0),
            PROBE_TIMEOUT_SECONDS,
        )
        return

    if isinstance(model, EagerModel):
        input_ids = torch.tensor([token_ids], device=model.device)
        with torch.no_grad():
            logits = model.hf_model(input_ids, use_cache=False).logits
        # ``.all()`` in a bool context syncs the device, so a kernel that failed asynchronously
        # surfaces here rather than in the next request.
        if not bool(torch.isfinite(logits[0, -1]).all()):
            raise RuntimeError("forward pass produced non-finite logits")
        return

    raise RuntimeError(f"unknown backend {type(model).__name__}")


async def _run(model: LoadedModel, config: Config) -> HealthResponse:
    response = HealthResponse(
        status="ok",
        model=config.custom_hf_model_id or config.override_model_id or config.model_id,
        backend="vllm" if isinstance(model, VLLMModel) else "eager",
        device=config.device,
    )
    started = time.monotonic()
    try:
        response.gpus = gpu_report()
        await _forward(model)
        response.probe_ms = round((time.monotonic() - started) * 1000, 1)
    except EngineNotReady as exc:
        response.status = "starting"
        response.error = str(exc)
    except TimeoutError:
        response.status = "unhealthy"
        response.error = f"probe forward pass did not finish in {PROBE_TIMEOUT_SECONDS:g}s"
        logger.error("Health probe timed out after %gs", PROBE_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 - the point is to report it, whatever it is
        response.status = "unhealthy"
        response.error = f"{type(exc).__name__}: {exc}"[:500]
        logger.exception("Health probe failed")
    return response


async def probe(model: LoadedModel, config: Config) -> HealthResponse:
    """Probe the model once; callers that arrive during a probe share its result."""
    global _inflight
    if _inflight is None or _inflight.done():
        _inflight = asyncio.ensure_future(_run(model, config))
    # ``shield`` keeps the probe alive when one waiter's client hangs up.
    return await asyncio.shield(_inflight)
