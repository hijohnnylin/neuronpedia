"""``/activation/raw`` -- residual stream vectors straight out of the model.

The only endpoint here that does not involve an SAE: it captures ``resid_post`` at the
requested layers and returns the raw vector at each prompt's final token. Callers use it to
embed prompts in the model's own basis (probing, nearest-neighbour search, steering-vector
construction), which is why it is worth serving even from a pod that loaded no SAEs at all.
"""

from __future__ import annotations

import logging

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from neuronpedia_inference.config import Config
from neuronpedia_inference.engine_adapter import (
    BackendUnsupported,
    assert_residual_available,
    capture_padded_cache_async,
)
from neuronpedia_inference.memory_cost import activation_raw_cost
from neuronpedia_inference.schemas import (
    ActivationRawLayer,
    ActivationRawPromptResult,
    ActivationRawRequest,
    ActivationRawResponse,
)
from neuronpedia_inference.shared import (
    Model,
    RecoverableOutOfMemory,
    recover_from_oom,
    with_request_lock,
)

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_BATCH_SIZE = 16

# 16-bit checkpoints carry ~3 decimal digits, so everything past the 4th is noise that only
# inflates the response. fp32 values are emitted as-is.
ROUND_DECIMALS = 4
_LOW_PRECISION_DTYPES = {"float16", "bfloat16", "float8"}


@router.post("/activation/raw", responses={200: {"model": ActivationRawResponse}})
@with_request_lock(exclusive=False, cost=activation_raw_cost)
async def activation_raw(request: ActivationRawRequest):
    config = Config.get_instance()
    config.check_requested_model(request.model)

    if request.hook_point != "residual_stream":
        return JSONResponse(
            content={"error": f"Unsupported hook_point {request.hook_point!r}. Only 'residual_stream' is supported."},
            status_code=400,
        )
    if request.type != "final_output_token":
        return JSONResponse(
            content={"error": f"Unsupported type {request.type!r}. Only 'final_output_token' is supported."},
            status_code=400,
        )
    if len(request.prompts) == 0:
        return JSONResponse(content={"error": "At least one prompt is required"}, status_code=400)
    if len(request.prompts) > MAX_BATCH_SIZE:
        return JSONResponse(
            content={"error": f"Batch size {len(request.prompts)} exceeds maximum of {MAX_BATCH_SIZE}"},
            status_code=400,
        )

    try:
        layers = _resolve_layers(request.layers)
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

    try:
        results = await _capture_final_token_residuals(request.prompts, layers)
    except BackendUnsupported as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        if recover_from_oom(e):
            return JSONResponse(content={"error": str(RecoverableOutOfMemory())}, status_code=503)
        logger.exception("Error processing raw activations")
        return JSONResponse(
            content={"error": "An error occurred while processing the request"},
            status_code=500,
        )

    return ActivationRawResponse(
        hook_point=request.hook_point,
        type=request.type,
        dtype=config.model_dtype,
        device=config.device or "unknown",
        results=results,
    )


def num_model_layers() -> int:
    """Layer count of the loaded model, preferring the value recorded at startup."""
    config = Config.get_instance()
    if config.num_layers:
        return int(config.num_layers)
    return int(Model.get_instance().n_layers)


def _resolve_layers(requested: list[int] | None) -> list[int]:
    """Deduplicated, ascending layer indices. Empty/absent means every layer."""
    n_layers = num_model_layers()
    if not requested:
        return list(range(n_layers))
    out_of_range = sorted({layer for layer in requested if layer < 0 or layer >= n_layers})
    if out_of_range:
        raise ValueError(
            f"Layers {out_of_range} are out of range; this model has {n_layers} layers (0-{n_layers - 1})."
        )
    return sorted(set(requested))


async def _capture_final_token_residuals(
    prompts: list[str],
    layers: list[int],
) -> list[ActivationRawPromptResult]:
    model = Model.get_instance()
    config = Config.get_instance()
    assert_residual_available(model, "/activation/raw")

    # Match the other activation endpoints: BOS is prepended to the STRING and tokenization
    # then runs with prepend_bos=False, so the token ids returned here are exactly the ones
    # the model saw.
    bos_token = model.tokenizer.bos_token
    prompts = [prompt if not bos_token or prompt.startswith(bos_token) else bos_token + prompt for prompt in prompts]

    all_tokens: list[torch.Tensor] = []
    all_str_tokens: list[list[str]] = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt, prepend_bos=False, truncate=False)[0]
        if len(tokens) > config.activation_token_limit:
            raise ValueError(f"Text too long: {len(tokens)} tokens, max is {config.activation_token_limit}")
        all_tokens.append(tokens)
        all_str_tokens.append(model.to_str_tokens(prompt, prepend_bos=False))

    max_len = max(len(tokens) for tokens in all_tokens)
    batch_size = len(all_tokens)
    pad_token_id = (
        model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
    )
    padded_tokens = torch.full(
        (batch_size, max_len),
        pad_token_id,
        dtype=all_tokens[0].dtype,
        device=all_tokens[0].device,
    )
    original_lengths: list[int] = []
    for i, tokens in enumerate(all_tokens):
        padded_tokens[i, : len(tokens)] = tokens
        original_lengths.append(len(tokens))

    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in layers]
    cache = await capture_padded_cache_async(model, padded_tokens, original_lengths, hook_names)

    # One [batch, d_model] gather per layer, done on whatever device the capture landed on so
    # only the final-token rows cross to the host -- the padded capture is
    # [batch, max_len, d_model] per layer and there is no reason to copy all of it.
    final_indices_by_prompt = [length - 1 for length in original_lengths]
    finals_by_layer: dict[int, torch.Tensor] = {}
    for layer, hook_name in zip(layers, hook_names):
        captured = cache[hook_name]
        rows = torch.arange(batch_size, device=captured.device)
        cols = torch.tensor(final_indices_by_prompt, device=captured.device)
        finals_by_layer[layer] = captured[rows, cols, :].float().cpu()

    round_to = ROUND_DECIMALS if config.model_dtype in _LOW_PRECISION_DTYPES else None
    results: list[ActivationRawPromptResult] = []
    for i in range(batch_size):
        token_ids = padded_tokens[i, : original_lengths[i]].detach().cpu().tolist()
        results.append(
            ActivationRawPromptResult(
                token_strings=all_str_tokens[i],
                token_ids=token_ids,
                activations=[
                    ActivationRawLayer(
                        layer=layer,
                        token_indices=[final_indices_by_prompt[i]],
                        values=[_serialize(finals_by_layer[layer][i], round_to)],
                    )
                    for layer in layers
                ],
            )
        )
    return results


def _serialize(vector: torch.Tensor, round_to: int | None) -> list[float]:
    values = vector.tolist()
    if round_to is None:
        return values
    return [round(value, round_to) for value in values]
