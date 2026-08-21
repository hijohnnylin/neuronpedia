import logging
from typing import Any, cast

import torch
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from interp_engine import special_token_positions

from neuronpedia_inference.config import Config
from neuronpedia_inference.engine_adapter import (
    BackendUnsupported,
    calculate_dfa_for_values,
    capture_activation_async,
    capture_cache_async,
)
from neuronpedia_inference.inference_utils.token_limit import reject_if_over_token_limit
from neuronpedia_inference.memory_cost import activation_single_cost
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.schemas import (
    ActivationSingleRequest,
    ActivationSingleResponse,
    ActivationValues,
)
from neuronpedia_inference.shared import LoadedModel, Model, with_request_lock

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/activation/single", responses={200: {"model": ActivationSingleResponse}})
@with_request_lock(exclusive=False, cost=activation_single_cost)
async def activation_single(
    request: ActivationSingleRequest = Body(
        ...,
        examples=[
            {
                "prompt": "The Jedi in Star Wars wield lightsabers.",
                "model": "gpt2-small",
                "source": "0-res-jb",
                "index": "14057",
            }
        ],
    ),
):
    model = Model.get_instance()
    config = Config.get_instance()
    sae_manager = SAEManager.get_instance()
    # Ensure exactly one of features or vector is provided
    if (request.source is not None and request.index is not None) == (
        request.vector is not None and request.hook is not None
    ):
        logger.error("Invalid request data: exactly one of layer/index or vector must be provided")
        return JSONResponse(
            content={"error": "Invalid request data: exactly one of layer/index or vector must be provided"},
            status_code=400,
        )

    prompt = request.prompt

    if request.source is not None and request.index is not None:
        source = request.source
        layer_num = get_layer_num_from_sae_id(source)
        index = int(request.index)

        sae = sae_manager.get_sae(source)

        # TODO: we assume that if either SAE or model prepends bos, then we should prepend bos
        # this is not exactly correct, but sometimes the SAE doesn't have the prepend_bos flag set
        # prepend_bos = sae.cfg.metadata.prepend_bos or model.cfg.tokenizer_prepends_bos
        prepend_bos = False
        # if the prompt doesn't start with the bos, prepend it
        bos_token = model.tokenizer.bos_token
        if not prompt.startswith(bos_token):
            prompt = bos_token + prompt

        tokens = model.to_tokens(
            prompt,
            prepend_bos=prepend_bos,
            truncate=False,
        )[0]

        too_long = reject_if_over_token_limit(len(tokens), config.activation_token_limit)
        if too_long is not None:
            return too_long

        str_tokens: list[str] = model.to_str_tokens(prompt, prepend_bos=prepend_bos)  # type: ignore
        try:
            result = await process_activations(model, source, index, tokens)

            # Calculate DFA if enabled. `n_values` because `process_activations` may have
            # dropped the leading BOS from `values`, which shifts `max_value_index` off the
            # attention pattern's own indexing (see `calculate_dfa_for_values`).
            if sae_manager.is_dfa_enabled(source):
                dfa_result = await calculate_dfa_for_values(
                    model,
                    sae,
                    layer_num,
                    index,
                    result.max_value_index,
                    tokens,
                    n_values=len(result.values),
                )
                result.dfa_values = dfa_result["dfa_values"]  # type: ignore
                result.dfa_target_index = dfa_result["dfa_target_index"]  # type: ignore
                result.dfa_max_value = dfa_result["dfa_max_value"]  # type: ignore
        except BackendUnsupported as e:
            return JSONResponse(content={"error": str(e)}, status_code=400)

    else:
        # Both are set on this path: the check above rejects any request that does not supply
        # exactly one of (source, index) / (vector, hook).
        vector = cast(list[float], request.vector)
        hook = cast(str, request.hook)
        prepend_bos = model.tok.tokenizer_prepends_bos
        tokens = model.to_tokens(
            prompt,
            prepend_bos=prepend_bos,
            truncate=False,
        )[0]
        too_long = reject_if_over_token_limit(len(tokens), config.activation_token_limit)
        if too_long is not None:
            return too_long

        str_tokens: list[str] = model.to_str_tokens(prompt, prepend_bos=prepend_bos)  # type: ignore
        try:
            cache = {hook: await capture_activation_async(model, tokens, hook)}
        except BackendUnsupported as e:
            return JSONResponse(content={"error": str(e)}, status_code=400)
        result = process_vector_activations(vector, cache, hook, sae_manager.device)  # type: ignore

    logger.info("Returning result: %s", result)

    # Built from the ids, then sliced wherever `str_tokens` is, so the two stay
    # index-aligned whichever branch above produced `result`. Callers use this to
    # drop scaffolding tokens without matching literals like "<bos>".
    special_positions = set(special_token_positions(tokens, model.tokenizer))
    tokens_is_special = [i in special_positions for i in range(len(str_tokens))]

    # if the model prepends the BOS token, then we need to remove the first token from the str_tokens
    if len(str_tokens) > len(result.values):
        str_tokens = str_tokens[1:]
        tokens_is_special = tokens_is_special[1:]

    return ActivationSingleResponse(activation=result, tokens=str_tokens, tokens_is_special=tokens_is_special)


def _get_safe_dtype(dtype: torch.dtype) -> torch.dtype:
    """
    Convert float16 to float32, leave other dtypes unchanged.
    """
    return torch.float32 if dtype == torch.float16 else dtype


def get_layer_num_from_sae_id(sae_id: str) -> int:
    return int(sae_id.split("-")[0]) if not sae_id.isdigit() else int(sae_id)


async def process_activations(
    model: LoadedModel,
    layer: str,
    index: int,
    tokens: torch.Tensor,
) -> ActivationValues:
    sae_manager = SAEManager.get_instance()
    hook_name = sae_manager.get_sae_hook(layer)
    sae_type = sae_manager.get_sae_type(layer)

    # zero out all values that are the BOS token
    bos_token_id = model.tokenizer.bos_token_id
    bos_indices = (tokens == bos_token_id).nonzero(as_tuple=True)[0]

    cache = await capture_cache_async(model, tokens, [hook_name])
    if sae_type == "neurons":
        return process_neuron_activations(cache, hook_name, index, sae_manager.device)
    return process_feature_activations(sae_manager.get_sae(layer), sae_type, cache, hook_name, index, bos_indices)


def process_neuron_activations(
    cache: dict[str, torch.Tensor],
    hook_name: str,
    index: int,
    device: str,
) -> ActivationValues:
    mlp_activation_data = cache[hook_name].to(device)
    values = torch.transpose(mlp_activation_data[0], 0, 1)[index].detach().tolist()
    max_value = max(values)
    return ActivationValues(
        values=values,
        max_value=max_value,
        max_value_index=values.index(max_value),
    )


def process_feature_activations(
    sae: Any,
    sae_type: str,
    cache: dict[str, torch.Tensor],
    hook_name: str,
    index: int,
    bos_indices: list[int],
) -> ActivationValues:
    if sae_type == "saelens-1":
        return process_saelens_activations(sae, cache, hook_name, index, bos_indices)
    raise ValueError(f"Unsupported SAE type: {sae_type}")


def process_saelens_activations(
    sae: Any,
    cache: dict[str, torch.Tensor],
    hook_name: str,
    index: int,
    bos_indices: list[int],
) -> ActivationValues:
    # if the cache[hook_name] is not on the same device as the sae, move it to the sae's device
    cached_value = cache[hook_name]
    if cached_value.device != sae.device:
        cached_value = cached_value.to(sae.device)
    feature_acts = sae.encode(cached_value)
    values = torch.transpose(feature_acts.squeeze(0), 0, 1)[index].detach().tolist()

    # zero out all values that are the BOS token
    for idx in bos_indices:
        values[idx] = 0
    # if the first token was the BOS token, then offset outputs by one removing teh first token
    if len(bos_indices) > 0:
        values = values[1:]

    max_value = max(values)
    return ActivationValues(
        values=values,
        max_value=max_value,
        max_value_index=values.index(max_value),
    )


def process_vector_activations(
    vector: torch.Tensor | list[float],
    cache: dict[str, torch.Tensor],
    hook_name: str,
    device: torch.device,
) -> ActivationValues:
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector, device=device)
    # not normalizing it for now
    # vector = vector / torch.linalg.norm(vector)
    activations = cache[hook_name].to(device)
    # ensure vector has the same dtype as activations
    vector = vector.to(dtype=activations.dtype)
    feature_acts = torch.matmul(activations, vector)
    values = feature_acts.squeeze(0).detach().tolist()
    max_value = max(values)
    return ActivationValues(
        values=values,
        max_value=max_value,
        max_value_index=values.index(max_value),
    )
