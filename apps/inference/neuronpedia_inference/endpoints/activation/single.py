import logging
from typing import Any

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from neuronpedia_inference_client.models.activation_single_post200_response import (
    ActivationSinglePost200Response,
)
from neuronpedia_inference_client.models.activation_single_post200_response_activation import (
    ActivationSinglePost200ResponseActivation,
)
from neuronpedia_inference_client.models.activation_single_post_request import (
    ActivationSinglePostRequest,
)
from transformer_lens import ActivationCache, HookedTransformer

from neuronpedia_inference.config import Config
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.shared import (
    Model,
    calculate_per_source_dfa,
    get_layer_num_from_sae_id,
    with_request_lock,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/activation/single")
@with_request_lock()
async def activation_single(
    request: ActivationSinglePostRequest,
):
    model = Model.get_instance()
    config = Config.get_instance()
    sae_manager = SAEManager.get_instance()
    # Ensure exactly one of features or vector is provided
    if (request.source is not None and request.index is not None) == (
        request.vector is not None and request.hook is not None
    ):
        logger.error(
            "Invalid request data: exactly one of layer/index or vector must be provided"
        )
        return JSONResponse(
            content={
                "error": "Invalid request data: exactly one of layer/index or vector must be provided"
            },
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
        prepend_bos = sae.cfg.prepend_bos or model.cfg.tokenizer_prepends_bos

        tokens = model.to_tokens(
            prompt,
            prepend_bos=prepend_bos,
            truncate=False,
        )[0]

        if len(tokens) > config.token_limit:
            logger.error(
                "Text too long: %s tokens, max is %s",
                len(tokens),
                config.token_limit,
            )
            return JSONResponse(
                content={
                    "error": f"Text too long: {len(tokens)} tokens, max is {config.token_limit}"
                },
                status_code=400,
            )

        str_tokens: list[str] = model.to_str_tokens(prompt, prepend_bos=prepend_bos)  # type: ignore
        result = process_activations(model, source, index, tokens)

        # Calculate DFA if enabled
        if sae_manager.is_dfa_enabled(source):
            dfa_result = calculate_dfa(
                model,
                sae,
                layer_num,
                index,
                result.max_value_index,
                tokens,
            )
            result.dfa_values = dfa_result["dfa_values"]  # type: ignore
            result.dfa_target_index = dfa_result["dfa_target_index"]  # type: ignore
            result.dfa_max_value = dfa_result["dfa_max_value"]  # type: ignore

    else:
        vector = request.vector
        hook = request.hook
        prepend_bos = model.cfg.tokenizer_prepends_bos
        tokens = model.to_tokens(
            prompt,
            prepend_bos=prepend_bos,
            truncate=False,
        )[0]
        if len(tokens) > config.token_limit:
            logger.error(
                "Text too long: %s tokens, max is %s",
                len(tokens),
                config.token_limit,
            )
            return JSONResponse(
                content={
                    "error": f"Text too long: {len(tokens)} tokens, max is {config.token_limit}"
                },
                status_code=400,
            )

        str_tokens: list[str] = model.to_str_tokens(prompt, prepend_bos=prepend_bos)  # type: ignore
        _, cache = model.run_with_cache(tokens)
        result = process_vector_activations(vector, cache, hook, sae_manager.device)  # type: ignore

    logger.info("Returning result: %s", result)

    return ActivationSinglePost200Response(activation=result, tokens=str_tokens)


def process_activations(
    model: HookedTransformer, layer: str, index: int, tokens: torch.Tensor
) -> ActivationSinglePost200ResponseActivation:
    sae_manager = SAEManager.get_instance()
    _, cache = model.run_with_cache(tokens)
    hook_name = sae_manager.get_sae_hook(layer)
    sae_type = sae_manager.get_sae_type(layer)

    if sae_type == "neurons":
        return process_neuron_activations(cache, hook_name, index, sae_manager.device)
    if sae_manager.get_sae(layer) is not None:
        return process_feature_activations(
            sae_manager.get_sae(layer),
            sae_type,
            cache,
            hook_name,
            index,
        )
    raise ValueError(f"Invalid layer: {layer}")


def process_neuron_activations(
    cache: ActivationCache | dict[str, torch.Tensor],
    hook_name: str,
    index: int,
    device: str,
) -> ActivationSinglePost200ResponseActivation:
    mlp_activation_data = cache[hook_name].to(device)
    values = torch.transpose(mlp_activation_data[0], 0, 1)[index].detach().tolist()
    max_value = max(values)
    return ActivationSinglePost200ResponseActivation(
        values=values,
        max_value=max_value,
        max_value_index=values.index(max_value),
    )


def process_feature_activations(
    sae: Any,
    sae_type: str,
    cache: ActivationCache | dict[str, torch.Tensor],
    hook_name: str,
    index: int,
) -> ActivationSinglePost200ResponseActivation:
    if sae_type == "saelens-1":
        return process_saelens_activations(sae, cache, hook_name, index)
    raise ValueError(f"Unsupported SAE type: {sae_type}")


def process_saelens_activations(
    sae: Any,
    cache: ActivationCache | dict[str, torch.Tensor],
    hook_name: str,
    index: int,
) -> ActivationSinglePost200ResponseActivation:
    feature_acts = sae.encode(cache[hook_name])
    values = torch.transpose(feature_acts.squeeze(0), 0, 1)[index].detach().tolist()
    max_value = max(values)
    return ActivationSinglePost200ResponseActivation(
        values=values,
        max_value=max_value,
        max_value_index=values.index(max_value),
    )


def process_vector_activations(
    vector: torch.Tensor | list[float],
    cache: ActivationCache | dict[str, torch.Tensor],
    hook_name: str,
    device: torch.device,
) -> ActivationSinglePost200ResponseActivation:
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
    return ActivationSinglePost200ResponseActivation(
        values=values,
        max_value=max_value,
        max_value_index=values.index(max_value),
    )


def calculate_dfa(
    model: HookedTransformer,
    sae: Any,
    layer_num: int,
    index: int,
    max_value_index: int,
    tokens: torch.Tensor,
) -> dict[str, list[float] | int | float]:
    _, cache = model.run_with_cache(tokens)
    v = cache["v", layer_num]  # [batch, src_pos, n_heads, d_head]
    attn_weights = cache["pattern", layer_num]  # [batch, n_heads, dest_pos, src_pos]

    per_src_dfa = calculate_per_source_dfa(
        model=model,
        encoder=sae,
        v=v,
        attn_weights=attn_weights,
        feature_index=index,
        max_value_index=max_value_index,
    )

    dfa_values = per_src_dfa[0].tolist()

    return {
        "dfa_values": dfa_values,
        "dfa_target_index": max_value_index,
        "dfa_max_value": max(dfa_values),
    }
