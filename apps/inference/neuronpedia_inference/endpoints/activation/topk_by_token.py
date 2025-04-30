import logging

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from neuronpedia_inference_client.models.activation_topk_by_token_post200_response import (
    ActivationTopkByTokenPost200Response,
)
from neuronpedia_inference_client.models.activation_topk_by_token_post200_response_results_inner import (
    ActivationTopkByTokenPost200ResponseResultsInner,
)
from neuronpedia_inference_client.models.activation_topk_by_token_post200_response_results_inner_top_features_inner import (
    ActivationTopkByTokenPost200ResponseResultsInnerTopFeaturesInner,
)
from neuronpedia_inference_client.models.activation_topk_by_token_post_request import (
    ActivationTopkByTokenPostRequest,
)
from transformer_lens import ActivationCache

from neuronpedia_inference.config import Config
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.shared import Model, with_request_lock

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5

router = APIRouter()


@router.post("/activation/topk-by-token")
@with_request_lock()
async def activation_topk_by_token(
    request: ActivationTopkByTokenPostRequest,
):
    model = Model.get_instance()
    config = Config.get_instance()
    sae_manager = SAEManager.get_instance()
    prompt = request.prompt
    source = request.source
    top_k = request.top_k if request.top_k is not None else DEFAULT_TOP_K

    ignore_bos = request.ignore_bos

    sae = sae_manager.get_sae(source)

    prepend_bos = sae.cfg.prepend_bos or model.cfg.tokenizer_prepends_bos

    tokens = model.to_tokens(
        prompt,
        prepend_bos=prepend_bos,
        truncate=False,
    )[0]

    if len(tokens) > config.TOKEN_LIMIT:
        logger.error(
            "Text too long: %s tokens, max is %s",
            len(tokens),
            config.TOKEN_LIMIT,
        )
        return JSONResponse(
            content={
                "error": f"Text too long: {len(tokens)} tokens, max is {config.TOKEN_LIMIT}"
            },
            status_code=400,
        )

    str_tokens = model.to_str_tokens(prompt, prepend_bos=prepend_bos)
    _, cache = model.run_with_cache(tokens)

    hook_name = sae_manager.get_sae_hook(source)
    sae_type = sae_manager.get_sae_type(source)

    activations_by_index = get_activations_by_index(
        sae_type,
        source,
        cache,
        hook_name,
    )

    # Get top k activations for each token
    top_k_values, top_k_indices = torch.topk(activations_by_index.T, k=top_k)

    # if we are ignoring BOS and the model prepends BOS, we shift everything over by one
    if ignore_bos and prepend_bos:
        str_tokens = str_tokens[1:]
        top_k_values = top_k_values[1:]
        top_k_indices = top_k_indices[1:]

    results = []
    for token_idx, (token, values, indices) in enumerate(
        zip(str_tokens, top_k_values, top_k_indices)
    ):
        token_result = ActivationTopkByTokenPost200ResponseResultsInner(
            token=token,  # type: ignore
            token_position=token_idx,
            top_features=[
                ActivationTopkByTokenPost200ResponseResultsInnerTopFeaturesInner(
                    feature_index=int(idx.item()),
                    activation_value=float(val.item()),
                )
                for val, idx in zip(values, indices)
            ],
        )
        results.append(token_result)

    logger.info("Returning result: %s", {"layer": source, "results": results})

    return ActivationTopkByTokenPost200Response(
        results=results,
        tokens=str_tokens,  # type: ignore
    )


# Keep the get_activations_by_index function from the original code
def get_activations_by_index(
    sae_type: str,
    selected_layer: str,
    cache: ActivationCache | dict[str, torch.Tensor],
    hook_name: str,
) -> torch.Tensor:
    if sae_type == "neurons":
        mlp_activation_data = cache[hook_name].to(Config.get_instance().DEVICE)
        return torch.transpose(mlp_activation_data[0], 0, 1)

    activation_data = cache[hook_name].to(Config.get_instance().DEVICE)
    feature_activation_data = (
        SAEManager.get_instance().get_sae(selected_layer).encode(activation_data)
    )
    return torch.transpose(feature_activation_data.squeeze(0), 0, 1)
