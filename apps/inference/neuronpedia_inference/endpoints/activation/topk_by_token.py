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
from nnterp import StandardizedTransformer
from transformer_lens import ActivationCache

# from transformer_lens.model_bridge import TransformerBridge
from neuronpedia_inference.config import Config
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.shared import Model, with_request_lock, is_nnterp_model, get_nnterp_layer_output

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5

router = APIRouter()


def get_layer_num_from_sae_id(sae_id: str) -> int:
    return int(sae_id.split("-")[0]) if not sae_id.isdigit() else int(sae_id)


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

    # sae = sae_manager.get_sae(source)

    prepend_bos = False

    # if the request doesn't start with the bos, prepend it
    bos_token = Model.get_instance().tokenizer.bos_token
    if not prompt.startswith(bos_token):
        prompt = bos_token + prompt

    if is_nnterp_model(model):
        tokens = model.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ][0]
    else:
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

    if is_nnterp_model(model):
        tokenizer = model.tokenizer
        str_tokens = tokenizer.tokenize(prompt)
        str_tokens = [tokenizer.convert_tokens_to_string([t]) for t in str_tokens]
    else:
        str_tokens = model.to_str_tokens(prompt, prepend_bos=prepend_bos)

    hook_name = sae_manager.get_sae_hook(source)
    sae_type = sae_manager.get_sae_type(source)

    # if isinstance(model, TransformerBridge) and tokens.ndim == 1:
    #     tokens = tokens.unsqueeze(0)
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    if is_nnterp_model(model):
        layer_num = get_layer_num_from_sae_id(source)
        # For vLLM models, pass tokens as a list (vLLM doesn't accept tensors directly)
        trace_input = tokens.squeeze(0).tolist() if getattr(model, 'is_vllm', False) else tokens
        with model.trace(trace_input):
            if "resid_post" in hook_name:
                outputs = get_nnterp_layer_output(model, layer_num).save()
            else:
                raise ValueError(f"Unsupported hook name for nnsight: {hook_name}")
        cache = {hook_name: outputs}
    else:
        _, cache = model.run_with_cache(tokens)

    activations_by_index = get_activations_by_index(
        sae_type,
        source,
        cache,
        hook_name,
    )

    # Get top k activations for each token
    top_k_values, top_k_indices = torch.topk(activations_by_index.T, k=top_k)

    # if we are ignoring BOS and the model prepends BOS, we shift everything over by one
    if ignore_bos:
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
        mlp_activation_data = cache[hook_name].to(Config.get_instance().device)
        return torch.transpose(mlp_activation_data[0], 0, 1)

    activation_data = cache[hook_name].to(Config.get_instance().device)
    feature_activation_data = (
        SAEManager.get_instance().get_sae(selected_layer).encode(activation_data)
    )
    return torch.transpose(feature_activation_data.squeeze(0), 0, 1)
