import logging

import torch
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from neuronpedia_inference_client.models.activation_topk_by_token_batch_post200_response import (
    ActivationTopkByTokenBatchPost200Response,
)
from neuronpedia_inference_client.models.activation_topk_by_token_batch_post200_response_results_inner import (
    ActivationTopkByTokenBatchPost200ResponseResultsInner,
)
from neuronpedia_inference_client.models.activation_topk_by_token_batch_post_request import (
    ActivationTopkByTokenBatchPostRequest,
)
from neuronpedia_inference_client.models.activation_topk_by_token_post200_response_results_inner import (
    ActivationTopkByTokenPost200ResponseResultsInner,
)
from neuronpedia_inference_client.models.activation_topk_by_token_post200_response_results_inner_top_features_inner import (
    ActivationTopkByTokenPost200ResponseResultsInnerTopFeaturesInner,  # noqa: E501
)
from nnterp import StandardizedTransformer
from transformer_lens import ActivationCache, HookedTransformer

# from transformer_lens.model_bridge import TransformerBridge
from neuronpedia_inference.config import Config
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.shared import Model, with_request_lock, is_nnterp_model, get_nnterp_layer_output

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5

# Maximum number of prompts that can be processed in a single batch
MAX_BATCH_SIZE = 4

router = APIRouter()


def get_layer_num_from_sae_id(sae_id: str) -> int:
    return int(sae_id.split("-")[0]) if not sae_id.isdigit() else int(sae_id)


@router.post("/activation/topk-by-token-batch")
@with_request_lock()
async def activation_topk_by_token_batch(
    request: ActivationTopkByTokenBatchPostRequest = Body(
        ...,
        example={
            "prompts": [
                "The Jedi in Star Wars wield lightsabers.",
                "The Force is strong with this one.",
            ],
            "model": "gpt2-small",
            "source": "0-res-jb",
            "ignore_bos": True,
        },
    ),
):
    model = Model.get_instance()
    config = Config.get_instance()

    # Validate batch size
    prompts = request.prompts
    if len(prompts) == 0:
        return JSONResponse(
            content={"error": "At least one prompt is required"},
            status_code=400,
        )

    if len(prompts) > MAX_BATCH_SIZE:
        return JSONResponse(
            content={
                "error": f"Batch size {len(prompts)} exceeds maximum of {MAX_BATCH_SIZE}"
            },
            status_code=400,
        )

    source = request.source
    top_k = request.top_k if request.top_k is not None else DEFAULT_TOP_K
    ignore_bos = request.ignore_bos

    prepend_bos = False

    # Tokenize all prompts
    all_tokens = []
    all_str_tokens = []

    for prompt in prompts:
        # if the request doesn't start with the bos, prepend it
        bos_token = model.tokenizer.bos_token
        if not prompt.startswith(bos_token):
            prompt = bos_token + prompt

        if is_nnterp_model(model):
            tokens = model.tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]
        else:
            tokens = model.to_tokens(
                prompt,
                prepend_bos=prepend_bos,
                truncate=False,
            )[0]

        batch_token_limit = config.token_limit / MAX_BATCH_SIZE
        if len(tokens) > batch_token_limit:
            logger.error(
                "Text too long: %s tokens, max is %s",
                len(tokens),
                batch_token_limit,
            )
            return JSONResponse(
                content={
                    "error": f"Text too long: {len(tokens)} tokens, max is {batch_token_limit} for batch requests"
                },
                status_code=400,
            )

        if is_nnterp_model(model):
            tokenizer = model.tokenizer
            str_tokens = tokenizer.tokenize(prompt)
            str_tokens = [tokenizer.convert_tokens_to_string([t]) for t in str_tokens]
        else:
            str_tokens = model.to_str_tokens(prompt, prepend_bos=prepend_bos)

        all_tokens.append(tokens)
        all_str_tokens.append(str_tokens)

    # Process all prompts in batch
    batch_results = process_topk_batch(
        model, source, top_k, ignore_bos, all_tokens, all_str_tokens
    )

    logger.info("Returning %d results", len(batch_results))

    return ActivationTopkByTokenBatchPost200Response(results=batch_results)


def process_topk_batch(
    model: HookedTransformer | StandardizedTransformer,  # | TransformerBridge
    source: str,
    top_k: int,
    ignore_bos: bool,
    tokens_list: list[torch.Tensor],
    str_tokens_list: list[list[str]],
) -> list[ActivationTopkByTokenBatchPost200ResponseResultsInner]:
    """
    Process multiple token sequences in a single batch for GPU efficiency.
    Returns results in the same order as input.
    """
    sae_manager = SAEManager.get_instance()
    hook_name = sae_manager.get_sae_hook(source)
    sae_type = sae_manager.get_sae_type(source)

    # Pad sequences to the same length
    max_len = max(len(tokens) for tokens in tokens_list)
    batch_size = len(tokens_list)

    # Create padded batch tensor
    if is_nnterp_model(model):
        pad_token_id = (
            model.tokenizer.pad_token_id
            if model.tokenizer.pad_token_id is not None
            else model.tokenizer.eos_token_id
        )
    else:
        pad_token_id = (
            model.tokenizer.pad_token_id
            if model.tokenizer.pad_token_id is not None
            else model.tokenizer.eos_token_id
        )

    padded_tokens = torch.full(
        (batch_size, max_len),
        pad_token_id,
        dtype=tokens_list[0].dtype,
        device=tokens_list[0].device,
    )

    # Track original lengths
    original_lengths = []

    for i, tokens in enumerate(tokens_list):
        padded_tokens[i, : len(tokens)] = tokens
        original_lengths.append(len(tokens))

    # Run batch inference
    if is_nnterp_model(model):
        layer_num = get_layer_num_from_sae_id(source)
        # For vLLM models, pass tokens as a list (vLLM doesn't accept tensors directly)
        trace_input = padded_tokens.tolist() if getattr(model, 'is_vllm', False) else padded_tokens
        with model.trace(trace_input):
            if "resid_post" in hook_name:
                outputs = get_nnterp_layer_output(model, layer_num).save()
            else:
                raise ValueError(f"Unsupported hook name for nnsight: {hook_name}")
        cache = {hook_name: outputs}
    else:
        _, cache = model.run_with_cache(padded_tokens)

    # Process each prompt separately to handle different lengths
    results = []
    for i in range(batch_size):
        seq_len = original_lengths[i]
        str_tokens = str_tokens_list[i]

        # Extract single sequence from batch
        seq_cache = {hook_name: cache[hook_name][i : i + 1, :seq_len]}

        # Get activations for this sequence
        activations_by_index = get_activations_by_index(
            sae_type,
            source,
            seq_cache,
            hook_name,
        )

        # Get top k activations for each token
        # activations_by_index has shape [num_features, num_tokens]
        # We want top k features for each token
        top_k_values, top_k_indices = torch.topk(activations_by_index.T, k=top_k)

        # Apply ignore_bos if needed
        if ignore_bos:
            str_tokens = str_tokens[1:]
            top_k_values = top_k_values[1:]
            top_k_indices = top_k_indices[1:]

        # Build result for this prompt
        token_results = []
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
            token_results.append(token_result)

        results.append(
            ActivationTopkByTokenBatchPost200ResponseResultsInner(
                results=token_results,
                tokens=str_tokens,  # type: ignore
            )
        )

    return results


def get_activations_by_index(
    sae_type: str,
    selected_layer: str,
    cache: ActivationCache | dict[str, torch.Tensor],
    hook_name: str,
) -> torch.Tensor:
    """
    Get activations organized by feature index.
    Returns a tensor of shape [num_features, num_tokens].
    """
    if sae_type == "neurons":
        mlp_activation_data = cache[hook_name].to(Config.get_instance().device)
        return torch.transpose(mlp_activation_data[0], 0, 1)

    activation_data = cache[hook_name].to(Config.get_instance().device)
    feature_activation_data = (
        SAEManager.get_instance().get_sae(selected_layer).encode(activation_data)
    )
    return torch.transpose(feature_activation_data.squeeze(0), 0, 1)
