import logging

import torch
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from interp_engine import special_token_positions

from neuronpedia_inference.config import Config
from neuronpedia_inference.endpoints.activation.topk_by_token import build_token_results
from neuronpedia_inference.engine_adapter import (
    BackendUnsupported,
    capture_padded_cache_async,
)
from neuronpedia_inference.inference_utils.token_limit import reject_if_over_token_limit
from neuronpedia_inference.memory_cost import activation_topk_by_token_batch_cost
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.schemas import (
    ActivationTopkByTokenBatchRequest,
    ActivationTopkByTokenBatchResponse,
    ActivationTopkByTokenBatchResult,
)
from neuronpedia_inference.shared import LoadedModel, Model, with_request_lock

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5

# Maximum number of prompts that can be processed in a single batch
MAX_BATCH_SIZE = 4

router = APIRouter()


def get_layer_num_from_sae_id(sae_id: str) -> int:
    return int(sae_id.split("-")[0]) if not sae_id.isdigit() else int(sae_id)


@router.post("/activation/topk-by-token-batch", responses={200: {"model": ActivationTopkByTokenBatchResponse}})
@with_request_lock(exclusive=False, cost=activation_topk_by_token_batch_cost)
async def activation_topk_by_token_batch(
    request: ActivationTopkByTokenBatchRequest = Body(
        ...,
        examples=[
            {
                "prompts": [
                    "The Jedi in Star Wars wield lightsabers.",
                    "The Force is strong with this one.",
                ],
                "model": "gpt2-small",
                "source": "0-res-jb",
                "ignore_bos": True,
            }
        ],
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
            content={"error": f"Batch size {len(prompts)} exceeds maximum of {MAX_BATCH_SIZE}"},
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

        tokens = model.to_tokens(
            prompt,
            prepend_bos=prepend_bos,
            truncate=False,
        )[0]

        batch_token_limit = config.activation_token_limit / MAX_BATCH_SIZE
        too_long = reject_if_over_token_limit(len(tokens), batch_token_limit, suffix=" for batch requests")
        if too_long is not None:
            return too_long

        str_tokens = model.to_str_tokens(prompt, prepend_bos=prepend_bos)

        all_tokens.append(tokens)
        all_str_tokens.append(str_tokens)

    # Process all prompts in batch
    try:
        batch_results = await process_topk_batch(model, source, top_k, ignore_bos, all_tokens, all_str_tokens)
    except BackendUnsupported as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

    logger.info("Returning %d results", len(batch_results))

    return ActivationTopkByTokenBatchResponse(results=batch_results)


async def process_topk_batch(
    model: LoadedModel,
    source: str,
    top_k: int,
    ignore_bos: bool,
    tokens_list: list[torch.Tensor],
    str_tokens_list: list[list[str]],
) -> list[ActivationTopkByTokenBatchResult]:
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
    pad_token_id = (
        model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
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

    # Run batch inference (backend-aware: eager batched forward, or per-prompt vLLM capture)
    cache = await capture_padded_cache_async(model, padded_tokens, original_lengths, [hook_name])

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
        # We want top k features for each token. Clamped to the feature count: torch.topk
        # raises when k exceeds the axis it reduces, which turned an over-large top_k into
        # a 500.
        effective_top_k = max(1, min(top_k, activations_by_index.shape[0]))
        top_k_values, top_k_indices = torch.topk(activations_by_index.T, k=effective_top_k)

        # From the full id tensor, before any slicing, so it stays aligned with
        # the ids it was derived from.
        special_positions = set(special_token_positions(tokens_list[i][:seq_len], model.tokenizer))
        str_tokens, token_results = build_token_results(
            str_tokens,
            top_k_values,
            top_k_indices,
            special_positions,
            ignore_bos,
        )

        results.append(
            ActivationTopkByTokenBatchResult(
                results=token_results,
                tokens=str_tokens,
            )
        )

    return results


def get_activations_by_index(
    sae_type: str,
    selected_layer: str,
    cache: dict[str, torch.Tensor],
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
    feature_activation_data = SAEManager.get_instance().get_sae(selected_layer).encode(activation_data)
    return torch.transpose(feature_activation_data.squeeze(0), 0, 1)
