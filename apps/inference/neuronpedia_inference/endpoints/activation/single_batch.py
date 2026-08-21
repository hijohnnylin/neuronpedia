import logging
from typing import cast

import torch
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from interp_engine import special_token_positions

from neuronpedia_inference.config import Config
from neuronpedia_inference.endpoints.activation.single import (
    get_layer_num_from_sae_id,
    process_feature_activations,
    process_neuron_activations,
)
from neuronpedia_inference.engine_adapter import (
    BackendUnsupported,
    calculate_dfa_for_values,
    capture_padded_cache_async,
)
from neuronpedia_inference.inference_utils.token_limit import reject_if_over_token_limit
from neuronpedia_inference.memory_cost import activation_single_batch_cost
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.schemas import (
    ActivationSingleBatchRequest,
    ActivationSingleBatchResponse,
    ActivationSingleBatchResult,
    ActivationValues,
)
from neuronpedia_inference.shared import LoadedModel, Model, with_request_lock

logger = logging.getLogger(__name__)

router = APIRouter()

# Maximum number of prompts that can be processed in a single batch
MAX_BATCH_SIZE = 4


@router.post("/activation/single-batch", responses={200: {"model": ActivationSingleBatchResponse}})
@with_request_lock(exclusive=False, cost=activation_single_batch_cost)
async def activation_single_batch(
    request: ActivationSingleBatchRequest = Body(
        ...,
        examples=[
            {
                "prompts": [
                    "The Jedi in Star Wars wield lightsabers.",
                    "The Force is strong with this one.",
                ],
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

    # Ensure exactly one of features or vector is provided
    if (request.source is not None and request.index is not None) == (
        request.vector is not None and request.hook is not None
    ):
        logger.error("Invalid request data: exactly one of layer/index or vector must be provided")
        return JSONResponse(
            content={"error": "Invalid request data: exactly one of layer/index or vector must be provided"},
            status_code=400,
        )

    if request.source is not None and request.index is not None:
        source = request.source
        layer_num = get_layer_num_from_sae_id(source)
        index = int(request.index)

        sae = sae_manager.get_sae(source)

        # TODO: we assume that if either SAE or model prepends bos, then we should prepend bos
        # this is not exactly correct, but sometimes the SAE doesn't have the prepend_bos flag set
        # prepend_bos = sae.cfg.metadata.prepend_bos or model.cfg.tokenizer_prepends_bos
        prepend_bos = False

        # Tokenize all prompts
        all_tokens = []
        all_str_tokens = []

        for prompt in prompts:
            # if the prompt doesn't start with the bos, prepend it
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

            str_tokens: list[str] = model.to_str_tokens(prompt, prepend_bos=prepend_bos)  # type: ignore

            all_tokens.append(tokens)
            all_str_tokens.append(str_tokens)

        # Process all prompts in batch
        try:
            results = await process_activations_batch(model, source, index, all_tokens)

            # Calculate DFA if enabled (for each result)
            if sae_manager.is_dfa_enabled(source):
                for result, tokens in zip(results, all_tokens):
                    # `n_values` because `process_activations_batch` may have dropped the
                    # leading BOS from `values`, which shifts `max_value_index` off the
                    # attention pattern's own indexing (see `calculate_dfa_for_values`).
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

        all_tokens = []
        all_str_tokens = []

        for prompt in prompts:
            tokens = model.to_tokens(
                prompt,
                prepend_bos=prepend_bos,
                truncate=False,
            )[0]
            batch_token_limit = config.activation_token_limit / MAX_BATCH_SIZE
            too_long = reject_if_over_token_limit(len(tokens), batch_token_limit, suffix=" for batch requests")
            if too_long is not None:
                return too_long

            str_tokens: list[str] = model.to_str_tokens(prompt, prepend_bos=prepend_bos)  # type: ignore
            all_tokens.append(tokens)
            all_str_tokens.append(str_tokens)

        # Process all prompts in batch
        try:
            results = await process_vector_activations_batch(vector, all_tokens, hook, model, sae_manager.device)
        except BackendUnsupported as e:
            return JSONResponse(content={"error": str(e)}, status_code=400)

    logger.info("Returning %d results", len(results))

    # Build response in the same order as input
    response_results = [
        _build_result(model, result, tokens, str_tokens)
        for result, tokens, str_tokens in zip(results, all_tokens, all_str_tokens)
    ]

    return ActivationSingleBatchResponse(results=response_results)


def _build_result(
    model: LoadedModel,
    result: ActivationValues,
    tokens: torch.Tensor,
    str_tokens: list[str],
) -> ActivationSingleBatchResult:
    """Assemble one prompt's result, keeping tokens aligned with values.

    The feature path drops the BOS activation (``process_saelens_activations``, shared with
    ``/activation/single``), which leaves ``values`` one shorter than the tokenization, so
    ``tokens`` is trimmed to match. A mask over a misaligned array describes the wrong tokens,
    which is what would make ``tokens_is_special`` meaningless.
    """
    special_positions = set(special_token_positions(tokens, model.tokenizer))
    tokens_is_special = [i in special_positions for i in range(len(str_tokens))]

    if len(str_tokens) > len(result.values):
        str_tokens = str_tokens[1:]
        tokens_is_special = tokens_is_special[1:]

    return ActivationSingleBatchResult(activation=result, tokens=str_tokens, tokens_is_special=tokens_is_special)


async def process_activations_batch(
    model: LoadedModel,
    layer: str,
    index: int,
    tokens_list: list[torch.Tensor],
) -> list[ActivationValues]:
    """
    Process multiple token sequences in a single batch for GPU efficiency.
    Returns results in the same order as input.
    """
    sae_manager = SAEManager.get_instance()
    hook_name = sae_manager.get_sae_hook(layer)
    sae_type = sae_manager.get_sae_type(layer)

    # Get BOS token ID for masking
    bos_token_id = model.tokenizer.bos_token_id

    # Pad sequences to the same length
    max_len = max(len(tokens) for tokens in tokens_list)
    batch_size = len(tokens_list)

    # Create padded batch tensor and attention mask
    pad_token_id = (
        model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
    )

    padded_tokens = torch.full(
        (batch_size, max_len),
        pad_token_id,
        dtype=tokens_list[0].dtype,
        device=tokens_list[0].device,
    )

    # Track original lengths and BOS indices for each sequence
    original_lengths = []
    all_bos_indices = []

    for i, tokens in enumerate(tokens_list):
        padded_tokens[i, : len(tokens)] = tokens
        original_lengths.append(len(tokens))
        # Find BOS indices for this sequence
        bos_indices = (tokens == bos_token_id).nonzero(as_tuple=True)[0].tolist()
        all_bos_indices.append(bos_indices)

    # Run batch inference (backend-aware). Right-padded + causal attention => real positions
    # are unaffected by trailing pad tokens (eager batched forward, or per-prompt vLLM capture).
    cache = await capture_padded_cache_async(model, padded_tokens, original_lengths, [hook_name])

    # Process each result separately
    results = []
    for i in range(batch_size):
        # Extract the non-padded portion for this sequence
        seq_len = original_lengths[i]

        if sae_type == "neurons":
            # Extract single sequence from batch
            seq_cache = {hook_name: cache[hook_name][i : i + 1, :seq_len]}
            result = process_neuron_activations(seq_cache, hook_name, index, sae_manager.device)
        elif sae_manager.get_sae(layer) is not None:
            # Extract single sequence from batch
            seq_cache = {hook_name: cache[hook_name][i : i + 1, :seq_len]}
            result = process_feature_activations(
                sae_manager.get_sae(layer),
                sae_type,
                seq_cache,
                hook_name,
                index,
                all_bos_indices[i],
            )
        else:
            raise ValueError(f"Invalid layer: {layer}")

        results.append(result)

    return results


async def process_vector_activations_batch(
    vector: torch.Tensor | list[float],
    tokens_list: list[torch.Tensor],
    hook_name: str,
    model: LoadedModel,
    device: str | torch.device,
) -> list[ActivationValues]:
    """
    Process multiple token sequences with a custom vector in a single batch for GPU efficiency.
    Returns results in the same order as input.
    """
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector, device=device)

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

    # Run batch inference (backend-aware)
    cache = await capture_padded_cache_async(model, padded_tokens, original_lengths, [hook_name])

    # Get activations for the batch
    activations = cache[hook_name].to(device)

    # Ensure vector has the same dtype as activations
    vector = vector.to(dtype=activations.dtype)

    # Process each sequence separately
    results = []
    for i in range(batch_size):
        seq_len = original_lengths[i]
        # Extract activations for this sequence (non-padded portion)
        seq_activations = activations[i : i + 1, :seq_len]

        # Apply vector projection
        feature_acts = torch.matmul(seq_activations, vector)
        values = feature_acts.squeeze(0).detach().tolist()
        max_value = max(values)

        result = ActivationValues(
            values=values,
            max_value=max_value,
            max_value_index=values.index(max_value),
        )
        results.append(result)

    return results
