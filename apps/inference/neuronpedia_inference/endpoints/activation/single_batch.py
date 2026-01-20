import logging
from typing import Any

import einops
import torch
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from neuronpedia_inference_client.models.activation_single_batch_post200_response import (
    ActivationSingleBatchPost200Response,
)
from neuronpedia_inference_client.models.activation_single_batch_post200_response_results_inner import (
    ActivationSingleBatchPost200ResponseResultsInner,
)
from neuronpedia_inference_client.models.activation_single_batch_post_request import (
    ActivationSingleBatchPostRequest,
)
from neuronpedia_inference_client.models.activation_single_post200_response_activation import (
    ActivationSinglePost200ResponseActivation,
)
from nnterp import StandardizedTransformer
from transformer_lens import ActivationCache, HookedTransformer

from neuronpedia_inference.config import Config
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.shared import Model, with_request_lock, is_nnterp_model, get_nnterp_layer_output

logger = logging.getLogger(__name__)

router = APIRouter()

# Maximum number of prompts that can be processed in a single batch
MAX_BATCH_SIZE = 4


@router.post("/activation/single-batch")
@with_request_lock()
async def activation_single_batch(
    request: ActivationSingleBatchPostRequest = Body(
        ...,
        example={
            "prompts": [
                "The Jedi in Star Wars wield lightsabers.",
                "The Force is strong with this one.",
            ],
            "model": "gpt2-small",
            "source": "0-res-jb",
            "index": "14057",
        },
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
            content={
                "error": f"Batch size {len(prompts)} exceeds maximum of {MAX_BATCH_SIZE}"
            },
            status_code=400,
        )

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
                str_tokens: list[str] = tokenizer.tokenize(prompt)
                str_tokens = [
                    tokenizer.convert_tokens_to_string([t]) for t in str_tokens
                ]
            else:
                str_tokens: list[str] = model.to_str_tokens(
                    prompt, prepend_bos=prepend_bos
                )  # type: ignore

            all_tokens.append(tokens)
            all_str_tokens.append(str_tokens)

        # Process all prompts in batch
        results = process_activations_batch(model, source, index, all_tokens)

        # Calculate DFA if enabled (for each result)
        if sae_manager.is_dfa_enabled(source):
            for i, (result, tokens) in enumerate(zip(results, all_tokens)):
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

        all_tokens = []
        all_str_tokens = []

        for prompt in prompts:
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

            str_tokens: list[str] = model.to_str_tokens(prompt, prepend_bos=prepend_bos)  # type: ignore
            all_tokens.append(tokens)
            all_str_tokens.append(str_tokens)

        # Process all prompts in batch
        results = process_vector_activations_batch(
            vector, all_tokens, hook, model, sae_manager.device
        )

    logger.info("Returning %d results", len(results))

    # Build response in the same order as input
    response_results = [
        ActivationSingleBatchPost200ResponseResultsInner(
            activation=result, tokens=str_tokens
        )
        for result, str_tokens in zip(results, all_str_tokens)
    ]

    return ActivationSingleBatchPost200Response(results=response_results)


def _get_safe_dtype(dtype: torch.dtype) -> torch.dtype:
    """
    Convert float16 to float32, leave other dtypes unchanged.
    """
    return torch.float32 if dtype == torch.float16 else dtype


def _safe_cast(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """
    Safely cast a tensor to the target dtype, creating a copy if needed.
    Convert float16 to float32, leave other dtypes unchanged.
    """
    safe_dtype = _get_safe_dtype(tensor.dtype)
    if safe_dtype != tensor.dtype or safe_dtype != target_dtype:
        return tensor.to(target_dtype)
    return tensor


def get_layer_num_from_sae_id(sae_id: str) -> int:
    return int(sae_id.split("-")[0]) if not sae_id.isdigit() else int(sae_id)


def process_activations_batch(
    model: HookedTransformer | StandardizedTransformer,  # | TransformerBridge
    layer: str,
    index: int,
    tokens_list: list[torch.Tensor],
) -> list[ActivationSinglePost200ResponseActivation]:
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

    # Track original lengths and BOS indices for each sequence
    original_lengths = []
    all_bos_indices = []

    for i, tokens in enumerate(tokens_list):
        padded_tokens[i, : len(tokens)] = tokens
        original_lengths.append(len(tokens))
        # Find BOS indices for this sequence
        bos_indices = (tokens == bos_token_id).nonzero(as_tuple=True)[0].tolist()
        all_bos_indices.append(bos_indices)

    # Run batch inference
    if is_nnterp_model(model):
        layer_num = get_layer_num_from_sae_id(layer)
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

    # Process each result separately
    results = []
    for i in range(batch_size):
        # Extract the non-padded portion for this sequence
        seq_len = original_lengths[i]

        if sae_type == "neurons":
            # Extract single sequence from batch
            seq_cache = {hook_name: cache[hook_name][i : i + 1, :seq_len]}
            result = process_neuron_activations(
                seq_cache, hook_name, index, sae_manager.device
            )
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
    bos_indices: list[int],
) -> ActivationSinglePost200ResponseActivation:
    if sae_type == "saelens-1":
        return process_saelens_activations(sae, cache, hook_name, index, bos_indices)
    raise ValueError(f"Unsupported SAE type: {sae_type}")


def process_saelens_activations(
    sae: Any,
    cache: ActivationCache | dict[str, torch.Tensor],
    hook_name: str,
    index: int,
    bos_indices: list[int],
) -> ActivationSinglePost200ResponseActivation:
    # if the cache[hook_name] is not on the same device as the sae, move it to the sae's device
    cached_value = cache[hook_name]
    if cached_value.device != sae.device:
        cached_value = cached_value.to(sae.device)
    feature_acts = sae.encode(cached_value)
    values = torch.transpose(feature_acts.squeeze(0), 0, 1)[index].detach().tolist()

    # zero out all values that are the BOS token
    for idx in bos_indices:
        values[idx] = 0

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


def process_vector_activations_batch(
    vector: torch.Tensor | list[float],
    tokens_list: list[torch.Tensor],
    hook_name: str,
    model: HookedTransformer | StandardizedTransformer,  # | TransformerBridge
    device: torch.device,
) -> list[ActivationSinglePost200ResponseActivation]:
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
    _, cache = model.run_with_cache(padded_tokens)

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

        result = ActivationSinglePost200ResponseActivation(
            values=values,
            max_value=max_value,
            max_value_index=values.index(max_value),
        )
        results.append(result)

    return results


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

    # Determine the safe dtype for operations
    v_dtype = _get_safe_dtype(v.dtype)
    attn_weights_dtype = _get_safe_dtype(attn_weights.dtype)
    sae_dtype = _get_safe_dtype(sae.W_enc.dtype)

    # Use the highest precision dtype
    op_dtype = max(v_dtype, attn_weights_dtype, sae_dtype, key=lambda x: x.itemsize)

    # Check if the model uses GQA
    use_gqa = (
        hasattr(model.cfg, "n_key_value_heads")
        and model.cfg.n_key_value_heads is not None
        and model.cfg.n_key_value_heads < model.cfg.n_heads
    )

    if use_gqa:
        n_query_heads = attn_weights.shape[1]
        n_kv_heads = v.shape[2]
        expansion_factor = n_query_heads // n_kv_heads
        v = v.repeat_interleave(expansion_factor, dim=2)

    # Cast tensors to operation dtype
    v = _safe_cast(v, op_dtype)
    attn_weights = _safe_cast(attn_weights, op_dtype)

    v_cat = einops.rearrange(
        v, "batch src_pos n_heads d_head -> batch src_pos (n_heads d_head)"
    )
    attn_weights_bcast = einops.repeat(
        attn_weights,
        "batch n_heads dest_pos src_pos -> batch dest_pos src_pos (n_heads d_head)",
        d_head=model.cfg.d_head,
    )
    decomposed_z_cat = attn_weights_bcast * v_cat.unsqueeze(1)

    # Cast SAE weights to operation dtype
    W_enc = _safe_cast(sae.W_enc[:, index], op_dtype)

    per_src_pos_dfa = einops.einsum(
        decomposed_z_cat,
        W_enc,
        "batch dest_pos src_pos d_model, d_model -> batch dest_pos src_pos",
    )
    per_src_dfa = per_src_pos_dfa[torch.arange(1), torch.tensor([max_value_index]), :]
    dfa_values = per_src_dfa[0].tolist()
    return {
        "dfa_values": dfa_values,
        "dfa_target_index": max_value_index,
        "dfa_max_value": max(dfa_values),
    }
