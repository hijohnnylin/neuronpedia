import logging
from typing import Any

import einops
import torch
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

# from transformer_lens.model_bridge import TransformerBridge
from neuronpedia_inference.config import Config
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.shared import Model, with_request_lock, is_nnterp_model, get_nnterp_layer_output, extract_hidden_states
from neuronpedia_inference_client.models.activation_single_post200_response import (
    ActivationSinglePost200Response,
)
from neuronpedia_inference_client.models.activation_single_post200_response_activation import (
    ActivationSinglePost200ResponseActivation,
)
from neuronpedia_inference_client.models.activation_single_post_request import (
    ActivationSinglePostRequest,
)
from nnterp import StandardizedTransformer
from transformer_lens import ActivationCache, HookedTransformer

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/activation/single")
@with_request_lock()
async def activation_single(
    request: ActivationSinglePostRequest = Body(
        ...,
        example={
            "prompt": "The Jedi in Star Wars wield lightsabers.",
            "model": "gpt2-small",
            "source": "0-res-jb",
            "index": "14057",
        },
    ),
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
        # prepend_bos = sae.cfg.metadata.prepend_bos or model.cfg.tokenizer_prepends_bos
        prepend_bos = False
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
            str_tokens: list[str] = tokenizer.tokenize(prompt)
            str_tokens = [tokenizer.convert_tokens_to_string([t]) for t in str_tokens]
        else:
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

    # if the model prepends the BOS token, then we need to remove the first token from the str_tokens
    if len(str_tokens) > len(result.values):
        str_tokens = str_tokens[1:]

    return ActivationSinglePost200Response(activation=result, tokens=str_tokens)


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


def process_activations(
    model: HookedTransformer | StandardizedTransformer,  # | TransformerBridge
    layer: str,
    index: int,
    tokens: torch.Tensor,
) -> ActivationSinglePost200ResponseActivation:
    sae_manager = SAEManager.get_instance()
    hook_name = sae_manager.get_sae_hook(layer)
    sae_type = sae_manager.get_sae_type(layer)

    # zero out all values that are the BOS token
    bos_token_id = model.tokenizer.bos_token_id
    bos_indices = (tokens == bos_token_id).nonzero(as_tuple=True)[0]

    if is_nnterp_model(model):
        layer_num = get_layer_num_from_sae_id(layer)
        # For vLLM models, pass tokens as a list (vLLM doesn't accept tensors directly)
        trace_input = tokens.tolist() if getattr(model, 'is_vllm', False) else tokens
        with model.trace(trace_input):
            if "resid_post" in hook_name:
                outputs = get_nnterp_layer_output(model, layer_num).save()
            else:
                raise ValueError(f"Unsupported hook name for nnsight: {hook_name}")
        # Extract hidden states from tuple if needed (some layers return (hidden_states, cache))
        cache = {hook_name: extract_hidden_states(outputs)}
        return process_feature_activations(
            sae_manager.get_sae(layer),
            sae_type,
            cache,
            hook_name,
            index,
            bos_indices,
        )

    # if isinstance(model, TransformerBridge) and tokens.ndim == 1:
    #     tokens = tokens.unsqueeze(0)
    _, cache = model.run_with_cache(tokens)

    if sae_type == "neurons":
        return process_neuron_activations(cache, hook_name, index, sae_manager.device)
    if sae_manager.get_sae(layer) is not None:
        return process_feature_activations(
            sae_manager.get_sae(layer), sae_type, cache, hook_name, index, bos_indices
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
    # if the first token was the BOS token, then offset outputs by one removing teh first token
    if len(bos_indices) > 0:
        values = values[1:]

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
