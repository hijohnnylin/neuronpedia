import logging

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from interp_engine import special_token_positions

from neuronpedia_inference.config import Config
from neuronpedia_inference.engine_adapter import BackendUnsupported, capture_cache_async
from neuronpedia_inference.inference_utils.token_limit import reject_if_over_token_limit
from neuronpedia_inference.memory_cost import activation_topk_by_token_cost
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.schemas import (
    ActivationTopkByTokenFeature,
    ActivationTopkByTokenRequest,
    ActivationTopkByTokenResponse,
    ActivationTopkByTokenResult,
)
from neuronpedia_inference.shared import Model, with_request_lock

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5

router = APIRouter()


def get_layer_num_from_sae_id(sae_id: str) -> int:
    return int(sae_id.split("-")[0]) if not sae_id.isdigit() else int(sae_id)


def build_token_results(
    str_tokens: list[str],
    top_k_values: torch.Tensor,
    top_k_indices: torch.Tensor,
    special_positions: set[int],
    ignore_bos: bool,
) -> tuple[list[str], list[ActivationTopkByTokenResult]]:
    """Per-token TopK results, plus the tokens they describe.

    ``ignore_bos`` drops the BOS these endpoints prepend at position 0.
    Everything per-position moves together, and ``token_position`` stays an index
    into the prompt rather than into what survived the slice — it used to be the
    latter, so every reported position was one too low whenever ``ignore_bos``
    was set, and ``special_positions`` (computed from the unsliced ids) would
    have inherited the same skew.

    Shared by the batch and non-batch endpoints, which return the same per-token
    object and must not drift on any of this.
    """
    offset = 1 if ignore_bos else 0
    if offset:
        str_tokens = str_tokens[offset:]
        top_k_values = top_k_values[offset:]
        top_k_indices = top_k_indices[offset:]

    results = [
        ActivationTopkByTokenResult(
            token=token,
            token_position=offset + local_idx,
            is_special=(offset + local_idx) in special_positions,
            top_features=[
                ActivationTopkByTokenFeature(
                    feature_index=int(idx.item()),
                    activation_value=float(val.item()),
                )
                for val, idx in zip(values, indices)
            ],
        )
        for local_idx, (token, values, indices) in enumerate(zip(str_tokens, top_k_values, top_k_indices))
    ]
    return str_tokens, results


@router.post("/activation/topk-by-token", responses={200: {"model": ActivationTopkByTokenResponse}})
@with_request_lock(exclusive=False, cost=activation_topk_by_token_cost)
async def activation_topk_by_token(
    request: ActivationTopkByTokenRequest,
):
    model = Model.get_instance()
    config = Config.get_instance()
    sae_manager = SAEManager.get_instance()
    prompt = request.prompt
    source = request.source
    top_k = request.top_k if request.top_k is not None else DEFAULT_TOP_K

    ignore_bos = request.ignore_bos

    # Resolve + validate the source up front (matches /activation/single): this populates
    # the SAE's metadata so get_sae_hook/get_sae_type below return real values, and raises
    # an informative AssertionError ("Found 0 entries when searching for <model>/<source>")
    # for an unknown source instead of later crashing on a None hook name.
    sae_manager.ensure_source(source)

    prepend_bos = False

    # if the request doesn't start with the bos, prepend it
    bos_token = Model.get_instance().tokenizer.bos_token
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

    str_tokens = model.to_str_tokens(prompt, prepend_bos=prepend_bos)

    hook_name = sae_manager.get_sae_hook(source)
    sae_type = sae_manager.get_sae_type(source)

    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    try:
        cache = await capture_cache_async(model, tokens, [hook_name])
    except BackendUnsupported as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

    activations_by_index = get_activations_by_index(
        sae_type,
        source,
        cache,
        hook_name,
    )

    # The results loop below zips str_tokens with the per-position activations, and zip
    # stops at the shorter one. So a capture covering fewer positions than the prompt has
    # tokens would silently return a truncated 200 that looks like a valid answer. Refuse
    # instead: a partial result here is indistinguishable from a correct one downstream.
    captured_len = activations_by_index.shape[1]
    if captured_len != len(str_tokens):
        logger.error(
            "Captured %s token position(s) for a %s token prompt",
            captured_len,
            len(str_tokens),
        )
        return JSONResponse(
            content={
                "error": f"Captured activations cover {captured_len} token position(s) but the "
                f"prompt has {len(str_tokens)} tokens; refusing to return a truncated result."
            },
            status_code=500,
        )

    # Get top k activations for each token. Clamped to the feature count: torch.topk raises
    # when k exceeds the axis it reduces, which turned an over-large top_k into a 500.
    n_features = activations_by_index.shape[0]
    effective_top_k = max(1, min(top_k, n_features))
    top_k_values, top_k_indices = torch.topk(activations_by_index.T, k=effective_top_k)

    # From the full id tensor, before any slicing, so it stays aligned with the
    # ids it was derived from.
    special_positions = set(special_token_positions(tokens[0], model.tokenizer))
    str_tokens, results = build_token_results(
        str_tokens,  # type: ignore
        top_k_values,
        top_k_indices,
        special_positions,
        ignore_bos,
    )

    return ActivationTopkByTokenResponse(
        results=results,
        tokens=str_tokens,
    )


# Keep the get_activations_by_index function from the original code
def get_activations_by_index(
    sae_type: str,
    selected_layer: str,
    cache: dict[str, torch.Tensor],
    hook_name: str,
) -> torch.Tensor:
    if sae_type == "neurons":
        mlp_activation_data = cache[hook_name].to(Config.get_instance().device)
        return torch.transpose(mlp_activation_data[0], 0, 1)

    activation_data = cache[hook_name].to(Config.get_instance().device)
    feature_activation_data = SAEManager.get_instance().get_sae(selected_layer).encode(activation_data)
    return torch.transpose(feature_activation_data.squeeze(0), 0, 1)
