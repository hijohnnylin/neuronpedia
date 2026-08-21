import logging

import torch
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from interp_engine import (
    EagerModel,
    VLLMModel,
    is_linear_attention_layer,
    run_with_cache,
)

from neuronpedia_inference.config import Config
from neuronpedia_inference.engine_adapter import vllm_attention_unsupported_reason
from neuronpedia_inference.inference_utils.token_limit import reject_if_over_token_limit
from neuronpedia_inference.memory_cost import attention_cost
from neuronpedia_inference.schemas import (
    ActivationAttentionRequest,
    ActivationAttentionResponse,
)
from neuronpedia_inference.shared import Model, with_request_lock

logger = logging.getLogger(__name__)

router = APIRouter()

# Sparsification constants. These MUST match the HeadVis metrics pipeline
# (utils/neuronpedia-utils/neuronpedia_utils/headvis/compute-head-metrics.py) so
# that custom-text attention rows render identically to the stored top sequences.
SPARSE_TOPK_PER_ROW = 8
SPARSE_THRESHOLD = 0.005
VALUE_DECIMALS = 4


@router.post("/activation/attention", responses={200: {"model": ActivationAttentionResponse}})
@with_request_lock(exclusive=False, cost=attention_cost)
async def activation_attention(
    request: ActivationAttentionRequest = Body(
        ...,
        examples=[
            {
                "prompt": "When Mary and John went to the store, John gave a drink to Mary.",
                "model": "gpt2-small",
                "layer": 5,
                "head": 1,
            }
        ],
    ),
):
    model = Model.get_instance()
    config = Config.get_instance()

    # Resolve layer/head counts and the per-layer attention kind for validation. Both
    # backends read the same `layer_types` config field, so the linear-attention guard
    # below covers them equally -- it used to sit inside the EagerModel branch, which left
    # the vLLM path to reconstruct a softmax for layers that have none.
    if not isinstance(model, EagerModel | VLLMModel):
        return JSONResponse(
            content={"error": "Attention patterns are only supported on the interp-engine and vLLM backends."},
            status_code=400,
        )
    num_layers = model.n_layers
    if isinstance(model, EagerModel):
        num_heads = model.n_heads
        is_linear = model.arch.is_linear_attention_layer(request.layer)
        # Eager reads the real softmax out of the model, so no config quirk can be missed.
        unsupported: tuple[str, ...] = ()
    else:
        num_heads = model._attn_dims["n_heads"]
        is_linear = is_linear_attention_layer(model._attn_dims, request.layer)
        unsupported = model._attn_dims.get("unsupported", ())

    if not (0 <= request.layer < num_layers):
        return JSONResponse(
            content={"error": f"Invalid layer: {request.layer}. Must be in [0, {num_layers})."},
            status_code=400,
        )
    if num_heads is not None and not (0 <= request.head < num_heads):
        return JSONResponse(
            content={"error": f"Invalid head: {request.head}. Must be in [0, {num_heads})."},
            status_code=400,
        )
    if is_linear:
        return JSONResponse(
            content={"error": f"Layer {request.layer} is a linear-attention layer with no softmax attention pattern."},
            status_code=400,
        )
    # The vLLM path rebuilds the softmax from captured q/k, so a config term it cannot
    # reproduce yields a plausible-looking pattern that is not the model's. Refusing is the
    # only honest answer; returning the wrong numbers is what this check exists to prevent.
    if unsupported:
        logger.error(
            "Refusing attention for %s: unsupported config for off-kernel recompute: %s",
            request.model,
            "; ".join(unsupported),
        )
        return JSONResponse(
            content={
                "error": "Attention patterns are not supported for this model on the vLLM "
                "backend: " + "; ".join(unsupported)
            },
            status_code=400,
        )
    # Same reasoning, but about the deployment rather than the model: a sharded pod has no
    # rank holding every head, so there is no pattern to return.
    sharding_reason = vllm_attention_unsupported_reason(model) if isinstance(model, VLLMModel) else None
    if sharding_reason is not None:
        logger.error("Refusing attention for %s: %s", request.model, sharding_reason)
        return JSONResponse(
            content={"error": f"Attention patterns are not available on this instance: {sharding_reason}."},
            status_code=400,
        )

    # Tokenize, mirroring /activation/single: prepend BOS to the raw prompt (when
    # the model has a BOS token and the prompt doesn't already start with it) then
    # tokenize without letting the tokenizer add its own special tokens.
    prompt = request.prompt
    prepend_bos = False
    bos_token = model.tokenizer.bos_token
    if bos_token is not None and not prompt.startswith(bos_token):
        prompt = bos_token + prompt

    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos, truncate=False)[0]

    too_long = reject_if_over_token_limit(len(tokens), config.activation_token_limit)
    if too_long is not None:
        return too_long

    str_tokens: list[str] = model.to_str_tokens(prompt, prepend_bos=prepend_bos)  # type: ignore

    # Extract the [q, k] attention pattern for the requested (layer, head). Attention-sink models
    # (gpt-oss) intentionally do not sum to 1 across keys; we never renormalize.
    #
    # `capture_attention` would serve both backends in one call, and is deliberately not used: it
    # returns the whole triple, so the eager arm would also rebuild the pre-softmax scores through
    # the re-dispatched attention -- another [heads, q, q] per layer that this endpoint discards.
    # The vLLM arm has no such choice, since one off-kernel recompute produces all three.
    if isinstance(model, EagerModel):
        ids = tokens.unsqueeze(0) if tokens.ndim == 1 else tokens
        cache = run_with_cache(model, ids, [("attn_probs", request.layer)])
        attn = cache.get("attn_probs", request.layer)
        if attn is None:
            return JSONResponse(
                content={"error": f"No attention probabilities for layer {request.layer}."},
                status_code=400,
            )
        # (batch, n_heads, q, k) -> (q, k).
        attention = attn[0, request.head].float().detach().cpu()
    else:
        # Off-kernel recompute of probs from post-rope q/k, with the sliding-window band and any
        # attention sinks reapplied (the fused kernel's, not ours).
        token_ids = tokens.tolist() if tokens.ndim == 1 else tokens[0].tolist()
        res = await model.capture_attention(token_ids, [request.layer])
        # [n_heads, q, k] -> (q, k) for the requested head.
        attention = res[request.layer]["probs"][request.head].float().detach().cpu()

    sparse = _sparsify_attention(attention)

    logger.info(
        "Returning attention for layer %s head %s (%s tokens, %s nonzero)",
        request.layer,
        request.head,
        sparse["seq_len"],
        len(sparse["attention_values"]),
    )
    return ActivationAttentionResponse(tokens=str_tokens, **sparse)


def _sparsify_attention(attention: torch.Tensor) -> dict:
    """Convert a dense [q, k] attention matrix to the sparse COO format used by
    the HeadVis pipeline.

    Keeps the top-K keys per query row (>= threshold), rounds values, and encodes
    each kept entry as a flat index ``q * seq_len + k``. ``max_activation`` is the
    largest attention weight excluding row 0 and column 0 (the BOS / position-0
    attention sink), matching compute-head-metrics.py.
    """
    seq_len = int(attention.shape[0])

    attention_indices: list[int] = []
    attention_values: list[float] = []

    if seq_len >= 2:
        k = min(SPARSE_TOPK_PER_ROW, seq_len)
        topk_values, topk_indices = torch.topk(attention, k=k, dim=-1)
        topk_values_np = topk_values.numpy()
        topk_indices_np = topk_indices.numpy()

        entries: list[tuple[int, int, float]] = []
        for q in range(seq_len):
            for j in range(k):
                value = float(topk_values_np[q, j])
                if value < SPARSE_THRESHOLD:
                    continue
                key = int(topk_indices_np[q, j])
                entries.append((q, key, round(value, VALUE_DECIMALS)))

        # Sort by (q, k) for stable, readable output (matches the pipeline).
        entries.sort(key=lambda e: (e[0], e[1]))
        for q, key, value in entries:
            attention_indices.append(q * seq_len + key)
            attention_values.append(value)

        # Max attention excluding the position-0 attention sink (row 0 / col 0).
        interior = attention[1:, 1:]
        max_activation = round(float(interior.max()), VALUE_DECIMALS) if interior.numel() > 0 else 0.0
    else:
        max_activation = 0.0

    return {
        "seq_len": seq_len,
        "attention_indices": attention_indices,
        "attention_values": attention_values,
        "max_activation": max_activation,
    }
