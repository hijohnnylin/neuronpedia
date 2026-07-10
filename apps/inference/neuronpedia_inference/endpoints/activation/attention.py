import logging

import nnsight
import torch
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from nnterp import StandardizedTransformer
from nnterp.rename_utils import RenamingError
from pydantic import BaseModel
from transformer_lens import HookedTransformer

from neuronpedia_inference.config import Config
from neuronpedia_inference.shared import Model, with_request_lock

logger = logging.getLogger(__name__)

router = APIRouter()

# Sparsification constants. These MUST match the HeadVis metrics pipeline
# (utils/neuronpedia-utils/neuronpedia_utils/headvis/compute-head-metrics.py) so
# that custom-text attention rows render identically to the stored top sequences.
SPARSE_TOPK_PER_ROW = 8
SPARSE_THRESHOLD = 0.005
VALUE_DECIMALS = 4


class ActivationAttentionPostRequest(BaseModel):
    prompt: str
    model: str
    # Integer layer + head. Attention heads are not SAE/source-based, so we index
    # the model's attention layers/query-heads directly.
    layer: int
    head: int


@router.post("/activation/attention")
@with_request_lock()
async def activation_attention(
    request: ActivationAttentionPostRequest = Body(
        ...,
        example={
            "prompt": "When Mary and John went to the store, John gave a drink to Mary.",
            "model": "gpt2-small",
            "layer": 5,
            "head": 1,
        },
    ),
):
    model = Model.get_instance()
    config = Config.get_instance()

    # Resolve layer/head counts for validation.
    if isinstance(model, StandardizedTransformer):
        num_layers = model.num_layers
        num_heads = model.num_heads
    elif isinstance(model, HookedTransformer):
        num_layers = model.cfg.n_layers
        num_heads = model.cfg.n_heads
    else:
        return JSONResponse(
            content={
                "error": "Attention patterns are only supported on TransformerLens and NNsight engines."
            },
            status_code=400,
        )

    if not (0 <= request.layer < num_layers):
        return JSONResponse(
            content={
                "error": f"Invalid layer: {request.layer}. Must be in [0, {num_layers})."
            },
            status_code=400,
        )
    if num_heads is not None and not (0 <= request.head < num_heads):
        return JSONResponse(
            content={
                "error": f"Invalid head: {request.head}. Must be in [0, {num_heads})."
            },
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

    if isinstance(model, StandardizedTransformer):
        tokens = model.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ][0]
    else:
        tokens = model.to_tokens(prompt, prepend_bos=prepend_bos, truncate=False)[0]

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

    if isinstance(model, StandardizedTransformer):
        tokenizer = model.tokenizer
        str_tokens: list[str] = tokenizer.tokenize(prompt)
        str_tokens = [tokenizer.convert_tokens_to_string([t]) for t in str_tokens]
    else:
        str_tokens: list[str] = model.to_str_tokens(prompt, prepend_bos=prepend_bos)  # type: ignore

    # Extract the [q, k] attention pattern for the requested (layer, head).
    if isinstance(model, StandardizedTransformer):
        if not model.attn_probs_available:
            return JSONResponse(
                content={
                    "error": "Attention probabilities are not available for this NNsight model."
                },
                status_code=400,
            )
        try:
            with model.trace(tokens):
                saved = nnsight.save(model.attention_probabilities[request.layer])
        except RenamingError as exc:
            # Hybrid models (e.g. Qwen3.6) only have softmax attention on their
            # full-attention layers; the interleaved linear-attention layers
            # have no attention pattern to return.
            return JSONResponse(
                content={
                    "error": f"No attention probabilities for layer {request.layer}: {exc}"
                },
                status_code=400,
            )
        # (batch, n_heads, q, k) -> (q, k)
        attention = saved[0, request.head].float().detach().cpu()
    else:
        _, cache = model.run_with_cache(tokens)
        # cache["pattern", layer] -> (batch, n_heads, dest/q, src/k)
        attention = cache["pattern", request.layer][0, request.head].float().detach().cpu()

    result = _sparsify_attention(attention)
    result["tokens"] = str_tokens

    logger.info(
        "Returning attention for layer %s head %s (%s tokens, %s nonzero)",
        request.layer,
        request.head,
        result["seq_len"],
        len(result["attention_values"]),
    )
    return JSONResponse(content=result)


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
