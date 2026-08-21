import logging
import re

import numpy as np
import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from neuronpedia_inference.config import Config
from neuronpedia_inference.engine_adapter import (
    BackendUnsupported,
    capture_padded_cache_async,
)
from neuronpedia_inference.memory_cost import activation_source_cost
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.schemas import (
    ActivationSourceRequest,
    ActivationSourceResponse,
    ActivationSourceResult,
)
from neuronpedia_inference.shared import (
    Model,
    RecoverableOutOfMemory,
    recover_from_oom,
    with_request_lock,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Maximum number of prompts that can be processed in a single batch
MAX_BATCH_SIZE = 4

ROUND_DECIMALS = 3


@router.post("/activation/source", responses={200: {"model": ActivationSourceResponse}})
@with_request_lock(exclusive=False, cost=activation_source_cost)
async def activation_source(
    request: ActivationSourceRequest,
):
    config = Config.get_instance()
    config.check_requested_model(request.model)

    # MAX_BATCH_SIZE only divided the per-prompt token limit below; the prompt COUNT was
    # never checked, so the batch (and with it the padded capture and the response) was
    # bounded only by what the client chose to send.
    if len(request.prompts) == 0:
        return JSONResponse(
            content={"error": "At least one prompt is required"},
            status_code=400,
        )
    if len(request.prompts) > MAX_BATCH_SIZE:
        return JSONResponse(
            content={"error": f"Batch size {len(request.prompts)} exceeds maximum of {MAX_BATCH_SIZE}"},
            status_code=400,
        )

    # if the request doesn't start with the bos, prepend it
    bos_token = Model.get_instance().tokenizer.bos_token
    # iterate through prompts and prepend bos if needed
    processed_prompts = []
    for prompt in request.prompts:
        if not prompt.startswith(bos_token):
            prompt = bos_token + prompt
        processed_prompts.append(prompt)
    request.prompts = processed_prompts

    try:
        logger.info("Processing activations")
        processor = ActivationProcessor()
        result = await processor.process_activations_batch(request, request.prompts)
        logger.info("Activations result processed successfully")

        return ActivationSourceResponse(results=result)
    except BackendUnsupported as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        # An allocation OOM here is retryable and does not poison the CUDA context, so it
        # gets a 503 and a reclaimed allocator rather than being flattened into a 500 that
        # leaves the next request to inherit the fragmentation.
        if recover_from_oom(e):
            return JSONResponse(content={"error": str(RecoverableOutOfMemory())}, status_code=503)
        logger.error(f"Error processing activations: {str(e)}")
        import traceback

        logger.error("Stack trace: %s", traceback.format_exc())
        return JSONResponse(
            content={"error": "An error occurred while processing the request"},
            status_code=500,
        )


class ActivationProcessor:
    async def process_activations_batch(
        self, request: ActivationSourceRequest, prompts: list[str]
    ) -> list[ActivationSourceResult]:
        """
        Process multiple prompts in parallel using batched GPU operations.
        Returns results in the same order as input prompts.
        """
        model = Model.get_instance()
        sae_manager = SAEManager.get_instance()
        config = Config.get_instance()

        # Tokenize all prompts
        all_tokens = []
        all_str_tokens = []

        for prompt in prompts:
            tokens = model.to_tokens(
                prompt,
                prepend_bos=False,
                truncate=False,
            )[0]

            # if prompts is an array of one string, the max is config.activation_token_limit
            # if prompts is an array of multiple strings, the max is config.activation_token_limit / MAX_BATCH_SIZE
            if isinstance(prompts, list) and len(prompts) == 1:
                batch_token_limit = config.activation_token_limit
            else:
                batch_token_limit = config.activation_token_limit / MAX_BATCH_SIZE
            if len(tokens) > batch_token_limit:
                if isinstance(prompts, list) and len(prompts) == 1:
                    raise ValueError(
                        f"Text too long: {len(tokens)} tokens, max is {config.activation_token_limit} for single string requests"
                    )
                raise ValueError(
                    f"Text too long: {len(tokens)} tokens, max is {config.activation_token_limit / MAX_BATCH_SIZE} for batch requests"
                )

            str_tokens = model.to_str_tokens(prompt, prepend_bos=False)

            all_tokens.append(tokens)
            all_str_tokens.append(str_tokens)

        # Pad sequences to the same length
        max_len = max(len(tokens) for tokens in all_tokens)
        batch_size = len(all_tokens)

        # Determine pad token
        pad_token_id = (
            model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
        )

        # Create padded batch tensor
        padded_tokens = torch.full(
            (batch_size, max_len),
            pad_token_id,
            dtype=all_tokens[0].dtype,
            device=all_tokens[0].device,
        )

        # Track original lengths
        original_lengths = []
        for i, tokens in enumerate(all_tokens):
            padded_tokens[i, : len(tokens)] = tokens
            original_lengths.append(len(tokens))

        hook_name = sae_manager.get_sae_hook(request.source)

        cache = await capture_padded_cache_async(model, padded_tokens, original_lengths, [hook_name])
        sae = sae_manager.get_sae(request.source)
        device = Config.get_instance().device

        # Convert feature_activation_data to sparse format for all prompts
        results: list[ActivationSourceResult] = []
        for i in range(batch_size):
            seq_len = original_lengths[i]
            str_tokens = all_str_tokens[i]

            # Encode ONE sequence at a time. `encode` is position-wise, so this is
            # numerically identical to encoding the padded batch and slicing afterwards, but
            # peak memory is [seq_len, d_sae] instead of [batch, max_len, d_sae] -- on both
            # the GPU and (via the host copy below) in RAM. It also skips the padding.
            with torch.no_grad():
                activation_data = cache[hook_name][i : i + 1, :seq_len].to(device)
                prompt_activations = sae.encode(activation_data)[0].float().cpu().numpy()

            # Find non-zero activations
            nonzero_indices = np.nonzero(prompt_activations)
            token_indices = nonzero_indices[0]
            feature_indices = nonzero_indices[1]
            activation_values = prompt_activations[nonzero_indices]

            # Build sparse dictionary: feature_index -> [[token_index, activation_value], ...]
            active_features: dict[str, list[list[float]]] = {}
            for token_idx, feature_idx, activation_value in zip(token_indices, feature_indices, activation_values):
                if token_idx == 0:
                    continue
                feature_key = str(int(feature_idx))
                if feature_key not in active_features:
                    active_features[feature_key] = []
                active_features[feature_key].append([int(token_idx), round(float(activation_value), ROUND_DECIMALS)])

            results.append(
                ActivationSourceResult(
                    tokens=str_tokens,
                    activeFeatures=active_features,
                )
            )

        return results

    @staticmethod
    def _get_layer_num(sae_id: str) -> int:
        """Get layer number from SAE ID."""
        try:
            return int(sae_id.split("-")[0]) if not sae_id.isdigit() else int(sae_id)

        except ValueError as e:
            if "blocks" in sae_id:
                pattern = r"blocks\.(\d+)\.hook"
                match = re.search(pattern, sae_id)
                if match:
                    return int(match.group(1))
                raise ValueError(f"Can't retrieve layer number from SAE ID: {sae_id}") from e
            if "layer" in sae_id:
                pattern = r"layer_(\d+)"
                match = re.search(pattern, sae_id)
                if match:
                    return int(match.group(1))
                raise ValueError(f"Can't retrieve layer number from SAE ID: {sae_id}") from e
            raise ValueError(f"Can't retrieve layer number from SAE ID: {sae_id}") from e
