import logging

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from neuronpedia_inference.config import Config
from neuronpedia_inference.endpoints.activation.all import (
    ActivationProcessor as SinglePromptActivationProcessor,
)
from neuronpedia_inference.engine_adapter import (
    BackendUnsupported,
    capture_padded_cache_async,
)
from neuronpedia_inference.memory_cost import activation_all_batch_cost
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.schemas import (
    ActivationAllBatchRequest,
    ActivationAllBatchResponse,
    ActivationAllBatchResult,
    ActivationAllResponse,
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


@router.post("/activation/all-batch", responses={200: {"model": ActivationAllBatchResponse}})
@with_request_lock(exclusive=False, cost=activation_all_batch_cost)
async def activation_all_batch(
    request: ActivationAllBatchRequest,
):
    sae_manager = SAEManager.get_instance()
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

    config.check_requested_model(request.model)
    if request.source_set not in sae_manager.get_valid_sae_sets():
        logger.error(
            "Invalid source set: %s, valid sets are %s",
            request.source_set,
            sae_manager.get_valid_sae_sets(),
        )
        return JSONResponse(content={"error": "Invalid source set"}, status_code=400)

    if len(request.selected_sources) == 0:
        request.selected_sources = sae_manager.sae_set_to_saes[request.source_set]

    # Prepend BOS token to prompts that don't have it
    bos_token = Model.get_instance().tokenizer.bos_token
    processed_prompts = []
    for prompt in prompts:
        if not prompt.startswith(bos_token):
            processed_prompts.append(bos_token + prompt)
        else:
            processed_prompts.append(prompt)

    # # Removed this check because our SAE manager will just load and unload as
    # # needed (though it will be a little slower)
    # # Check if the number of requested layers exceeds the maximum
    # if len(request.selected_sources) > config.max_loaded_saes:
    #     logger.error(
    #         "Number of requested layers (%s) exceeds the maximum allowed (%s)",
    #         len(request.selected_sources),
    #         config.max_loaded_saes,
    #     )
    #     return JSONResponse(
    #         content={
    #             "error": (
    #                 f"Number of requested SAEs ({len(request.selected_sources)})"
    #                 f" exceeds the maximum allowed ({config.max_loaded_saes})"
    #             )
    #         },
    #         status_code=400,
    #     )

    try:
        logger.info("Processing activations for %d prompts", len(processed_prompts))
        processor = ActivationProcessor()

        # Process all prompts in parallel using batched GPU operations
        results = await processor.process_activations_batch(request, processed_prompts)

        logger.info("Activations results processed successfully")

        # Build response with results array
        response_results = [
            ActivationAllBatchResult(
                activations=result.activations,
                tokens=result.tokens,
            )
            for result in results
        ]

        return ActivationAllBatchResponse(results=response_results)
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


class ActivationProcessor(SinglePromptActivationProcessor):
    """Batched sibling of the single-prompt processor.

    Only the capture is batched: one padded forward for the whole batch, then each prompt's
    slice is reduced with the inherited per-source streaming top-K. Everything downstream of
    the capture (the reduction, DFA, layer-number parsing) is identical to the single-prompt
    path and is inherited rather than copied -- the duplicate copy of it that used to live
    here was a second place for the unbounded-memory reduction to be fixed.
    """

    async def process_activations_batch(
        self, request: ActivationAllBatchRequest, prompts: list[str]
    ) -> list[ActivationAllResponse]:
        """
        Process multiple prompts in parallel using batched GPU operations.
        Returns results in the same order as input prompts.
        """
        model = Model.get_instance()
        config = Config.get_instance()
        sae_manager = SAEManager.get_instance()

        # Get the first sae and check if prepend bos is true
        # first_layer = request.selected_sources[0]
        # prepend_bos = sae_manager.get_sae(first_layer).cfg.metadata.prepend_bos
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
            if len(tokens) > batch_token_limit:
                raise ValueError(f"Text too long: {len(tokens)} tokens, max is {batch_token_limit} for batch requests")

            str_tokens = model.to_str_tokens(prompt, prepend_bos=prepend_bos)

            # Validate sort_by_token_indexes for this prompt
            for token_index in request.sort_by_token_indexes:
                if token_index >= len(str_tokens) or token_index < 0:
                    raise ValueError(
                        f"Sort by token index {token_index} is out of range for "
                        f"the given prompt, which only has {len(str_tokens)} tokens."
                    )

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

        # Ensure metadata (hook/type) exists before capture — same reason as the
        # single-prompt path: get_sae_hook does not load on demand. Metadata only, so this
        # does not drag every selected source onto the GPU (see sae_cache).
        for source in request.selected_sources:
            sae_manager.ensure_source(source)

        # Capture only the hook points the selected sources need (backend-aware).
        hook_names = list(dict.fromkeys(sae_manager.get_sae_hook(s) for s in request.selected_sources))
        cache = await capture_padded_cache_async(model, padded_tokens, original_lengths, hook_names)

        # Process each prompt's results from the batch
        results = []
        for i in range(batch_size):
            seq_len = original_lengths[i]
            str_tokens = all_str_tokens[i]

            # Extract this sequence's hook activations
            seq_cache = {hook: cache[hook][i : i + 1, :seq_len] for hook in hook_names}

            # Process this prompt's activations
            sorted_activations = self._stream_top_activations(request, seq_cache, str_tokens)
            # DFA reuses per-layer value+attn_probs within a single prompt only.
            self._dfa_cache = {}
            feature_activations = await self._format_result_and_calculate_dfa(
                sorted_activations, model, all_tokens[i], request
            )

            result = ActivationAllResponse(
                activations=feature_activations,
                tokens=str_tokens,
            )
            results.append(result)

        return results
