import logging
import re

import numpy as np
import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from neuronpedia_inference_client.models.activation_source_post200_response import (
    ActivationSourcePost200Response,
)
from neuronpedia_inference_client.models.activation_source_post200_response_results_inner import (
    ActivationSourcePost200ResponseResultsInner,
)
from neuronpedia_inference_client.models.activation_source_post_request import (
    ActivationSourcePostRequest,
)
from nnterp import StandardizedTransformer

# from transformer_lens.model_bridge import TransformerBridge
from neuronpedia_inference.config import Config
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.shared import Model, with_request_lock

logger = logging.getLogger(__name__)

router = APIRouter()

# Maximum number of prompts that can be processed in a single batch
MAX_BATCH_SIZE = 4

ROUND_DECIMALS = 3


@router.post("/activation/source")
@with_request_lock()
async def activation_source(
    request: ActivationSourcePostRequest,
):
    config = Config.get_instance()
    if request.model not in config.get_valid_model_ids():
        logger.error(
            "Invalid model: %s, valid models are %s",
            request.model,
            config.get_valid_model_ids(),
        )
        return JSONResponse(content={"error": "Invalid model"}, status_code=400)

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
        result = processor.process_activations_batch(request, request.prompts)
        logger.info("Activations result processed successfully")

        return ActivationSourcePost200Response(results=result)
    except Exception as e:
        logger.error(f"Error processing activations: {str(e)}")
        import traceback

        logger.error("Stack trace: %s", traceback.format_exc())
        return JSONResponse(
            content={"error": "An error occurred while processing the request"},
            status_code=500,
        )


class ActivationProcessor:
    @torch.no_grad()
    def process_activations_batch(
        self, request: ActivationSourcePostRequest, prompts: list[str]
    ) -> list[ActivationSourcePost200ResponseResultsInner]:
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
            if isinstance(model, StandardizedTransformer):
                tokens = model.tokenizer(
                    prompt, add_special_tokens=False, return_tensors="pt"
                )["input_ids"][0]
            else:
                tokens = model.to_tokens(
                    prompt,
                    prepend_bos=False,
                    truncate=False,
                )[0]

            # if prompts is an array of one string, the max is config.token_limit
            # if prompts is an array of multiple strings, the max is config.token_limit / MAX_BATCH_SIZE
            if isinstance(prompts, list) and len(prompts) == 1:
                batch_token_limit = config.token_limit
            else:
                batch_token_limit = config.token_limit / MAX_BATCH_SIZE
            if len(tokens) > batch_token_limit:
                if isinstance(prompts, list) and len(prompts) == 1:
                    raise ValueError(
                        f"Text too long: {len(tokens)} tokens, max is {config.token_limit} for single string requests"
                    )
                raise ValueError(
                    f"Text too long: {len(tokens)} tokens, max is {config.token_limit / MAX_BATCH_SIZE} for batch requests"
                )

            if isinstance(model, StandardizedTransformer):
                tokenizer = model.tokenizer
                str_tokens = tokenizer.tokenize(prompt)

                str_tokens = [
                    tokenizer.convert_tokens_to_string([t]) for t in str_tokens
                ]
            else:
                str_tokens = model.to_str_tokens(prompt, prepend_bos=False)

            all_tokens.append(tokens)
            all_str_tokens.append(str_tokens)

        # Pad sequences to the same length
        max_len = max(len(tokens) for tokens in all_tokens)
        batch_size = len(all_tokens)

        # Determine pad token
        if isinstance(model, StandardizedTransformer):
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

        # Calculate max layer needed
        max_layer = self._get_layer_num(request.source) + 1
        if isinstance(model, StandardizedTransformer):
            if max_layer >= model.num_layers:
                max_layer = None
        elif max_layer >= model.cfg.n_layers:
            max_layer = None

        layer_num = self._get_layer_num(request.source)
        hook_name = sae_manager.get_sae_hook(request.source)

        with torch.no_grad():
            if isinstance(model, StandardizedTransformer):
                cache = {}
                with model.trace(padded_tokens):
                    if "resid_post" in hook_name:
                        outputs = model.layers_output[layer_num].save()
                    elif "resid_pre" in hook_name:
                        if layer_num == 0:
                            outputs = model.embeddings_output.save()
                        else:
                            outputs = model.layers_output[layer_num - 1].save()
                    else:
                        raise ValueError(
                            f"Unsupported hook name for nnsight: {hook_name}"
                        )
                    cache[hook_name] = outputs
            else:
                if max_layer:
                    _, cache = model.run_with_cache(
                        padded_tokens, stop_at_layer=max_layer
                    )
                else:
                    _, cache = model.run_with_cache(padded_tokens)

        activation_data = cache[hook_name].to(Config.get_instance().device)
        feature_activation_data = (
            SAEManager.get_instance().get_sae(request.source).encode(activation_data)
        )

        feature_activation_data = feature_activation_data.float().cpu().numpy()

        # Convert feature_activation_data to sparse format for all prompts
        results: list[ActivationSourcePost200ResponseResultsInner] = []
        for i in range(batch_size):
            seq_len = original_lengths[i]
            str_tokens = all_str_tokens[i]

            # Get activations for this prompt (shape: [seq_len, num_features])
            prompt_activations = feature_activation_data[i, :seq_len, :]

            # Find non-zero activations
            nonzero_indices = np.nonzero(prompt_activations)
            token_indices = nonzero_indices[0]
            feature_indices = nonzero_indices[1]
            activation_values = prompt_activations[nonzero_indices]

            # Build sparse dictionary: feature_index -> [[token_index, activation_value], ...]
            active_features: dict[str, list[list[float]]] = {}
            for token_idx, feature_idx, activation_value in zip(
                token_indices, feature_indices, activation_values
            ):
                if token_idx == 0:
                    continue
                feature_key = str(int(feature_idx))
                if feature_key not in active_features:
                    active_features[feature_key] = []
                active_features[feature_key].append(
                    [int(token_idx), round(float(activation_value), ROUND_DECIMALS)]
                )

            results.append(
                ActivationSourcePost200ResponseResultsInner(
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

        except ValueError:
            if "blocks" in sae_id:
                pattern = r"blocks\.(\d+)\.hook"
                match = re.search(pattern, sae_id)
                if match:
                    return int(match.group(1))
                raise ValueError(f"Can't retrieve layer number from SAE ID: {sae_id}")
            if "layer" in sae_id:
                pattern = r"layer_(\d+)"
                match = re.search(pattern, sae_id)
                if match:
                    return int(match.group(1))
                raise ValueError(f"Can't retrieve layer number from SAE ID: {sae_id}")
            raise ValueError(f"Can't retrieve layer number from SAE ID: {sae_id}")
