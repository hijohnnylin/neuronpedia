import logging
import re
from typing import Any

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from neuronpedia_inference_client.models.activation_all_post200_response import (
    ActivationAllPost200Response,
)
from neuronpedia_inference_client.models.activation_all_post200_response_activations_inner import (
    ActivationAllPost200ResponseActivationsInner,
)
from neuronpedia_inference_client.models.activation_all_post_request import (
    ActivationAllPostRequest,
)
from transformer_lens import ActivationCache

from neuronpedia_inference.config import Config
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.shared import (
    Model,
    calculate_per_source_dfa,
    get_activations_by_index,
    get_layer_num_from_sae_id,
    safe_cast,
    with_request_lock,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/activation/all")
@with_request_lock()
async def activation_all(
    request: ActivationAllPostRequest,
):
    sae_manager = SAEManager.get_instance()
    config = Config.get_instance()
    if request.model not in config.get_valid_model_ids():
        logger.error(
            "Invalid model: %s, valid models are %s",
            request.model,
            config.get_valid_model_ids(),
        )
        return JSONResponse(content={"error": "Invalid model"}, status_code=400)
    if request.source_set not in sae_manager.get_valid_sae_sets():
        logger.error(
            "Invalid source set: %s, valid sets are %s",
            request.source_set,
            sae_manager.get_valid_sae_sets(),
        )
        return JSONResponse(content={"error": "Invalid source set"}, status_code=400)

    if len(request.selected_sources) == 0:
        request.selected_sources = sae_manager.sae_set_to_saes[request.source_set]

    # Check if the number of requested layers exceeds the maximum
    if len(request.selected_sources) > config.max_loaded_saes:
        logger.error(
            "Number of requested layers (%s) exceeds the maximum allowed (%s)",
            len(request.selected_sources),
            config.max_loaded_saes,
        )
        return JSONResponse(
            content={
                "error": f"Number of requested SAEs ({len(request.selected_sources)}) exceeds the maximum allowed ({config.max_loaded_saes})"
            },
            status_code=400,
        )

    # get feature filter
    feature_filter = request.feature_filter
    if feature_filter and len(request.selected_sources) != 1:
        logger.error("Feature filter can only be used with a single layer")
        return JSONResponse(
            content={"error": "Feature filter can only be used with a single layer"},
            status_code=400,
        )

    try:
        logger.info("Processing activations")
        processor = ActivationProcessor()
        result = processor.process_activations(request)
        logger.info("Activations result processed successfully")

        return result
    except HTTPException:
        # Let FastAPI handle client errors we raise explicitly
        raise
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
    def process_activations(
        self, request: ActivationAllPostRequest
    ) -> ActivationAllPost200Response:
        model = Model.get_instance()
        sae_manager = SAEManager.get_instance()
        max_layer = max(self._get_layer_num(s) for s in request.selected_sources) + 1
        if max_layer >= model.cfg.n_layers:
            max_layer = None

        # Get the first sae and check if prepend bos is true, then pass to token getter
        first_layer = request.selected_sources[0]
        prepend_bos = sae_manager.get_sae(first_layer).cfg.prepend_bos

        _, str_tokens, cache = self._tokenize_and_get_cache(
            request.prompt, prepend_bos, max_layer
        )
        # ensure sort_by_token_indexes doesn't have any out of range indexes
        for token_index in request.sort_by_token_indexes:
            if token_index >= len(str_tokens) or token_index < 0:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Sort by token index {token_index} is out of range for the "
                        f"given prompt, which only has {len(str_tokens)} tokens."
                    ),
                )

        source_activations = self._process_sources(request, cache)

        sorted_activations = self._sort_and_filter_results(source_activations, request)

        # get pyright checks to pass
        assert request.num_results is not None
        assert request.offset is not None

        start = request.offset
        end = start + request.num_results
        sorted_activations = sorted_activations[start:end]

        feature_activations = self._format_result_and_calculate_dfa(
            sorted_activations, cache, request
        )
        table_counts = self._calculate_table_counts(
            source_activations, str_tokens, request.source_set
        )

        return ActivationAllPost200Response(
            activations=feature_activations,
            tokens=str_tokens,
            counts=table_counts.tolist(),
        )

    def _tokenize_and_get_cache(
        self, text: str, prepend_bos: bool, max_layer: int | None = None
    ) -> tuple[torch.Tensor, list[str], ActivationCache]:
        """Process input text and return tokens, string tokens, and cache."""
        model = Model.get_instance()
        config = Config.get_instance()
        tokens = model.to_tokens(text, prepend_bos=prepend_bos, truncate=False)[0]
        if len(tokens) > config.token_limit:
            raise ValueError(
                f"Text too long: {len(tokens)} tokens, max is {config.token_limit}"
            )

        str_tokens = model.to_str_tokens(text, prepend_bos=prepend_bos)

        with torch.no_grad():
            if max_layer:
                _, cache = model.run_with_cache(tokens, stop_at_layer=max_layer)
            else:
                _, cache = model.run_with_cache(tokens)
        return tokens, str_tokens, cache  # type: ignore

    def _process_sources(
        self,
        request: ActivationAllPostRequest,
        cache: ActivationCache,
    ) -> list[dict[str, Any]]:
        """Process activations for each selected layer."""
        sae_manager = SAEManager.get_instance()
        source_activations = []
        for selected_source in request.selected_sources:
            layer_num = self._get_layer_num(selected_source)
            hook_name = sae_manager.get_sae_hook(selected_source)
            sae_type = sae_manager.get_sae_type(selected_source)

            activations_by_index = get_activations_by_index(
                sae_type, selected_source, cache, hook_name
            )

            # replace activations with only those in the feature list
            if request.feature_filter:
                new_activations_by_index = torch.zeros_like(activations_by_index)
                new_activations_by_index[request.feature_filter] = activations_by_index[
                    request.feature_filter
                ]
                activations_by_index = new_activations_by_index

            source_activations.append(
                self._process_source_activations(
                    activations_by_index,
                    layer_num,
                    request.sort_by_token_indexes,
                )
            )

        return source_activations

    def _process_source_activations(
        self,
        activations_by_index: torch.Tensor,
        layer_num: int,
        sort_by_token_indexes: list[int],
    ) -> dict[str, Any]:
        """Process activations for a single layer."""
        max_values, max_indices = torch.max(activations_by_index, dim=1)
        layer_num_tensor = torch.full(max_values.shape, layer_num).to(
            Config.get_instance().device
        )
        indices_num_tensor = torch.arange(0, max_values.size(0)).to(
            Config.get_instance().device
        )

        if sort_by_token_indexes:
            sum_values = activations_by_index[:, sort_by_token_indexes].sum(dim=1)
        else:
            sum_values = torch.full(max_values.shape, 0).to(
                Config.get_instance().device
            )

        return {
            "layer_num": layer_num_tensor,
            "indices": indices_num_tensor,
            "max_values": max_values,
            "max_indices": max_indices,
            "sum_values": sum_values,
            "activations": activations_by_index,
        }

    def _sort_and_filter_results(
        self,
        source_activations: list[dict[str, Any]],
        request: ActivationAllPostRequest,
    ) -> list[list[float]]:
        """Sort and filter activations based on request parameters."""
        device = Config.get_instance().device
        all_activations = torch.cat(
            [
                torch.cat(
                    (
                        source["layer_num"].unsqueeze(1).to(device),
                        source["indices"].unsqueeze(1).to(device),
                        source["max_values"].unsqueeze(1).to(torch.float32).to(device),
                        source["max_indices"].unsqueeze(1).to(device),
                        source["sum_values"].unsqueeze(1).to(torch.float32).to(device),
                        source["activations"].to(torch.float32).to(device),
                    ),
                    dim=1,
                )
                for source in source_activations
            ],
            dim=0,
        )

        if request.sort_by_token_indexes:
            _, sorted_indices = torch.sort(all_activations[:, 4], descending=True)
        else:
            _, sorted_indices = torch.sort(all_activations[:, 2], descending=True)

        sorted_activations = all_activations[sorted_indices]

        if request.ignore_bos and Model.get_instance().cfg.default_prepend_bos:
            sorted_activations = sorted_activations[sorted_activations[:, 3] != 0]

        return sorted_activations.tolist()

    def _format_result_and_calculate_dfa(
        self,
        sorted_activations: list[list[float]],
        cache: ActivationCache,
        request: ActivationAllPostRequest,
    ) -> list[ActivationAllPost200ResponseActivationsInner]:
        """Format results and if needed, calculate DFA values for sorted activations."""

        feature_activations: list[ActivationAllPost200ResponseActivationsInner] = []
        for result in sorted_activations:
            source = (
                f"{int(result[0])}-{request.source_set}"
                if request.source_set != "neurons"
                else str(int(result[0]))
            )
            feature_index = int(result[1])
            max_value = result[2]
            max_value_index = int(result[3])
            sum_values = result[4]

            feature_activation = ActivationAllPost200ResponseActivationsInner(
                source=source,
                index=feature_index,
                values=result[5:],
                sum_values=sum_values,
                max_value=max_value,
                max_value_index=max_value_index,
            )
            if SAEManager.get_instance().is_dfa_enabled(source):
                dfa_values = self._calculate_dfa_values(
                    cache,
                    int(result[0]),
                    feature_index,
                    max_value_index,
                    request.source_set,
                )
                feature_activation.dfa_values = dfa_values[0].tolist()
                feature_activation.dfa_target_index = max_value_index
                feature_activation.dfa_max_value = max(dfa_values[0].tolist())

            feature_activations.append(feature_activation)

        return feature_activations

    def _calculate_dfa_values(
        self,
        cache: ActivationCache,
        layer_num: int,
        idx: int,
        max_value_index: int,
        source_set: str,
    ) -> torch.Tensor:
        """Calculate DFA values for a specific feature, supporting both standard and GQA models."""
        model = Model.get_instance()
        encoder = SAEManager.get_instance().get_sae(f"{layer_num}-{source_set}")
        v = cache["v", layer_num]
        attn_weights = cache["pattern", layer_num]

        result = calculate_per_source_dfa(
            model=model,
            encoder=encoder,
            v=v,
            attn_weights=attn_weights,
            feature_index=idx,
            max_value_index=max_value_index,
        )

        # Cast the result back to the original dtype of v
        return safe_cast(result, v.dtype)

    def _calculate_table_counts(
        self,
        source_activations: list[dict[str, Any]],
        str_tokens: list[str],
        source_set: str,
    ) -> torch.Tensor:
        """Calculate table counts for activating features."""
        table_max_layer = max(
            self._get_layer_num(s)
            for s in SAEManager.get_instance().sae_set_to_saes[source_set]
        )
        table_counts = torch.zeros((table_max_layer + 1, len(str_tokens)))

        for source_activation in source_activations:
            layer_num = int(source_activation["layer_num"][0])
            activating_features = (source_activation["activations"] > 0).sum(dim=0)
            table_counts[layer_num, :] = activating_features

        return table_counts

    @staticmethod
    def _get_layer_num(sae_id: str) -> int:
        """Get layer number from SAE ID."""
        try:
            return get_layer_num_from_sae_id(sae_id)

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
