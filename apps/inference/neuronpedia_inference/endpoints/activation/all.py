import logging
import re
from typing import Protocol

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from neuronpedia_inference.config import Config
from neuronpedia_inference.engine_adapter import (
    BackendUnsupported,
    DfaInputs,
    DfaResult,
    capture_cache_async,
    capture_dfa_inputs,
    dfa_from_v_and_probs,
)
from neuronpedia_inference.memory_cost import activation_all_cost
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.schemas import (
    ActivationAllFeature,
    ActivationAllRequest,
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

# Result rows are [layer_num, feature_index, max_value, max_value_index, sum_values,
# *activations], so the activation vector starts at column 5 and the two possible sort
# keys sit at columns 2 and 4.
_ROW_HEADER_WIDTH = 5
_SORT_KEY_COL_MAX_VALUE = 2
_SORT_KEY_COL_SUM_VALUES = 4

# Ceiling on `num_results`. The running top-K buffer is [num_results, n_tokens + 5] and the
# response carries every one of those rows as JSON floats, so an unbounded value is both a
# memory and a payload problem. The webapp asks for 50 and caps itself at 100.
MAX_NUM_RESULTS = 1000


class ActivationTopKRequest(Protocol):
    """Shared fields used by single-prompt and batch activation top-K helpers."""

    num_results: int | None
    source_set: str
    selected_sources: list[str]
    ignore_bos: bool
    sort_by_token_indexes: list[int]


@router.post("/activation/all", responses={200: {"model": ActivationAllResponse}})
@with_request_lock(exclusive=False, cost=activation_all_cost)
async def activation_all(
    request: ActivationAllRequest,
):
    sae_manager = SAEManager.get_instance()
    config = Config.get_instance()
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

    # if the request doesn't start with the bos, prepend it
    bos_token = Model.get_instance().tokenizer.bos_token
    if not request.prompt.startswith(bos_token):
        request.prompt = bos_token + request.prompt

    # # Removed this check because our SAE manager will just load and unload as needed (though it will be a little slower)
    # # Check if the number of requested layers exceeds the maximum
    # if len(request.selected_sources) > config.max_loaded_saes:
    #     logger.error(
    #         "Number of requested layers (%s) exceeds the maximum allowed (%s)",
    #         len(request.selected_sources),
    #         config.max_loaded_saes,
    #     )
    #     return JSONResponse(
    #         content={
    #             "error": f"Number of requested SAEs ({len(request.selected_sources)}) exceeds the maximum allowed ({config.max_loaded_saes})"
    #         },
    #         status_code=400,
    #     )

    try:
        logger.info("Processing activations")
        processor = ActivationProcessor()
        result = await processor.process_activations(request)
        logger.info("Activations result processed successfully")

        return result
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
    def __init__(self) -> None:
        # Per-request memo of engine value+attn_probs caches, keyed by layer (DFA reuse).
        self._dfa_cache: dict[int, DfaInputs] = {}

    async def process_activations(self, request: ActivationAllRequest) -> ActivationAllResponse:
        model = Model.get_instance()

        # Get the first sae and check if prepend bos is true, then pass to token getter
        # first_layer = request.selected_sources[0]
        # prepend_bos = sae_manager.get_sae(first_layer).cfg.metadata.prepend_bos
        prepend_bos = False
        # if the prompt doesn't start with the bos, prepend it
        bos_token = model.tokenizer.bos_token
        if not request.prompt.startswith(bos_token):
            request.prompt = bos_token + request.prompt

        tokens, str_tokens, cache = await self._tokenize_and_get_cache(request.prompt, prepend_bos, request)

        # ensure sort_by_token_indexes doesn't have any out of range indexes
        # TODO: return a better error for this (currently returns a 500 error)
        for token_index in request.sort_by_token_indexes:
            if token_index >= len(str_tokens) or token_index < 0:
                raise ValueError(
                    f"Sort by token index {token_index} is out of range for the given prompt, which only has {len(str_tokens)} tokens."
                )

        sorted_activations = self._stream_top_activations(request, cache, str_tokens)
        feature_activations = await self._format_result_and_calculate_dfa(sorted_activations, model, tokens, request)

        # The schema's optional `counts` (a [layer, token] histogram) is intentionally left
        # unset: no client reads it, and filling it costs a full pass over every source's
        # [d_sae, n_tokens] block plus a layers x tokens array in the payload.
        return ActivationAllResponse(
            activations=feature_activations,
            tokens=str_tokens,
        )

    async def _tokenize_and_get_cache(
        self,
        text: str,
        prepend_bos: bool,
        request: ActivationAllRequest,
    ) -> tuple[torch.Tensor, list[str], dict[str, torch.Tensor]]:
        """Process input text and return tokens, string tokens, and per-source-hook cache."""
        model = Model.get_instance()
        config = Config.get_instance()
        sae_manager = SAEManager.get_instance()

        tokens = model.to_tokens(
            text,
            prepend_bos=prepend_bos,
            truncate=False,
        )[0]
        if len(tokens) > config.activation_token_limit:
            raise ValueError(f"Text too long: {len(tokens)} tokens, max is {config.activation_token_limit}")

        str_tokens = model.to_str_tokens(text, prepend_bos=prepend_bos)

        # Ensure each selected source has metadata (hook/type) before capture: get_sae_hook
        # alone does not load, so a source missing from the startup set would otherwise hand
        # capture a None hook. Deliberately NOT get_sae -- that would stage every selected
        # source onto the GPU here (see sae_cache), when the encode loop below wants exactly
        # one at a time.
        for source in request.selected_sources:
            sae_manager.ensure_source(source)

        # Capture only the hook points the selected sources need (backend-aware).
        hook_names = list(dict.fromkeys(sae_manager.get_sae_hook(s) for s in request.selected_sources))
        cache = await capture_cache_async(model, tokens, hook_names)
        return tokens, str_tokens, cache

    def _stream_top_activations(
        self,
        request: ActivationTopKRequest,
        cache: dict[str, torch.Tensor],
        str_tokens: list[str],
    ) -> list[list[float]]:
        """Reduce the selected sources to the global top-K, one source at a time.

        Peak GPU memory is ONE source's ``[d_sae, n_tokens]`` encode plus the
        ``[num_results, n_tokens + 5]`` result buffer, so it does not grow with the number
        of selected sources. That matters because an empty ``selected_sources`` expands to
        the whole set (the all-layers default in the UI): holding every source at once and
        then sorting a copy of the concatenation cost ~9.4 GiB for 26 gemmascope-res-65k
        sources at 550 tokens, against roughly 2.4 GiB of real headroom outside the vLLM
        pool.
        """
        sae_manager = SAEManager.get_instance()
        # OpenAPI default is 25; treat an explicit null the same way.
        num_results = min(
            request.num_results if request.num_results is not None else 25,
            MAX_NUM_RESULTS,
        )

        top_rows: torch.Tensor | None = None
        for selected_source in request.selected_sources:
            layer_num = self._get_layer_num(selected_source)
            hook_name = sae_manager.get_sae_hook(selected_source)
            sae_type = sae_manager.get_sae_type(selected_source)

            activations_by_index = self._get_activations_by_index(sae_type, selected_source, cache, hook_name)

            if request.ignore_bos and Model.get_instance().default_prepend_bos:
                activations_by_index[:, 0] = 0

            candidates = self._source_top_rows(activations_by_index, layer_num, request, num_results)
            top_rows = self._merge_top_rows(top_rows, candidates, request, num_results)

            # Drop this source's [d_sae, n_tokens] block before encoding the next one. The
            # caching allocator then reuses the same block, which is the whole point.
            del activations_by_index, candidates

        if top_rows is None:
            return []
        return top_rows.tolist()

    def _source_top_rows(
        self,
        activations_by_index: torch.Tensor,
        layer_num: int,
        request: ActivationTopKRequest,
        num_results: int,
    ) -> torch.Tensor:
        """The best ``num_results`` rows from ONE source, in descending sort-key order.

        Reducing per source before merging is exact, not an approximation: the sort key is
        per-row and rows are never combined across sources, so the global top-K can draw at
        most K rows from any single source.
        """
        device = Config.get_instance().device
        max_values, max_indices = torch.max(activations_by_index, dim=1)

        if request.sort_by_token_indexes:
            # Re-check against the ACTUAL captured width, not len(str_tokens) as the caller
            # did: these are GPU advanced indices, so an out-of-range one triggers a
            # device-side assert that poisons the CUDA context and kills the whole server
            # rather than failing this one request. Tripping this means the backend captured
            # fewer positions than the prompt has tokens.
            captured_len = activations_by_index.shape[1]
            out_of_range = [i for i in request.sort_by_token_indexes if i < 0 or i >= captured_len]
            if out_of_range:
                raise ValueError(
                    f"Sort by token indexes {out_of_range} are out of range for the captured "
                    f"activations, which cover only {captured_len} token position(s)."
                )
            sum_values = activations_by_index[:, request.sort_by_token_indexes].sum(dim=1)
            sort_key = sum_values
        else:
            sum_values = torch.zeros_like(max_values)
            sort_key = max_values

        keep = min(num_results, int(sort_key.shape[0]))
        # Stable, so equal keys keep feature-index order. Ties are the common case rather
        # than a curiosity: on a short prompt most features never fire, so the tail of the
        # top-K is all exact zeros.
        _, order = torch.sort(sort_key.to(torch.float32), descending=True, stable=True)
        order = order[:keep]

        return torch.cat(
            (
                torch.full((keep, 1), layer_num, dtype=torch.float32, device=device),
                order.unsqueeze(1).to(torch.float32).to(device),
                max_values[order].unsqueeze(1).to(torch.float32).to(device),
                max_indices[order].unsqueeze(1).to(torch.float32).to(device),
                sum_values[order].unsqueeze(1).to(torch.float32).to(device),
                activations_by_index[order].to(torch.float32).to(device),
            ),
            dim=1,
        )

    def _merge_top_rows(
        self,
        top_rows: torch.Tensor | None,
        candidates: torch.Tensor,
        request: ActivationTopKRequest,
        num_results: int,
    ) -> torch.Tensor:
        """Fold one source's rows into the running top-K.

        ``top_rows`` is concatenated ahead of ``candidates`` and the sort is stable, so among
        equal keys the earlier source wins -- the same order the single global sort over the
        full concatenation used to produce.
        """
        if top_rows is None:
            return candidates
        key_col = _SORT_KEY_COL_SUM_VALUES if request.sort_by_token_indexes else _SORT_KEY_COL_MAX_VALUE
        merged = torch.cat((top_rows, candidates), dim=0)
        _, order = torch.sort(merged[:, key_col], descending=True, stable=True)
        return merged[order[:num_results]]

    def _get_activations_by_index(
        self,
        sae_type: str,
        selected_source: str,
        cache: dict[str, torch.Tensor],
        hook_name: str,
    ) -> torch.Tensor:
        """Get activations by index for a specific layer and SAE type."""
        if sae_type == "neurons":
            mlp_activation_data = cache[hook_name].to(Config.get_instance().device)
            return torch.transpose(mlp_activation_data[0], 0, 1)

        activation_data = cache[hook_name].to(Config.get_instance().device)
        feature_activation_data = SAEManager.get_instance().get_sae(selected_source).encode(activation_data)
        return torch.transpose(feature_activation_data.squeeze(0), 0, 1)

    async def _format_result_and_calculate_dfa(
        self,
        sorted_activations: list[list[float]],
        model: object,
        tokens: torch.Tensor,
        request: ActivationTopKRequest,
    ) -> list[ActivationAllFeature]:
        """Format results and if needed, calculate DFA values for sorted activations.

        DFA (attention probs x value) runs on EagerModel and vLLM (off-kernel recompute);
        per-layer capture is memoized in ``self._dfa_cache`` and reused across features.
        """
        feature_activations: list[ActivationAllFeature] = []
        for result in sorted_activations:
            source = (
                f"{int(result[0])}-{request.source_set}" if request.source_set != "neurons" else str(int(result[0]))
            )
            feature_index = int(result[1])
            max_value = result[2]
            max_value_index = int(result[3])
            sum_values = result[4]

            feature_activation = ActivationAllFeature(
                source=source,
                index=feature_index,
                values=result[_ROW_HEADER_WIDTH:],
                sum_values=sum_values,
                max_value=max_value,
                max_value_index=max_value_index,
            )
            if SAEManager.get_instance().is_dfa_enabled(source):
                dfa = await self._calculate_dfa_values(
                    model,
                    tokens,
                    int(result[0]),
                    feature_index,
                    max_value_index,
                    request.source_set,
                )
                feature_activation.dfa_values = dfa["dfa_values"]
                feature_activation.dfa_target_index = dfa["dfa_target_index"]
                feature_activation.dfa_max_value = dfa["dfa_max_value"]

            feature_activations.append(feature_activation)

        return feature_activations

    async def _dfa_layer_cache(self, model: object, tokens: torch.Tensor, layer_num: int):
        """Memoized backend-aware (value, attn_probs, dims) for a layer (reused across features)."""
        cached = self._dfa_cache.get(layer_num)
        if cached is None:
            cached = await capture_dfa_inputs(model, tokens, layer_num)
            self._dfa_cache[layer_num] = cached
        return cached

    async def _calculate_dfa_values(
        self,
        model: object,
        tokens: torch.Tensor,
        layer_num: int,
        idx: int,
        max_value_index: int,
        source_set: str,
    ) -> DfaResult:
        """DFA for one feature (GQA-aware), backend-agnostic via the shared engine helper."""
        encoder = SAEManager.get_instance().get_sae(f"{layer_num}-{source_set}")
        v, attn_weights, dims = await self._dfa_layer_cache(model, tokens, layer_num)
        return dfa_from_v_and_probs(v, attn_weights, encoder.W_enc, idx, max_value_index, **dims)

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
