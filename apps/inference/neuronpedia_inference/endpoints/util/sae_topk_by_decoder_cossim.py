import logging
import math

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from neuronpedia_inference.sae_manager import SAE_TYPE, SAEManager
from neuronpedia_inference.schemas import (
    NPFeature,
    UtilSaeTopkByDecoderCossimFeature,
    UtilSaeTopkByDecoderCossimRequest,
    UtilSaeTopkByDecoderCossimResponse,
)
from neuronpedia_inference.shared import with_request_lock

logger = logging.getLogger(__name__)

router = APIRouter()

# Ceiling on `num_results`. The similarity computation itself is one pass over the decoder and
# does not care, but the response loop below reads each result back with `.item()` -- one
# GPU->CPU sync each -- while holding a concurrency slot. Uncapped, `num_results` at the width
# of a 1M-feature SAE turns one request into a million syncs, which is a slot held for minutes
# and a response nobody asked for. The UI asks for 10.
MAX_NUM_RESULTS = 128


@router.post("/util/sae-topk-by-decoder-cossim", responses={200: {"model": UtilSaeTopkByDecoderCossimResponse}})
@with_request_lock(exclusive=False)
async def sae_topk_by_decoder_cossim(
    request: UtilSaeTopkByDecoderCossimRequest,
):
    # Ensure exactly one of features or vector is provided
    if (request.feature is not None) == (request.vector is not None):
        logger.error("Invalid request data: exactly one of feature or vector must be provided")
        return JSONResponse(
            content={"error": "Invalid request data: exactly one of feature or vector must be provided"},
            status_code=400,
        )

    num_results = request.num_results
    source = request.source
    model = request.model

    try:
        sae_manager = SAEManager.get_instance()
        if sae_manager.get_sae_type(source) != SAE_TYPE.SAELENS:
            raise ValueError(f"Invalid SAE ID or type: {source}")
        # get_sae, not sae_data["sae"]: under paging the latter is None and the weights are
        # on the host until the cache stages them in.
        sae = sae_manager.get_sae(source)
        if sae is None:
            raise ValueError(f"Invalid SAE ID or type: {source}")

        if request.feature:
            index = request.feature.index
            n_features = sae.W_dec.shape[0]
            if not 0 <= index < n_features:
                raise ValueError(f"Feature index {index} is out of range for {source}, which has {n_features} features")
            feature_vector = sae.W_dec[index].clone()
        else:
            feature_vector = torch.tensor(request.vector, device=sae.W_dec.device, dtype=sae.W_dec.dtype)

        result = get_top_k_by_decoder_cosine_similarity(source, model, feature_vector, num_results)

        return UtilSaeTopkByDecoderCossimResponse(
            feature=request.feature,
            topk_decoder_cossim_features=result,
        )
    except ValueError as e:
        logger.error("Error processing request: %s", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return JSONResponse(
            content={"error": "An error occurred while processing the request"},
            status_code=500,
        )


def get_top_k_by_decoder_cosine_similarity(source: str, model: str, feature_vector: torch.Tensor, num_results: int = 5):
    sae_manager = SAEManager.get_instance()
    if sae_manager.get_sae_type(source) != SAE_TYPE.SAELENS:
        raise ValueError(f"Invalid SAE ID or type: {source}")

    sae = sae_manager.get_sae(source)
    if sae is None:
        raise ValueError(f"Invalid SAE ID or type: {source}")
    W_dec = sae.W_dec

    # Every intermediate here is [n_features], never [n_features, d_model]. The previous
    # version masked NaN rows out with `W_dec[~isnan(W_dec).any(dim=1)]`, and boolean-mask
    # indexing copies: that was a full-size bool temporary plus a full COPY of the decoder
    # on every request (~11 GiB combined for a 1M-feature SAE at d_model 2304), for a result
    # that is one scalar per feature.
    query = feature_vector.to(device=W_dec.device, dtype=W_dec.dtype)
    dots = torch.mv(W_dec, query).float()
    # `dtype=` accumulates in fp32 without materializing a converted copy of W_dec.
    w_norms = torch.linalg.vector_norm(W_dec, dim=1, dtype=torch.float32)
    query_norm = torch.linalg.vector_norm(query, dtype=torch.float32)
    # eps matches torch.nn.functional.cosine_similarity's default.
    cosine_similarities = dots / (w_norms * query_norm).clamp_min(1e-8)

    # A NaN decoder row yields a NaN similarity, which -inf demotes below every real one --
    # the same exclusion the mask used to perform.
    cosine_similarities = torch.nan_to_num(cosine_similarities, nan=float("-inf"))

    k = max(1, min(num_results, MAX_NUM_RESULTS, int(W_dec.shape[0])))
    top_k_values, top_k_indices = torch.topk(cosine_similarities, k=k)

    results: list[UtilSaeTopkByDecoderCossimFeature] = []
    for val, idx in zip(top_k_values, top_k_indices):
        cosine_sim_value = val.detach().cpu().item()
        # Error case, but don't fail the request over it. Note this has to reject any
        # non-finite value, not just NaN: demoted (NaN-bearing) decoder rows carry -inf, and
        # `-inf != -inf` is False, so a NaN-only check let them through to be serialized as
        # `-Infinity`, which is not valid JSON.
        if not isinstance(cosine_sim_value, int | float) or not math.isfinite(cosine_sim_value):
            cosine_sim_value = 0.0

        results.append(
            UtilSaeTopkByDecoderCossimFeature(
                feature=NPFeature(
                    source=source,
                    index=int(idx.item()),
                    model=model,
                ),
                cosine_similarity=cosine_sim_value,
            )
        )
    return results
