import logging

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from neuronpedia_inference_client.models.np_feature import NPFeature
from neuronpedia_inference_client.models.util_sae_topk_by_decoder_cossim_post200_response import (
    UtilSaeTopkByDecoderCossimPost200Response,
)
from neuronpedia_inference_client.models.util_sae_topk_by_decoder_cossim_post200_response_topk_decoder_cossim_features_inner import (
    UtilSaeTopkByDecoderCossimPost200ResponseTopkDecoderCossimFeaturesInner,
)
from neuronpedia_inference_client.models.util_sae_topk_by_decoder_cossim_post_request import (
    UtilSaeTopkByDecoderCossimPostRequest,
)

from neuronpedia_inference.sae_manager import SAE_TYPE, SAEManager
from neuronpedia_inference.shared import with_request_lock

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/util/sae-topk-by-decoder-cossim")
@with_request_lock()
async def sae_topk_by_decoder_cossim(
    request: UtilSaeTopkByDecoderCossimPostRequest,
):
    # Ensure exactly one of features or vector is provided
    if (request.feature is not None) == (request.vector is not None):
        logger.error(
            "Invalid request data: exactly one of feature or vector must be provided"
        )
        return JSONResponse(
            content={
                "error": "Invalid request data: exactly one of feature or vector must be provided"
            },
            status_code=400,
        )

    num_results = request.num_results
    source = request.source
    model = request.model

    try:
        sae_manager = SAEManager.get_instance()
        sae_data = sae_manager.sae_data.get(source)
        if not sae_data or sae_data["type"] != SAE_TYPE.SAELENS:
            raise ValueError(f"Invalid SAE ID or type: {source}")
        sae = sae_data["sae"]

        if request.feature:
            index = request.feature.index
            feature_vector = sae.W_dec[index].clone()
        else:
            feature_vector = torch.tensor(
                request.vector, device=sae.W_dec.device, dtype=sae.W_dec.dtype
            )

        result = get_top_k_by_decoder_cosine_similarity(
            source, model, feature_vector, num_results
        )

        return UtilSaeTopkByDecoderCossimPost200Response(
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


def get_top_k_by_decoder_cosine_similarity(
    source: str, model: str, feature_vector: torch.Tensor, num_results: int = 5
):
    sae_manager = SAEManager.get_instance()
    sae_data = sae_manager.sae_data.get(source)
    if not sae_data or sae_data["type"] != SAE_TYPE.SAELENS:
        raise ValueError(f"Invalid SAE ID or type: {source}")

    sae = sae_data["sae"]

    # Filter out NaN values in W_dec before computing cosine similarity
    valid_mask = ~torch.isnan(sae.W_dec).any(dim=1)
    valid_W_dec = sae.W_dec[valid_mask]

    cosine_similarities = torch.nn.functional.cosine_similarity(
        feature_vector.unsqueeze(0).to(sae.W_dec.device), valid_W_dec
    )

    # Create a full tensor with -inf for invalid indices and actual similarities for valid ones
    full_cosine_similarities = torch.full(
        (sae.W_dec.shape[0],),
        float("-inf"),
        device=sae.W_dec.device,
        dtype=cosine_similarities.dtype,
    )
    full_cosine_similarities[valid_mask] = cosine_similarities
    cosine_similarities = full_cosine_similarities
    top_k_values, top_k_indices = torch.topk(cosine_similarities, k=num_results)

    results: list[
        UtilSaeTopkByDecoderCossimPost200ResponseTopkDecoderCossimFeaturesInner
    ] = []
    for val, idx in zip(top_k_values, top_k_indices):
        cosine_sim_value = val.detach().cpu().item()
        # Handle NaN values by replacing with 0.0 # this is an error case but we don't want to fail the request
        if (
            not isinstance(cosine_sim_value, (int, float))
            or cosine_sim_value != cosine_sim_value
        ):
            cosine_sim_value = 0.0

        results.append(
            UtilSaeTopkByDecoderCossimPost200ResponseTopkDecoderCossimFeaturesInner(
                feature=NPFeature(
                    source=source,
                    index=int(idx.item()),
                    model=model,
                ),
                cosine_similarity=cosine_sim_value,
            )
        )
    return results
