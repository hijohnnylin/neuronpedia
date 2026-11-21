import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from neuronpedia_inference.sae_manager import SAE_TYPE, SAEManager
from neuronpedia_inference.schemas import (
    UtilSaeVectorRequest,
    UtilSaeVectorResponse,
)
from neuronpedia_inference.shared import with_request_lock

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/util/sae-vector", responses={200: {"model": UtilSaeVectorResponse}})
@with_request_lock(exclusive=False)
async def sae_vector(request: UtilSaeVectorRequest):
    source = request.source
    index = request.index

    sae_manager = SAEManager.get_instance()
    if sae_manager.get_sae_type(source) == SAE_TYPE.NEURONS:
        return JSONResponse(
            content={"error": f"Source {source} is an MLP neuron layer, which has no SAE vector"},
            status_code=400,
        )
    sae = sae_manager.get_sae(source)

    result = sae.W_dec[index].detach().tolist()

    logger.info("Returning result: %s", result)

    return UtilSaeVectorResponse(vector=result, hookName=sae.cfg.metadata.hook_name)
