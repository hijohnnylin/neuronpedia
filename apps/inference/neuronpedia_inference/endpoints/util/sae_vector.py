import logging

from fastapi import APIRouter

from neuronpedia_inference.sae_manager import SAEManager
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

    sae = SAEManager.get_instance().get_sae(source)

    result = sae.W_enc[:, index].detach().tolist()

    logger.info("Returning result: %s", result)

    return UtilSaeVectorResponse(vector=result, hookName=sae.cfg.metadata.hook_name)
