import logging

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from neuronpedia_inference.config import Config
from neuronpedia_inference.engine_adapter import (
    BackendUnsupported,
    capture_activation_async,
)
from neuronpedia_inference.inference_utils.token_limit import reject_if_over_token_limit
from neuronpedia_inference.memory_cost import similarity_matrix_cost
from neuronpedia_inference.sae_manager import SAE_TYPE, SAEManager
from neuronpedia_inference.schemas import (
    SimilarityMatrixRequest,
    SimilarityMatrixResponse,
)
from neuronpedia_inference.shared import (
    Model,
    with_request_lock,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/util/similarity-matrix-pred", responses={200: {"model": SimilarityMatrixResponse}})
@with_request_lock(exclusive=False, cost=similarity_matrix_cost)
async def similarity_matrix(request: SimilarityMatrixRequest):
    model = Model.get_instance()
    source = request.sourceId

    sae_manager = SAEManager.get_instance()
    if sae_manager.get_sae_type(source) == SAE_TYPE.NEURONS:
        return JSONResponse(
            content={"error": f"Source {source} is an MLP neuron layer, which has no temporal SAE"},
            status_code=400,
        )
    sae = sae_manager.get_sae(source)

    # if the sae architecture is not temporal, throw error
    if sae.cfg.architecture() != "temporal":
        logger.error("SAE architecture is not temporal")
        return JSONResponse(
            content={"error": "SAE architecture is not temporal"},
            status_code=400,
        )

    # tokenize the text
    prepend_bos = model.tokenizer_prepends_bos
    tokens = model.to_tokens(
        request.text,
        prepend_bos=prepend_bos,
        truncate=False,
    )[0]
    logger.info("tokens: %s", tokens)

    # Every cost here is quadratic in the token count -- the [L, L] similarity matrix on the
    # GPU, its host copy, and the L^2 floats in the response -- and this endpoint had no
    # length check at all (it tokenizes with truncate=False).
    config = Config.get_instance()
    too_long = reject_if_over_token_limit(len(tokens), config.activation_token_limit)
    if too_long is not None:
        return too_long

    str_tokens = model.to_str_tokens(request.text, prepend_bos=prepend_bos)
    logger.info("str_tokens: %s", str_tokens)

    hook_name = sae.cfg.metadata.hook_name
    try:
        activation = await capture_activation_async(model, tokens, hook_name)
    except BackendUnsupported as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    # The vLLM backend returns captures on the CPU regardless of where the SAE lives.
    _, z_pred = sae.encode_with_predictions(activation.to(sae.device))

    # Extract single batch sample
    pred_LD = z_pred[0]  # Shape: [L, D]

    # Center the predictions
    pred_centered_LD = pred_LD - torch.mean(pred_LD, dim=0, keepdim=True)

    # Normalize along the D dimension
    pred_LD_normalized = torch.nn.functional.normalize(
        pred_centered_LD.float(), p=2, dim=-1
    )  # L x D, normalized along D

    # Compute cosine similarity: pred_LD @ pred_LD.T -> L x L
    cosine_sim_LL = pred_LD_normalized @ pred_LD_normalized.T
    cosine_sim_np = cosine_sim_LL.detach().cpu().numpy()

    # Remove the BOS token. With no BOS (Qwen3), position 0 is a real token, so keep it.
    bos_token_id = model.tokenizer.bos_token_id
    if bos_token_id is not None and len(tokens) > 0 and int(tokens[0]) == bos_token_id:
        str_tokens = str_tokens[1:]
        cosine_sim_np = cosine_sim_np[1:, 1:]

    return JSONResponse(
        content={"similarity_matrix": cosine_sim_np.tolist(), "tokens": str_tokens},
        status_code=200,
    )
