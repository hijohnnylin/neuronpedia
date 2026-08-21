"""`/tokenize`: split text into token ids and their string forms.

A thin wrapper over the loaded model's tokenizer, exposed so that callers tokenize
text exactly the way this server does. That includes whether a beginning-of-sequence
token gets prepended, which varies by model family and would otherwise shift every
token index the caller works with.
"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from neuronpedia_inference.config import Config
from neuronpedia_inference.schemas import (
    TokenizeRequest,
    TokenizeResponse,
)
from neuronpedia_inference.shared import Model, with_request_lock

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/tokenize", responses={200: {"model": TokenizeResponse}})
@with_request_lock(exclusive=False)
async def tokenize(request: TokenizeRequest):
    model = Model.get_instance()
    token_limit = Config.get_instance().token_limit

    # Only the engine's EagerModel advertises the model's own BOS behaviour, so
    # assume prepending for any other backend unless the caller decided for us.
    prepend_bos = request.prepend_bos
    if prepend_bos is None:
        prepend_bos = getattr(model, "default_prepend_bos", True)

    token_ids = model.to_tokens(request.text, prepend_bos=prepend_bos, truncate=False)[0]

    if len(token_ids) > token_limit:
        logger.error("Text too long: %s tokens, max is %s", len(token_ids), token_limit)
        return JSONResponse(
            content={"error": f"Text too long: {len(token_ids)} tokens, max is {token_limit}"},
            status_code=400,
        )

    return TokenizeResponse(
        tokens=token_ids.tolist(),
        token_strings=model.to_str_tokens(request.text, prepend_bos=prepend_bos),  # type: ignore
        prepend_bos=prepend_bos,
    )
