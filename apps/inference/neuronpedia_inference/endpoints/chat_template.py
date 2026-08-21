"""`/apply-chat-template`: render chat messages and return per-token span metadata.

This is the single source of truth for message boundaries. The frontend consumes the
returned per-token spans (role / channel / section / message index) to group tokens into
message bubbles, replacing the per-family state machines in `jlens-chat-format.ts`.

Span computation lives in the engine's `Tokenize.message_spans` (family-agnostic: it renders
the model's real chat template over a growing message prefix and diffs token counts). This
endpoint just exposes it over HTTP and works with whichever backend is loaded (the engine
`EagerModel` exposes `.tok`; other backends get a `Tokenize` built from their `.tokenizer`).
"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from neuronpedia_inference.config import Config
from neuronpedia_inference.engine_adapter import get_tokenize
from neuronpedia_inference.schemas import (
    ApplyChatTemplateRequest,
    ApplyChatTemplateResponse,
    TokenSpan,
)
from neuronpedia_inference.shared import Model, with_request_lock

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/apply-chat-template", responses={200: {"model": ApplyChatTemplateResponse}})
@with_request_lock(exclusive=False)
async def apply_chat_template(request: ApplyChatTemplateRequest):
    config = Config.get_instance()
    tok = get_tokenize(Model.get_instance())

    messages = [m.model_dump(exclude_none=True) for m in request.messages]

    try:
        spans = tok.message_spans(
            messages,
            add_generation_prompt=request.add_generation_prompt,
            **request.chat_template_kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - template rendering is tokenizer-dependent
        logger.error("apply_chat_template failed: %s", exc)
        return JSONResponse(
            content={"error": f"Failed to apply chat template: {exc}"},
            status_code=400,
        )

    if len(spans) > config.token_limit:
        return JSONResponse(
            content={"error": f"Rendered chat too long: {len(spans)} tokens, max is {config.token_limit}"},
            status_code=400,
        )

    prompt = tok.apply_chat_template(
        messages,
        add_generation_prompt=request.add_generation_prompt,
        tokenize=False,
        **request.chat_template_kwargs,
    )

    return ApplyChatTemplateResponse(
        prompt=prompt if isinstance(prompt, str) else "",
        tokens=[s.token_id for s in spans],
        spans=[
            TokenSpan(
                position=s.position,
                token_id=s.token_id,
                token_str=s.token_str,
                message_index=s.message_index,
                role=s.role,
                channel=s.channel,
                section=s.section,
            )
            for s in spans
        ],
    )
