"""Wire models for ``/v1/apply-chat-template``.

This endpoint is the single source of truth for message boundaries: the per-token spans it
returns are what let a client group tokens into message bubbles without reimplementing any
model family's chat format.

Hand-written rather than generated, so the annotations stay loose (see the note in
``activation.py``).
"""

from neuronpedia_inference.schemas.common import BaseSchema


class ChatMessage(BaseSchema):
    """One message to render through the model's chat template."""

    role: str
    content: str
    # Optional harmony-style channel (analysis/final/commentary); surfaced back on spans.
    channel: str | None = None


class ApplyChatTemplateRequest(BaseSchema):
    """Messages to render, plus the template switches to render them under."""

    messages: list[ChatMessage]
    add_generation_prompt: bool = True
    # Passed through to the tokenizer's chat template (e.g. {"enable_thinking": false}).
    chat_template_kwargs: dict = {}


class TokenSpan(BaseSchema):
    """One rendered token and the message it belongs to."""

    position: int
    token_id: int
    token_str: str
    message_index: int | None
    role: str | None
    channel: str | None
    section: str


class ApplyChatTemplateResponse(BaseSchema):
    """The rendered prompt, its token ids, and the per-token span metadata."""

    prompt: str
    tokens: list[int]
    spans: list[TokenSpan]
