"""Wire models for ``/v1/steer/*``.

``completion`` steers a raw text completion, ``completion-chat`` steers a chat exchange.
Both take the same steering knobs and can return STEERED and DEFAULT output side by side so
a caller can diff them, which is why ``types`` is a list rather than a flag.
"""

from enum import StrEnum
from typing import Annotated

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr

from neuronpedia_inference.schemas.common import BaseSchema, NPLogprob


class NPSteerMethod(StrEnum):
    """How a steering vector is combined with the residual stream."""

    SIMPLE_ADDITIVE = "SIMPLE_ADDITIVE"
    ORTHOGONAL_DECOMP = "ORTHOGONAL_DECOMP"
    PROJECTION_CAP = "PROJECTION_CAP"


class NPSteerType(StrEnum):
    """Whether a given output was produced with steering applied or without it."""

    STEERED = "STEERED"
    DEFAULT = "DEFAULT"


class NPSteerFeature(BaseSchema):
    """An SAE feature to steer with, and how hard."""

    model: StrictStr
    source: StrictStr
    index: StrictInt
    strength: StrictFloat
    steering_vector: list[StrictFloat] | None = None


class NPSteerVector(BaseSchema):
    """
    A raw vector for steering, including its hook and strength
    """

    steering_vector: list[StrictFloat]
    strength: StrictFloat
    hook: StrictStr


class NPSteerChatMessage(BaseSchema):
    """One message in a steered chat exchange."""

    content: StrictStr = Field(description="The chat message")
    role: StrictStr = Field(description='The role of the message (eg "model", "user", etc)')


class NPSteerChatResult(BaseSchema):
    """
    The formatted and unformatted (\"raw\") chat messages
    """

    chat_template: list[NPSteerChatMessage]
    raw: StrictStr
    type: NPSteerType | None = None
    logprobs: list[NPLogprob] | None = Field(
        default=None,
        description='Per-token scores for the generated text, in output order. Absent unless logprobs were both requested and reported by the serving backend, so read its absence as "not available" rather than "no candidates".',
    )


class NPSteerCompletionOutput(BaseSchema):
    """
    A streamed steering/default response. Output is either the whole response or a chunk, depending on response type.
    """

    type: NPSteerType
    output: StrictStr
    logprobs: list[NPLogprob] | None = Field(
        default=None,
        description='Per-token scores for the generated text, in output order. Absent unless logprobs were both requested and reported by the serving backend, so read its absence as "not available" rather than "no candidates".',
    )


class SteerCompletionRequest(BaseSchema):
    """
    Base request for steering
    """

    prompt: StrictStr = Field(description="Text to pass the model for completion")
    model: StrictStr = Field(description="Name of the model")
    steer_method: NPSteerMethod
    normalize_steering: StrictBool
    types: Annotated[list[NPSteerType], Field(min_length=1)] = Field(
        description="Array that specifies whether or not to generate STEERED output, DEFAULT (non-steered) output, or both."
    )
    features: list[NPSteerFeature] | None = Field(default=None, description="Features to steer towards or away from")
    vectors: list[NPSteerVector] | None = None
    n_completion_tokens: Annotated[int, Field(strict=True, ge=1)] = Field(
        description="Number of completion tokens to generate"
    )
    temperature: Annotated[float, Field(strict=True, ge=0)]
    strength_multiplier: StrictFloat = Field(description="The steering strength will be multiplied by this number")
    freq_penalty: StrictFloat
    seed: StrictFloat
    stream: StrictBool | None = Field(
        default=False,
        description='Whether or not to stream responses using Server Side Events (SSE). Note that the OpenAPI spec does not support SSE - you will receive multiple responses with the same format as non-streaming, except with the "output" field chunked.',
    )
    n_logprobs: Annotated[int, Field(le=10, strict=True, ge=0)] | None = Field(
        default=0,
        description="How many candidate tokens to report for each generated position. 0 asks for none. Backends that cannot report scores omit `logprobs` whatever this is set to.",
    )


class SteerCompletionResponse(BaseSchema):
    """
    The steering/default responses.
    """

    outputs: list[NPSteerCompletionOutput]


class SteerAssistantAxisTurn(BaseSchema):
    """One assistant turn's projection onto the persona axes."""

    pc_values: dict[str, StrictFloat] | None = Field(
        default=None, description="Dict mapping PC title to projection value (pre-cap)"
    )
    pc_values_post_cap: dict[str, StrictFloat] | None = Field(
        default=None, description="Dict mapping PC title to projection value (post-cap)"
    )
    snippet: StrictStr | None = Field(default=None, description="Truncated conversation content for this turn")


class SteerAssistantAxis(BaseSchema):
    """Persona monitoring for one steer type, across the assistant's turns."""

    type: NPSteerType | None = None
    pc_titles: list[StrictStr] | None = Field(
        default=None, description="List of principal component titles/descriptions"
    )
    turns: list[SteerAssistantAxisTurn] | None = None


class SteerCompletionChatRequest(BaseSchema):
    """Steer a chat exchange rather than a bare completion."""

    prompt: list[NPSteerChatMessage] = Field(description="Array of chat messages to pass to the model")
    model: StrictStr = Field(description="Name of the model")
    steer_method: NPSteerMethod
    normalize_steering: StrictBool
    types: Annotated[list[NPSteerType], Field(min_length=1)] = Field(
        description="Array that specifies whether or not to generate STEERED output, DEFAULT (non-steered) output, or both."
    )
    features: list[NPSteerFeature] | None = Field(default=None, description="Features to steer towards or away from")
    vectors: list[NPSteerVector] | None = None
    n_completion_tokens: Annotated[int, Field(strict=True, ge=1)] = Field(
        description="Number of completion tokens to generate"
    )
    temperature: Annotated[float, Field(strict=True, ge=0)]
    strength_multiplier: StrictFloat = Field(description="The steering strength will be multiplied by this number")
    freq_penalty: StrictFloat
    seed: StrictFloat
    stream: StrictBool | None = Field(
        default=False,
        description='Whether or not to stream responses using Server Side Events (SSE). Note that the OpenAPI spec does not support SSE - you will receive multiple responses with the same format as non-streaming, except with the "output" field chunked.',
    )
    n_logprobs: Annotated[int, Field(le=10, strict=True, ge=0)] | None = Field(
        default=0,
        description="How many candidate tokens to report for each generated position. 0 asks for none. Backends that cannot report scores omit `logprobs` whatever this is set to.",
    )
    is_assistant_axis: StrictBool | None = False
    steer_special_tokens: StrictBool


class SteerCompletionChatResponse(BaseSchema):
    """
    The steering/default chat responses.
    """

    assistant_axis: list[SteerAssistantAxis] | None = Field(
        default=None, description="Persona monitoring data for assistant turns, one entry per steer type"
    )
    outputs: list[NPSteerChatResult]
    input: NPSteerChatResult
