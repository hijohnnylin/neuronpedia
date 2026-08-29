"""Wire models for ``/v1/steer/*``.

``completion`` steers a raw text completion, ``completion-chat`` steers a chat exchange.
Both take the same steering knobs and can return STEERED and DEFAULT output side by side so
a caller can diff them, which is why ``types`` is a list rather than a flag.
"""

from enum import StrEnum
from typing import Annotated

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

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


class NPAxisNormalize(StrEnum):
    """How an activation is scaled before it is projected onto a readout axis."""

    L2 = "l2"
    NONE = "none"


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


class NPAxisRender(BaseSchema):
    """How a conversation has to be templated for an axis's numbers to mean anything.

    A projection onto a fitted direction only holds if the conversation reaches the model rendered
    the way it was during fitting, so these conditions travel with the axis. They are applied
    before generation and so change the text itself, which is why every axis in one request has to
    agree about them.
    """

    blank_system_prompt: StrictBool = Field(
        default=False,
        description=(
            "The fit used an empty system turn, so a caller-supplied system prompt is blanked before rendering."
        ),
    )
    template_kwargs: dict[StrictStr, StrictStr] = Field(
        default_factory=dict,
        description=(
            "Extra keyword arguments for the chat template. Llama 3.1 injects the current date "
            "into its system block, so a fit on that model pins `date_string` or drifts off "
            "distribution as the calendar moves."
        ),
    )


class NPAxisSource(BaseSchema):
    """A published axis to fetch, instead of sending its vectors inline.

    The artifact is one folder holding `axis.yaml` and `axis.safetensors`. Everything about the
    axis comes from there, so nothing but `id` may be sent beside this.
    """

    hf_repo_id: StrictStr = Field(description='Hugging Face model repo, e.g. "neuronpedia/persona-axes"')
    hf_folder: StrictStr = Field(description="Folder within that repo holding the two files")
    revision: StrictStr | None = Field(
        default=None,
        description=(
            "Branch, tag or commit sha to read. Defaults to the repo's default branch, and the "
            "commit it resolved to comes back as `sourceRevision` on the readout -- an artifact "
            "that changes under you changes what its readings mean, so pin one to compare across "
            "time."
        ),
    )


class NPAxis(BaseSchema):
    """A readout axis supplied with the request, rather than named by id from this server's assets.

    Either send `source` on its own, or send `direction` and `layer` with as much of the rest as
    the axis has. With everything else defaulted the reading is the dot product of the direction
    with the mean residual-stream activation over each assistant turn, which is the axis you can
    build without a calibration corpus. The optional fields refine that in three steps::

        h          = mean resid_post activation over the assistant turn, at `layer`
        x          = h - preNormMean                    # default: no-op
        x          = x / max(||x||, 1e-12)              # only when normalize is "l2"
        raw        = dot(x - postNormMean, direction)
        value      = (raw - center) / (raw >= center ? scalePos : scaleNeg)
        percentile = interpolated against quantilesPos / quantilesNeg, bounded to [-1, 1]

    So `value` is in the axis's own units until `center` and the two scales put it on a readable
    one, and a `percentile` is reported only once the quantile tables say what the fitting corpus
    looked like. `value` is never clipped: a reading past 1 says the axis is being read off the
    distribution it was fitted on, which is a signal rather than something to pin to the boundary.

    Nothing here says whether a pole is one you would rather see. That is an editorial call about
    a trait rather than a property of a fit, so it belongs to whatever displays the reading.
    """

    id: StrictStr = Field(
        description=(
            "What this axis is reported under, and unique across `axes` and `customAxes` in one "
            "request. Conventionally `<author>_<name>`, though nothing here parses it."
        )
    )
    source: NPAxisSource | None = Field(
        default=None, description="Fetch the axis from a published artifact instead of sending it inline"
    )

    direction: list[StrictFloat] | None = Field(
        default=None,
        description="The fitted direction. Required without `source`, and must be the model's hidden size",
    )
    layer: StrictInt | None = Field(
        default=None, description="Layer to read `resid_post` at. Required without `source`"
    )

    author: StrictStr | None = Field(default=None, description="Who fitted this axis. Reported as-is")
    pole_positive: StrictStr | None = Field(default=None, description='Name of the + pole, e.g. "toxic"')
    pole_negative: StrictStr | None = Field(default=None, description='Name of the - pole, e.g. "respectful"')
    pole_positive_description: StrictStr | None = None
    pole_negative_description: StrictStr | None = None
    display_name: StrictStr | None = Field(
        default=None, description="A label for an axis whose two poles are not one. Nothing parses it"
    )
    caveat: StrictStr | None = Field(
        default=None, description="A known limitation, worth showing beside this axis's values"
    )

    normalize: NPAxisNormalize = NPAxisNormalize.NONE
    pre_norm_mean: list[StrictFloat] | None = Field(
        default=None, description="Subtracted from the activation before normalizing. Rarely needed"
    )
    post_norm_mean: list[StrictFloat] | None = Field(
        default=None,
        description="Subtracted after normalizing. Interchangeable with `preNormMean` when normalize is none",
    )

    center: StrictFloat = Field(default=0.0, description="Where the axis reads zero, in raw projection units")
    scale_pos: StrictFloat = Field(default=1.0, description="Divisor above `center`. May not be zero")
    scale_neg: StrictFloat = Field(default=1.0, description="Divisor below `center`. May not be zero")

    quantiles_pos: list[StrictFloat] | None = Field(
        default=None,
        description=(
            "Distance from `center` at each level of this pole's half of the fitting corpus, "
            "ascending. Sent with `quantilesNeg`, and the two are what a percentile is read off."
        ),
    )
    quantiles_neg: list[StrictFloat] | None = None
    quantile_levels: list[StrictFloat] | None = Field(
        default=None,
        description="Levels the two tables are sampled at. Defaults to evenly spaced 0 to 1",
    )

    render: NPAxisRender = Field(default_factory=NPAxisRender)

    @model_validator(mode="after")
    def check_one_shape(self) -> "NPAxis":
        """Either a source or an inline definition, and never a source with fields beside it.

        Refused rather than merged: a caller who sends both cannot be told which one produced the
        numbers they get back, and half of a published axis overridden by hand is not the axis
        anyone can go and look at.
        """
        if self.source is None:
            missing = [name for name in ("direction", "layer") if getattr(self, name) is None]
            if missing:
                raise ValueError(f"axis {self.id!r}: {missing} required without a `source`")
            return self

        overridden = sorted(
            name
            for name, value in self.__dict__.items()
            if name not in ("id", "source", "render", "normalize", "center", "scale_pos", "scale_neg")
            and value is not None
        )
        # The four scalars above cannot be told apart from their defaults, so they are not checked;
        # the loaded artifact overwrites them either way.
        if overridden:
            raise ValueError(f"axis {self.id!r}: {overridden} cannot be sent with a `source`, which carries them")
        return self


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


class SteerAxisTurn(BaseSchema):
    """One assistant turn's reading on a single axis, as a measurement and as a percentile.

    `value` is calibrated against the axis's own spread, so it passes 1 for roughly 2% of turns
    by construction and is never clipped -- how far past the calibration corpus a turn sits is
    what says an axis is being read off distribution. `percentile` is the same reading expressed
    as the share of that corpus it is past, which cannot leave [-1, 1]. Display the percentile
    and keep the value: a gauge reading "102%" looks broken, and a clipped value would delete
    the diagnostic. `percentile` is absent for an axis whose asset ships no quantile tables.
    """

    value: StrictFloat | None = Field(default=None, description="Axis value for this turn (pre-cap)")
    value_post_cap: StrictFloat | None = Field(
        default=None, description="Axis value for this turn measured under steering (post-cap)"
    )
    percentile: StrictFloat | None = Field(
        default=None,
        description=(
            "Where `value` falls in the axis's calibration corpus, signed by pole and bounded "
            "to [-1, 1]: 0.97 is further along this pole than 97% of that corpus. Absent when "
            "the axis ships no quantile tables."
        ),
    )
    percentile_post_cap: StrictFloat | None = Field(
        default=None, description="`percentile` for this turn measured under steering (post-cap)"
    )
    snippet: StrictStr | None = Field(default=None, description="Truncated conversation content for this turn")


class SteerAxisReadout(BaseSchema):
    """One axis read across the assistant's turns, for one steer type.

    Keyed by `id` rather than by `title`: a title is a display string that may be reworded,
    and these readouts are persisted by callers.
    """

    id: StrictStr = Field(description="Axis id, as requested in `axes` or `customAxes`")
    author: StrictStr = Field(description="Who fitted this axis; the `<author>_` prefix of `id`")
    title: StrictStr = Field(description="Display label for the axis")
    type: NPSteerType | None = None
    layer: StrictInt | None = Field(default=None, description="Layer the axis was fitted and read at")
    caveat: StrictStr | None = Field(
        default=None, description="A known limitation of this axis, worth showing beside its values"
    )
    pole_positive: StrictStr | None = Field(
        default=None, description='What a positive reading means, e.g. "toxic". Absent when the axis names no poles'
    )
    pole_negative: StrictStr | None = Field(
        default=None, description='What a negative reading means, e.g. "respectful"'
    )
    pole_positive_description: StrictStr | None = None
    pole_negative_description: StrictStr | None = None
    source_revision: StrictStr | None = Field(
        default=None,
        description=(
            "The commit this axis was read from, for an axis fetched with `source`. What makes a "
            "reading reproducible: the same revision is the same axis, whatever the branch has "
            "moved on to since."
        ),
    )
    turns: list[SteerAxisTurn] | None = None


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
    custom_axes: list[NPAxis] | None = Field(
        default=None,
        description=(
            "Readout axes to report for the generated turns, sent inline or fetched from a "
            "published artifact. This server ships none of its own, so every axis a request "
            "wants measured is named and defined here. Two entries sharing an id is a 400."
        ),
    )
    steer_special_tokens: StrictBool


class SteerCompletionChatResponse(BaseSchema):
    """
    The steering/default chat responses.
    """

    axes: list[SteerAxisReadout] | None = Field(
        default=None,
        description="Axis readouts for the assistant turns, one entry per requested axis per steer type",
    )
    outputs: list[NPSteerChatResult]
    input: NPSteerChatResult
