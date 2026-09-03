"""Wire models for ``/v1/steer/*``.

``completion`` steers a raw text completion, ``completion-chat`` steers a chat exchange.
Both take the same steering knobs and can return STEERED and DEFAULT output side by side so
a caller can diff them, which is why ``types`` is a list rather than a flag.
"""

from enum import StrEnum
from typing import Annotated

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from neuronpedia_inference.schemas.common import BaseSchema, ExactSchema, NPLogprob


class NPSteerMethod(StrEnum):
    """How a steering vector is combined with the residual stream."""

    SIMPLE_ADDITIVE = "SIMPLE_ADDITIVE"
    ORTHOGONAL_DECOMP = "ORTHOGONAL_DECOMP"
    PROJECTION_CAP = "PROJECTION_CAP"


class NPSteerType(StrEnum):
    """Whether a given output was produced with steering applied or without it."""

    STEERED = "STEERED"
    DEFAULT = "DEFAULT"


class NPNormalize(StrEnum):
    """How an activation is scaled before a direction is projected onto it."""

    L2 = "l2"
    NONE = "none"


class NPCaptureSite(StrEnum):
    """Where in a layer an activation is read."""

    RESID_POST = "resid_post"


class NPTokenSelection(StrEnum):
    """Which of a conversation's messages a reading is reported for."""

    ASSISTANT_TURNS = "assistant_turns"
    ALL_TURNS = "all_turns"


class NPPooling(StrEnum):
    """How one message's token activations collapse to a single vector."""

    MEAN = "mean"
    LAST = "last"
    MAX = "max"


class NPReadSpec(ExactSchema):
    """How to get the activation a direction is projected onto.

    A fitted direction only means anything against activations gathered the way the fit gathered
    them, so these three travel with the vector rather than being this server's convention. Everything
    here was hard-coded until now, and the defaults are exactly what was hard-coded.

    Distinct from `render`, which is about the text: a render condition changes what the model is
    shown, so every read in one request has to agree about it. These change only how the captured
    activations are reduced, so two reads in one request may differ freely -- reading one layer two
    ways costs one forward, not two.
    """

    site: NPCaptureSite = Field(default=NPCaptureSite.RESID_POST, description="Where in the layer to read")
    tokens: NPTokenSelection = Field(
        default=NPTokenSelection.ASSISTANT_TURNS,
        description=(
            "Which messages get a reading. `assistant_turns` reports the model's own turns, which is "
            "what a persona fit is about; `all_turns` reports every message, including the user's."
        ),
    )
    pool: NPPooling = Field(
        default=NPPooling.MEAN,
        description=(
            "How a message's tokens collapse. `mean` is the whole turn's average, `last` its final "
            "token, `max` the per-dimension maximum."
        ),
    )


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


class NPRenderConditions(ExactSchema):
    """How a conversation has to be templated for a vector's numbers to mean anything.

    A projection onto a fitted direction only holds if the conversation reaches the model rendered
    the way it was during fitting, so these conditions travel with the vector. They are applied
    before generation and so change the text itself, which is why every read in one request has to
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


class NPVectorSource(ExactSchema):
    """A published vector to fetch, instead of sending its tensors inline.

    The artifact is one folder holding `vector.yaml` and `vector.safetensors`. Everything about the
    vector comes from there, so nothing but `id` may be sent beside this.
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


class NPVectorRead(ExactSchema):
    """One vector to read off the generated conversation, supplied with the request.

    Purely computational: a direction, where to read the activation it applies to, and the
    arithmetic that turns the projection into a number. Nothing here says what the number *means* --
    a trait, a class, a position between two named poles -- because that is the caller's to name,
    and this server has no catalogue to look it up in.

    Either send `source` on its own, or send `direction` and `layer` with as much of the rest as the
    fit has. With everything else defaulted the reading is the dot product of the direction with the
    mean residual-stream activation over each assistant turn, which is what you can read without a
    calibration corpus. The optional fields refine that in three steps::

        h          = activation at `layer`, gathered as `read` says (default: the mean
                     resid_post over each assistant turn)
        x          = h - preNormMean                    # default: no-op
        x          = x / max(||x||, 1e-12)              # only when normalize is "l2"
        raw        = dot(x - postNormMean, direction)
        value      = (raw - center) / (raw >= center ? scalePos : scaleNeg)
        percentile = interpolated against quantilesPos / quantilesNeg, bounded to [-1, 1]

    So `value` is in the fit's own units until `center` and the two scales put it on a readable
    one, and a `percentile` is reported only once the quantile tables say what the fitting corpus
    looked like. `value` is never clipped: a reading past 1 says the vector is being read off the
    distribution it was fitted on, which is a signal rather than something to pin to the boundary.

    **No labels.** There is no field here for an author, a title, a caveat or what either end means.
    A caller who holds the catalogue already has those beside the row this payload came from, and
    sending them here only to have them echoed back would make this server a courier for display
    text it cannot check. A `source` artifact is the exception, and states them in `vector.yaml`.
    """

    id: StrictStr = Field(
        description=(
            "What this reading is reported under, and unique across `reads` in one request. "
            "Conventionally `<author>_<name>`, though nothing here parses it."
        )
    )
    source: NPVectorSource | None = Field(
        default=None, description="Fetch the vector from a published artifact instead of sending it inline"
    )

    direction: list[StrictFloat] | None = Field(
        default=None,
        description="The fitted direction. Required without `source`, and must be the model's hidden size",
    )
    layer: StrictInt | None = Field(default=None, description="Layer to read at. Required without `source`")
    read: NPReadSpec | None = Field(
        default=None,
        description=(
            "How to gather the activation this direction is projected onto. Defaults to the mean over "
            "each assistant turn's `resid_post`, which is how every vector fitted so far was read."
        ),
    )

    normalize: NPNormalize = NPNormalize.NONE
    pre_norm_mean: list[StrictFloat] | None = Field(
        default=None, description="Subtracted from the activation before normalizing. Rarely needed"
    )
    post_norm_mean: list[StrictFloat] | None = Field(
        default=None,
        description="Subtracted after normalizing. Interchangeable with `preNormMean` when normalize is none",
    )

    center: StrictFloat = Field(default=0.0, description="Where the reading is zero, in raw projection units")
    scale_pos: StrictFloat = Field(default=1.0, description="Divisor above `center`. May not be zero")
    scale_neg: StrictFloat = Field(default=1.0, description="Divisor below `center`. May not be zero")

    quantiles_pos: list[StrictFloat] | None = Field(
        default=None,
        description=(
            "Distance from `center` at each level of the positive half of the fitting corpus, "
            "ascending. Sent with `quantilesNeg`, and the two are what a percentile is read off."
        ),
    )
    quantiles_neg: list[StrictFloat] | None = None
    quantile_levels: list[StrictFloat] | None = Field(
        default=None,
        description="Levels the two tables are sampled at. Defaults to evenly spaced 0 to 1",
    )

    render: NPRenderConditions = Field(default_factory=NPRenderConditions)

    @model_validator(mode="after")
    def check_one_shape(self) -> "NPVectorRead":
        """Either a source or an inline definition, and never a source with fields beside it.

        Refused rather than merged: a caller who sends both cannot be told which one produced the
        numbers they get back, and half of a published vector overridden by hand is not the vector
        anyone can go and look at.
        """
        if self.source is None:
            missing = [name for name in ("direction", "layer") if getattr(self, name) is None]
            if missing:
                raise ValueError(f"vector {self.id!r}: {missing} required without a `source`")
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
            raise ValueError(f"vector {self.id!r}: {overridden} cannot be sent with a `source`, which carries them")
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


class SteerReadoutTurn(BaseSchema):
    """One message's reading of a single vector, as a measurement and as a percentile.

    `value` is calibrated against the fit's own spread, so it passes 1 for roughly 2% of readings
    by construction and is never clipped -- how far past the calibration corpus a reading sits is
    what says a vector is being read off distribution. `percentile` is the same reading expressed
    as the share of that corpus it is past, which cannot leave [-1, 1]. Display the percentile
    and keep the value: a gauge reading "102%" looks broken, and a clipped value would delete
    the diagnostic. `percentile` is absent for a vector that ships no quantile tables.

    One entry per message the read's `tokens` selection picked, in conversation order.
    """

    value: StrictFloat | None = Field(default=None, description="The reading (pre-cap)")
    value_post_cap: StrictFloat | None = Field(
        default=None, description="The same reading measured under steering (post-cap)"
    )
    percentile: StrictFloat | None = Field(
        default=None,
        description=(
            "Where `value` falls in the fitting corpus, signed and bounded to [-1, 1]: 0.97 is "
            "further along than 97% of that corpus. Absent when the vector ships no quantile tables."
        ),
    )
    percentile_post_cap: StrictFloat | None = Field(
        default=None, description="`percentile` for this reading measured under steering (post-cap)"
    )
    snippet: StrictStr | None = Field(default=None, description="Truncated conversation content for this message")


class SteerVectorReadout(BaseSchema):
    """One vector read across a generated conversation, for one steer type.

    Keyed by `id` rather than by `title`: a title is a display string that may be reworded,
    and these readouts are persisted by callers.

    **The label fields say what the artifact said, and nothing more.** A vector sent inline carries
    no labels -- the request has no fields for them, because a caller who holds the catalogue can
    label a reading without a round trip through this server -- so `author` reads `custom`, `title`
    repeats the id, and the poles and caveat are absent. They are populated only for a vector
    fetched with `source`, where inference is the party that read `vector.yaml`.
    """

    id: StrictStr = Field(description="Vector id, as requested in `reads`")
    author: StrictStr = Field(description="Who fitted this vector, per its artifact. `custom` for anything sent inline")
    title: StrictStr = Field(description="Display label from the artifact, or the id for anything sent inline")
    type: NPSteerType | None = None
    layer: StrictInt | None = Field(default=None, description="Layer the vector was fitted and read at")
    caveat: StrictStr | None = Field(
        default=None, description="A known limitation stated by the artifact, worth showing beside its values"
    )
    pole_positive: StrictStr | None = Field(
        default=None,
        description=(
            'What a positive reading means, e.g. "toxic". Absent unless the artifact names both '
            "ends, which a probe or a steering vector need not"
        ),
    )
    pole_negative: StrictStr | None = Field(
        default=None, description='What a negative reading means, e.g. "respectful"'
    )
    pole_positive_description: StrictStr | None = None
    pole_negative_description: StrictStr | None = None
    source_revision: StrictStr | None = Field(
        default=None,
        description=(
            "The commit this vector was read from, for one fetched with `source`. What makes a "
            "reading reproducible: the same revision is the same vector, whatever the branch has "
            "moved on to since."
        ),
    )
    turns: list[SteerReadoutTurn] | None = None


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
    reads: list[NPVectorRead] | None = Field(
        default=None,
        description=(
            "Vectors to read off the generated conversation, sent inline or fetched from a "
            "published artifact. This server ships none of its own, so every vector a request "
            "wants read is named and defined here. Two entries sharing an id is a 400."
        ),
    )
    steer_special_tokens: StrictBool


class SteerCompletionChatResponse(BaseSchema):
    """
    The steering/default chat responses.
    """

    readouts: list[SteerVectorReadout] | None = Field(
        default=None,
        description="One entry per requested vector per steer type, in the order the vectors were sent",
    )
    outputs: list[NPSteerChatResult]
    input: NPSteerChatResult
