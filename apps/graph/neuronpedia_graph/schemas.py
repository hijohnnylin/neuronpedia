"""The wire format for this server.

These models are the source of truth: FastAPI derives ``openapi.json`` from them, and the
TypeScript types the webapp compiles against are generated from that. Before they existed, the
request models lived here but were validated by hand out of ``await req.json()`` -- so FastAPI
never saw them and none of it reached a spec -- and every response was a hand-built dict
mirrored by hand in ``apps/webapp/lib/utils/graph.ts``.

**Field names stay snake_case.** Unlike ``apps/inference`` and ``apps/autointerp``, which alias
to camelCase, this server's shapes are public in three directions: ``/api/graph/tokenize`` and
``/api/steer-logits`` forward responses nearly verbatim with snake_case in their swagger, and
``/api/steer-logits`` publishes :class:`SteerFeature`'s own field names because it forwards
``features`` untouched. See the note in the repo's AGENTS.md on when aliasing is safe.

Deliberately **not** modelled here: the graph JSON that ``/generate-graph`` uploads to S3. Its
keys come from ``circuit_tracer``'s ``build_model`` and are pinned by the published
``graph-schema.json``, the public ``/graph/validator`` page and rows already in the bucket.
"""

import os

from pydantic import BaseModel, ConfigDict, Field

# Lives here rather than in server.py because it is a field default and server.py imports this
# module, not the other way round. Nothing else reads it.
DEFAULT_MAX_FEATURE_NODES = int(os.getenv("MAX_FEATURE_NODES", "10000"))


class GraphSchema(BaseModel):
    """Base for every wire model here.

    Deliberately has no ``alias_generator``: see the module docstring.
    """

    model_config = ConfigDict(populate_by_name=True, validate_assignment=True, protected_namespaces=())


class GraphChatMessage(GraphSchema):
    """One structured chat turn.

    Roles come back canonical rather than as the template's literal label, so a gemma
    ``model`` turn is reported as ``assistant``.
    """

    role: str
    content: str


# ------------------------------------------------------------------------------- requests --


class GraphGenerationRequest(GraphSchema):
    prompt: str = ""
    # Structured chat turns. When provided, the server renders the prompt string
    # via the model's real chat template (frontend does no chat-template
    # special-casing). The stored graph prompt stays a plain string.
    messages: list[GraphChatMessage] | None = None
    model_id: str
    batch_size: int = 48
    max_n_logits: int = 10
    desired_logit_prob: float = 0.95
    node_threshold: float = 0.8
    edge_threshold: float = 0.98
    slug_identifier: str
    max_feature_nodes: int = DEFAULT_MAX_FEATURE_NODES
    signed_url: str | None = None
    user_id: str | None = None
    compress: bool = False
    enable_qk_tracing: bool = False
    qk_top_fraction: float = 0.6
    qk_topk: int = 10


class ForwardPassRequest(GraphSchema):
    prompt: str = ""
    messages: list[GraphChatMessage] | None = None
    max_n_logits: int = 10
    desired_logit_prob: float = 0.95


class SteerFeature(GraphSchema):
    """One feature intervention.

    These field names are public: ``/api/steer-logits`` documents them in its swagger and
    forwards the array here untouched, so renaming one is an API break.
    """

    layer: int
    index: int
    token_active_position: int
    steer_position: int | None = None
    steer_generated_tokens: bool = False
    delta: float | None = None
    ablate: bool = False


class SteerRequest(GraphSchema):
    model_id: str
    prompt: str = ""
    messages: list[GraphChatMessage] | None = None
    features: list[SteerFeature]
    n_tokens: int = 10
    top_k: int = 5
    temperature: float = 0.0
    freq_penalty: float = 0
    seed: int | None = None
    freeze_attention: bool = False


class ParseChatPromptRequest(GraphSchema):
    prompt: str


# ------------------------------------------------------------------------------ responses --


class CheckBusyResponse(GraphSchema):
    busy: bool


class ParseChatPromptResponse(GraphSchema):
    # None when the model has no chat template, or the prompt is plain text with no
    # recognizable turn headers. Callers should then treat it as a raw prompt.
    messages: list[GraphChatMessage] | None
    is_chat: bool
    # The request's prompt with any leading BOS removed, per the model's own tokenizer. The
    # non-chat case has no `messages` to carry that strip, so this is what a caller seeding a
    # prompt editor should use.
    prompt: str
    # Token positions `/steer` silently drops features at, as indices into the tokenization of
    # `prompt`. A steer UI hides its controls at these rather than matching token strings, so
    # what it hides cannot drift from what the server refuses.
    unsteerable_positions: list[int]


class SalientLogit(GraphSchema):
    token: str
    token_id: int
    probability: float


class ForwardPassResponse(GraphSchema):
    """The salient-logit payload.

    A failed pass returns ``{"error": ...}`` with a 200 instead of this, which the spec does
    not describe -- callers check for ``error`` before reading any of these fields. Keeping
    that shape out of the model is what lets the five fields below stay required.
    """

    prompt: str
    input_tokens: list[str]
    salient_logits: list[SalientLogit]
    total_salient_tokens: int
    cumulative_probability: float


class TopLogit(GraphSchema):
    token: str
    prob: float


class LogitsByToken(GraphSchema):
    token: str
    top_logits: list[TopLogit]


class SteerResponse(GraphSchema):
    """Default and steered generations, plus per-token logits for each.

    The SCREAMING_SNAKE keys are the wire names, so they are spelled out as aliases rather than
    left to the field names: ``/api/steer-logits`` returns this object nearly verbatim and its
    swagger documents them in this form.
    """

    default_generation: str = Field(alias="DEFAULT_GENERATION")
    steered_generation: str = Field(alias="STEERED_GENERATION")
    default_logits_by_token: list[LogitsByToken] = Field(alias="DEFAULT_LOGITS_BY_TOKEN")
    steered_logits_by_token: list[LogitsByToken] = Field(alias="STEERED_LOGITS_BY_TOKEN")

    # `serialize_by_alias` is what puts the SCREAMING_SNAKE names on the wire; without it the
    # aliases would only be accepted on input and the response would come back snake_case.
    model_config = ConfigDict(serialize_by_alias=True)


class GraphGenerationResponse(GraphSchema):
    """The receipt from an upload.

    Only describes the ``signed_url`` path, which is the only one the webapp uses. Called
    without a ``signed_url`` the endpoint returns the ``circuit_tracer`` graph model itself,
    whose schema is not ours to publish (see the module docstring).
    """

    success: str | None = None
    error: str | None = None
