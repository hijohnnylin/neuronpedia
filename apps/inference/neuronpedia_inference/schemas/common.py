"""Types shared by more than one endpoint."""

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr
from pydantic.alias_generators import to_camel


class BaseSchema(BaseModel):
    """Base for every wire model in this package.

    The wire is camelCase while the python attributes stay snake_case: this server's only
    caller is the TypeScript webapp, and camelCase there matches the casing of the public
    ``/api`` surface it re-exports into. ``serialize_by_alias`` is what makes that safe --
    without it a bare ``model_dump()`` silently emits snake_case, and several handlers dump
    that way. ``populate_by_name`` keeps snake_case accepted on input, so requests predating
    the switch still validate.

    ``protected_namespaces=()`` is separately load-bearing: most request bodies carry a
    field literally named ``model``, which collides with pydantic's ``model_`` namespace
    and warns without it.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
        validate_assignment=True,
        protected_namespaces=(),
    )

    def to_wire_json(self) -> str:
        """Serialize the way the generated client's ``to_json`` did.

        ``exclude_none`` is the load-bearing flag, not a style choice: an optional field
        that was never set is omitted rather than sent as ``null``. The steer endpoints
        rely on it, where a caller predating ``logprobs`` expects the key to be absent.
        """
        return self.model_dump_json(exclude_none=True)


class ExactSchema(BaseSchema):
    """Base for a payload where a field this server does not know is an error.

    Tolerating unknown fields is the right default and stays the default everywhere else: it
    is what lets an older server keep serving a newer client, and a response grow a field
    without every reader being redeployed first.

    It is the wrong default for a payload that is somebody's stored blob forwarded verbatim.
    A vector read is exactly that -- the caller holds a table of fitting parameters and hands
    them over -- and every field in one changes the number that comes back. So an unrecognized
    key there is a parameter that was meant to apply and did not, and ignoring it answers with
    a plausible reading measured some other way, which nothing downstream can tell from a
    correct one. Refusing the request names the key instead.

    Both spellings stay accepted, since ``populate_by_name`` makes the snake_case attribute a
    valid input name beside its camelCase alias. What is refused is a name that is neither.

    Only for payloads whose fields are a closed set of parameters. Not for a request envelope
    or a response, where refusing an unknown field would make a version skew an outage.
    """

    model_config = ConfigDict(extra="forbid")


class PublicFrameSchema(BaseSchema):
    """Base for a payload whose field names are themselves a public contract.

    The aliasing above assumes the webapp is free to rename on the way out, which holds for
    anything it reads and reshapes. It does not hold for a payload the webapp forwards
    verbatim -- the lens NDJSON frames go straight into ``/api/lens/prompt``'s response and
    into the stored share blobs, both snake_case and both with existing readers.

    Aliasing those and renaming them back at each consumer is what this avoids. That
    arrangement cost a bug once already: a second consumer was added, did not translate, and
    nothing failed to compile because both sides were hand-written types that agreed with
    each other. Naming the fields here the way they are consumed leaves nothing to keep in
    sync.

    Everything else -- ``populate_by_name``, ``validate_assignment``, ``protected_namespaces``
    -- is inherited; only the alias generator is dropped.
    """

    model_config = ConfigDict(alias_generator=None)


class HealthResponse(BaseSchema):
    """Liveness only -- says the process is up, not that a model finished loading."""

    status: StrictStr


class NPFeature(BaseSchema):
    """
    A feature in Neuronpedia, identified by model, source, and index.
    """

    model: StrictStr
    source: StrictStr
    index: StrictInt


class NPLogprobTop(BaseSchema):
    """
    One token the model considered at a single position
    """

    token: StrictStr = Field(description="The candidate token, as text")
    logprob: StrictFloat = Field(description="Natural log of the probability the model gave this candidate")


class NPLogprob(BaseSchema):
    """
    What the model was weighing at one position of the output
    """

    token: StrictStr = Field(description="The token that was actually emitted, as text")
    logprob: StrictFloat = Field(description="Natural log of the probability of the emitted token")
    top_logprobs: list[NPLogprobTop] = Field(
        description="The highest-scoring candidates at this position, most likely first"
    )
