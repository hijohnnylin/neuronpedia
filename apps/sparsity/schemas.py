"""The wire format for this server.

These models are the source of truth: FastAPI derives ``openapi.json`` from them, and the
TypeScript types the webapp compiles against are generated from that. Before they existed the
same shapes were hand-copied into two webapp files that agreed with this server only by
convention, so a rename here compiled fine and broke at runtime.

**Field names stay snake_case.** Unlike ``apps/inference`` and ``apps/autointerp``, which alias
to camelCase, this server's responses are forwarded nearly verbatim by the documented public
route ``/api/sparsity/connected-neurons`` -- so these names are already a public contract.
See the note in the repo's AGENTS.md on when aliasing is safe.
"""

from pydantic import BaseModel, ConfigDict, Field


class SparsitySchema(BaseModel):
    """Base for every wire model here.

    Deliberately has no ``alias_generator``: see the module docstring. ``populate_by_name`` and
    ``validate_assignment`` match the other apps' bases so the only difference is the casing.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        # `HealthResponse.model` collides with pydantic's `model_` namespace without this.
        protected_namespaces=(),
    )


class TraceNode(SparsitySchema):
    """One hop in a circuit trace: a neuron reached through a residual channel.

    ``children`` and ``parents`` are the same relationship in opposite directions -- a node in
    ``trace_forward`` carries ``children``, one in ``trace_backward`` carries ``parents``, and
    never both. They are modelled on one type because every consumer walks the two traces with
    the same code.
    """

    layer: int
    neuron: int
    read_weight: float = Field(description="Weight with which this neuron reads the channel")
    via_channel: int = Field(description="Residual channel connecting this neuron to the previous hop")
    write_weight: float = Field(description="Weight with which the previous hop writes the channel")
    children: list["TraceNode"] | None = None
    parents: list["TraceNode"] | None = None


class ChannelNeuron(SparsitySchema):
    """A neuron reading from or writing to a residual channel, with its weight."""

    neuron_id: int
    weight: float


class HealthResponse(SparsitySchema):
    """Liveness plus the dimensions needed to bounds-check a request."""

    status: str
    model: str
    num_layers: int
    mlp_size: int
    d_model: int


class NeuronConnectionsResponse(SparsitySchema):
    """Circuit traces downstream and upstream of one MLP neuron."""

    layer: int
    neuron: int
    trace_forward: list[TraceNode]
    trace_backward: list[TraceNode]


class ChannelConnectionsResponse(SparsitySchema):
    """Every neuron touching one residual channel, grouped by layer.

    The ``*_by_layer`` keys are layer indices. They are ints in python but JSON object keys are
    always strings, so they arrive as ``"0"``, ``"1"`` and so on.
    """

    channel_id: int
    readers_by_layer: dict[int, list[ChannelNeuron]]
    writers_by_layer: dict[int, list[ChannelNeuron]]
