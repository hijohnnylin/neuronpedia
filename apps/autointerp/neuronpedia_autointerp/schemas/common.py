"""Types shared by more than one endpoint."""

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictStr
from pydantic.alias_generators import to_camel


class BaseSchema(BaseModel):
    """Base for every wire model in this package.

    The wire is camelCase while the python attributes stay snake_case, matching
    apps/inference; see the note in its schemas/common.py for why. ``serialize_by_alias``
    makes a bare ``model_dump()`` emit aliases, and ``populate_by_name`` keeps snake_case
    accepted on input, so requests predating the switch still validate.

    ``protected_namespaces=()`` is separately load-bearing: two request bodies carry a
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


class NPActivation(BaseSchema):
    """
    An activation record containing tokens and their corresponding activation values
    """

    tokens: list[StrictStr] = Field(description="List of tokens for this text")
    values: list[StrictFloat] = Field(description="Activation values corresponding to each token")
