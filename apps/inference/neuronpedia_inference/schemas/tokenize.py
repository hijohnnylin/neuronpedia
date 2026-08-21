"""Wire models for ``/v1/tokenize``."""

from pydantic import Field, StrictBool, StrictInt, StrictStr

from neuronpedia_inference.schemas.common import BaseSchema


class TokenizeRequest(BaseSchema):
    """
    Text to split into tokens, plus how to treat the BOS token
    """

    model: StrictStr = Field(description="Model whose tokenizer should be used")
    text: StrictStr = Field(description="Text to split into tokens")
    prepend_bos: StrictBool | None = Field(
        default=None,
        description="Force a beginning-of-sequence token on or off. Omit to follow whatever the model normally does, which is what the other endpoints assume.",
    )


class TokenizeResponse(BaseSchema):
    """The token ids, their rendered text, and whether BOS was prepended."""

    tokens: list[StrictInt] = Field(description="One token id per token, in order")
    token_strings: list[StrictStr] = Field(
        description="The same tokens rendered back to text, index-aligned with `tokens`"
    )
    prepend_bos: StrictBool = Field(description="Whether a beginning-of-sequence token was actually prepended")
