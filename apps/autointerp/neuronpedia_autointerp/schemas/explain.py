"""Wire models for ``POST /v1/explain/default``."""

from pydantic import Field, StrictStr

from neuronpedia_autointerp.schemas.common import BaseSchema, NPActivation


class ExplainDefaultRequest(BaseSchema):
    """
    Request model for generating explanations of neuron/feature behavior
    """

    activations: list[NPActivation] = Field(description="List of activation records to analyze")
    openrouter_key: StrictStr = Field(description="API key for OpenRouter service")
    model: StrictStr = Field(description="Model identifier to use for explanation generation")


class ExplainDefaultResponse(BaseSchema):
    """
    A natural-language explanation of what the given activations have in common
    """

    explanation: StrictStr = Field(description="The generated explanation for the given set of activations")
