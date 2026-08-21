"""Wire models for ``POST /v1/score/embedding`` and ``POST /v1/score/fuzz-detection``."""

from enum import Enum

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr

from neuronpedia_autointerp.schemas.common import BaseSchema, NPActivation


class ScoreFuzzDetectionType(str, Enum):
    """
    Type of scoring method, either fuzz or detection
    """

    FUZZ = "FUZZ"
    DETECTION = "DETECTION"


class ScoreEmbeddingRequest(BaseSchema):
    """
    Request model for scoring explanations using embedding similarity
    """

    activations: list[NPActivation] = Field(description="List of activation records to analyze")
    explanation: StrictStr = Field(description="The explanation to evaluate")


class ScoreEmbeddingBreakdownItem(BaseSchema):
    """
    One scored example from an embedding run. With exception of fixing similarity to change to number instead of array of number, type is copied from https://github.com/EleutherAI/sae-auto-interp/blob/3659ff3bfefbe2628d37484e5bcc0087a5b10a27/sae_auto_interp/scorers/embedding/embedding.py#L20
    """

    text: StrictStr = Field(description="The text that was used to evaluate the similarity")
    distance: StrictInt = Field(description="Quantile or neighbor distance")
    similarity: StrictFloat = Field(description="What is the similarity of the example to the explanation")


class ScoreEmbeddingResponse(BaseSchema):
    """
    An embedding-similarity score, with the per-example results it was computed from
    """

    score: StrictFloat = Field(description="The score from 0 to 1")
    breakdown: list[ScoreEmbeddingBreakdownItem] = Field(description="Detailed breakdown of the embedding outputs")


class ScoreFuzzDetectionRequest(BaseSchema):
    """
    Request model for scoring explanations using fuzzing or detection methods
    """

    activations: list[NPActivation] = Field(description="List of activation records to analyze")
    explanation: StrictStr = Field(description="The explanation to evaluate")
    openrouter_key: StrictStr = Field(description="API key for OpenRouter service")
    model: StrictStr = Field(description="Model identifier to use for scoring")
    type: ScoreFuzzDetectionType


class ScoreFuzzDetectionBreakdownItem(BaseSchema):
    """
    One scored example from a fuzz or detection run. Type copied from https://github.com/EleutherAI/sae-auto-interp/blob/3659ff3bfefbe2628d37484e5bcc0087a5b10a27/sae_auto_interp/scorers/classifier/sample.py#L19
    """

    str_tokens: list[StrictStr] | None = Field(default=None, description="List of strings")
    activations: list[StrictFloat] | None = Field(default=None, description="List of floats")
    distance: StrictInt | None = Field(default=None, description="Quantile or neighbor distance")
    ground_truth: StrictBool | None = Field(default=None, description="Whether the example is activating or not")
    prediction: StrictBool | None = Field(
        default=False, description="Whether the model predicted the example activating or not"
    )
    highlighted: StrictBool | None = Field(default=False, description="Whether the sample is highlighted")
    probability: StrictFloat | None = Field(default=0.0, description="The probability of the example activating")
    correct: StrictBool | None = Field(default=False, description="Whether the prediction is correct")


class ScoreFuzzDetectionResponse(BaseSchema):
    """
    A fuzz or detection score, with the per-example results it was computed from
    """

    score: StrictFloat = Field(description="The score from 0 to 1")
    breakdown: list[ScoreFuzzDetectionBreakdownItem]
