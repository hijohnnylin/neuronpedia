"""Wire models for ``/v1/util/*`` -- lookups that need the loaded SAEs but no generation."""

from pydantic import Field, StrictFloat, StrictInt, StrictStr

from neuronpedia_inference.schemas.common import BaseSchema, NPFeature


class UtilSaeVectorRequest(BaseSchema):
    """
    Get the raw vector for an SAE feature
    """

    model: StrictStr
    source: StrictStr
    index: StrictInt


class UtilSaeVectorResponse(BaseSchema):
    """The decoder vector for one feature, and the hook it belongs to."""

    vector: list[StrictFloat]
    # camelCase on the wire, unlike every other field on this endpoint. Kept as-is because
    # renaming it would break existing clients for no gain.
    hook_name: StrictStr = Field(alias="hookName")


class UtilSaeTopkByDecoderCossimRequest(BaseSchema):
    """Find the features whose decoder directions are closest to a feature or vector."""

    feature: NPFeature | None = None
    vector: list[StrictFloat] | None = Field(
        default=None, description="Custom vector to find the top features by cossim for."
    )
    model: StrictStr = Field(description="Model to compare the vector or feature against.")
    source: StrictStr = Field(description="Source/SAE ID to compare the vector or feature against.")
    num_results: StrictInt = Field(
        description="Number of top features to return. Clamped to 128, and to the number of features in the source."
    )


class UtilSaeTopkByDecoderCossimFeature(BaseSchema):
    """One nearby feature and how close it is."""

    feature: NPFeature | None = None
    cosine_similarity: StrictFloat | None = None


class UtilSaeTopkByDecoderCossimResponse(BaseSchema):
    """The query feature echoed back, alongside its nearest neighbours."""

    feature: NPFeature | None = None
    topk_decoder_cossim_features: list[UtilSaeTopkByDecoderCossimFeature] | None = None


# Docstrings on these classes become the schema `description` in the published spec, so they
# describe the endpoint to a caller. Notes to ourselves go in comments like this one: this
# request keeps its camelCase field names because they are the endpoint's existing wire
# format, having never gone through the generator's snake_case convention.
class SimilarityMatrixRequest(BaseSchema):
    """Predicted similarity matrix for one feature over a piece of text."""

    modelId: str  # noqa: N815
    sourceId: str  # noqa: N815
    index: int
    text: str


class SimilarityMatrixResponse(BaseSchema):
    """Token-by-token cosine similarities, with the BOS token dropped from both axes."""

    similarity_matrix: list[list[float]]
    tokens: list[str]
