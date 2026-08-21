"""The wire format for this server.

These models are the source of truth: FastAPI derives ``/openapi.json`` from them, and
everything downstream -- the committed ``openapi.json``, the TypeScript types the webapp
compiles against, the published client SDKs -- is generated outward from that. So a
change to a request or response shape is written here and nowhere else.

Class names become the schema keys in ``openapi.json`` and therefore the type names in
the published SDKs, so they mirror the URL path they belong to: ``POST /v1/score/fuzz-detection``
carries ``ScoreFuzzDetectionRequest`` and ``ScoreFuzzDetectionResponse``. Renaming one is
a breaking change for SDK consumers.
"""

from neuronpedia_autointerp.schemas.common import BaseSchema, NPActivation
from neuronpedia_autointerp.schemas.explain import (
    ExplainDefaultRequest,
    ExplainDefaultResponse,
)
from neuronpedia_autointerp.schemas.score import (
    ScoreEmbeddingBreakdownItem,
    ScoreEmbeddingRequest,
    ScoreEmbeddingResponse,
    ScoreFuzzDetectionBreakdownItem,
    ScoreFuzzDetectionRequest,
    ScoreFuzzDetectionResponse,
    ScoreFuzzDetectionType,
)

__all__ = [
    "BaseSchema",
    "ExplainDefaultRequest",
    "ExplainDefaultResponse",
    "NPActivation",
    "ScoreEmbeddingBreakdownItem",
    "ScoreEmbeddingRequest",
    "ScoreEmbeddingResponse",
    "ScoreFuzzDetectionBreakdownItem",
    "ScoreFuzzDetectionRequest",
    "ScoreFuzzDetectionResponse",
    "ScoreFuzzDetectionType",
]
