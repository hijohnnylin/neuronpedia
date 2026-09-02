"""Vector reads: asset loading, backend-agnostic capture, and projection."""

from .snippets import truncate_content
from .vector_data import (
    CaptureKey,
    Pooling,
    ReadSpec,
    RenderConditions,
    TokenSelection,
    VectorAsset,
    load_vector,
    project_vector,
    project_vector_with_percentile,
)
from .vector_request import (
    VectorRequestError,
    asset_from_payload,
    resolve_request_reads,
)

__all__ = [
    "CaptureKey",
    "Pooling",
    "ReadSpec",
    "RenderConditions",
    "TokenSelection",
    "VectorAsset",
    "VectorRequestError",
    "asset_from_payload",
    "load_vector",
    "project_vector",
    "project_vector_with_percentile",
    "resolve_request_reads",
    "truncate_content",
]
