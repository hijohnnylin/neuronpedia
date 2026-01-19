"""Text generation utilities with steering vector support."""

from .compat import LegacyExperiment, load_legacy_role_trait_config
from .config import GenerationConfig
from .vllm_steer_model import (
    AddSpec,
    AblationSpec,
    CaptureHandle,
    ChatResponse,
    LayerSteeringSpec,
    MessageBoundary,
    ProjectionCapSpec,
    SteeringOp,
    SteeringSpec,
    VLLMSteerModel,
    VLLMSteeringConfig,
    compute_message_boundaries,
)

__all__ = [
    "GenerationConfig",
    "VLLMSteerModel",
    "VLLMSteeringConfig",
    "AddSpec",
    "LayerSteeringSpec",
    "ProjectionCapSpec",
    "AblationSpec",
    "SteeringOp",
    "SteeringSpec",
    "CaptureHandle",
    "ChatResponse",
    "MessageBoundary",
    "compute_message_boundaries",
    "LegacyExperiment",
    "load_legacy_role_trait_config",
]
