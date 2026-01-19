"""Core steerllm module - steering specs and protocols."""

from steerllm.core.specs import (
    AddSpec,
    ProjectionCapSpec,
    AblationSpec,
    LayerSteeringSpec,
    SteeringSpec,
    SteeringOp,
)
from steerllm.core.capture import CaptureHandle, MessageBoundary, ChatResponse
from steerllm.core.protocols import SteeringBackend, TrainableSteeringBackend
from steerllm.core.exceptions import (
    SteerLLMError,
    BackendError,
    CaptureError,
    ValidationError,
)

__all__ = [
    # Specs
    "AddSpec",
    "ProjectionCapSpec",
    "AblationSpec",
    "LayerSteeringSpec",
    "SteeringSpec",
    "SteeringOp",
    # Capture
    "CaptureHandle",
    "MessageBoundary",
    "ChatResponse",
    # Protocols
    "SteeringBackend",
    "TrainableSteeringBackend",
    # Exceptions
    "SteerLLMError",
    "BackendError",
    "CaptureError",
    "ValidationError",
]
