"""steerllm - Multi-backend LLM steering library.

A clean API for applying steering vectors, projection caps, and ablations
to large language models during inference. Supports multiple backends
(vLLM, HuggingFace) with a unified interface.

Example
-------
Basic steering with vLLM::

    from steerllm import VLLMSteeringModel, SteeringSpec

    model = VLLMSteeringModel("Qwen/Qwen3-0.6B")
    steering = SteeringSpec.simple_add(layer=5, vector=direction, scale=2.0)

    texts, _ = await model.generate(
        ["What is consciousness?"],
        max_tokens=100,
        steering_spec=steering,
    )

Activation capture::

    texts, handles = await model.generate(
        ["The meaning of life is"],
        capture_layers=[5, 10, 15],
    )

    async with handles[0] as handle:
        await handle.fetch()
        layer_5 = handle.captures[5][0]["hidden"]

Training with HuggingFace::

    from steerllm.backends.huggingface import HFSteeringModel

    model = HFSteeringModel("Qwen/Qwen3-0.6B", target_layers=[5])
    optimizer = Adam(model.get_trainable_parameters(), lr=1e-4)

    for batch in dataloader:
        loss = compute_loss(model(**batch))
        loss.backward()
        optimizer.step()
"""

__version__ = "0.1.0"

# Core specs - always available
from steerllm.core.specs import (
    AddSpec,
    ProjectionCapSpec,
    AblationSpec,
    LayerSteeringSpec,
    SteeringSpec,
    SteeringOp,
)

# Capture types - always available
from steerllm.core.capture import (
    CaptureHandle,
    MessageBoundary,
    ChatResponse,
)

# Protocols - always available
from steerllm.core.protocols import (
    SteeringBackend,
    TrainableSteeringBackend,
    SyncWrapperMixin,
)

# Exceptions - always available
from steerllm.core.exceptions import (
    SteerLLMError,
    BackendError,
    CaptureError,
    ValidationError,
)

__all__ = [
    # Version
    "__version__",
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
    "SyncWrapperMixin",
    # Exceptions
    "SteerLLMError",
    "BackendError",
    "CaptureError",
    "ValidationError",
    # Backends (lazy-loaded)
    "VLLMSteeringModel",
    "HFSteeringModel",
]


# Lazy imports for optional backends
def __getattr__(name: str):
    if name == "VLLMSteeringModel":
        from steerllm.backends.vllm import VLLMSteeringModel
        return VLLMSteeringModel
    if name == "HFSteeringModel":
        from steerllm.backends.huggingface import HFSteeringModel
        return HFSteeringModel
    raise AttributeError(f"module 'steerllm' has no attribute {name!r}")
