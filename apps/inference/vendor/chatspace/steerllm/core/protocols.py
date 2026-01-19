"""Backend protocols for steerllm.

Defines the interface that all steering backends must implement.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, Sequence, runtime_checkable

import torch

from steerllm.core.capture import CaptureHandle
from steerllm.core.specs import SteeringSpec


@runtime_checkable
class SteeringBackend(Protocol):
    """Protocol for steering-enabled LLM backends.

    All backends must implement this interface. The protocol is async-first;
    sync wrappers are provided via mixin.
    """

    @property
    def hidden_size(self) -> int:
        """Model's hidden dimension size."""
        ...

    @property
    def layer_count(self) -> int:
        """Number of transformer layers."""
        ...

    @property
    def model_name(self) -> str:
        """Model identifier."""
        ...

    async def generate(
        self,
        prompts: list[str],
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        steering_spec: SteeringSpec | None = None,
        capture_layers: Sequence[int] | None = None,
        **sampling_kwargs: Any,
    ) -> tuple[list[str], list[CaptureHandle] | None]:
        """Generate text with optional steering and capture.

        Parameters
        ----------
        prompts :
            Input prompts for generation.
        max_tokens :
            Maximum tokens to generate per prompt.
        temperature :
            Sampling temperature.
        steering_spec :
            Optional steering configuration.
        capture_layers :
            Optional layer indices to capture activations from.
        **sampling_kwargs :
            Additional backend-specific sampling parameters.

        Returns
        -------
        tuple[list[str], list[CaptureHandle] | None]
            Generated texts and capture handles (if capture_layers provided).
        """
        ...

    async def chat(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        steering_spec: SteeringSpec | None = None,
        capture_layers: Sequence[int] | None = None,
        **sampling_kwargs: Any,
    ) -> tuple[list[str], list[CaptureHandle] | None]:
        """Chat-style generation with optional steering and capture.

        Parameters
        ----------
        messages :
            Single conversation (list of message dicts) or batch of
            conversations (list of lists).
        max_tokens :
            Maximum tokens to generate.
        temperature :
            Sampling temperature.
        steering_spec :
            Optional steering configuration.
        capture_layers :
            Optional layer indices to capture activations from.
        **sampling_kwargs :
            Additional backend-specific sampling parameters.

        Returns
        -------
        tuple[list[str], list[CaptureHandle] | None]
            Generated responses and capture handles (if capture_layers provided).
        """
        ...


@runtime_checkable
class TrainableSteeringBackend(SteeringBackend, Protocol):
    """Extended protocol for backends that support training.

    In addition to inference, trainable backends can:
    - Return trainable parameters for optimization
    - Set/get steering vectors directly
    - Save/load steering checkpoints
    """

    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return parameters that should receive gradients."""
        ...

    def set_steering_vector(self, layer: int, vector: torch.Tensor) -> None:
        """Set a steering vector at a layer."""
        ...

    def get_steering_vector(self, layer: int) -> torch.Tensor | None:
        """Get current steering vector at a layer."""
        ...

    def save_steering(self, path: str) -> None:
        """Save steering vectors to disk."""
        ...

    def load_steering(self, path: str) -> None:
        """Load steering vectors from disk."""
        ...


class SyncWrapperMixin:
    """Mixin providing sync wrappers for async methods.

    Backends can inherit from this to get generate_sync() and chat_sync().
    """

    def generate_sync(
        self,
        prompts: list[str],
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        steering_spec: SteeringSpec | None = None,
        capture_layers: Sequence[int] | None = None,
        **sampling_kwargs: Any,
    ) -> tuple[list[str], list[CaptureHandle] | None]:
        """Synchronous wrapper for generate().

        Creates a new event loop if needed. For use in non-async contexts.
        """
        return asyncio.run(
            self.generate(  # type: ignore
                prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                steering_spec=steering_spec,
                capture_layers=capture_layers,
                **sampling_kwargs,
            )
        )

    def chat_sync(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        steering_spec: SteeringSpec | None = None,
        capture_layers: Sequence[int] | None = None,
        **sampling_kwargs: Any,
    ) -> tuple[list[str], list[CaptureHandle] | None]:
        """Synchronous wrapper for chat().

        Creates a new event loop if needed. For use in non-async contexts.
        """
        return asyncio.run(
            self.chat(  # type: ignore
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                steering_spec=steering_spec,
                capture_layers=capture_layers,
                **sampling_kwargs,
            )
        )
