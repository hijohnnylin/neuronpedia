"""Backward-compatible steering wrapper around steerllm.VLLMSteeringModel.

This module provides the chatspace-compatible API that wraps steerllm's
VLLMSteeringModel. The wrapper preserves the original chatspace API including:

- VLLMSteeringConfig dataclass
- CaptureHandle with identical interface
- Static helper methods (simple_steering, simple_projection_cap, simple_ablation)
- Conditional return types (list vs tuple based on capture_layers)
- Environment variable compatibility (CHATSPACE_* and STEERLLM_*)

Typical usage::

    cfg = VLLMSteeringConfig(model_name="Qwen/Qwen3-0.6B")
    model = VLLMSteerModel(cfg, bootstrap_layers=(target_layer,))

    steering_spec = SteeringSpec(layers={
        target_layer: LayerSteeringSpec(operations=[
            AddSpec(vector=torch.randn(model.hidden_size), scale=1.0)
        ])
    })

    outputs, handles = await model.generate(
        ["...prompt..."],
        sampling_params,
        steering_spec=steering_spec,
        capture_layers=target_layer
    )
"""

from __future__ import annotations

import asyncio
import os
import warnings
import weakref
import logging
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from collections.abc import AsyncGenerator
from typing import Any, Literal, Sequence, overload

import torch
from vllm import SamplingParams

# Import the underlying steerllm model
from steerllm import VLLMSteeringModel as _SteerLLMModel

# Re-export specs and types from steerllm
from steerllm.core.specs import (
    AddSpec,
    ProjectionCapSpec,
    AblationSpec,
    LayerSteeringSpec,
    SteeringSpec,
    SteeringOp,
)
from steerllm.core.capture import (
    MessageBoundary,
    ChatResponse,
)


logger = logging.getLogger(__name__)


@dataclass
class VLLMSteeringConfig:
    """Configuration for vLLM-based steerable model."""

    model_name: str = "Qwen/Qwen3-32B"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    dtype: str = "auto"
    bootstrap_layers: tuple[int, ...] = ()


def _cleanup_capture_handle_and_warn(
    model_ref: weakref.ref,
    shm_names: list[str],
    accessed_container: list[bool],
) -> None:
    """Cleanup callback for CaptureHandle finalization."""
    accessed = accessed_container[0]

    if shm_names and not accessed:
        warnings.warn(
            f"CaptureHandle held {len(shm_names)} shared memory regions "
            f"but was never accessed! This wastes memory. "
            f"Use 'async with handle:' or call 'await handle.close()' explicitly.",
            ResourceWarning,
            stacklevel=2
        )

    model = model_ref()
    if model is not None and shm_names:
        try:
            logger.debug(f"Finalizer: {len(shm_names)} shm segments will be cleaned by TTL")
        except Exception as e:
            logger.debug(f"Finalizer cleanup note: {e}")


class CaptureHandle:
    """Handle for lazily fetching activation captures for a single request.

    This is the chatspace-compatible CaptureHandle that wraps steerllm's handle.

    Attributes
    ----------
    request_id : str
        Internal request identifier used for fetching captures.
    layer_indices : tuple[int, ...]
        Layer indices that were captured for this request.
    message_boundaries : tuple[MessageBoundary, ...] | None
        Optional message boundary information for chat-style captures.
    """

    def __init__(
        self,
        request_id: str,
        layer_indices: tuple[int, ...],
        model: "VLLMSteerModel",
        message_boundaries: tuple[MessageBoundary, ...] | None = None,
        _inner_handle: Any = None,
    ):
        self.request_id = request_id
        self.layer_indices = layer_indices
        self._model_ref = weakref.ref(model)
        self._captures: dict[int, list[dict[str, Any]]] | None = None
        self.message_boundaries = message_boundaries
        self._inner_handle = _inner_handle

        # Shared memory tracking (for compatibility)
        self._shm_names: list[str] = []
        self._shm_objects: list[SharedMemory] = []
        self._accessed = False
        self._closed = False

        self._accessed_container = [False]

        self._finalizer = weakref.finalize(
            self,
            _cleanup_capture_handle_and_warn,
            weakref.ref(model),
            self._shm_names,
            self._accessed_container,
        )

    async def __aenter__(self):
        """Enter async context manager."""
        return self

    async def __aexit__(self, *args):
        """Exit async context manager and release shared memory."""
        await self.close()

    async def close(self):
        """Explicitly release shared memory resources."""
        if self._closed:
            return

        self._closed = True

        # Close inner handle
        if self._inner_handle is not None:
            await self._inner_handle.close()

        # Detach finalizer
        self._finalizer.detach()

    async def fetch(self) -> dict[int, list[dict[str, Any]]]:
        """Fetch captures from workers (idempotent)."""
        if self._captures is None:
            if self._inner_handle is None:
                raise RuntimeError(
                    f"Cannot fetch captures for request {self.request_id}: "
                    "no inner handle available. This indicates the CaptureHandle "
                    "was created without valid capture data."
                )
            self._captures = await self._inner_handle.fetch()
        return self._captures

    @property
    def captures(self) -> dict[int, list[dict[str, Any]]]:
        """Get captures (must call fetch() first)."""
        if self._captures is None:
            raise RuntimeError(
                f"Captures not fetched yet for request {self.request_id}. "
                "Call: await handle.fetch()"
            )
        self._accessed = True
        self._accessed_container[0] = True
        return self._captures

    def get_message_activations(
        self,
        message_idx: int,
        layer_idx: int,
        *,
        include_generated: bool = False,
    ) -> torch.Tensor:
        """Get activations for a specific message from chat-style captures."""
        if self._captures is None:
            raise RuntimeError(
                f"Captures not fetched yet for request {self.request_id}. "
                "Call: await handle.fetch() or await model.fetch_captures_batch([handle])"
            )

        if self.message_boundaries is None:
            raise RuntimeError(
                "Message boundaries not available for this capture. "
                "This handle was not created from a chat() call with message tracking."
            )

        if message_idx < 0 or message_idx >= len(self.message_boundaries):
            raise ValueError(
                f"message_idx {message_idx} out of range [0, {len(self.message_boundaries)})"
            )

        if layer_idx not in self._captures:
            raise ValueError(
                f"layer_idx {layer_idx} not in captured layers: {list(self._captures.keys())}"
            )

        full_hidden = self._captures[layer_idx][0]["hidden"]

        boundary = self.message_boundaries[message_idx]
        start = boundary.start_token
        end = boundary.end_token

        if include_generated and message_idx == len(self.message_boundaries) - 1:
            end = full_hidden.shape[0]

        return full_hidden[start:end]


def compute_message_boundaries(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    chat_kwargs: dict[str, Any],
) -> tuple[MessageBoundary, ...]:
    """Compute token boundaries for each message in a conversation."""
    boundaries: list[MessageBoundary] = []
    current_offset = 0

    for i, msg in enumerate(messages):
        partial_conv = messages[:i + 1]

        partial_text = tokenizer.apply_chat_template(
            partial_conv,
            tokenize=False,
            **chat_kwargs,
        )

        partial_tokens = tokenizer(partial_text, return_tensors="pt")
        total_len = partial_tokens.input_ids.shape[1]

        boundary = MessageBoundary(
            role=msg["role"],
            content=msg["content"],
            start_token=current_offset,
            end_token=total_len,
        )
        boundaries.append(boundary)
        current_offset = total_len

    return tuple(boundaries)


class VLLMSteerModel:
    """Backward-compatible wrapper around steerllm.VLLMSteeringModel.

    This class preserves the original chatspace API while delegating to steerllm.
    """

    def __init__(
        self,
        cfg: VLLMSteeringConfig,
        *,
        bootstrap_layers: Sequence[int] | None = None,
        shm_ttl_seconds: int | None = None,
        shm_max_gb: float | None = None,
        decode_buffer_size: int | None = None,
        **vllm_kwargs,
    ) -> None:
        self.cfg = cfg

        # Support both CHATSPACE_* and STEERLLM_* env vars (CHATSPACE takes precedence)
        _shm_ttl = (
            shm_ttl_seconds if shm_ttl_seconds is not None
            else int(os.getenv("CHATSPACE_SHM_TTL", os.getenv("STEERLLM_SHM_TTL", "600")))
        )
        _shm_max_gb = (
            shm_max_gb if shm_max_gb is not None
            else float(os.getenv("CHATSPACE_MAX_SHM_GB", os.getenv("STEERLLM_MAX_SHM_GB", "128")))
        )
        _decode_buffer_size = (
            decode_buffer_size if decode_buffer_size is not None
            else int(os.getenv("CHATSPACE_DECODE_BUFFER_SIZE", os.getenv("STEERLLM_DECODE_BUFFER_SIZE", "128")))
        )

        # Determine bootstrap layers
        init_layers: tuple[int, ...]
        if bootstrap_layers is not None:
            init_layers = tuple(int(idx) for idx in bootstrap_layers)
        else:
            init_layers = tuple(int(idx) for idx in cfg.bootstrap_layers)

        # Create the underlying steerllm model
        self._inner = _SteerLLMModel(
            model_name=cfg.model_name,
            tensor_parallel_size=cfg.tensor_parallel_size,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_model_len=cfg.max_model_len,
            dtype=cfg.dtype,
            bootstrap_layers=init_layers,
            shm_ttl_seconds=_shm_ttl,
            shm_max_gb=_shm_max_gb,
            decode_buffer_size=_decode_buffer_size,
            **vllm_kwargs,
        )

    @property
    def hidden_size(self) -> int:
        """Model's hidden dimension."""
        return self._inner.hidden_size

    @property
    def layer_count(self) -> int:
        """Number of transformer layers."""
        return self._inner.layer_count

    @property
    def tokenizer(self):
        """Tokenizer instance."""
        return self._inner.tokenizer

    @property
    def llm(self):
        """Access the underlying AsyncLLMEngine (for tests)."""
        return self._inner._engine

    @llm.setter
    def llm(self, value):
        """Set the underlying AsyncLLMEngine (for test mocking)."""
        self._inner._engine = value

    # ------------------------------------------------------------------
    # Static helper methods (preserved for compatibility)
    # ------------------------------------------------------------------

    @staticmethod
    def simple_steering(layer: int, vector: torch.Tensor, scale: float = 1.0) -> SteeringSpec:
        """Create a simple additive steering spec for a single layer."""
        norm = float(vector.norm().item())
        unit = vector / norm if norm > 0 else vector
        return SteeringSpec(
            layers={layer: LayerSteeringSpec(operations=[AddSpec(vector=unit, scale=norm * scale)])}
        )

    @staticmethod
    def simple_projection_cap(
        layer: int, vector: torch.Tensor, min: float | None = None, max: float | None = None
    ) -> SteeringSpec:
        """Create a projection cap steering spec for a single layer."""
        if min is None and max is None:
            raise ValueError("Must specify at least one of min or max")
        norm = float(vector.norm().item())
        if norm == 0:
            raise ValueError("Projection cap vector must have nonzero norm")
        unit = vector / norm
        return SteeringSpec(
            layers={
                layer: LayerSteeringSpec(operations=[ProjectionCapSpec(vector=unit, min=min, max=max)])
            }
        )

    @staticmethod
    def simple_ablation(layer: int, vector: torch.Tensor, scale: float = 1.0) -> SteeringSpec:
        """Create an ablation steering spec for a single layer."""
        norm = float(vector.norm().item())
        if norm == 0:
            raise ValueError("Ablation vector must have nonzero norm")
        unit = vector / norm
        return SteeringSpec(
            layers={layer: LayerSteeringSpec(operations=[AblationSpec(vector=unit, scale=scale)])}
        )

    # ------------------------------------------------------------------
    # Main API methods
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompts: list[str] | str,
        sampling_params: SamplingParams | None = None,
        *,
        capture_layers: int | Sequence[int] | None = None,
        steering_spec: SteeringSpec | None = None,
        raw_output: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ) -> list[str] | tuple[list[str], list[CaptureHandle]] | list[Any] | tuple[list[Any], list[CaptureHandle]] | AsyncGenerator[str, None]:
        """Generate text with optional activation capture and per-request steering.

        Returns list[str] when capture_layers is None, or tuple[list[str], list[CaptureHandle]]
        when capture_layers is provided. When stream=True, returns an AsyncGenerator[str, None]
        that yields text deltas.
        """
        # Handle streaming mode
        if stream:
            if capture_layers is not None:
                raise ValueError("Streaming mode does not support capture_layers")
            if raw_output:
                raise ValueError("Streaming mode does not support raw_output")
            # For streaming, only single prompt is supported
            prompt = prompts if isinstance(prompts, str) else prompts[0]
            return self._inner.generate_stream(
                prompt,
                sampling_params=sampling_params,
                steering_spec=steering_spec,
                **kwargs,
            )

        if isinstance(prompts, str):
            prompts = [prompts]

        # Convert capture_layers to sequence
        layers_seq: Sequence[int] | None = None
        if capture_layers is not None:
            if isinstance(capture_layers, int):
                layers_seq = [capture_layers]
            else:
                layers_seq = list(capture_layers)

        # Call inner model - return type depends on capture_layers
        if capture_layers is None:
            results = await self._inner.generate(
                prompts,
                sampling_params=sampling_params,
                steering_spec=steering_spec,
                capture_layers=None,
                raw_output=raw_output,
                **kwargs,
            )
            return results
        else:
            results, inner_handles = await self._inner.generate(
                prompts,
                sampling_params=sampling_params,
                steering_spec=steering_spec,
                capture_layers=layers_seq,
                raw_output=raw_output,
                **kwargs,
            )
            # Wrap handles in chatspace CaptureHandle
            handles = []
            if inner_handles:
                for ih in inner_handles:
                    handle = CaptureHandle(
                        request_id=ih.request_id,
                        layer_indices=ih.layer_indices,
                        model=self,
                        message_boundaries=ih.message_boundaries,
                        _inner_handle=ih,
                    )
                    handles.append(handle)
            return results, handles

    @overload
    async def chat(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        sampling_params: SamplingParams | None = None,
        *,
        chat_options: dict[str, Any] | None = None,
        capture_layers: None = None,
        steering_spec: SteeringSpec | None = None,
        raw_output: Literal[False] = False,
        **sampling_kwargs: Any,
    ) -> list[ChatResponse]: ...

    @overload
    async def chat(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        sampling_params: SamplingParams | None = None,
        *,
        chat_options: dict[str, Any] | None = None,
        capture_layers: None = None,
        steering_spec: SteeringSpec | None = None,
        raw_output: Literal[True] = True,
        **sampling_kwargs: Any,
    ) -> list[Any]: ...

    @overload
    async def chat(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        sampling_params: SamplingParams | None = None,
        *,
        chat_options: dict[str, Any] | None = None,
        capture_layers: int | Sequence[int],
        steering_spec: SteeringSpec | None = None,
        raw_output: Literal[False] = False,
        **sampling_kwargs: Any,
    ) -> tuple[list[ChatResponse], list[CaptureHandle]]: ...

    @overload
    async def chat(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        sampling_params: SamplingParams | None = None,
        *,
        chat_options: dict[str, Any] | None = None,
        capture_layers: int | Sequence[int],
        steering_spec: SteeringSpec | None = None,
        raw_output: Literal[True] = True,
        **sampling_kwargs: Any,
    ) -> tuple[list[Any], list[CaptureHandle]]: ...

    async def chat(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        sampling_params: SamplingParams | None = None,
        *,
        chat_options: dict[str, Any] | None = None,
        capture_layers: int | Sequence[int] | None = None,
        steering_spec: SteeringSpec | None = None,
        raw_output: bool = False,
        **sampling_kwargs: Any,
    ) -> list[ChatResponse] | list[Any] | tuple[list[ChatResponse], list[CaptureHandle]] | tuple[list[Any], list[CaptureHandle]]:
        """Execute chat-style generation with optional steering and activation capture."""
        # Convert capture_layers to sequence
        layers_seq: Sequence[int] | None = None
        if capture_layers is not None:
            if isinstance(capture_layers, int):
                layers_seq = [capture_layers]
            else:
                layers_seq = list(capture_layers)

        # Call inner model - return type depends on capture_layers
        if capture_layers is None:
            results = await self._inner.chat(
                messages,
                sampling_params=sampling_params,
                steering_spec=steering_spec,
                capture_layers=None,
                chat_options=chat_options,
                raw_output=raw_output,
                **sampling_kwargs,
            )
            return results
        else:
            results, inner_handles = await self._inner.chat(
                messages,
                sampling_params=sampling_params,
                steering_spec=steering_spec,
                capture_layers=layers_seq,
                chat_options=chat_options,
                raw_output=raw_output,
                **sampling_kwargs,
            )
            # Wrap handles in chatspace CaptureHandle
            handles = []
            if inner_handles:
                for ih in inner_handles:
                    handle = CaptureHandle(
                        request_id=ih.request_id,
                        layer_indices=ih.layer_indices,
                        model=self,
                        message_boundaries=ih.message_boundaries,
                        _inner_handle=ih,
                    )
                    handles.append(handle)
            return results, handles

    async def fetch_captures_batch(
        self,
        handles: Sequence["CaptureHandle"],
    ) -> None:
        """Fetch captures for multiple handles in a single RPC call.

        This is more efficient than calling handle.fetch() individually when
        fetching many captures, as it batches the GPU→shm transfers and performs
        a single stream synchronization.

        Parameters
        ----------
        handles :
            Sequence of CaptureHandle objects to fetch captures for.

        Note
        ----
        Mutates handles in-place by populating their _captures field.
        Handles that already have captures are skipped.
        """
        # Filter handles that need fetching and have inner handles
        to_fetch = [h for h in handles if h._captures is None and h._inner_handle is not None]
        if not to_fetch:
            return

        # Extract inner handles and call steerllm batch fetch
        inner_handles = [h._inner_handle for h in to_fetch]
        await self._inner.fetch_captures_batch(inner_handles)

        # Copy captures from inner handles to wrapper handles
        for handle in to_fetch:
            if handle._inner_handle._captures is not None:
                handle._captures = handle._inner_handle._captures

    # ------------------------------------------------------------------
    # Sync wrappers (deprecated)
    # ------------------------------------------------------------------

    def generate_sync(self, *args, **kwargs) -> list[str]:
        """Synchronous wrapper for generate(). DEPRECATED."""
        warnings.warn(
            "generate_sync() is deprecated. Use async generate() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return asyncio.run(self.generate(*args, **kwargs))

    def chat_sync(self, *args, **kwargs) -> list[str]:
        """Synchronous wrapper for chat(). DEPRECATED."""
        warnings.warn(
            "chat_sync() is deprecated. Use async chat() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return asyncio.run(self.chat(*args, **kwargs))
