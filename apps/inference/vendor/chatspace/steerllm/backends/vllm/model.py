"""vLLM steering model implementation.

This module provides the VLLMSteeringModel class which wraps vLLM's AsyncLLMEngine
with steering vector injection and activation capture capabilities.

This is a standalone implementation that does not depend on chatspace.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
import weakref
from multiprocessing.shared_memory import SharedMemory
from collections.abc import AsyncGenerator
from typing import Any, Sequence

import torch

from steerllm.core.capture import CaptureHandle, ChatResponse, MessageBoundary
from steerllm.core.exceptions import BackendError
from steerllm.core.protocols import SyncWrapperMixin
from steerllm.core.specs import (
    AddSpec,
    AblationSpec,
    LayerSteeringSpec,
    ProjectionCapSpec,
    SteeringSpec,
)
from steerllm.backends.vllm import runtime as steering_runtime

logger = logging.getLogger(__name__)


def _parse_dtype(dtype_str: str) -> torch.dtype:
    """Parse dtype string to torch.dtype."""
    if not dtype_str.startswith("torch."):
        raise ValueError(f"Unexpected dtype format: {dtype_str}")
    name = dtype_str.split(".", maxsplit=1)[1]
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return dtype


class _VLLMCaptureHandle:
    """Internal capture handle that wraps vLLM worker captures.

    This handle fetches captures from workers via shared memory IPC
    and provides access to the captured tensors.
    """

    def __init__(
        self,
        request_id: str,
        layer_indices: tuple[int, ...],
        model: "VLLMSteeringModel",
        message_boundaries: tuple[MessageBoundary, ...] | None = None,
    ) -> None:
        self.request_id = request_id
        self.layer_indices = layer_indices
        self.message_boundaries = message_boundaries
        self._model_ref = weakref.ref(model)
        self._captures: dict[int, list[dict[str, Any]]] | None = None
        self._shm_objects: list[SharedMemory] = []
        self._shm_names: list[str] = []
        self._closed = False
        self._accessed = False

        # Create mutable container for finalizer tracking
        self._accessed_container = [False]

        # Weak finalizer for cleanup
        def _cleanup_handler(
            shm_objects: list[SharedMemory],
            model_ref: weakref.ref,
            shm_names: list[str],
            request_id: str,
            accessed_container: list[bool],
        ):
            # Emit warning if handle had shared memory but was never accessed
            if shm_objects and not accessed_container[0]:
                import warnings

                warnings.warn(
                    f"CaptureHandle for request {request_id} held {len(shm_objects)} "
                    "shared memory regions but was never accessed. Use 'async with handle:' "
                    "or call 'await handle.close()' to properly release resources.",
                    ResourceWarning,
                    stacklevel=2,
                )

            # Client-side cleanup
            for shm in shm_objects:
                try:
                    shm.close()
                except Exception:
                    pass

            # Worker-side cleanup (best-effort)
            model = model_ref()
            if model is not None and shm_names:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(
                            model._collective_rpc("release_shared_memory", shm_names)
                        )
                except Exception:
                    pass

        self._finalizer = weakref.finalize(
            self,
            _cleanup_handler,
            self._shm_objects,
            self._model_ref,
            self._shm_names,
            self.request_id,
            self._accessed_container,
        )

    async def __aenter__(self) -> "_VLLMCaptureHandle":
        """Async context manager entry."""
        await self.fetch()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Release shared memory and cleanup resources."""
        if self._closed:
            return

        self._closed = True

        # Client-side cleanup: unmap shared memory
        for shm in self._shm_objects:
            try:
                shm.close()
            except Exception as e:
                logger.warning(f"Failed to close shared memory {shm.name}: {e}")

        # Worker-side cleanup: send RPC to unlink shared memory segments
        model = self._model_ref()
        if model is not None and self._shm_names:
            try:
                await model._collective_rpc("release_shared_memory", self._shm_names)
                logger.debug(f"Released {len(self._shm_names)} shared memory segments")
            except Exception as e:
                logger.warning(f"Failed to release shared memory: {e}")

        # Clear references
        self._shm_objects.clear()

        # Detach finalizer since we cleaned up explicitly
        self._finalizer.detach()

    async def fetch(self) -> dict[int, list[dict[str, Any]]]:
        """Fetch captures from workers (idempotent)."""
        if self._captures is None:
            model = self._model_ref()
            if model is None:
                raise RuntimeError("Model has been garbage collected")
            self._captures = await model._fetch_request_captures(
                self.request_id,
                shm_objects_list=self._shm_objects,
            )
            # Extract names for cleanup RPC
            self._shm_names = [shm.name for shm in self._shm_objects]
        return self._captures

    @property
    def captures(self) -> dict[int, list[dict[str, Any]]]:
        """Get captures (must call fetch() first)."""
        if self._captures is None:
            raise RuntimeError(
                f"Captures not fetched yet for request {self.request_id}. "
                "Call: await handle.fetch()"
            )
        # Mark as accessed for finalizer tracking
        self._accessed = True
        self._accessed_container[0] = True
        return self._captures


class VLLMSteeringModel(SyncWrapperMixin):
    """vLLM steering backend with zero-copy shared memory capture.

    Provides per-request steering configuration, allowing different requests
    in the same batch to use different steering vectors (heterogeneous batching).

    Parameters
    ----------
    model_name :
        HuggingFace model identifier or path.
    tensor_parallel_size :
        Number of GPUs for tensor parallelism.
    gpu_memory_utilization :
        Fraction of GPU memory to use.
    max_model_len :
        Maximum sequence length. None for auto-detection.
    dtype :
        Model dtype ("auto", "float16", "bfloat16").
    bootstrap_layers :
        Layer indices to pre-warm for steering.
    shm_ttl_seconds :
        Worker-side TTL for shared memory segments.
    shm_max_gb :
        Maximum total shared memory usage.
    **vllm_kwargs :
        Additional arguments passed to vLLM engine.

    Example
    -------
    >>> model = VLLMSteeringModel("Qwen/Qwen3-0.6B")
    >>> steering = SteeringSpec.simple_add(layer=5, vector=v, scale=1.0)
    >>> texts, handles = await model.generate(prompts, steering_spec=steering)
    """

    def __init__(
        self,
        model_name: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        dtype: str = "auto",
        bootstrap_layers: tuple[int, ...] = (),
        shm_ttl_seconds: int | None = None,
        shm_max_gb: float | None = None,
        decode_buffer_size: int | None = None,
        **vllm_kwargs: Any,
    ) -> None:
        # Lazy import to check for vLLM
        try:
            from vllm import SamplingParams, AsyncEngineArgs
        except ImportError as e:
            raise BackendError(
                "vLLM backend requires vllm. "
                "Install with: pip install steerllm[vllm]"
            ) from e

        # Shared memory configuration (default from env vars if not specified)
        self._shm_ttl_seconds = (
            shm_ttl_seconds
            if shm_ttl_seconds is not None
            else int(os.getenv("STEERLLM_SHM_TTL", "600"))
        )
        self._shm_max_gb = (
            shm_max_gb
            if shm_max_gb is not None
            else float(os.getenv("STEERLLM_MAX_SHM_GB", "128"))
        )
        self._decode_buffer_size = (
            decode_buffer_size
            if decode_buffer_size is not None
            else int(os.getenv("STEERLLM_DECODE_BUFFER_SIZE", "128"))
        )

        # Ensure eager execution (required for steering patches)
        enforce_eager = vllm_kwargs.get("enforce_eager", True)
        if not enforce_eager:
            logger.warning(
                "vLLM steering requires enforce_eager=True; overriding user-supplied value."
            )
            enforce_eager = True

        llm_kwargs = {
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "enforce_eager": enforce_eager,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        llm_kwargs.update(vllm_kwargs)
        llm_kwargs.setdefault(
            "worker_extension_cls", steering_runtime.STEERING_WORKER_EXTENSION
        )

        # Install patches before engine creation
        steering_runtime.ensure_layer_patch_installed()
        steering_runtime.ensure_collective_rpc_gateway_installed()

        # Create engine args (engine initialized on first use)
        self._engine_args = AsyncEngineArgs(model=model_name, **llm_kwargs)
        self._engine = None
        self._engine_init_lock = asyncio.Lock()

        self._model_name = model_name
        self._init_layers = tuple(int(idx) for idx in bootstrap_layers)

        # Load model config to get dimensions before engine init
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Handle multimodal models where hidden_size is nested
        if hasattr(model_config, "text_config") and hasattr(
            model_config.text_config, "hidden_size"
        ):
            self._hidden_size = model_config.text_config.hidden_size
            self._layer_count = model_config.text_config.num_hidden_layers
        else:
            self._hidden_size = model_config.hidden_size
            self._layer_count = model_config.num_hidden_layers

        self._vector_dtype: torch.dtype | None = None
        self._tokenizer = None

    @property
    def hidden_size(self) -> int:
        """Model's hidden dimension."""
        return self._hidden_size

    @property
    def layer_count(self) -> int:
        """Number of transformer layers."""
        return self._layer_count

    @property
    def model_name(self) -> str:
        """Model identifier."""
        return self._model_name

    @property
    def tokenizer(self) -> Any:
        """Tokenizer instance (lazy-loaded)."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        return self._tokenizer

    async def _ensure_engine_initialized(self) -> None:
        """Initialize AsyncLLMEngine and workers on first use."""
        async with self._engine_init_lock:
            if self._engine is not None:
                return

            from vllm import AsyncLLMEngine

            # Create async engine
            self._engine = AsyncLLMEngine.from_engine_args(self._engine_args)

            # Initialize worker state with shared memory config
            setup_info = await self._collective_rpc(
                "initialize_worker_state",
                self._init_layers,
                self._shm_ttl_seconds,
                self._shm_max_gb,
                self._decode_buffer_size,
            )
            if not setup_info:
                raise RuntimeError("Failed to initialize steering state on workers.")

            first = setup_info[0]
            # Verify dimensions match what we loaded from config
            worker_hidden_size = int(first["hidden_size"])
            worker_layer_count = int(first["layer_count"])
            if worker_hidden_size != self._hidden_size:
                raise RuntimeError(
                    f"Worker hidden_size {worker_hidden_size} doesn't match config {self._hidden_size}"
                )
            if worker_layer_count != self._layer_count:
                raise RuntimeError(
                    f"Worker layer_count {worker_layer_count} doesn't match config {self._layer_count}"
                )
            self._vector_dtype = _parse_dtype(first["dtype"])

    async def _collective_rpc(
        self,
        op: str,
        *args: Any,
        timeout: float | None = None,
    ) -> list[Any]:
        """Call a steering RPC on all workers."""
        if self._engine is None:
            raise RuntimeError("Engine not initialized. Call an async method first.")
        return await self._engine.collective_rpc(
            steering_runtime.STEERING_RPC_METHOD,
            timeout=timeout,
            args=steering_runtime.rpc_args(op, *args),
        )

    def _serialize_steering_spec(
        self, spec: SteeringSpec
    ) -> tuple[dict[str, Any], list[SharedMemory]]:
        """Serialize a SteeringSpec for RPC transmission.

        Returns (serialized_spec, list_of_shm_objects_to_cleanup).
        Caller must clean up shm objects after RPC completes.
        """
        # Validate layer indices before serialization
        for layer_idx in spec.layers.keys():
            if layer_idx < 0 or layer_idx >= self._layer_count:
                raise ValueError(
                    f"Layer index {layer_idx} out of range [0, {self._layer_count}). "
                    f"Model {self._model_name} has {self._layer_count} layers."
                )

        serialized_layers = {}
        shm_objects: list[SharedMemory] = []

        for layer_idx, layer_spec in spec.layers.items():
            serialized_ops = []

            for op in layer_spec.operations:
                if isinstance(op, AddSpec):
                    # Materialize and send pre-scaled vector
                    vector = op.materialize()
                    if vector.numel() != self._hidden_size:
                        raise ValueError(
                            f"Steering vector dimension mismatch at layer {layer_idx}: "
                            f"expected {self._hidden_size}, got {vector.numel()}."
                        )
                    # Use serialize_tensor for shm-based encoding
                    payload, shm = steering_runtime.serialize_tensor(
                        vector.to(dtype=self._vector_dtype)
                    )
                    shm_objects.append(shm)
                    serialized_ops.append(
                        {
                            "type": "add",
                            "vector": payload,
                            "params": None,  # Scale already applied
                        }
                    )

                elif isinstance(op, ProjectionCapSpec):
                    if op.vector.numel() != self._hidden_size:
                        raise ValueError(
                            f"Projection cap vector dimension mismatch at layer {layer_idx}: "
                            f"expected {self._hidden_size}, got {op.vector.numel()}."
                        )
                    # Use serialize_tensor for shm-based encoding
                    payload, shm = steering_runtime.serialize_tensor(
                        op.vector.to(dtype=self._vector_dtype)
                    )
                    shm_objects.append(shm)
                    serialized_ops.append(
                        {
                            "type": "cap",
                            "vector": payload,
                            "params": (op.min, op.max),
                        }
                    )

                elif isinstance(op, AblationSpec):
                    if op.vector.numel() != self._hidden_size:
                        raise ValueError(
                            f"Ablation vector dimension mismatch at layer {layer_idx}: "
                            f"expected {self._hidden_size}, got {op.vector.numel()}."
                        )
                    # Use serialize_tensor for shm-based encoding
                    payload, shm = steering_runtime.serialize_tensor(
                        op.vector.to(dtype=self._vector_dtype)
                    )
                    shm_objects.append(shm)
                    serialized_ops.append(
                        {
                            "type": "ablation",
                            "vector": payload,
                            "params": float(op.scale),
                        }
                    )

            serialized_layers[int(layer_idx)] = {"operations": serialized_ops}

        return {"layers": serialized_layers}, shm_objects

    async def _fetch_request_captures(
        self,
        request_id: str,
        shm_objects_list: list[SharedMemory],
    ) -> dict[int, list[dict[str, Any]]]:
        """Fetch captures from workers via shared memory."""
        results = await self._collective_rpc("fetch_captures", request_id)

        # Process first worker's result (all workers have same captures for single-GPU)
        if not results or results[0] is None:
            return {}

        worker_result = results[0]
        captures: dict[int, list[dict[str, Any]]] = {}

        for layer_idx_str, capture_data in worker_result.items():
            layer_idx = int(layer_idx_str)

            tensor = steering_runtime.deserialize_tensor(
                capture_data,
                shm_objects_list=shm_objects_list,
            )
            captures[layer_idx] = [{"hidden": tensor}]

        return captures

    async def fetch_captures_batch(
        self,
        handles: Sequence[_VLLMCaptureHandle],
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
        await self._ensure_engine_initialized()

        # Filter to handles that need fetching
        to_fetch = [h for h in handles if h._captures is None]
        if not to_fetch:
            return

        # Extract request IDs
        request_ids = [h.request_id for h in to_fetch]

        # Batch fetch all captures in single RPC
        results = await self._collective_rpc("fetch_batch_captures", request_ids)

        # Process first worker's result (all workers have same captures for single-GPU)
        if not results or results[0] is None:
            return

        worker_result = results[0]

        # Distribute results to handles
        for handle in to_fetch:
            request_data = worker_result.get(handle.request_id)
            if request_data is None:
                handle._captures = {}
                continue

            captures: dict[int, list[dict[str, Any]]] = {}
            for layer_idx_str, capture_data in request_data.items():
                layer_idx = int(layer_idx_str)
                tensor = steering_runtime.deserialize_tensor(
                    capture_data,
                    shm_objects_list=handle._shm_objects,
                )
                captures[layer_idx] = [{"hidden": tensor}]

            handle._captures = captures

    async def generate(
        self,
        prompts: list[str],
        sampling_params: Any | None = None,
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        steering_spec: SteeringSpec | None = None,
        capture_layers: Sequence[int] | None = None,
        raw_output: bool = False,
        **sampling_kwargs: Any,
    ) -> tuple[list[str], list[CaptureHandle] | None] | tuple[list[Any], list[CaptureHandle] | None]:
        """Generate text with optional steering and capture.

        Parameters
        ----------
        prompts :
            Input prompts for generation.
        sampling_params :
            Optional vLLM SamplingParams object. If provided, max_tokens,
            temperature, and sampling_kwargs are ignored.
        max_tokens :
            Maximum tokens to generate per prompt (ignored if sampling_params provided).
        temperature :
            Sampling temperature (ignored if sampling_params provided).
        steering_spec :
            Optional steering configuration.
        capture_layers :
            Optional layer indices to capture activations from.
        raw_output :
            If True, return full vLLM RequestOutput objects instead of text strings.
        **sampling_kwargs :
            Additional sampling parameters (ignored if sampling_params provided).

        Returns
        -------
        tuple[list[str], list[CaptureHandle] | None]
            Generated texts and capture handles (or None if no capture).
            If raw_output=True, returns RequestOutput objects instead of strings.
        """
        from vllm import SamplingParams

        await self._ensure_engine_initialized()

        # Use provided sampling params or create from kwargs
        if sampling_params is not None:
            params = sampling_params
        else:
            params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                **sampling_kwargs,
            )

        # Convert capture_layers to tuple
        layers_tuple: tuple[int, ...] | None = None
        if capture_layers is not None:
            if isinstance(capture_layers, int):
                layers_tuple = (capture_layers,)
            else:
                layers_tuple = tuple(capture_layers)

            # Validate layer indices
            for layer_idx in layers_tuple:
                if layer_idx < 0 or layer_idx >= self._layer_count:
                    raise ValueError(
                        f"Capture layer index {layer_idx} is out of range. "
                        f"Model has {self._layer_count} layers (valid range: 0-{self._layer_count - 1})"
                    )

        # Generate request IDs
        request_ids = [f"steerllm_{uuid.uuid4().hex}" for _ in prompts]

        # Register captures
        handles: list[CaptureHandle] | None = None
        if layers_tuple is not None:
            handles = []
            for req_id in request_ids:
                await self._collective_rpc("register_capture", req_id, list(layers_tuple))
                internal_handle = _VLLMCaptureHandle(
                    request_id=req_id,
                    layer_indices=layers_tuple,
                    model=self,
                )
                # Wrap in CaptureHandle
                handle = CaptureHandle(
                    request_id=req_id,
                    layer_indices=layers_tuple,
                    fetch_fn=internal_handle.fetch,
                    cleanup_fn=internal_handle.close,
                )
                handles.append(handle)

        # Register steering specs
        serialized_spec = None
        steering_shm_objects: list[SharedMemory] = []
        if steering_spec is not None and not steering_spec.is_empty():
            serialized_spec, steering_shm_objects = self._serialize_steering_spec(
                steering_spec
            )
            for req_id in request_ids:
                await self._collective_rpc(
                    "register_steering_spec", req_id, serialized_spec
                )
            # All workers have cloned from shm - safe to cleanup
            for shm in steering_shm_objects:
                try:
                    shm.close()
                    shm.unlink()
                except Exception as e:
                    logger.warning(f"Failed to cleanup steering shm {shm.name}: {e}")

        try:
            # Generate all prompts concurrently
            async def process_one(i: int, prompt: str) -> Any:
                req_id = request_ids[i]
                final_output = None
                async for output in self._engine.generate(
                    prompt, params, request_id=req_id
                ):
                    final_output = output

                if final_output is None:
                    raise RuntimeError(f"No output for prompt: {prompt}")

                if raw_output:
                    return final_output
                return final_output.outputs[0].text

            # Launch all requests concurrently
            tasks = [process_one(i, p) for i, p in enumerate(prompts)]
            results = await asyncio.gather(*tasks)

            return list(results), handles

        finally:
            # Clean up steering specs
            if serialized_spec is not None:
                cleanup_tasks = [
                    self._collective_rpc("unregister_steering_spec", req_id)
                    for req_id in request_ids
                ]
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*cleanup_tasks, return_exceptions=True),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Steering cleanup timed out for batch of {len(request_ids)} requests"
                    )

    async def generate_stream(
        self,
        prompt: str,
        sampling_params: Any | None = None,
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        steering_spec: SteeringSpec | None = None,
        **sampling_kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream text generation, yielding text deltas.

        This method streams generation for a single prompt, yielding incremental
        text as it's generated. Unlike generate(), this does not support batching
        or activation capture.

        Parameters
        ----------
        prompt :
            Input prompt for generation.
        sampling_params :
            Optional vLLM SamplingParams object. If provided, max_tokens,
            temperature, and sampling_kwargs are ignored.
        max_tokens :
            Maximum tokens to generate (ignored if sampling_params provided).
        temperature :
            Sampling temperature (ignored if sampling_params provided).
        steering_spec :
            Optional steering configuration.
        **sampling_kwargs :
            Additional sampling parameters (ignored if sampling_params provided).

        Yields
        ------
        str
            Incremental text deltas as they are generated.
        """
        from vllm import SamplingParams

        await self._ensure_engine_initialized()

        # Use provided sampling params or create from kwargs
        if sampling_params is not None:
            params = sampling_params
        else:
            params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                **sampling_kwargs,
            )

        # Generate request ID
        request_id = f"steerllm_{uuid.uuid4().hex}"

        # Register steering spec if provided
        serialized_spec = None
        steering_shm_objects: list[SharedMemory] = []
        if steering_spec is not None and not steering_spec.is_empty():
            serialized_spec, steering_shm_objects = self._serialize_steering_spec(
                steering_spec
            )
            await self._collective_rpc(
                "register_steering_spec", request_id, serialized_spec
            )
            # All workers have cloned from shm - safe to cleanup
            for shm in steering_shm_objects:
                try:
                    shm.close()
                    shm.unlink()
                except Exception as e:
                    logger.warning(f"Failed to cleanup steering shm {shm.name}: {e}")

        try:
            prev_text = ""
            async for output in self._engine.generate(
                prompt, params, request_id=request_id
            ):
                current_text = output.outputs[0].text
                delta = current_text[len(prev_text):]
                prev_text = current_text
                if delta:
                    yield delta
        finally:
            # Clean up steering spec
            if serialized_spec is not None:
                try:
                    await asyncio.wait_for(
                        self._collective_rpc("unregister_steering_spec", request_id),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Steering cleanup timed out for request {request_id}"
                    )

    async def chat(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        sampling_params: Any | None = None,
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        steering_spec: SteeringSpec | None = None,
        capture_layers: Sequence[int] | None = None,
        chat_options: dict[str, Any] | None = None,
        raw_output: bool = False,
        **sampling_kwargs: Any,
    ) -> tuple[list[ChatResponse], list[CaptureHandle] | None] | tuple[list[Any], list[CaptureHandle] | None]:
        """Chat-style generation with optional steering and capture.

        Parameters
        ----------
        messages :
            Single conversation or batch of conversations.
        sampling_params :
            Optional vLLM SamplingParams object. If provided, max_tokens,
            temperature, and sampling_kwargs are ignored.
        max_tokens :
            Maximum tokens to generate (ignored if sampling_params provided).
        temperature :
            Sampling temperature (ignored if sampling_params provided).
        steering_spec :
            Optional steering configuration.
        capture_layers :
            Optional layer indices to capture.
        chat_options :
            Options passed to apply_chat_template().
        raw_output :
            If True, return full vLLM RequestOutput objects instead of ChatResponse.
        **sampling_kwargs :
            Additional sampling parameters (ignored if sampling_params provided).

        Returns
        -------
        tuple[list[ChatResponse], list[CaptureHandle] | None]
            Chat responses and capture handles.
            If raw_output=True, returns RequestOutput objects instead of ChatResponse.
        """
        # Normalize to batch format
        single_conversation = isinstance(messages, list) and (
            len(messages) == 0 or isinstance(messages[0], dict)
        )
        conversations = [messages] if single_conversation else messages

        # Format prompts using chat template
        chat_kwargs = {"add_generation_prompt": True, "tokenize": False}
        if chat_options:
            chat_kwargs.update(chat_options)

        # Track prefill text for each conversation (assistant prefix when continue_final_message)
        prefill_texts: list[str] = []
        prompts = []
        for conv in conversations:
            # Check if we're continuing a final assistant message
            has_prefill = (
                conv
                and conv[-1].get("role") == "assistant"
                and chat_kwargs.get("continue_final_message", False)
            )
            prefill_text = conv[-1].get("content", "") if has_prefill else ""
            prefill_texts.append(prefill_text)

            prompt = self.tokenizer.apply_chat_template(conv, **chat_kwargs)
            prompts.append(prompt)

        # Generate
        results, handles = await self.generate(
            prompts,
            sampling_params=sampling_params,
            max_tokens=max_tokens,
            temperature=temperature,
            steering_spec=steering_spec,
            capture_layers=capture_layers,
            raw_output=raw_output,
            **sampling_kwargs,
        )

        # Return raw output if requested
        if raw_output:
            # Add message boundaries to handles if capturing
            if handles is not None:
                for i, handle in enumerate(handles):
                    boundaries = self._compute_message_boundaries(
                        conversations[i], chat_kwargs
                    )
                    handle.message_boundaries = boundaries
            return results, handles

        # Convert to ChatResponse
        responses = []
        for i, text in enumerate(results):
            responses.append(ChatResponse(prefill=prefill_texts[i], generated=text))

        # Add message boundaries to handles if capturing
        if handles is not None:
            for i, handle in enumerate(handles):
                boundaries = self._compute_message_boundaries(
                    conversations[i], chat_kwargs
                )
                handle.message_boundaries = boundaries

        return responses, handles

    def _compute_message_boundaries(
        self,
        messages: list[dict[str, Any]],
        chat_kwargs: dict[str, Any],
    ) -> tuple[MessageBoundary, ...]:
        """Compute token boundaries for each message."""
        boundaries: list[MessageBoundary] = []
        current_offset = 0

        for i, msg in enumerate(messages):
            partial_conv = messages[: i + 1]
            # Note: chat_kwargs already contains tokenize=False
            partial_text = self.tokenizer.apply_chat_template(
                partial_conv, **chat_kwargs
            )
            partial_tokens = self.tokenizer(partial_text, return_tensors="pt")
            total_len = partial_tokens.input_ids.shape[1]

            boundary = MessageBoundary(
                role=msg.get("role", "unknown"),
                content=msg.get("content", ""),
                start_token=current_offset,
                end_token=total_len,
            )
            boundaries.append(boundary)
            current_offset = total_len

        return tuple(boundaries)
