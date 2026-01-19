"""Worker-side utilities for steering vector control inside vLLM workers.

These helpers are executed inside vLLM worker processes via collective RPCs.
They patch decoder layers in Qwen, Llama, and Gemma models so steering vectors
can be injected and activations captured at runtime.

Note: This is a standalone reimplementation for steerllm, not dependent on chatspace.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import wraps
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


# =============================================================================
# Tensor Serialization
# =============================================================================

def create_shared_tensor(tensor: torch.Tensor) -> tuple[dict[str, Any], SharedMemory]:
    """Create shared memory segment for tensor (client-side).

    Returns (metadata_dict, SharedMemory_object).
    Caller is responsible for cleanup via shm.close() and shm.unlink().

    Uses uint8 view for dtype-agnostic transfer - works uniformly for
    float32, float16, bfloat16, etc.
    """
    cpu_tensor = tensor.detach().cpu().contiguous()
    nbytes = cpu_tensor.numel() * cpu_tensor.element_size()
    dtype_name = str(cpu_tensor.dtype).removeprefix("torch.")

    if nbytes == 0:
        # Handle empty tensors - create minimal shm segment
        nbytes = 1  # SharedMemory requires size > 0

    shm_name = f"steerllm_{uuid.uuid4().hex}"
    shm = SharedMemory(create=True, size=nbytes, name=shm_name)

    if cpu_tensor.numel() > 0:
        # uint8 view for dtype-agnostic transfer
        shm_array = np.ndarray(nbytes, dtype=np.uint8, buffer=shm.buf)
        byte_view = cpu_tensor.view(torch.uint8).flatten()
        shm_array[:] = byte_view.numpy()

    metadata = {
        "encoding": "shm",
        "shm_name": shm_name,
        "shape": list(cpu_tensor.shape),
        "dtype": dtype_name,
        "nbytes": cpu_tensor.numel() * cpu_tensor.element_size(),  # Original nbytes (may be 0)
    }
    return metadata, shm


def serialize_tensor(tensor: torch.Tensor) -> tuple[dict[str, Any], SharedMemory]:
    """Serialize tensor via shared memory for RPC transport (client→worker).

    Returns (metadata_dict, SharedMemory_object).
    Caller must clean up after RPC completes: shm.close(); shm.unlink()

    Uses shared memory for zero-copy transfer between client and worker processes.
    """
    return create_shared_tensor(tensor)


def deserialize_tensor(
    payload: dict[str, Any],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    shm_objects_list: list[SharedMemory] | None = None,
) -> torch.Tensor:
    """Deserialize tensor from shared memory.

    Parameters
    ----------
    payload :
        Serialized tensor metadata (shm_name, shape, dtype, nbytes).
    device :
        Target device for the tensor.
    dtype :
        Target dtype for the tensor (overrides payload dtype if specified).
    shm_objects_list :
        If provided, the SharedMemory object will be appended to keep it alive.
    """
    dtype_str = payload.get("dtype")
    shape = payload.get("shape")
    shm_name = payload.get("shm_name")
    nbytes = payload.get("nbytes")

    if dtype_str is None or shape is None:
        raise TypeError(f"Missing required tensor metadata: dtype={dtype_str}, shape={shape}")
    if shm_name is None:
        raise TypeError("Missing shm_name in payload")
    if nbytes is None:
        raise TypeError("Missing nbytes in payload")

    target_dtype = getattr(torch, dtype_str, None)
    if target_dtype is None:
        raise TypeError(f"Unsupported tensor dtype: {dtype_str}")
    shape_tuple = tuple(int(dim) for dim in shape)

    # Open existing shared memory segment
    shm = SharedMemory(name=shm_name, create=False)
    if shm_objects_list is not None:
        shm_objects_list.append(shm)

    # Handle empty tensors
    if nbytes == 0:
        tensor = torch.empty(shape_tuple, dtype=target_dtype)
    else:
        # Read as uint8 bytes, reinterpret to target dtype
        np_array = np.ndarray(nbytes, dtype=np.uint8, buffer=shm.buf)
        tensor = torch.frombuffer(bytearray(np_array), dtype=torch.uint8)
        tensor = tensor.view(target_dtype).reshape(shape_tuple).clone()

    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def _intern_vector(state: "_SteeringState", tensor: torch.Tensor) -> torch.Tensor:
    """Intern a steering vector to enable gather/scatter optimization.

    Returns a cached tensor if one with identical content exists,
    otherwise caches and returns the input tensor. This ensures
    vectors with the same content share the same `id()`, enabling
    the gather/scatter path to batch them together.

    Uses SHA256 hash of tensor bytes for content-based deduplication.
    LRU eviction keeps cache size bounded.
    """
    # Hash tensor content (view as uint8 for universal dtype support)
    tensor_bytes = tensor.detach().cpu().view(torch.uint8).numpy().tobytes()
    content_hash = hashlib.sha256(tensor_bytes).hexdigest()

    # Check cache
    if content_hash in state.vector_cache:
        # Move to end (most recently used)
        state.vector_cache.move_to_end(content_hash)
        return state.vector_cache[content_hash]

    # Not in cache - add it
    state.vector_cache[content_hash] = tensor

    # LRU eviction if over limit
    while len(state.vector_cache) > state.vector_cache_max_size:
        state.vector_cache.popitem(last=False)  # Remove oldest

    return tensor


def _create_shared_tensor(
    tensor: torch.Tensor,
    state: _SteeringState,
    stream: torch.cuda.Stream | None = None,
) -> dict[str, Any]:
    """Create shared memory segment for tensor with direct GPU→shm transfer.

    Uses torch.from_numpy() on shm buffer and copy_() for direct DMA transfer,
    eliminating intermediate CPU tensor allocation.

    NOTE: Does NOT sync - caller must sync the stream after all copies are queued.

    Parameters
    ----------
    tensor :
        Tensor to share (can be on any device).
    state :
        Worker steering state for tracking active segments.
    stream :
        CUDA stream to use for async copy. If None and tensor is on GPU,
        uses default stream (blocking).

    Returns
    -------
    dict
        Metadata dict with encoding="shm".
    """
    nbytes = tensor.numel() * tensor.element_size()
    shm_name = f"steerllm_{uuid.uuid4().hex}"

    # Create shared memory segment
    shm = SharedMemory(create=True, size=nbytes, name=shm_name)

    # Create shm-backed tensor via uint8 view (works for any dtype)
    shm_array = np.ndarray(nbytes, dtype=np.uint8, buffer=shm.buf)
    shm_tensor = torch.from_numpy(shm_array)

    # View source tensor as uint8 for dtype-agnostic transfer
    byte_view = tensor.detach().contiguous().view(torch.uint8).flatten()

    # Direct GPU→shm copy via DMA (or CPU→shm if already on CPU)
    if tensor.device.type == "cuda" and stream is not None:
        with torch.cuda.stream(stream):
            shm_tensor.copy_(byte_view, non_blocking=True)
    else:
        shm_tensor.copy_(byte_view)

    # Track in state with timestamp
    with state.shm_lock:
        state.active_shared_memory[shm_name] = (shm, time.time())

    dtype_name = str(tensor.dtype).removeprefix("torch.")
    return {
        "encoding": "shm",
        "shm_name": shm_name,
        "shape": list(tensor.shape),
        "dtype": dtype_name,
        "nbytes": nbytes,
    }


def _get_env_int(name: str, default: int) -> int:
    """Parse integer environment variable safely."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse boolean environment flag."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


# Configuration from environment
_CAPTURE_METADATA_ENABLED = _env_flag("STEERLLM_CAPTURE_METADATA", True)

# Enable torch.compile for slow path steering loop (reduces kernel launch overhead)
# Benchmarks show 2x speedup for heterogeneous batches when compiling the entire
# loop vs individual ops. Individual op compilation has too much dispatch overhead.
# Disabled by default since the fast path handles most common cases.
_COMPILE_STEERING = _env_flag("STEERLLM_COMPILE_STEERING", False)

# Optional precision override for projection caps
_PROJECTION_CAP_PRECISION: torch.dtype | None = None

# =============================================================================
# Compiled Slow Path Loop
# =============================================================================
# Compiling the entire slow path loop gives 2x speedup vs compiling individual ops.
# Individual op compilation has too much dispatch overhead (~30-50μs per call).
# Loop-level compilation amortizes the dispatch overhead across all iterations.

_compiled_slow_path: Callable | None = None


def _slow_path_loop_impl(
    hidden: torch.Tensor,
    seq_lens: list[int],
    request_layer_specs: list[Any],
) -> torch.Tensor:
    """Apply per-request steering in a loop.

    This function is compiled when STEERLLM_COMPILE_STEERING=1.
    torch.compile traces through the loop, dict lookups, and conditionals.

    Args:
        hidden: Hidden states tensor [total_tokens, hidden_size]
        seq_lens: Sequence lengths for each request
        request_layer_specs: Pre-extracted layer specs for each request (None if no steering)

    Returns:
        Modified hidden states
    """
    start = 0
    for i, layer_spec in enumerate(request_layer_specs):
        seq_len = seq_lens[i]
        end = start + seq_len

        if layer_spec is not None and layer_spec.operations:
            for op_type, vec, params in layer_spec.operations:
                # Vector already on correct device/dtype from deserialization
                if op_type == "add":
                    hidden[start:end] = hidden[start:end] + vec
                elif op_type == "cap":
                    cap_min, cap_max = params
                    proj = hidden[start:end] @ vec
                    # Apply clamping based on which bounds are set
                    if cap_min is not None and cap_max is not None:
                        clamped = torch.clamp(proj, min=cap_min, max=cap_max)
                    elif cap_min is not None:
                        clamped = torch.clamp(proj, min=cap_min)
                    elif cap_max is not None:
                        clamped = torch.clamp(proj, max=cap_max)
                    else:
                        clamped = proj
                    diff = (clamped - proj).unsqueeze(-1)
                    hidden[start:end] = hidden[start:end] + diff * vec
                elif op_type == "ablation":
                    scale = params
                    proj = hidden[start:end] @ vec
                    adjustment = ((scale - 1) * proj).unsqueeze(-1)
                    hidden[start:end] = hidden[start:end] + adjustment * vec

        start = end
    return hidden


def _get_compiled_slow_path() -> Callable:
    """Get or create compiled slow path loop."""
    global _compiled_slow_path
    if _compiled_slow_path is not None:
        return _compiled_slow_path

    try:
        _compiled_slow_path = torch.compile(_slow_path_loop_impl, dynamic=True)
        logger.debug("Compiled slow path loop for steering")
    except Exception as e:
        logger.warning(f"torch.compile failed for slow path: {e}, using uncompiled")
        _compiled_slow_path = _slow_path_loop_impl

    return _compiled_slow_path


# =============================================================================
# Gather/Scatter Path (optimized for heterogeneous batches)
# =============================================================================


def _gather_scatter_steering(
    hidden: torch.Tensor,
    seq_lens: list[int],
    request_layer_specs: list[Any],
) -> torch.Tensor:
    """Apply steering using gather/scatter for better memory access patterns.

    Instead of N non-contiguous slice operations, this:
    1. Groups tokens by operation signature (op_type, vec identity, params)
    2. Gathers all tokens needing same operation to contiguous buffer
    3. Applies steering to contiguous buffer (single efficient op)
    4. Scatters results back to original positions

    For 32 requests using the same steering vector, this reduces
    32 slice operations to 1 gather + 1 op + 1 scatter.
    """
    # Group tokens by operation signature
    # Key: (op_type, id(vec), params_tuple)
    # Value: {'vec': tensor, 'params': params, 'indices': list[int]}
    op_groups: dict[tuple, dict] = {}

    start = 0
    for i, layer_spec in enumerate(request_layer_specs):
        seq_len = seq_lens[i]
        end = start + seq_len

        if layer_spec is not None and layer_spec.operations:
            for op_type, vec, params in layer_spec.operations:
                # Create hashable key for this operation
                # Use id(vec) to group by vector identity (same tensor object)
                if op_type == "cap":
                    params_key = params  # (min, max) tuple
                elif op_type == "ablation":
                    params_key = (params,)  # (scale,) tuple
                else:
                    params_key = ()  # add has no params

                op_key = (op_type, id(vec), params_key)

                if op_key not in op_groups:
                    op_groups[op_key] = {
                        'vec': vec,
                        'op_type': op_type,
                        'params': params,
                        'indices': [],
                    }
                # Add all token indices for this request
                op_groups[op_key]['indices'].extend(range(start, end))

        start = end

    # If no operations, return unchanged
    if not op_groups:
        return hidden

    # Apply each unique operation using gather/scatter
    for op_data in op_groups.values():
        indices = torch.tensor(op_data['indices'], device=hidden.device, dtype=torch.long)
        vec = op_data['vec']
        op_type = op_data['op_type']
        params = op_data['params']
        # Vector already on correct device/dtype from deserialization

        # Gather to contiguous buffer
        gathered = hidden[indices]  # [num_tokens, hidden_size] - contiguous!

        # Apply operation to contiguous buffer
        if op_type == "add":
            gathered = gathered + vec
        elif op_type == "cap":
            cap_min, cap_max = params
            proj = gathered @ vec  # [num_tokens]
            # Apply clamping based on which bounds are set
            if cap_min is not None and cap_max is not None:
                clamped = torch.clamp(proj, min=cap_min, max=cap_max)
            elif cap_min is not None:
                clamped = torch.clamp(proj, min=cap_min)
            elif cap_max is not None:
                clamped = torch.clamp(proj, max=cap_max)
            else:
                clamped = proj
            diff = (clamped - proj).unsqueeze(-1)  # [num_tokens, 1]
            gathered = gathered + diff * vec
        elif op_type == "ablation":
            scale = params
            proj = gathered @ vec  # [num_tokens]
            adjustment = ((scale - 1) * proj).unsqueeze(-1)  # [num_tokens, 1]
            gathered = gathered + adjustment * vec

        # Scatter back to original positions
        hidden[indices] = gathered

    return hidden


_compiled_gather_scatter: Callable | None = None


def _get_compiled_gather_scatter() -> Callable:
    """Get or create compiled gather/scatter function."""
    global _compiled_gather_scatter
    if _compiled_gather_scatter is not None:
        return _compiled_gather_scatter

    try:
        _compiled_gather_scatter = torch.compile(_gather_scatter_steering, dynamic=True)
        logger.debug("Compiled gather/scatter steering")
    except Exception as e:
        logger.warning(f"torch.compile failed for gather/scatter: {e}, using uncompiled")
        _compiled_gather_scatter = _gather_scatter_steering

    return _compiled_gather_scatter


# =============================================================================
# Fused Gather Path (GPU-native index construction)
# =============================================================================


def _fused_add_impl(
    hidden: torch.Tensor,
    seq_lens: list[int],
    dispatch: _OpDispatch,
    uniform_seq_lens: bool,
) -> torch.Tensor:
    """Apply additive steering using pre-computed dispatch info.

    Three paths:
    1. All-same (dispatch.all_same): simple broadcast, fastest
    2. Repeat-interleave (uniform_seq_lens): 4-5x faster than slow loop
    3. Masking (non-uniform): 1.2x faster than slow loop
    """
    # FAST PATH: all requests use identical vector - simple broadcast
    if dispatch.all_same:
        return hidden + dispatch.first_vec

    device = hidden.device

    if uniform_seq_lens:
        # REPEAT-INTERLEAVE PATH: expand vectors using broadcasting
        seq_len = seq_lens[0]
        num_requests = len(seq_lens)

        # Build per-request vector tensor (zeros for unsteered)
        hidden_size = hidden.shape[-1]
        vec_per_request = torch.zeros(num_requests, hidden_size, device=device, dtype=hidden.dtype)
        vec_idx = 0
        for i, has_op in enumerate(dispatch.request_has_op):
            if has_op:
                vec_per_request[i] = dispatch.vecs[vec_idx]
                vec_idx += 1

        # Expand to all tokens and apply
        vec_expanded = vec_per_request.repeat_interleave(seq_len, dim=0)
        hidden = hidden + vec_expanded
        return hidden

    # MASKING PATH: GPU-constructed indices for non-uniform seq_lens
    seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.long)
    cumsum = torch.cumsum(seq_lens_t, dim=0)
    starts = torch.cat([torch.zeros(1, device=device, dtype=torch.long), cumsum[:-1]])

    # Collect steered request indices
    steered_requests = [i for i, has in enumerate(dispatch.request_has_op) if has]
    if not steered_requests:
        return hidden

    steered_t = torch.tensor(steered_requests, device=device, dtype=torch.long)
    steered_starts = starts[steered_t]
    steered_lens = seq_lens_t[steered_t]

    # Build masked indices
    max_len = steered_lens.max().item()
    offsets = torch.arange(max_len, device=device, dtype=torch.long)
    all_indices = steered_starts.unsqueeze(1) + offsets.unsqueeze(0)
    mask = offsets.unsqueeze(0) < steered_lens.unsqueeze(1)
    valid_indices = all_indices[mask]

    # Build assignment tensor and apply
    assignments = torch.arange(len(steered_requests), device=device, dtype=torch.long)
    assign_expanded = assignments.repeat_interleave(steered_lens)

    vecs_stacked = torch.stack(dispatch.vecs)
    gathered = hidden[valid_indices]
    vec_for_each = vecs_stacked[assign_expanded]
    hidden[valid_indices] = gathered + vec_for_each

    return hidden


def _fused_add_gpu_native(
    hidden: torch.Tensor,
    seq_lens: list[int],
    request_layer_specs: list[Any],
    uniform_seq_lens: bool,
) -> torch.Tensor:
    """Wrapper for benchmark compatibility - extracts dispatch and calls impl."""
    add_dispatch, _, _ = _extract_dispatch_info(request_layer_specs)
    if add_dispatch is None:
        return hidden
    return _fused_add_impl(hidden, seq_lens, add_dispatch, uniform_seq_lens)


def _fused_cap_impl(
    hidden: torch.Tensor,
    seq_lens: list[int],
    dispatch: _OpDispatch,
    uniform_seq_lens: bool,
) -> torch.Tensor:
    """Apply projection cap steering using pre-computed dispatch info.

    Three paths:
    1. All-same (dispatch.all_same): simple batched cap, fastest
    2. Repeat-interleave (uniform_seq_lens): 7-8x faster than slow loop
    3. Masking (non-uniform): 2-3x faster than slow loop
    """
    # FAST PATH: all requests use identical vector and params
    if dispatch.all_same:
        cap_min, cap_max = dispatch.first_params
        proj = (hidden * dispatch.first_vec).sum(dim=-1)
        clamped = torch.clamp(proj, cap_min, cap_max)
        diff = (clamped - proj).unsqueeze(-1)
        return hidden + diff * dispatch.first_vec

    device = hidden.device
    dtype = hidden.dtype
    num_requests = len(seq_lens)
    hidden_size = hidden.shape[-1]

    # Extract params lists
    cap_mins = [p[0] for p in dispatch.params]
    cap_maxs = [p[1] for p in dispatch.params]

    if uniform_seq_lens:
        # REPEAT-INTERLEAVE PATH: expand vectors/params using broadcasting
        seq_len = seq_lens[0]

        # Build per-request tensors (zeros for unsteered requests)
        vec_per_request = torch.zeros(num_requests, hidden_size, device=device, dtype=dtype)
        min_per_request = torch.zeros(num_requests, device=device, dtype=dtype)
        max_per_request = torch.zeros(num_requests, device=device, dtype=dtype)
        vec_idx = 0
        for i, has_op in enumerate(dispatch.request_has_op):
            if has_op:
                vec_per_request[i] = dispatch.vecs[vec_idx]
                min_per_request[i] = cap_mins[vec_idx]
                max_per_request[i] = cap_maxs[vec_idx]
                vec_idx += 1

        # Expand to all tokens
        vecs_expanded = vec_per_request.repeat_interleave(seq_len, dim=0)
        mins_expanded = min_per_request.repeat_interleave(seq_len)
        maxs_expanded = max_per_request.repeat_interleave(seq_len)

        # Apply cap: project, clamp, adjust
        proj = (hidden * vecs_expanded).sum(dim=-1)
        clamped = torch.clamp(proj, mins_expanded, maxs_expanded)
        diff = (clamped - proj).unsqueeze(-1)
        hidden = hidden + diff * vecs_expanded
        return hidden

    # MASKING PATH: GPU-constructed indices for non-uniform seq_lens
    seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.long)
    cumsum = torch.cumsum(seq_lens_t, dim=0)
    starts = torch.cat([torch.zeros(1, device=device, dtype=torch.long), cumsum[:-1]])

    # Collect steered request indices
    steered_requests = [i for i, has in enumerate(dispatch.request_has_op) if has]
    if not steered_requests:
        return hidden

    steered_t = torch.tensor(steered_requests, device=device, dtype=torch.long)
    steered_starts = starts[steered_t]
    steered_lens = seq_lens_t[steered_t]

    # Build masked indices
    max_len = steered_lens.max().item()
    offsets = torch.arange(max_len, device=device, dtype=torch.long)
    all_indices = steered_starts.unsqueeze(1) + offsets.unsqueeze(0)
    mask = offsets.unsqueeze(0) < steered_lens.unsqueeze(1)
    valid_indices = all_indices[mask]

    # Build assignment tensor and apply
    assignments = torch.arange(len(steered_requests), device=device, dtype=torch.long)
    assign_expanded = assignments.repeat_interleave(steered_lens)

    vecs_stacked = torch.stack(dispatch.vecs)
    mins_t = torch.tensor(cap_mins, device=device, dtype=dtype)
    maxs_t = torch.tensor(cap_maxs, device=device, dtype=dtype)

    gathered = hidden[valid_indices]
    vec_for_each = vecs_stacked[assign_expanded]
    min_for_each = mins_t[assign_expanded]
    max_for_each = maxs_t[assign_expanded]

    proj = (gathered * vec_for_each).sum(dim=-1)
    clamped = torch.clamp(proj, min_for_each, max_for_each)
    diff = (clamped - proj).unsqueeze(-1)
    hidden[valid_indices] = gathered + diff * vec_for_each

    return hidden


def _fused_cap_gpu_native(
    hidden: torch.Tensor,
    seq_lens: list[int],
    request_layer_specs: list[Any],
    uniform_seq_lens: bool,
) -> torch.Tensor:
    """Wrapper for benchmark compatibility - extracts dispatch and calls impl."""
    _, cap_dispatch, _ = _extract_dispatch_info(request_layer_specs)
    if cap_dispatch is None:
        return hidden
    return _fused_cap_impl(hidden, seq_lens, cap_dispatch, uniform_seq_lens)


def _fused_ablation_impl(
    hidden: torch.Tensor,
    seq_lens: list[int],
    dispatch: _OpDispatch,
    uniform_seq_lens: bool,
) -> torch.Tensor:
    """Apply ablation steering using pre-computed dispatch info.

    Three paths:
    1. All-same (dispatch.all_same): simple batched ablation, fastest
    2. Repeat-interleave (uniform_seq_lens): 7-8x faster than slow loop
    3. Masking (non-uniform): 2-3x faster than slow loop
    """
    # FAST PATH: all requests use identical vector and scale
    if dispatch.all_same:
        proj = (hidden * dispatch.first_vec).sum(dim=-1)
        adjustment = ((dispatch.first_params - 1) * proj).unsqueeze(-1)
        return hidden + adjustment * dispatch.first_vec

    device = hidden.device
    dtype = hidden.dtype
    num_requests = len(seq_lens)
    hidden_size = hidden.shape[-1]

    # dispatch.params is the list of scales for ablation
    ablation_scales = dispatch.params

    if uniform_seq_lens:
        # REPEAT-INTERLEAVE PATH: expand vectors/scales using broadcasting
        seq_len = seq_lens[0]

        # Build per-request tensors (zeros for unsteered, scale=1 means no-op)
        vec_per_request = torch.zeros(num_requests, hidden_size, device=device, dtype=dtype)
        scale_per_request = torch.ones(num_requests, device=device, dtype=dtype)
        vec_idx = 0
        for i, has_op in enumerate(dispatch.request_has_op):
            if has_op:
                vec_per_request[i] = dispatch.vecs[vec_idx]
                scale_per_request[i] = ablation_scales[vec_idx]
                vec_idx += 1

        # Expand to all tokens
        vecs_expanded = vec_per_request.repeat_interleave(seq_len, dim=0)
        scales_expanded = scale_per_request.repeat_interleave(seq_len)

        # Apply ablation: project, scale adjustment
        proj = (hidden * vecs_expanded).sum(dim=-1)
        adjustment = ((scales_expanded - 1) * proj).unsqueeze(-1)
        hidden = hidden + adjustment * vecs_expanded
        return hidden

    # MASKING PATH: GPU-constructed indices for non-uniform seq_lens
    seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.long)
    cumsum = torch.cumsum(seq_lens_t, dim=0)
    starts = torch.cat([torch.zeros(1, device=device, dtype=torch.long), cumsum[:-1]])

    # Collect steered request indices
    steered_requests = [i for i, has in enumerate(dispatch.request_has_op) if has]
    if not steered_requests:
        return hidden

    steered_t = torch.tensor(steered_requests, device=device, dtype=torch.long)
    steered_starts = starts[steered_t]
    steered_lens = seq_lens_t[steered_t]

    # Build masked indices
    max_len = steered_lens.max().item()
    offsets = torch.arange(max_len, device=device, dtype=torch.long)
    all_indices = steered_starts.unsqueeze(1) + offsets.unsqueeze(0)
    mask = offsets.unsqueeze(0) < steered_lens.unsqueeze(1)
    valid_indices = all_indices[mask]

    # Build assignment tensor and apply
    assignments = torch.arange(len(steered_requests), device=device, dtype=torch.long)
    assign_expanded = assignments.repeat_interleave(steered_lens)

    vecs_stacked = torch.stack(dispatch.vecs)
    scales_t = torch.tensor(ablation_scales, device=device, dtype=dtype)

    gathered = hidden[valid_indices]
    vec_for_each = vecs_stacked[assign_expanded]
    scale_for_each = scales_t[assign_expanded]

    proj = (gathered * vec_for_each).sum(dim=-1)
    adjustment = ((scale_for_each - 1) * proj).unsqueeze(-1)
    hidden[valid_indices] = gathered + adjustment * vec_for_each

    return hidden


def _fused_ablation_gpu_native(
    hidden: torch.Tensor,
    seq_lens: list[int],
    request_layer_specs: list[Any],
    uniform_seq_lens: bool,
) -> torch.Tensor:
    """Wrapper for benchmark compatibility - extracts dispatch and calls impl."""
    _, _, abl_dispatch = _extract_dispatch_info(request_layer_specs)
    if abl_dispatch is None:
        return hidden
    return _fused_ablation_impl(hidden, seq_lens, abl_dispatch, uniform_seq_lens)


# =============================================================================
# Clock Scheduler for Multi-Op Batching
# =============================================================================


def _premerge_adds(
    operations: list[tuple[str, torch.Tensor, Any]],
) -> list[tuple[str, torch.Tensor, Any]]:
    """Merge consecutive add operations within a single request.

    [add(v1), add(v2), cap, add(v3)] → [add(v1+v2), cap, add(v3)]

    This is always valid since adds commute with each other.
    Reduces the number of operations and enables better batching.
    """
    if not operations or len(operations) <= 1:
        return operations

    merged: list[tuple[str, torch.Tensor, Any]] = []
    i = 0
    while i < len(operations):
        op_type, vec, params = operations[i]

        if op_type == "add":
            # Collect consecutive adds
            combined_vec = vec
            j = i + 1
            while j < len(operations) and operations[j][0] == "add":
                combined_vec = combined_vec + operations[j][1]
                j += 1
            merged.append(("add", combined_vec, None))
            i = j
        else:
            merged.append(operations[i])
            i += 1

    return merged


@dataclass
class _ClockSlot:
    """Ops at one position in the virtual clock.

    Each slot has three phases: ADD → ABLATE → CAP.
    Within each phase, ops are batched across all requests that have
    that op type at this slot position.
    """
    # Add phase
    add_vecs: list[torch.Tensor] = field(default_factory=list)
    add_request_mask: list[bool] = field(default_factory=list)

    # Ablation phase
    ablate_vecs: list[torch.Tensor] = field(default_factory=list)
    ablate_scales: list[float] = field(default_factory=list)
    ablate_request_mask: list[bool] = field(default_factory=list)

    # Cap phase
    cap_vecs: list[torch.Tensor] = field(default_factory=list)
    cap_params: list[tuple[float, float]] = field(default_factory=list)
    cap_request_mask: list[bool] = field(default_factory=list)


def _build_clock_schedule(
    request_layer_specs: list[Any],
) -> list[_ClockSlot]:
    """Build clock schedule using greedy list scheduling.

    1. Pre-merge consecutive adds within each request
    2. Schedule all "ready" ops at each slot (greedy)
    3. Return list of slots for execution

    Args:
        request_layer_specs: Per-request layer specs with operations

    Returns:
        List of _ClockSlot, one per clock cycle needed
    """
    num_requests = len(request_layer_specs)

    # Step 1: Pre-merge consecutive adds
    ops_per_request: list[list[tuple[str, torch.Tensor, Any]]] = []
    for spec in request_layer_specs:
        if spec is not None and spec.operations:
            merged = _premerge_adds(list(spec.operations))
            ops_per_request.append(merged)
        else:
            ops_per_request.append([])

    # Step 2: Greedy scheduling - schedule all ready ops per slot
    op_cursors = [0] * num_requests  # Next op index per request
    slots: list[_ClockSlot] = []

    while any(op_cursors[r] < len(ops_per_request[r]) for r in range(num_requests)):
        # Initialize slot with empty masks
        slot = _ClockSlot(
            add_request_mask=[False] * num_requests,
            ablate_request_mask=[False] * num_requests,
            cap_request_mask=[False] * num_requests,
        )

        # Collect and schedule all ready ops
        for r in range(num_requests):
            if op_cursors[r] >= len(ops_per_request[r]):
                continue

            op_type, vec, params = ops_per_request[r][op_cursors[r]]

            if op_type == "add":
                slot.add_vecs.append(vec)
                slot.add_request_mask[r] = True
            elif op_type == "ablation":
                slot.ablate_vecs.append(vec)
                slot.ablate_scales.append(params)
                slot.ablate_request_mask[r] = True
            elif op_type == "cap":
                slot.cap_vecs.append(vec)
                # Normalize params (handle None bounds)
                cap_min, cap_max = params
                normalized = (
                    cap_min if cap_min is not None else float('-inf'),
                    cap_max if cap_max is not None else float('inf'),
                )
                slot.cap_params.append(normalized)
                slot.cap_request_mask[r] = True

            op_cursors[r] += 1

        slots.append(slot)

    return slots


def _check_slot_all_same(
    vecs: list[torch.Tensor],
    request_mask: list[bool],
    params: list[Any] | None = None,
) -> tuple[bool, torch.Tensor | None, Any]:
    """Check if all ops in a slot phase are identical.

    Returns (all_same, first_vec, first_params).
    """
    if not vecs:
        return True, None, None

    first_vec = vecs[0]
    first_params = params[0] if params else None
    all_same = True

    # Check if all requests have this op (no gaps)
    if not all(request_mask):
        all_same = False
    else:
        # Check if all vecs are identical (by identity)
        for v in vecs[1:]:
            if v is not first_vec:
                all_same = False
                break

        # Check params if present
        if all_same and params:
            for p in params[1:]:
                if p != first_params:
                    all_same = False
                    break

    return all_same, first_vec, first_params


def _execute_clock_schedule(
    hidden: torch.Tensor,
    seq_lens: list[int],
    schedule: list[_ClockSlot],
    uniform_seq_lens: bool,
) -> torch.Tensor:
    """Execute clock schedule, batching ops at each slot.

    For each slot, runs: ADD phase → ABLATE phase → CAP phase.
    Skips phases with no ops.
    """
    for slot in schedule:
        # ADD phase
        if slot.add_vecs:
            all_same, first_vec, _ = _check_slot_all_same(
                slot.add_vecs, slot.add_request_mask
            )
            add_dispatch = _OpDispatch(
                vecs=slot.add_vecs,
                request_has_op=slot.add_request_mask,
                all_same=all_same,
                first_vec=first_vec,
            )
            hidden = _fused_add_impl(hidden, seq_lens, add_dispatch, uniform_seq_lens)

        # ABLATE phase
        if slot.ablate_vecs:
            all_same, first_vec, first_params = _check_slot_all_same(
                slot.ablate_vecs, slot.ablate_request_mask, slot.ablate_scales
            )
            abl_dispatch = _OpDispatch(
                vecs=slot.ablate_vecs,
                request_has_op=slot.ablate_request_mask,
                all_same=all_same,
                first_vec=first_vec,
                params=slot.ablate_scales,
                first_params=first_params,
            )
            hidden = _fused_ablation_impl(hidden, seq_lens, abl_dispatch, uniform_seq_lens)

        # CAP phase
        if slot.cap_vecs:
            all_same, first_vec, first_params = _check_slot_all_same(
                slot.cap_vecs, slot.cap_request_mask, slot.cap_params
            )
            cap_dispatch = _OpDispatch(
                vecs=slot.cap_vecs,
                request_has_op=slot.cap_request_mask,
                all_same=all_same,
                first_vec=first_vec,
                params=slot.cap_params,
                first_params=first_params,
            )
            hidden = _fused_cap_impl(hidden, seq_lens, cap_dispatch, uniform_seq_lens)

    return hidden


# =============================================================================
# Single-Op Dispatch (existing fast paths)
# =============================================================================


@dataclass
class _OpDispatch:
    """Pre-computed dispatch info for a single operation type."""
    vecs: list[torch.Tensor]
    request_has_op: list[bool]
    all_same: bool
    first_vec: torch.Tensor | None = None
    # For cap/ablation params
    params: list[Any] | None = None
    first_params: Any = None


def _extract_dispatch_info(
    request_layer_specs: list[Any],
) -> tuple[_OpDispatch | None, _OpDispatch | None, _OpDispatch | None]:
    """Extract all dispatch info in a single pass through specs.

    Returns (add_dispatch, cap_dispatch, ablation_dispatch).
    Each is None if no operations of that type exist.
    """
    num_requests = len(request_layer_specs)

    # Add state
    add_vecs: list[torch.Tensor] = []
    add_has: list[bool] = [False] * num_requests
    add_first: torch.Tensor | None = None
    add_all_same = True

    # Cap state
    cap_vecs: list[torch.Tensor] = []
    cap_params: list[tuple[float, float]] = []
    cap_has: list[bool] = [False] * num_requests
    cap_first: torch.Tensor | None = None
    cap_first_params: tuple[float, float] | None = None
    cap_all_same = True

    # Ablation state
    abl_vecs: list[torch.Tensor] = []
    abl_scales: list[float] = []
    abl_has: list[bool] = [False] * num_requests
    abl_first: torch.Tensor | None = None
    abl_first_scale: float | None = None
    abl_all_same = True

    for i, layer_spec in enumerate(request_layer_specs):
        if layer_spec is None or not layer_spec.operations:
            # No ops for this request - breaks "all same" if we've seen any
            if add_first is not None:
                add_all_same = False
            if cap_first is not None:
                cap_all_same = False
            if abl_first is not None:
                abl_all_same = False
            continue

        for op_type, vec, params in layer_spec.operations:
            if op_type == "add":
                add_vecs.append(vec)
                add_has[i] = True
                if add_first is None:
                    add_first = vec
                elif vec is not add_first:
                    add_all_same = False

            elif op_type == "cap":
                cap_vecs.append(vec)
                p = (
                    params[0] if params[0] is not None else float('-inf'),
                    params[1] if params[1] is not None else float('inf'),
                )
                cap_params.append(p)
                cap_has[i] = True
                if cap_first is None:
                    cap_first = vec
                    cap_first_params = p
                elif vec is not cap_first or p != cap_first_params:
                    cap_all_same = False

            elif op_type == "ablation":
                abl_vecs.append(vec)
                abl_scales.append(params)
                abl_has[i] = True
                if abl_first is None:
                    abl_first = vec
                    abl_first_scale = params
                elif vec is not abl_first or params != abl_first_scale:
                    abl_all_same = False

    # Build dispatch objects (None if no ops of that type)
    add_dispatch = None
    if add_vecs:
        add_dispatch = _OpDispatch(
            vecs=add_vecs,
            request_has_op=add_has,
            all_same=add_all_same and all(add_has),
            first_vec=add_first,
        )

    cap_dispatch = None
    if cap_vecs:
        cap_dispatch = _OpDispatch(
            vecs=cap_vecs,
            request_has_op=cap_has,
            all_same=cap_all_same and all(cap_has),
            first_vec=cap_first,
            params=cap_params,
            first_params=cap_first_params,
        )

    abl_dispatch = None
    if abl_vecs:
        abl_dispatch = _OpDispatch(
            vecs=abl_vecs,
            request_has_op=abl_has,
            all_same=abl_all_same and all(abl_has),
            first_vec=abl_first,
            params=abl_scales,
            first_params=abl_first_scale,
        )

    return add_dispatch, cap_dispatch, abl_dispatch


def _fused_gather_steering(
    hidden: torch.Tensor,
    seq_lens: list[int],
    request_layer_specs: list[Any],
) -> torch.Tensor:
    """Apply steering using GPU-native fused operations.

    Dispatch strategy:
    1. Check for multi-op case (any request has >1 op after add merging)
    2. If multi-op: use clock scheduler to preserve operation order
    3. If single-op: use fast single-pass extraction

    Benchmarks show (single-op case):
    - All-same: 9-12x faster than slow loop
    - Repeat-interleave: 2-5x faster than slow loop
    - Masking: 1.2-2.8x faster than slow loop
    """
    # Check for multi-op case: any request has >1 operation
    # Use raw op count - clock scheduler will handle add merging
    max_ops = 0
    for spec in request_layer_specs:
        if spec is not None and spec.operations:
            max_ops = max(max_ops, len(spec.operations))

    uniform_seq_lens = len(set(seq_lens)) == 1

    # Multi-op case: use clock scheduler (handles add merging internally)
    if max_ops > 1:
        schedule = _build_clock_schedule(request_layer_specs)
        if schedule:
            return _execute_clock_schedule(hidden, seq_lens, schedule, uniform_seq_lens)
        return hidden

    # Single-op case: use fast single-pass extraction
    add_dispatch, cap_dispatch, abl_dispatch = _extract_dispatch_info(request_layer_specs)

    if not (add_dispatch or cap_dispatch or abl_dispatch):
        return hidden

    # Apply each operation type with pre-computed dispatch info
    if add_dispatch:
        hidden = _fused_add_impl(hidden, seq_lens, add_dispatch, uniform_seq_lens)
    if cap_dispatch:
        hidden = _fused_cap_impl(hidden, seq_lens, cap_dispatch, uniform_seq_lens)
    if abl_dispatch:
        hidden = _fused_ablation_impl(hidden, seq_lens, abl_dispatch, uniform_seq_lens)

    return hidden


@dataclass
class _ProjectionCapConfig:
    """Projection capping parameters for a layer."""
    unit_vector: torch.Tensor
    min: float | None
    max: float | None


@dataclass
class _AblationConfig:
    """Ablation parameters for a layer."""
    unit_vector: torch.Tensor
    scale: float


@dataclass
class _SteeringState:
    """Track steering metadata for a worker.

    This is the per-worker state that maintains:
    - Active capture requests and their buffers
    - Per-request steering specifications
    - Shared memory tracking for zero-copy IPC
    """

    hidden_size: int
    dtype: torch.dtype
    device: torch.device

    # Per-request activation capture
    active_capture_requests: dict[str, set[int]] = field(default_factory=dict)
    request_captures: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)
    request_prefill_buffers: dict[str, dict[int, list[torch.Tensor]]] = field(default_factory=dict)
    request_decode_buffers: dict[str, dict[int, list[torch.Tensor]]] = field(default_factory=dict)
    request_last_phase: dict[str, str] = field(default_factory=dict)
    request_token_counts: dict[str, int] = field(default_factory=dict)

    # Per-request steering (request_id -> deserialized spec)
    request_steering_specs: dict[str, Any] = field(default_factory=dict)

    # Vector interning cache (hash -> tensor) with LRU eviction
    # This enables gather/scatter optimization across RPC by deduplicating identical vectors
    vector_cache: OrderedDict[str, torch.Tensor] = field(default_factory=OrderedDict)
    vector_cache_max_size: int = 1000  # ~8MB for bf16 4096-dim vectors

    # Per-step batch metadata from model runner
    step_metadata: dict[int, dict[str, Any]] = field(default_factory=dict)
    global_step: int = 0

    # Async transfer infrastructure
    transfer_stream: torch.cuda.Stream | None = None
    request_pending_transfers: dict[str, dict[int, tuple[torch.Tensor, torch.cuda.Event]]] = field(default_factory=dict)

    # Shared memory IPC
    active_shared_memory: dict[str, tuple[Any, float]] = field(default_factory=dict)
    shm_lock: threading.Lock = field(default_factory=threading.Lock)
    shm_cleanup_thread: Any = None

    # Shared memory configuration
    shm_ttl_seconds: int = 600
    shm_max_gb: float = 128.0

    # Capture configuration
    decode_buffer_size: int = 128


# Module-level state
_WORKER_STATE: _SteeringState | None = None


def get_worker_state() -> _SteeringState | None:
    """Get the current worker's steering state."""
    return _WORKER_STATE


def set_worker_state(state: _SteeringState) -> None:
    """Set the current worker's steering state."""
    global _WORKER_STATE
    _WORKER_STATE = state


class _SteeredModelWrapper(nn.Module):
    """Wrap vLLM model to apply steering after forward execution."""

    def __init__(self, model: nn.Module, state: _SteeringState) -> None:
        super().__init__()
        object.__setattr__(self, "_wrapped_model", model)
        self._steering_state = state

    def __getattr__(self, name: str) -> Any:
        if name in {"_wrapped_model", "_steering_state"}:
            return object.__getattribute__(self, name)
        return getattr(self._wrapped_model, name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._wrapped_model(*args, **kwargs)

    def unwrap(self) -> nn.Module:
        return self._wrapped_model


# =============================================================================
# Decoder Layer Patching
# =============================================================================

_PATCH_TARGETS: Sequence[tuple[str, str]] = (
    # Qwen models
    ("vllm.model_executor.models.qwen2", "Qwen2DecoderLayer"),
    ("vllm.model_executor.models.qwen2_moe", "Qwen2MoeDecoderLayer"),
    ("vllm.model_executor.models.qwen2_vl", "Qwen2VLDecoderLayer"),
    ("vllm.model_executor.models.qwen3", "Qwen3DecoderLayer"),
    ("vllm.model_executor.models.qwen3_moe", "Qwen3MoeDecoderLayer"),
    ("vllm.model_executor.models.qwen3_next", "Qwen3NextDecoderLayer"),
    ("vllm.model_executor.models.qwen3_vl", "Qwen3DecoderLayer"),
    # Llama models
    ("vllm.model_executor.models.llama", "LlamaDecoderLayer"),
    ("vllm.model_executor.models.llama4", "Llama4DecoderLayer"),
    ("vllm.model_executor.models.llama_eagle", "LlamaDecoderLayer"),
    ("vllm.model_executor.models.llama_eagle3", "LlamaDecoderLayer"),
    ("vllm.model_executor.models.llama4_eagle", "Llama4DecoderLayer"),
    # Gemma models
    ("vllm.model_executor.models.gemma", "GemmaDecoderLayer"),
    ("vllm.model_executor.models.gemma2", "Gemma2DecoderLayer"),
    ("vllm.model_executor.models.gemma3", "Gemma3DecoderLayer"),
)

_PATCHED_CLASSES: set[type] = set()
_PATCH_INSTALLED = False


def _extract_hidden_from_output(output: Any) -> torch.Tensor | None:
    """Extract hidden state tensor from layer output.

    vLLM Qwen/Llama layers return (delta, residual) where the full hidden
    state is residual + delta. We combine them to match HuggingFace outputs.
    """
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (tuple, list)):
        if len(output) >= 2:
            first, second = output[0], output[1]
            if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
                # vLLM returns (delta, residual), we want residual + delta
                return second + first
        if len(output) > 0 and isinstance(output[0], torch.Tensor):
            return output[0]

    if isinstance(output, dict) and "last_hidden_state" in output:
        return output["last_hidden_state"]

    if hasattr(output, "last_hidden_state"):
        hidden = output.last_hidden_state
        if isinstance(hidden, torch.Tensor):
            return hidden

    raise TypeError(f"Cannot extract hidden state from {type(output).__name__}")


def _reconstruct_output_with_hidden(
    output: Any,
    original_hidden: torch.Tensor,
    new_hidden: torch.Tensor,
) -> Any:
    """Reconstruct layer output with modified hidden states.

    For vLLM (delta, residual) format, we compute new_delta = new_hidden - residual.
    """
    if isinstance(output, torch.Tensor):
        return new_hidden

    if isinstance(output, tuple):
        if len(output) >= 2:
            first, second = output[0], output[1]
            if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
                # output is (delta, residual)
                # original: delta + residual = original_hidden
                # new: new_delta + residual = new_hidden
                # new_delta = new_hidden - residual
                new_delta = new_hidden - second
                return (new_delta,) + output[1:]
        if len(output) > 0 and isinstance(output[0], torch.Tensor):
            return (new_hidden,) + output[1:]

    if isinstance(output, list):
        if len(output) >= 2:
            first, second = output[0], output[1]
            if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
                new_delta = new_hidden - second
                return [new_delta] + output[1:]
        if len(output) > 0 and isinstance(output[0], torch.Tensor):
            return [new_hidden] + output[1:]

    return output


# =============================================================================
# Steering Operations
# =============================================================================

def _apply_projection_cap(
    hidden: torch.Tensor,
    config: _ProjectionCapConfig,
) -> torch.Tensor:
    """Apply projection capping to hidden states.

    Clamps the component of hidden states along the direction.
    """
    vec = config.unit_vector
    # Vector already on correct device/dtype from deserialization

    # Compute projection: (hidden @ vec)
    proj = hidden @ vec  # [seq_len]

    # Clamp projection
    clamped = proj.clone()
    if config.min is not None:
        clamped = torch.clamp(clamped, min=config.min)
    if config.max is not None:
        clamped = torch.clamp(clamped, max=config.max)

    # Apply correction: hidden += (clamped - proj) * vec
    diff = (clamped - proj).unsqueeze(-1)  # [seq_len, 1]
    return hidden + diff * vec


def _apply_ablation(
    hidden: torch.Tensor,
    config: _AblationConfig,
) -> torch.Tensor:
    """Apply ablation (component scaling) to hidden states.

    Scales the component along the direction by the given factor.
    """
    vec = config.unit_vector
    # Vector already on correct device/dtype from deserialization

    # Compute projection
    proj = hidden @ vec  # [seq_len]

    # Apply scaling: hidden += (scale - 1) * proj * vec
    adjustment = ((config.scale - 1) * proj).unsqueeze(-1)  # [seq_len, 1]
    return hidden + adjustment * vec


def _apply_layer_steering_to_hidden(
    hidden: torch.Tensor,
    layer_spec: Any,
    state: _SteeringState,
) -> torch.Tensor:
    """Apply steering operations to hidden states.

    Operations are applied in sequence order.
    Each operation is a tuple: (op_type, vector, params)

    Note: torch.compile is applied at the loop level in the slow path,
    not here. Individual op compilation adds too much dispatch overhead.
    """
    ops = layer_spec.operations
    if not ops:
        return hidden

    for op_type, vec, params in ops:
        # Vector already on correct device/dtype from deserialization
        if op_type == "add":
            hidden = hidden + vec
        elif op_type == "cap":
            cap_min, cap_max = params
            hidden = _apply_projection_cap(
                hidden, _ProjectionCapConfig(unit_vector=vec, min=cap_min, max=cap_max)
            )
        elif op_type == "ablation":
            hidden = _apply_ablation(
                hidden, _AblationConfig(unit_vector=vec, scale=params)
            )

    return hidden


def _apply_per_request_steering(
    output: Any,
    state: _SteeringState,
    layer_idx: int,
    request_ids: list[str],
    seq_lens: list[int] | None,
    cached_hidden: torch.Tensor | None = None,
) -> Any:
    """Apply per-request steering by slicing and transforming hidden states.

    Uses a fast path when all requests have identical steering specs (by identity),
    avoiding per-request slicing overhead. Falls back to slow path for heterogeneous
    batches where different requests have different steering configurations.
    """
    hidden = cached_hidden if cached_hidden is not None else _extract_hidden_from_output(output)
    if hidden is None or hidden.dim() != 2:
        return output

    # Single request without seq_lens
    if seq_lens is None or len(seq_lens) != len(request_ids):
        if len(request_ids) == 1:
            req_id = request_ids[0]
            spec = state.request_steering_specs.get(req_id)
            if spec is not None and layer_idx in spec.layers:
                layer_spec = spec.layers[layer_idx]
                if layer_spec.operations:
                    new_hidden = _apply_layer_steering_to_hidden(hidden, layer_spec, state)
                    return _reconstruct_output_with_hidden(output, hidden, new_hidden)
        return output

    # Fast path: check if all requests use the same layer spec (by identity)
    # This allows a single batched operation instead of per-request slicing
    first_layer_spec = None
    all_same_spec = True
    any_has_spec = False

    for req_id in request_ids:
        spec = state.request_steering_specs.get(req_id)
        if spec is not None and layer_idx in spec.layers:
            layer_spec = spec.layers[layer_idx]
            if layer_spec.operations:  # Non-empty operations list
                if first_layer_spec is None:
                    first_layer_spec = layer_spec
                    any_has_spec = True
                elif layer_spec is not first_layer_spec:
                    all_same_spec = False
                    break
        else:
            # This request has no spec for this layer
            if any_has_spec:
                # Mixed: some have spec, some don't
                all_same_spec = False
                break

    # Uniform batch fast path: single batched operation
    if all_same_spec and first_layer_spec is not None:
        # Compute total real tokens (excludes padding)
        total_real_tokens = sum(seq_lens)

        # Clone to avoid modifying original, apply steering to real tokens only
        transformed_hidden = hidden.clone()
        transformed_hidden[:total_real_tokens] = _apply_layer_steering_to_hidden(
            hidden[:total_real_tokens], first_layer_spec, state
        )
        return _reconstruct_output_with_hidden(output, hidden, transformed_hidden)

    # Slow path: per-request slicing (heterogeneous batch or mixed steered/unsteered)
    # Pre-extract layer specs for the compiled loop
    request_layer_specs = []
    for req_id in request_ids:
        spec = state.request_steering_specs.get(req_id)
        if spec is not None and layer_idx in spec.layers:
            request_layer_specs.append(spec.layers[layer_idx])
        else:
            request_layer_specs.append(None)

    transformed_hidden = hidden.clone()

    # GPU-native fused path for all heterogeneous batches
    # Uses repeat_interleave for uniform seq_lens (5.6x faster than slow loop)
    # or GPU-constructed indices for sparse/non-uniform cases (2.8x faster)
    #
    # Benchmarks show fused is always faster than gather/scatter and slow loop
    # for add operations. For mixed ops (cap/ablation), fused handles them too.
    transformed_hidden = _fused_gather_steering(transformed_hidden, seq_lens, request_layer_specs)

    return _reconstruct_output_with_hidden(output, hidden, transformed_hidden)


# =============================================================================
# Capture Logic
# =============================================================================

def _capture_hook_full(
    state: _SteeringState,
    layer_idx: int,
    hidden: torch.Tensor,
    request_ids: list[str],
    seq_lens: list[int] | None,
) -> None:
    """Capture activations for requests that have capture enabled."""
    if not state.active_capture_requests:
        return

    start_idx = 0
    for i, req_id in enumerate(request_ids):
        if req_id not in state.active_capture_requests:
            if seq_lens is not None and i < len(seq_lens):
                start_idx += seq_lens[i]
            continue

        capture_layers = state.active_capture_requests[req_id]
        if layer_idx not in capture_layers:
            if seq_lens is not None and i < len(seq_lens):
                start_idx += seq_lens[i]
            continue

        # Extract this request's hidden states
        if seq_lens is not None and i < len(seq_lens):
            seq_len = seq_lens[i]
            end_idx = start_idx + seq_len
            req_hidden = hidden[start_idx:end_idx].detach().clone()
            start_idx = end_idx
        else:
            req_hidden = hidden.detach().clone()

        # Store in request captures
        if req_id not in state.request_captures:
            state.request_captures[req_id] = {}

        if layer_idx not in state.request_captures[req_id]:
            state.request_captures[req_id][layer_idx] = req_hidden
        else:
            # Concatenate with existing captures (for decode tokens)
            existing = state.request_captures[req_id][layer_idx]
            state.request_captures[req_id][layer_idx] = torch.cat([existing, req_hidden], dim=0)


# =============================================================================
# Layer Patching
# =============================================================================

def _patch_decoder_layer_class(layer_cls: type) -> None:
    """Patch a decoder layer class to support steering and capture."""
    if layer_cls in _PATCHED_CLASSES:
        return

    original_forward = layer_cls.forward

    @wraps(original_forward)
    def _patched_forward(self, *args: Any, **kwargs: Any) -> Any:
        output = original_forward(self, *args, **kwargs)

        state = getattr(self, "_steerllm_state", None)
        layer_idx = getattr(self, "_steerllm_layer_index", None)

        if state is None or layer_idx is None:
            return output

        # Get batch metadata
        request_ids = None
        seq_lens = None
        current_step = state.global_step - 1
        if current_step >= 0:
            metadata = state.step_metadata.get(current_step)
            if metadata is not None:
                request_ids = metadata.get("request_ids")
                seq_lens = metadata.get("seq_lens")

        if not request_ids:
            return output

        # Check for per-request steering
        has_steering = False
        for req_id in request_ids:
            spec = state.request_steering_specs.get(req_id)
            if spec is not None and layer_idx in spec.layers:
                layer_spec = spec.layers[layer_idx]
                if layer_spec.operations:
                    has_steering = True
                    break

        # Extract hidden state once for both steering and capture
        cached_hidden = None
        if has_steering or state.active_capture_requests:
            cached_hidden = _extract_hidden_from_output(output)

        # Apply steering
        if has_steering:
            output = _apply_per_request_steering(
                output, state, layer_idx, request_ids, seq_lens, cached_hidden
            )
            if state.active_capture_requests:
                cached_hidden = _extract_hidden_from_output(output)

        # Capture activations
        if state.active_capture_requests:
            hidden = cached_hidden if cached_hidden is not None else _extract_hidden_from_output(output)
            if hidden is not None and hidden.dim() == 2:
                _capture_hook_full(state, layer_idx, hidden, request_ids, seq_lens)

        return output

    layer_cls.forward = _patched_forward
    _PATCHED_CLASSES.add(layer_cls)


def ensure_layer_patch_installed() -> None:
    """Patch known decoder layers to support steering."""
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return

    for module_name, class_name in _PATCH_TARGETS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        layer_cls = getattr(module, class_name, None)
        if layer_cls is None:
            continue
        _patch_decoder_layer_class(layer_cls)

    _PATCH_INSTALLED = True


def _resolve_layers(model: Any) -> list[Any]:
    """Return transformer layers for Qwen, Llama, and Gemma architectures."""
    # Multimodal models
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return list(lm.model.layers)
        if hasattr(lm, "layers"):
            return list(lm.layers)

    # Standard architectures
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    if hasattr(model, "layers"):
        return list(model.layers)

    raise ValueError(f"Cannot find layers in model {type(model).__name__}")


def _cleanup_stale_shared_memory(state: _SteeringState, stop_event: threading.Event) -> None:
    """Background thread that periodically cleans up stale shared memory segments.

    Parameters
    ----------
    state : _SteeringState
        Worker steering state containing active_shared_memory dict.
    stop_event : threading.Event
        Event to signal thread shutdown.
    """
    ttl_seconds = state.shm_ttl_seconds
    scan_interval = 60  # Scan every 60 seconds

    logger.info(f"Starting shared memory cleanup thread (TTL: {ttl_seconds}s, scan: {scan_interval}s)")

    while not stop_event.wait(scan_interval):
        # Snapshot current shared memory state with lock
        with state.shm_lock:
            if not state.active_shared_memory:
                continue
            active_items = list(state.active_shared_memory.items())

        now = time.time()
        stale_names = []

        # Find stale segments (outside lock)
        for shm_name, (shm, timestamp) in active_items:
            age = now - timestamp
            if age > ttl_seconds:
                stale_names.append(shm_name)

        # Clean up stale segments
        for shm_name in stale_names:
            # Pop from dict with lock held
            with state.shm_lock:
                if shm_name not in state.active_shared_memory:
                    continue  # Already cleaned up
                shm, timestamp = state.active_shared_memory.pop(shm_name)

            # Close and unlink outside lock
            try:
                age = now - timestamp
                shm.close()
                shm.unlink()
                logger.warning(
                    f"TTL expired: {shm_name} (age: {age:.1f}s, "
                    f"size: {shm.size / (1024**2):.2f}MB)"
                )
            except Exception as e:
                logger.error(f"Failed to cleanup stale shared memory {shm_name}: {e}")

    logger.info("Shared memory cleanup thread stopped")


def _patch_model_runner(worker: Any, state: _SteeringState) -> None:
    """Patch GPUModelRunner.execute_model to capture per-step batch metadata."""
    if not _CAPTURE_METADATA_ENABLED:
        logger.info("STEERLLM_CAPTURE_METADATA=0; skipping model runner patch.")
        return

    model_runner = worker.model_runner
    logger.info(
        f"_patch_model_runner: model_runner type={type(model_runner).__name__}, "
        f"has execute_model={hasattr(model_runner, 'execute_model')}"
    )

    if not hasattr(model_runner, "execute_model"):
        logger.error(
            f"model_runner does not have execute_model method! "
            f"Available methods: {[m for m in dir(model_runner) if not m.startswith('_')][:20]}"
        )
        return

    original_execute = getattr(model_runner, "_original_execute_model", None)
    if original_execute is not None:
        # Already patched
        logger.info("Model runner already patched, skipping")
        return

    original_execute = model_runner.execute_model
    logger.info(
        f"Patching model_runner.execute_model: original={original_execute}, "
        f"model_runner type={type(model_runner).__name__}"
    )

    def patched_execute_model(model_input: Any, *args: Any, **kwargs: Any) -> Any:
        """Intercept execute_model to capture batch metadata."""
        if not state.active_capture_requests and not state.request_steering_specs:
            return original_execute(model_input, *args, **kwargs)

        # Extract request IDs and sequence lengths from model_input
        try:
            request_ids = None
            seq_lens = None

            # V1 engine: model_input is SchedulerOutput
            # IMPORTANT: vLLM V1 orders the hidden state tensor as [CACHED, NEW]
            if hasattr(model_input, "scheduled_new_reqs") and hasattr(model_input, "scheduled_cached_reqs"):
                request_ids = []
                seq_lens = []

                # Process CACHED requests FIRST (they appear first in the tensor)
                cached_reqs_val = model_input.scheduled_cached_reqs
                if cached_reqs_val and hasattr(cached_reqs_val, "req_ids"):
                    cached = cached_reqs_val
                    if cached.num_reqs > 0 and cached.req_ids:
                        request_ids.extend(cached.req_ids)
                        seq_lens.extend([1] * len(cached.req_ids))

                # Process NEW requests SECOND (they appear after cached in the tensor)
                new_reqs_val = model_input.scheduled_new_reqs
                if new_reqs_val:
                    new_reqs = new_reqs_val
                    if not isinstance(new_reqs, list):
                        new_reqs = [new_reqs]

                    for req in new_reqs:
                        if hasattr(req, "req_id"):
                            request_ids.append(req.req_id)
                        if hasattr(req, "prompt_token_ids"):
                            seq_lens.append(len(req.prompt_token_ids))

            if request_ids:
                # Store metadata for this step
                current_step = state.global_step
                state.step_metadata[current_step] = {
                    "request_ids": request_ids,
                    "seq_lens": seq_lens,
                    "step": current_step,
                }
                state.global_step += 1

                # Clean up old metadata (keep last 1000 steps)
                if len(state.step_metadata) > 1000:
                    old_steps = sorted(state.step_metadata.keys())[:-1000]
                    for step in old_steps:
                        state.step_metadata.pop(step, None)

        except (AttributeError, TypeError, KeyError, IndexError) as e:
            logger.warning(
                f"Failed to extract metadata from model_input: {type(e).__name__}: {e}",
                exc_info=True
            )
        except Exception as e:
            logger.error(
                f"Unexpected error in metadata extraction: {type(e).__name__}: {e}",
                exc_info=True
            )
            raise

        # Call original execute_model
        return original_execute(model_input, *args, **kwargs)

    model_runner.execute_model = patched_execute_model
    model_runner._original_execute_model = original_execute


def initialize_worker_state(
    worker: Any,
    layer_indices: Sequence[int] | None = None,
    shm_ttl_seconds: int = 600,
    shm_max_gb: float = 128.0,
    decode_buffer_size: int = 128,
) -> dict[str, Any]:
    """Install steering patch on worker after model load.

    This is called via RPC on each worker to initialize steering infrastructure.

    Parameters
    ----------
    worker :
        vLLM worker instance with model_runner attribute.
    layer_indices :
        Optional list of layer indices to initialize (unused, for compatibility).
    shm_ttl_seconds :
        TTL for shared memory segments in seconds.
    shm_max_gb :
        Maximum shared memory usage in GB.
    decode_buffer_size :
        Buffer size for decode token batching.

    Returns
    -------
    dict[str, Any]
        Metadata about the initialized worker state.
    """
    ensure_layer_patch_installed()
    model = worker.model_runner.model
    layers = _resolve_layers(model)

    # Handle multimodal models where config is nested
    config = model.config
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        hidden_size = config.text_config.hidden_size
    elif hasattr(config, "hidden_size"):
        hidden_size = config.hidden_size
    else:
        raise RuntimeError(f"Could not resolve hidden_size from config of type {type(config)}")

    first_param = next(model.parameters(), None)
    if first_param is None:
        raise RuntimeError("Model has no parameters to infer device/dtype.")
    device = first_param.device
    dtype = first_param.dtype

    state = _SteeringState(
        hidden_size=int(hidden_size),
        dtype=dtype,
        device=device,
        shm_ttl_seconds=shm_ttl_seconds,
        shm_max_gb=shm_max_gb,
        decode_buffer_size=decode_buffer_size,
    )

    # Create CUDA stream for async transfers if available
    if device.type == "cuda":
        state.transfer_stream = torch.cuda.Stream(device=device)

    worker._steerllm_steering = state
    set_worker_state(state)

    # Start shared memory cleanup thread
    stop_event = threading.Event()
    cleanup_thread = threading.Thread(
        target=_cleanup_stale_shared_memory,
        args=(state, stop_event),
        daemon=True,
        name="steerllm-shm-cleanup",
    )
    cleanup_thread.start()
    state.shm_cleanup_thread = (cleanup_thread, stop_event)

    # Register atexit handler to stop thread on shutdown
    def _stop_cleanup_thread():
        if state.shm_cleanup_thread:
            thread, event = state.shm_cleanup_thread
            event.set()
            thread.join(timeout=5.0)
            logger.info("Shared memory cleanup thread stopped (atexit)")

    atexit.register(_stop_cleanup_thread)
    logger.info("Started shared memory cleanup thread")

    # Patch model runner to capture batch metadata
    _patch_model_runner(worker, state)

    # Wrap model if not already wrapped
    if not isinstance(worker.model_runner.model, _SteeredModelWrapper):
        worker.model_runner.model = _SteeredModelWrapper(model, state)

    # Attach state and layer indices to all layers
    layers = _resolve_layers(worker.model_runner.model)
    for layer_idx, layer in enumerate(layers):
        setattr(layer, "_steerllm_state", state)
        setattr(layer, "_steerllm_layer_index", layer_idx)

    return {
        "hidden_size": hidden_size,
        "layer_count": len(layers),
        "dtype": str(dtype),
        "device": str(device),
    }


# =============================================================================
# RPC Handlers
# =============================================================================

_RPC_HANDLERS: dict[str, Callable[..., Any]] = {}
_RPC_GATEWAY_INSTALLED = False

STEERING_RPC_METHOD = "_steerllm_steering_rpc"
STEERING_WORKER_EXTENSION = "steerllm.backends.vllm.runtime.SteeringWorkerExtension"


def _register_rpc(name: str, func: Callable[..., Any]) -> None:
    """Register an RPC handler."""
    _RPC_HANDLERS[name] = func


def rpc_args(op: str, *args: Any) -> tuple[Any, ...]:
    """Create RPC arguments tuple."""
    return (op, *args)


class SteeringWorkerExtension:
    """Worker mixin providing steering RPC handling."""

    def _steerllm_steering_rpc(self, op: str, *args: Any, **kwargs: Any) -> Any:
        handler = _RPC_HANDLERS.get(op)
        if handler is None:
            raise ValueError(f"Unknown steerllm steering RPC: {op}")
        return handler(self, *args, **kwargs)


def ensure_collective_rpc_gateway_installed() -> None:
    """Install RPC gateway on vLLM WorkerWrapperBase."""
    global _RPC_GATEWAY_INSTALLED
    if _RPC_GATEWAY_INSTALLED:
        return

    try:
        from vllm.worker.worker_base import WorkerWrapperBase
    except ImportError:
        return

    if hasattr(WorkerWrapperBase, STEERING_RPC_METHOD):
        _RPC_GATEWAY_INSTALLED = True
        return

    def _dispatch(self: Any, op: str, *args: Any, **kwargs: Any) -> Any:
        handler = _RPC_HANDLERS.get(op)
        if handler is None:
            raise ValueError(f"Unknown steerllm steering RPC: {op}")
        target = getattr(self, "worker", self)
        return handler(target, *args, **kwargs)

    setattr(WorkerWrapperBase, STEERING_RPC_METHOD, _dispatch)
    _RPC_GATEWAY_INSTALLED = True


# =============================================================================
# RPC Handler Implementations
# =============================================================================

def _rpc_register_capture(
    worker: Any,
    request_id: str,
    layer_indices: list[int],
) -> bool:
    """Register capture for a request."""
    state = get_worker_state()
    if state is None:
        return False

    state.active_capture_requests[request_id] = set(layer_indices)
    state.request_captures[request_id] = {}
    return True


_register_rpc("register_capture", _rpc_register_capture)


def _rpc_unregister_capture(
    worker: Any,
    request_id: str,
) -> bool:
    """Unregister capture for a request."""
    state = get_worker_state()
    if state is None:
        return False

    state.active_capture_requests.pop(request_id, None)
    state.request_captures.pop(request_id, None)
    return True


_register_rpc("unregister_capture", _rpc_unregister_capture)


def _rpc_register_steering_spec(
    worker: Any,
    request_id: str,
    spec_data: dict[str, Any],
) -> bool:
    """Register a steering spec for a request.

    spec_data is a dict with:
    - layers: dict[int, layer_spec]
    - Each layer_spec has operations: list of {type, vector, params}
      where vector is a shm tensor payload (shm_name, shape, dtype, nbytes)
    """
    state = get_worker_state()
    if state is None:
        return False

    # Deserialize spec
    layers = {}
    for layer_idx_str, layer_data in spec_data.get("layers", {}).items():
        layer_idx = int(layer_idx_str)
        ops = []
        for op_data in layer_data.get("operations", []):
            op_type = op_data["type"]
            vector_payload = op_data["vector"]
            params = op_data.get("params")

            vec = deserialize_tensor(
                vector_payload,
                device=state.device,
                dtype=state.dtype,
            )

            # Scale vector for additive steering
            if op_type == "add" and params is not None:
                vec = vec * params  # params is scale for add
                params = None  # Clear params after applying

            # Intern vector for content-based deduplication
            # This enables gather/scatter optimization across RPC
            vec = _intern_vector(state, vec)

            ops.append((op_type, vec, params))

        # Create a simple namespace for the layer spec
        class LayerSpec:
            pass
        layer_spec = LayerSpec()
        layer_spec.operations = ops
        layers[layer_idx] = layer_spec

    # Create full spec
    class Spec:
        pass
    spec = Spec()
    spec.layers = layers

    state.request_steering_specs[request_id] = spec
    return True


_register_rpc("register_steering_spec", _rpc_register_steering_spec)


def _rpc_unregister_steering_spec(
    worker: Any,
    request_id: str,
) -> bool:
    """Unregister steering spec for a request."""
    state = get_worker_state()
    if state is None:
        return False

    state.request_steering_specs.pop(request_id, None)
    return True


_register_rpc("unregister_steering_spec", _rpc_unregister_steering_spec)


def _rpc_fetch_captures(
    worker: Any,
    request_id: str,
) -> dict[str, Any] | None:
    """Fetch captures using zero-copy shared memory with batch sync.

    Uses _create_shared_tensor() with transfer_stream for direct GPU→shm DMA
    transfer, syncing the stream once after all copies are queued.
    """
    state = get_worker_state()
    if state is None:
        return None

    captures = state.request_captures.get(request_id)
    if captures is None:
        return None

    # Queue all GPU→shm copies on transfer_stream (non-blocking)
    result = {}
    for layer_idx, tensor in captures.items():
        result[str(layer_idx)] = _create_shared_tensor(
            tensor, state, stream=state.transfer_stream
        )

    # Sync transfer_stream ONCE after all copies queued
    if state.transfer_stream is not None:
        state.transfer_stream.synchronize()

    return result


_register_rpc("fetch_captures", _rpc_fetch_captures)


def _rpc_fetch_batch_captures(
    worker: Any,
    request_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Fetch captures for multiple requests in a single RPC call.

    Batches all GPU→shm transfers and syncs the transfer stream once at the end,
    reducing synchronization overhead when fetching many requests.

    Parameters
    ----------
    worker :
        vLLM worker instance (unused but required by RPC signature).
    request_ids :
        List of request IDs to fetch captures for.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from request_id to layer captures. Each layer capture is
        serialized tensor metadata for shared memory reconstruction.
    """
    state = get_worker_state()
    if state is None:
        return {}

    result: dict[str, dict[str, Any]] = {}

    # Queue all GPU→shm copies on transfer_stream (non-blocking)
    for request_id in request_ids:
        captures = state.request_captures.get(request_id)
        if captures is None:
            continue

        request_result: dict[str, Any] = {}
        for layer_idx, tensor in captures.items():
            request_result[str(layer_idx)] = _create_shared_tensor(
                tensor, state, stream=state.transfer_stream
            )
        result[request_id] = request_result

    # Sync transfer_stream ONCE after all copies queued
    if state.transfer_stream is not None:
        state.transfer_stream.synchronize()

    return result


_register_rpc("fetch_batch_captures", _rpc_fetch_batch_captures)


def _rpc_release_shared_memory(
    worker: Any,
    shm_names: list[str],
) -> bool:
    """Release shared memory segments."""
    state = get_worker_state()
    if state is None:
        return False

    with state.shm_lock:
        for name in shm_names:
            entry = state.active_shared_memory.pop(name, None)
            if entry is not None:
                shm, _ = entry
                try:
                    shm.close()
                    shm.unlink()
                except Exception as e:
                    logger.debug(f"Error releasing shared memory {name}: {e}")

    return True


_register_rpc("release_shared_memory", _rpc_release_shared_memory)


def _rpc_set_step_metadata(
    worker: Any,
    step: int,
    metadata: dict[str, Any],
) -> bool:
    """Set batch metadata for a step."""
    state = get_worker_state()
    if state is None:
        return False

    state.step_metadata[step] = metadata
    state.global_step = step + 1

    # Cleanup old metadata (keep last 10 steps)
    if len(state.step_metadata) > 10:
        old_steps = sorted(state.step_metadata.keys())[:-10]
        for s in old_steps:
            state.step_metadata.pop(s, None)

    return True


_register_rpc("set_step_metadata", _rpc_set_step_metadata)


# Register initialize_worker_state as an RPC handler
_register_rpc("initialize_worker_state", initialize_worker_state)


# Install RPC gateway at module load time
ensure_collective_rpc_gateway_installed()
