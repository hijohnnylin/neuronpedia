"""Utilities for coordinating steering vectors with vLLM workers.

This module re-exports from steerllm.backends.vllm.runtime for backward compatibility.
The original chatspace implementation has been migrated to steerllm.
"""

# Re-export runtime module from steerllm
from steerllm.backends.vllm import runtime

# Re-export commonly used symbols for code that imports them directly
from steerllm.backends.vllm.runtime import (
    ensure_layer_patch_installed,
    ensure_collective_rpc_gateway_installed,
    STEERING_WORKER_EXTENSION,
    STEERING_RPC_METHOD,
    serialize_tensor,
    deserialize_tensor,
    rpc_args,
)

__all__ = [
    "runtime",
    "ensure_layer_patch_installed",
    "ensure_collective_rpc_gateway_installed",
    "STEERING_WORKER_EXTENSION",
    "STEERING_RPC_METHOD",
    "serialize_tensor",
    "deserialize_tensor",
    "rpc_args",
]
