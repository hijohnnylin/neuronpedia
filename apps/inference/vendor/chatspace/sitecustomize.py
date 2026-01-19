"""Project-wide interpreter customization for vLLM steering patches.

Any Python process launched with this repository on its PYTHONPATH will import
this module during interpreter startup, ensuring Qwen decoder layers are
patched before vLLM performs CUDA-graph capture.
"""

# Import patching functions but DON'T call them yet.
# They will be called explicitly by VLLMSteerModel when needed.
# Calling them here is too early - vLLM modules aren't imported yet.
from chatspace.vllm_steering.runtime import (  # noqa: F401
    ensure_collective_rpc_gateway_installed,
    ensure_layer_patch_installed,
)
