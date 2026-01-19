"""vLLM backend for steerllm.

High-performance steering with per-request configuration, zero-copy shared
memory activation capture, and tensor parallelism support.

Example
-------
>>> from steerllm.backends.vllm import VLLMSteeringModel
>>> model = VLLMSteeringModel("Qwen/Qwen3-0.6B")
>>> texts, handles = await model.generate(prompts, capture_layers=[5])
"""

from steerllm.backends.vllm.model import VLLMSteeringModel

__all__ = ["VLLMSteeringModel"]
