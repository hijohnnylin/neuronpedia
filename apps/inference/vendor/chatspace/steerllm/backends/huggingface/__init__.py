"""HuggingFace backend for steerllm.

Provides trainable steering via PyTorch forward hooks on transformers models.
Supports training steering vectors with gradient descent.

Example
-------
>>> from steerllm.backends.huggingface import HFSteeringModel
>>> model = HFSteeringModel("Qwen/Qwen3-0.6B", target_layers=[5])
>>> optimizer = Adam(model.get_trainable_parameters(), lr=1e-4)
"""

from steerllm.backends.huggingface.model import HFSteeringModel
from steerllm.backends.huggingface.hooks import ResidualHook

__all__ = ["HFSteeringModel", "ResidualHook"]
