"""Training utilities for steerllm.

Provides high-level APIs for training steering vectors.

Note: Training utilities are under development for v0.2.
For now, use HFSteeringModel directly for training.
"""

# Training utilities will be added in v0.2
# For now, use HFSteeringModel directly:
#
# from steerllm.backends.huggingface import HFSteeringModel
# model = HFSteeringModel("Qwen/Qwen3-0.6B", target_layers=[5])
# optimizer = Adam(model.get_trainable_parameters(), lr=1e-4)
