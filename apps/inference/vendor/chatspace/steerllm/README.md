# steerllm

Multi-backend LLM steering library. Apply steering vectors, projection caps, and ablations to large language models during inference.

## Installation

```bash
# Core only (just specs and utilities)
pip install steerllm

# vLLM backend for production inference
pip install steerllm[vllm]

# HuggingFace for training
pip install steerllm[huggingface]

# Everything
pip install steerllm[all]
```

## Quick Start

### Basic Steering (vLLM)

```python
import asyncio
from steerllm import VLLMSteeringModel, SteeringSpec
import torch

async def main():
    model = VLLMSteeringModel("Qwen/Qwen3-0.6B")

    # Create steering spec
    direction = torch.randn(model.hidden_size)
    steering = SteeringSpec.simple_add(layer=5, vector=direction, scale=2.0)

    # Generate with steering
    texts, _ = await model.generate(
        ["What is consciousness?"],
        max_tokens=100,
        steering_spec=steering,
    )
    print(texts[0])

asyncio.run(main())
```

### Activation Capture

```python
async def capture_example():
    model = VLLMSteeringModel("Qwen/Qwen3-0.6B")

    texts, handles = await model.generate(
        ["The meaning of life is"],
        max_tokens=50,
        capture_layers=[5, 10, 15],
    )

    async with handles[0] as handle:
        await handle.fetch()
        layer_5 = handle.captures[5][0]["hidden"]
        print(f"Layer 5 shape: {layer_5.shape}")
```

### Multi-Operation Steering

Apply multiple operations per layer: additions, projection caps, and ablations.

```python
from steerllm import (
    SteeringSpec, LayerSteeringSpec,
    AddSpec, ProjectionCapSpec, AblationSpec
)

# Complex steering: add vector, cap projection, ablate direction
spec = SteeringSpec(layers={
    5: LayerSteeringSpec(operations=[
        AddSpec(vector=steering_direction, scale=2.0),
        ProjectionCapSpec(vector=cap_direction, min=-1.0, max=1.0),
    ]),
    10: LayerSteeringSpec(operations=[
        AblationSpec(vector=ablate_direction, scale=0.0),  # Full removal
    ]),
})

texts, _ = await model.generate(prompts, steering_spec=spec)
```

### Training (HuggingFace)

```python
from steerllm.backends.huggingface import HFSteeringModel
from torch.optim import Adam

model = HFSteeringModel("Qwen/Qwen3-0.6B", target_layers=[5])
optimizer = Adam(model.get_trainable_parameters(), lr=1e-4)

for batch in dataloader:
    outputs = model(**batch)
    loss = compute_loss(outputs)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

model.save_steering("./checkpoint")
```

## Features

- **Per-request steering**: Different requests can use different steering configs
- **Composable operations**: Add + Cap + Ablation in any order per layer
- **Zero-copy capture**: Shared memory IPC for fast activation transfer (vLLM)
- **Multi-backend**: vLLM for production, HuggingFace for training
- **Tensor parallelism**: Works with vLLM's TP support
- **Async-first API**: Native async support with sync wrappers

## Steering Operations

### AddSpec
Additive steering: `hidden += vector * scale`

```python
# Create from raw vector (will be normalized, scale captures magnitude)
spec = AddSpec.from_unnormalized(raw_vector, scale=1.0)

# Or with explicit unit vector and scale
spec = AddSpec(vector=unit_vector, scale=2.0)
```

### ProjectionCapSpec
Clamp projection onto a direction: bounds the component along a direction

```python
# Cap between -1 and 1
spec = ProjectionCapSpec(vector=direction, min=-1.0, max=1.0)

# One-sided caps
spec = ProjectionCapSpec(vector=direction, max=0.5)  # Upper only
spec = ProjectionCapSpec(vector=direction, min=0.0)  # Lower only
```

### AblationSpec
Scale component along direction: `scale=0` removes it entirely

```python
# Full ablation (remove component)
spec = AblationSpec(vector=direction, scale=0.0)

# Partial ablation (reduce by half)
spec = AblationSpec(vector=direction, scale=0.5)

# Amplification (double the component)
spec = AblationSpec(vector=direction, scale=2.0)
```

## Convenience Constructors

```python
# Simple single-layer steering
spec = SteeringSpec.simple_add(layer=5, vector=v, scale=1.0)
spec = SteeringSpec.simple_cap(layer=10, vector=v, min=-0.5, max=0.5)
spec = SteeringSpec.simple_ablation(layer=15, vector=v, scale=0.0)
```

## Chat-Style Generation

```python
messages = [
    {"role": "user", "content": "What is the capital of France?"}
]

responses, handles = await model.chat(
    messages,
    max_tokens=100,
    steering_spec=steering,
    capture_layers=[5],
)

print(responses[0].generated)  # Just the generated part
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STEERLLM_SHM_TTL` | 600 | Shared memory TTL in seconds |
| `STEERLLM_MAX_SHM_GB` | 128 | Maximum shared memory usage (GB) |
| `STEERLLM_CAPTURE_METADATA` | 1 | Enable batch metadata capture |

## Architecture

steerllm uses a modular architecture with:

- **Core specs** (`steerllm.core.specs`): Pure dataclass definitions, no backend deps
- **vLLM backend** (`steerllm.backends.vllm`): Production inference with decoder patching
- **HuggingFace backend** (`steerllm.backends.huggingface`): Training and validation

The vLLM backend patches decoder layers (Qwen, Llama, Gemma) to intercept forward passes
and apply steering operations. Per-request steering allows heterogeneous batching where
different requests in the same batch use different configurations.

## License

MIT
