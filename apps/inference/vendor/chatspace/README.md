# chatspace

A **vLLM steering runtime** for language model activation capture and interpretability research.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Steering Methods](#steering-methods)
- [Activation Capture](#activation-capture)
- [Concurrency Model](#concurrency-model)
- [Advanced Features](#advanced-features)
- [Development](#development)

---

## Overview

**chatspace** provides a production-ready system for applying steering vectors and capturing activations from vLLM-hosted language models with support for:

- **Additive steering vectors**: Inject concept directions into layer activations
- **Projection capping**: Clamp hidden state components along specific directions
- **Component ablation**: Scale or suppress features for circuit analysis
- **Per-request activation capture**: Capture hidden states during generation for analysis
- **Concurrent generation**: Thread-safe steering updates with async readers-writer lock
- **Tensor parallelism**: Works transparently with vLLM's multi-GPU parallelism

> **Note**: This repo also contains some research code related to dataset embedding and persona subspace analysis (collaboration with [persona-subspace](https://github.com/lu-christina/persona-subspace)), but the vLLM steering runtime is the primary public-facing feature.

---

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/chatspace.git
cd chatspace

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

**Requirements:**
- Python 3.10+
- PyTorch 2.0+
- vLLM 0.6.0+ (for steering features)
- CUDA-capable GPU (recommended)

---

## Quick Start

```python
import torch
import asyncio
from vllm import SamplingParams
from chatspace.generation.vllm_steer_model import (
    VLLMSteerModel,
    VLLMSteeringConfig,
)

async def main():
    # Initialize model
    cfg = VLLMSteeringConfig(
        model_name="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )
    model = VLLMSteerModel(cfg, bootstrap_layers=(2, 4, 6))

    # Generate baseline
    sampling = SamplingParams(temperature=0.0, max_tokens=64)
    baseline = await model.generate(
        "Question: What is the capital of France? Answer:",
        sampling
    )
    print("Baseline:", baseline[0])

    # Apply steering vector to layer 4
    steering_vector = torch.randn(model.hidden_size) * 100.0
    await model.set_layer_vector(4, steering_vector)

    # Generate with steering
    steered = await model.generate(
        "Question: What is the capital of France? Answer:",
        sampling
    )
    print("Steered:", steered[0])

    # Clear steering
    await model.clear_all_vectors()

if __name__ == "__main__":
    asyncio.run(main())
```

**Run the smoke test:**
```bash
uv run python scripts/steering_smoke.py \
    --model-name "Qwen/Qwen3-0.6B" \
    --layer 2 \
    --scale 5000.0 \
    --max-tokens 32
```

---

## Core Concepts

#### 1. VLLMSteerModel

The main interface for steering and generation. Wraps vLLM's `AsyncLLMEngine` and provides:

- **Async-first API**: All generation and steering operations are async
- **Multi-layer steering**: Apply different steering operations to different transformer layers
- **Thread-safe**: Concurrent generation requests are safe; steering updates block during generation
- **Worker coordination**: Steering vectors are broadcast to all tensor-parallel workers via RPC

#### 2. Steering Specifications

Steering state is organized into layer-wise specifications:

```python
from chatspace.generation.vllm_steer_model import (
    SteeringSpec,
    LayerSteeringSpec,
    AddSpec,
    ProjectionCapSpec,
    AblationSpec,
)

# Create a steering spec
spec = SteeringSpec(layers={
    2: LayerSteeringSpec(
        add=AddSpec(vector=unit_vec, scale=50.0),
    ),
    4: LayerSteeringSpec(
        add=AddSpec(vector=unit_vec, scale=100.0),
        projection_cap=ProjectionCapSpec(
            vector=direction_vec,
            min=-10.0,
            max=10.0,
        ),
    ),
})

# Apply spec (all steering updates happen atomically)
await model.apply_steering_spec(spec)

# Export current steering state
current_spec = model.export_steering_spec()
```

#### 3. Eager Execution Requirement

**IMPORTANT**: vLLM steering requires `enforce_eager=True` (enabled by default). CUDA graph compilation skips the Python-side steering hooks.

```python
# This is the default and recommended:
model = VLLMSteerModel(cfg)  # enforce_eager=True by default

# If you try to disable it, you'll get a warning:
model = VLLMSteerModel(cfg, enforce_eager=False)
# WARNING: vLLM steering requires enforce_eager=True; overriding user-supplied value.
```

---

## Steering Methods

#### Additive Steering

Add a fixed vector to layer activations:

```python
# Apply to specific layer
steering_vec = torch.randn(model.hidden_size) * 50.0
await model.set_layer_vector(layer_idx=4, vector=steering_vec)

# Or use the active layer (set via set_target_layer)
model.set_target_layer(4)
await model.set_vector(steering_vec)

# Clear steering for a layer
await model.clear_layer_vector(4)
```

**Use cases:**
- Concept steering (e.g., "make outputs more formal")
- Feature injection from trained steering vectors
- Behavioral modification (e.g., refusal prevention)

#### Projection Capping

Clamp the component of activations along a direction:

```python
import torch

# Define direction (will be normalized automatically)
direction = torch.randn(model.hidden_size)

# Cap projection to [-10, 10] range
await model.set_layer_projection_cap(
    layer_idx=6,
    vector=direction,
    min=-10.0,
    max=10.0,
)

# Remove cap
await model.clear_layer_projection_cap(6)
```

**Use cases:**
- Prevent extreme activations in specific directions
- Constrain steering vector effects
- Stabilize generation under strong steering

**Note**: Projection capping operates on the full hidden state (after adding steering vectors), computing `hidden @ direction` and clamping to `[min, max]`.

#### Component Ablation

Scale (amplify or suppress) activations along a direction:

```python
# Suppress component (scale < 1.0)
await model.set_layer_ablation(
    layer_idx=8,
    vector=direction,
    scale=0.1,  # Reduce to 10%
)

# Amplify component (scale > 1.0)
await model.set_layer_ablation(
    layer_idx=8,
    vector=direction,
    scale=2.0,  # Double the component
)

# Clear ablation
await model.clear_layer_ablation(8)
```

**Use cases:**
- Interpretability research (what happens when we remove a feature?)
- Circuit analysis (ablate specific features)
- Causal intervention experiments

---

## Activation Capture

Capture hidden states during generation for analysis, interpretability research, and debugging.

Both the `generate()` and `chat()` APIs support activation capture through the `capture_layers` parameter.

### Basic Usage

**Using `generate()` with raw prompts:**

```python
# Capture activations from layers 2, 4, 6
results, handles = await model.generate(
    ["What is 2+2?", "What is the capital of France?"],
    sampling,
    capture_layers=[2, 4, 6],
)

# Fetch captures (batched fetch is more efficient)
await model.fetch_captures_batch(handles)

# Access captures for each request
for i, handle in enumerate(handles):
    print(f"\nPrompt {i}: {results[i]}")

    for layer_idx in handle.layer_indices:
        # Each layer has a list (one per TP worker)
        captures = handle.captures[layer_idx]
        hidden = captures[0]["hidden"]  # [seq_len, hidden_size]
        print(f"  Layer {layer_idx}: {hidden.shape}")
```

**Using `chat()` with conversation messages:**

```python
# Capture activations from chat-style generation
messages = [
    {"role": "user", "content": "What is 2+2?"}
]

responses, handles = await model.chat(
    messages,
    sampling_params=sampling,
    capture_layers=[2, 4, 6],
)

# Fetch captures
await model.fetch_captures_batch(handles)

# Access captures
for layer_idx in handles[0].layer_indices:
    hidden = handles[0].captures[layer_idx][0]["hidden"]
    print(f"Layer {layer_idx}: {hidden.shape}")
```

**Batch chat with captures:**

```python
# Multiple conversations with capture
conversations = [
    [{"role": "user", "content": "What is 2+2?"}],
    [{"role": "user", "content": "Explain quantum computing."}],
    [{"role": "user", "content": "Write a haiku."}],
]

responses, handles = await model.chat(
    conversations,
    sampling_params=sampling,
    capture_layers=[4],
)

await model.fetch_captures_batch(handles)

for i, (response, handle) in enumerate(zip(responses, handles)):
    hidden = handle.captures[4][0]["hidden"]
    print(f"Conversation {i}: {hidden.shape[0]} tokens captured")
```

**Splitting activations by message:**

```python
# Multi-turn conversation with message-level analysis
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What is its population?"},
]

responses, handles = await model.chat(
    messages,
    sampling_params=sampling,
    capture_layers=[4],
)

await model.fetch_captures_batch(handles)
handle = handles[0]

# Access activations for each message separately
for i, boundary in enumerate(handle.message_boundaries):
    # Get activations for this specific message
    msg_acts = handle.get_message_activations(
        message_idx=i,
        layer_idx=4,
    )
    print(f"Message {i} ({boundary.role}): {msg_acts.shape}")
    print(f"  Content: {boundary.content[:50]}...")
    print(f"  Tokens: {boundary.num_tokens}")
    print(f"  Mean activation norm: {torch.norm(msg_acts, dim=-1).mean():.4f}")

# Get the last user message's activations including the generated response
last_user_idx = len(handle.message_boundaries) - 1
full_acts = handle.get_message_activations(
    message_idx=last_user_idx,
    layer_idx=4,
    include_generated=True,  # Include generated tokens
)
print(f"\nLast message + generated: {full_acts.shape}")
```

### Capture Format and Behavior

**Concatenated tensor format:**
- Returns a single tensor per layer containing all tokens processed
- Format: `[prefill_tokens + decode_tokens, hidden_size]`
- Both prefill (prompt) and decode (generated) tokens are concatenated in sequence order

**Length calculation:**
- Captured length = `prompt_tokens + (generated_tokens - 1)`
- The final generated token is sampled but never processed through the model
- Example: 15-token prompt generating 10 tokens → 24 captured activations (15 + 9)

**Isolation and thread safety:**
- Concurrent requests with capture enabled maintain proper per-request isolation
- Each request's captures are tracked independently via request IDs
- Captures are accumulated during generation and fetched after completion

### Slicing Captured Activations

```python
# Generate with capture
results, handles = await model.generate(
    prompt,
    sampling_params=SamplingParams(max_tokens=50),
    capture_layers=[4],
)
await model.fetch_captures_batch(handles)

# Extract the full concatenated tensor
full_hidden = handles[0].captures[4][0]["hidden"]  # [seq_len, hidden_size]

# Tokenize to get prompt length
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model.model_name)
prompt_tokens = tokenizer(prompt, return_tensors="pt")
prompt_len = prompt_tokens.input_ids.shape[1]

# Slice into prefill and decode phases
prefill_acts = full_hidden[:prompt_len]  # Prompt processing
decode_acts = full_hidden[prompt_len:]   # Generated tokens (excluding final)

print(f"Prefill shape: {prefill_acts.shape}")  # [prompt_len, hidden_size]
print(f"Decode shape: {decode_acts.shape}")    # [generated_tokens - 1, hidden_size]

# Analyze specific tokens
first_generated = decode_acts[0]  # First generated token's activations
last_captured = decode_acts[-1]   # Last captured token (second-to-last generated)
```

### Comparing Messages in a Conversation

```python
import torch.nn.functional as F

# Capture a multi-turn conversation
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."},
    {"role": "user", "content": "What is 10*10?"},
]

responses, handles = await model.chat(
    messages,
    sampling_params=sampling,
    capture_layers=[4, 8, 12],
)

await model.fetch_captures_batch(handles)
handle = handles[0]

# Compare activation patterns between the two user questions
user_msg_indices = [i for i, b in enumerate(handle.message_boundaries) if b.role == "user"]

for layer in [4, 8, 12]:
    acts_1 = handle.get_message_activations(user_msg_indices[0], layer)
    acts_2 = handle.get_message_activations(user_msg_indices[1], layer)

    # Compute mean representations
    mean_1 = acts_1.mean(dim=0)
    mean_2 = acts_2.mean(dim=0)

    # Cosine similarity between questions
    cos_sim = F.cosine_similarity(mean_1.unsqueeze(0), mean_2.unsqueeze(0)).item()
    print(f"Layer {layer} - similarity between user questions: {cos_sim:.4f}")

# Analyze how system prompt affects user message processing
system_acts = handle.get_message_activations(0, layer_idx=4)  # System message
first_user_acts = handle.get_message_activations(1, layer_idx=4)  # First user msg

print(f"\nSystem message tokens: {system_acts.shape[0]}")
print(f"First user message tokens: {first_user_acts.shape[0]}")
print(f"System mean norm: {torch.norm(system_acts, dim=-1).mean():.4f}")
print(f"User mean norm: {torch.norm(first_user_acts, dim=-1).mean():.4f}")
```

### Analyzing Steering Effects

**Using `generate()`:**

```python
import torch.nn.functional as F

# Capture baseline (no steering)
baseline_results, baseline_handles = await model.generate(
    prompt,
    sampling,
    capture_layers=[4],
)
await model.fetch_captures_batch(baseline_handles)
baseline_acts = baseline_handles[0].captures[4][0]["hidden"]

# Apply steering and capture again
steering_vec = torch.randn(model.hidden_size) * 50.0
await model.set_layer_vector(4, steering_vec)

steered_results, steered_handles = await model.generate(
    prompt,
    sampling,
    capture_layers=[4],
)
await model.fetch_captures_batch(steered_handles)
steered_acts = steered_handles[0].captures[4][0]["hidden"]

# Compare activations
delta = steered_acts - baseline_acts
print(f"Mean activation change: {delta.mean().item():.4f}")
print(f"Max activation change: {delta.abs().max().item():.4f}")

# Compute cosine similarity for each token
for i in range(min(5, baseline_acts.shape[0])):
    cos_sim = F.cosine_similarity(
        baseline_acts[i].unsqueeze(0),
        steered_acts[i].unsqueeze(0),
        dim=-1
    ).item()
    print(f"Token {i} cosine similarity: {cos_sim:.6f}")
```

**Using `chat()`:**

```python
import torch.nn.functional as F

messages = [
    {"role": "user", "content": "Explain the concept of recursion."}
]

# Capture baseline (no steering)
baseline_resp, baseline_handles = await model.chat(
    messages,
    sampling_params=sampling,
    capture_layers=[4],
)
await model.fetch_captures_batch(baseline_handles)
baseline_acts = baseline_handles[0].captures[4][0]["hidden"]

# Apply steering
steering_vec = torch.randn(model.hidden_size) * 50.0
await model.set_layer_vector(4, steering_vec)

# Capture with steering
steered_resp, steered_handles = await model.chat(
    messages,
    sampling_params=sampling,
    capture_layers=[4],
)
await model.fetch_captures_batch(steered_handles)
steered_acts = steered_handles[0].captures[4][0]["hidden"]

# Compare activations
delta = steered_acts - baseline_acts
print(f"Baseline response: {baseline_resp[0][:100]}...")
print(f"Steered response: {steered_resp[0][:100]}...")
print(f"Mean activation Δ: {delta.mean().item():.4f}")
print(f"Max activation Δ: {delta.abs().max().item():.4f}")
```

### Concurrent Capture (Isolation)

**Using `generate()` with concurrent tasks:**

```python
import asyncio

async def analyze_multiple_prompts():
    """Capture activations for multiple prompts concurrently."""
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing.",
        "Write a haiku about programming.",
    ]

    # Create concurrent tasks
    tasks = [
        model.generate(
            [prompt],
            sampling,
            capture_layers=[2, 4, 6],
        )
        for prompt in prompts
    ]

    # Run all captures concurrently
    results = await asyncio.gather(*tasks)

    # Unpack results (each is a (texts, handles) tuple)
    all_handles = [handles[0] for texts, handles in results]

    # Fetch all captures in one batch RPC
    await model.fetch_captures_batch(all_handles)

    # Analyze each prompt's captures
    for i, handle in enumerate(all_handles):
        print(f"\nPrompt {i}: {prompts[i][:50]}...")
        for layer in [2, 4, 6]:
            hidden = handle.captures[layer][0]["hidden"]
            norm = torch.norm(hidden, dim=-1).mean().item()
            print(f"  Layer {layer}: mean norm = {norm:.4f}")

# Run concurrent analysis
await analyze_multiple_prompts()
```

**Using `chat()` with batch conversations:**

```python
# Capture multiple conversations in a single batch
conversations = [
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "Explain quantum computing."}],
    [{"role": "user", "content": "Write a haiku about programming."}],
]

# Single call handles all conversations
responses, handles = await model.chat(
    conversations,
    sampling_params=sampling,
    capture_layers=[2, 4, 6],
)

# Fetch all captures at once
await model.fetch_captures_batch(handles)

# Analyze each conversation's captures
for i, (response, handle) in enumerate(zip(responses, handles)):
    print(f"\nConversation {i}:")
    print(f"Response: {response[:50]}...")
    for layer in [2, 4, 6]:
        hidden = handle.captures[layer][0]["hidden"]
        norm = torch.norm(hidden, dim=-1).mean().item()
        print(f"  Layer {layer}: mean norm = {norm:.4f}")
```

### Multi-Layer Analysis

```python
# Capture all layers for deep analysis
all_layers = list(range(model.layer_count))
results, handles = await model.generate(
    prompt,
    sampling,
    capture_layers=all_layers,
)
await model.fetch_captures_batch(handles)

# Compute activation norms across layers
layer_norms = {}
for layer_idx in all_layers:
    hidden = handles[0].captures[layer_idx][0]["hidden"]
    # Average norm across all tokens and hidden dimensions
    layer_norms[layer_idx] = torch.norm(hidden, dim=-1).mean().item()

# Plot layer-wise activation magnitudes
import matplotlib.pyplot as plt
plt.plot(list(layer_norms.keys()), list(layer_norms.values()))
plt.xlabel("Layer")
plt.ylabel("Mean Activation Norm")
plt.title("Activation Magnitude Across Layers")
plt.show()
```

### Validating Against HuggingFace

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load HuggingFace model for ground truth
hf_model = AutoModelForCausalLM.from_pretrained(
    model.model_name,
    torch_dtype=torch.float32,
    device_map="cuda",
)
hf_tokenizer = AutoTokenizer.from_pretrained(model.model_name)

# Generate with vLLM and capture
vllm_results, vllm_handles = await model.generate(
    prompt,
    sampling,
    capture_layers=[4],
)
await model.fetch_captures_batch(vllm_handles)
vllm_acts = vllm_handles[0].captures[4][0]["hidden"]

# Generate same sequence with HuggingFace and capture
full_text = prompt + vllm_results[0]
inputs = hf_tokenizer(full_text, return_tensors="pt").to("cuda")

hf_captures = {}
def capture_hook(module, args, output):
    hidden = output[0] if isinstance(output, tuple) else output
    hf_captures[4] = hidden.detach().cpu()

handle = hf_model.model.layers[4].register_forward_hook(capture_hook)
with torch.no_grad():
    hf_model(**inputs)
handle.remove()

hf_acts = hf_captures[4].squeeze(0)  # [seq_len, hidden_size]

# Compare (should be very similar if using same precision)
cos_sim = F.cosine_similarity(
    vllm_acts.flatten().unsqueeze(0),
    hf_acts.flatten().unsqueeze(0),
    dim=-1
).item()
mae = torch.mean(torch.abs(vllm_acts - hf_acts)).item()

print(f"Cosine similarity: {cos_sim:.6f}")  # Should be ~1.0
print(f"Mean absolute error: {mae:.6f}")     # Should be very small
```

---

## Concurrency Model

#### Readers-Writer Lock (AsyncRWLock)

`VLLMSteerModel` uses an async readers-writer lock to coordinate operations:

- **Read operations (concurrent)**: Multiple `generate()` calls can run simultaneously
- **Write operations (exclusive)**: Steering updates block until all in-flight requests complete

**Read operations** (acquire read lock):
- `generate()`
- `chat()`

**Write operations** (acquire write lock):
- `set_layer_vector()`, `set_vector()`
- `set_layer_projection_cap()`, `clear_layer_projection_cap()`
- `set_layer_ablation()`, `clear_layer_ablation()`
- `apply_steering_spec()`, `push_steering_spec()`, `pop_steering_spec()`
- `clear_layer_vector()`, `clear_all_vectors()`

#### Concurrent Generation Example

```python
import asyncio

async def generate_many(model, prompts, sampling):
    """Run multiple concurrent generation requests."""
    tasks = [
        model.generate(prompt, sampling)
        for prompt in prompts
    ]
    results = await asyncio.gather(*tasks)
    return [r[0] for r in results]

# This is safe and performant:
prompts = [f"Prompt {i}" for i in range(10)]
results = await generate_many(model, prompts, sampling)
```

**Important**: Steering changes during concurrent generation will wait for all in-flight requests to complete:

```python
async def concurrent_steer_test():
    # Start long generation
    gen_task = asyncio.create_task(
        model.generate("Write a long story...", sampling)
    )

    # Try to update steering (will block until generation completes)
    await model.set_layer_vector(4, new_steering_vec)

    result = await gen_task
    # Steering update applied AFTER generation completed
```

---

## Advanced Features

#### Steering Context Manager

Temporarily apply steering and restore previous state:

```python
# Save current steering state
spec = SteeringSpec(layers={
    4: LayerSteeringSpec(add=AddSpec(vector=vec, scale=100.0))
})

async with model.steering(spec):
    # Steering active within this block
    results = await model.generate(prompts, sampling)
    # ...

# Previous steering automatically restored after exiting block
```

#### Steering Stack

Push and pop steering configurations:

```python
# Save baseline
await model.push_steering_spec(baseline_spec)

# Apply intervention
await model.apply_steering_spec(intervention_spec)
results_1 = await model.generate(prompts, sampling)

# Try different steering
await model.apply_steering_spec(alternative_spec)
results_2 = await model.generate(prompts, sampling)

# Restore baseline
await model.pop_steering_spec()
results_baseline = await model.generate(prompts, sampling)
```

#### Chat-style Generation

The `chat()` method provides a convenient interface for OpenAI-style conversation generation with automatic chat template formatting.

**Basic usage:**

```python
from vllm import SamplingParams

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

sampling = SamplingParams(temperature=0.7, max_tokens=128)
responses = await model.chat(
    messages,
    sampling_params=sampling,
)

# Responses are ChatResponse objects with .prefill and .generated attributes
response = responses[0]
print(response.full_text())  # Get complete response text
print(f"Generated: {response.generated}")  # Just the generated part
print(f"Has prefill: {response.has_prefill}")  # False unless using continue_final_message
```

**Batched conversations:**

```python
# Multiple conversations at once
conversations = [
    [
        {"role": "user", "content": "What is 2+2?"},
    ],
    [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "Explain the Pythagorean theorem."},
    ],
]

responses = await model.chat(
    conversations,  # list[list[dict]] for batch
    temperature=0.7,
    max_tokens=256,
)
# Returns list[str] with one response per conversation
for i, response in enumerate(responses):
    print(f"Conversation {i}: {response}")
```

**Using sampling keyword arguments:**

```python
# Can pass sampling params directly as kwargs
responses = await model.chat(
    messages,
    temperature=0.8,
    top_p=0.95,
    max_tokens=100,
    # sampling_params=...  # OR provide SamplingParams object
)
```

**Custom chat template options:**

```python
responses = await model.chat(
    messages,
    sampling_params=sampling,
    chat_options={
        "add_generation_prompt": True,
        "chat_template": "custom_template",
        # Other tokenizer chat template options
    },
)
```

**Assistant response prefilling:**

```python
# Guide the model's response format by prefilling assistant content
prefill = '{"explanation": "'
messages = [
    {"role": "user", "content": "Explain quantum computing in JSON format"},
    {"role": "assistant", "content": prefill}  # Partial response
]

responses = await model.chat(
    messages,
    sampling_params=sampling,
    chat_options={
        "add_generation_prompt": False,  # Required when continuing
        "continue_final_message": True,   # Enable prefill mode
    },
)

# ChatResponse objects separate prefill from generated text
response = responses[0]
print(f"Prefill: {response.prefill}")  # '{"explanation": "'
print(f"Generated: {response.generated}")  # 'Quantum computing uses quantum bits..."}'
print(f"Full text: {response.full_text()}")  # Complete response

# Build conversation history easily
messages.append(response.to_message())  # Adds {"role": "assistant", "content": "..."}

# Force reasoning blocks (for hybrid models like Qwen)
reasoning_prefill = "<think>\n"
messages_with_reasoning = [
    {"role": "user", "content": "Solve this problem step by step"},
    {"role": "assistant", "content": reasoning_prefill}  # Start reasoning block
]

reasoning_responses = await model.chat(
    messages_with_reasoning,
    sampling_params=sampling,
    chat_options={
        "add_generation_prompt": False,
        "continue_final_message": True,
    },
)

# Access the complete reasoning output
print(reasoning_responses[0].full_text())  # "<think>\n...reasoning..."
```

**Accessing token-level details:**

```python
# Get full RequestOutput objects with token IDs and logprobs
outputs = await model.chat(
    messages,
    sampling_params=sampling,
    raw_output=True,
)

for output in outputs:
    # output is a RequestOutput object
    text = output.outputs[0].text
    token_ids = output.outputs[0].token_ids
    logprobs = output.outputs[0].logprobs
    print(f"Generated {len(token_ids)} tokens")
```

**Combining chat with steering:**

```python
# Apply steering and use chat API
await model.set_layer_vector(4, steering_vec)

messages = [
    {"role": "user", "content": "Write a formal email."},
]

# Chat API respects current steering configuration
responses = await model.chat(
    messages,
    sampling_params=sampling,
)

# Clear steering after
await model.clear_all_vectors()
```

**Multi-turn conversations:**

```python
# Build conversation history
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
]

# First turn
responses = await model.chat(conversation, sampling_params=sampling)
first_response = responses[0]

# Add assistant's reply to history using .to_message()
conversation.append(first_response.to_message())
conversation.append({"role": "user", "content": "Tell me more about it."})

# Continue conversation
responses = await model.chat(conversation, sampling_params=sampling)
print(responses[0].full_text())
```

#### Tensor Parallel Support

The steering runtime is designed to work with vLLM's tensor parallelism:

```python
cfg = VLLMSteeringConfig(
    model_name="Qwen/Qwen3-32B",
    tensor_parallel_size=4,  # Multi-GPU
)
model = VLLMSteerModel(cfg, bootstrap_layers=(0, 15, 31))

# Steering vectors are broadcast to all workers
await model.set_layer_vector(15, steering_vec)
```

**Implementation notes:**
- Steering vectors are broadcast to all TP ranks via `collective_rpc`
- Each worker stores the full-size vector (memory cost is `O(hidden_size)` per rank)
- No distributed operations needed in steering code (vLLM's `RowParallelLinear` handles allreduce)

---

## Development

### Running Tests

```bash
# All tests
uv run pytest tests/

# Specific test
uv run pytest tests/test_vllm_comprehensive_integration.py -v

# With coverage
uv run pytest tests/ --cov=chatspace --cov-report=html
```

**Important**: Always run tests with timeouts - bugs can cause GPU hangs.

### Project Structure

```
chatspace/
  chatspace/
    vllm_steering/       # vLLM steering runtime
      runtime.py       # Worker-side patching & RPC handlers
    generation/
      vllm_steer_model.py  # Client-side steering API
      base.py          # Abstract base classes
    hf_embed/            # SentenceTransformer embedding pipeline
    cli.py               # Command-line interface
  scripts/
    steering_smoke.py    # Quick steering verification
  tests/
    test_vllm_comprehensive_integration.py  # End-to-end tests
    test_*.py            # Unit tests
  README.md                # This file
```

---

## References

- **vLLM Documentation**: https://docs.vllm.ai/
- **Steering Vectors**: Representation Engineering papers (Li et al., 2023)
- **Activation Capture**: Interpretability research (Anthropic, OpenAI)

---

## License

[Your license here]

## Citation

```bibtex
@software{chatspace2025,
  title={chatspace: vLLM Steering Toolkit},
  author={Your Name},
  year={2025},
  url={https://github.com/your-org/chatspace}
}
```
