# RunPod Serverless - Neuronpedia Inference

RunPod Serverless deployment for Neuronpedia's completion-chat endpoint with steering vector support.

## Overview

This serverless handler is specifically configured for:
- **Model**: Llama 3.3 70B Instruct (AWQ quantized via `casperhansen/llama-3.3-70b-instruct-awq`)
- **Engine**: ChatSpace (vLLM-based)
- **Endpoint**: completion-chat (streaming only)
- **Feature**: assistant_axis persona monitoring enabled

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t neuronpedia-inference-serverless .
```

### 2. Push to Docker Hub (or your container registry)

```bash
docker tag neuronpedia-inference-serverless your-registry/neuronpedia-inference-serverless:latest
docker push your-registry/neuronpedia-inference-serverless:latest
```

### 3. Create RunPod Serverless Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Create a new endpoint using your container image
3. Configure the GPU type (recommended: A100 80GB or H100 for 70B model)
4. Set environment variables as needed (see Configuration section)

## API Usage

### Request Format

```json
{
  "input": {
    "prompt": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "types": ["STEERED"],
    "vectors": [
      {
        "hook": "blocks.40.hook_resid_post",
        "strength": 1.0,
        "steering_vector": [0.1, 0.2, ...]
      }
    ],
    "strength_multiplier": 1.0,
    "seed": 42,
    "temperature": 0.7,
    "freq_penalty": 0.0,
    "n_completion_tokens": 512,
    "steer_method": "SIMPLE_ADDITIVE",
    "normalize_steering": false,
    "steer_special_tokens": false
  }
}
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | array | required | Array of chat messages with `role` and `content` |
| `types` | array | `["DEFAULT"]` | Steer types: `["STEERED"]`, `["DEFAULT"]`, or `["STEERED", "DEFAULT"]` |
| `vectors` | array | `[]` | Steering vectors with `hook`, `strength`, and `steering_vector` |
| `strength_multiplier` | float | `1.0` | Global multiplier for steering strength |
| `seed` | int | `null` | Random seed for reproducibility |
| `temperature` | float | `0.7` | Sampling temperature |
| `freq_penalty` | float | `0.0` | Frequency penalty (ignored for vLLM) |
| `n_completion_tokens` | int | `512` | Maximum tokens to generate |
| `steer_method` | string | `"SIMPLE_ADDITIVE"` | `"SIMPLE_ADDITIVE"` or `"PROJECTION_CAP"` |
| `normalize_steering` | bool | `false` | Whether to normalize steering vectors |
| `steer_special_tokens` | bool | `false` | Whether to apply steering to special tokens |

### Response Format

The response is streamed as JSON objects. The final response includes:

```json
{
  "outputs": [
    {
      "raw": "Full tokenized output including prompt and response",
      "chat_template": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "Generated response"}
      ],
      "type": "STEERED"
    }
  ],
  "input": {
    "raw": "Tokenized input prompt",
    "chat_template": [...]
  },
  "assistant_axis": [
    {
      "type": "STEERED",
      "pc_titles": ["- Role-playing ↔️ + Assistant-like"],
      "turns": [
        {
          "pc_values": {"- Role-playing ↔️ + Assistant-like": 0.42},
          "snippet": "Generated response preview..."
        }
      ]
    }
  ]
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `meta-llama/Llama-3.3-70B-Instruct` | Base model identifier |
| `OVERRIDE_MODEL_ID` | `casperhansen/llama-3.3-70b-instruct-awq` | Actual model to load |
| `MODEL_DTYPE` | `bfloat16` | Model data type |
| `TOKEN_LIMIT` | `65536` | Maximum context length |
| `MAX_MODEL_LEN` | `65536` | vLLM max model length |
| `GPU_MEMORY_UTILIZATION` | `0.95` | GPU memory utilization ratio |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for tensor parallelism |
| `PERSONA_DATA_PATH` | `/app/data` | Path to persona PCA data |

## Project Structure

```
runpod_serverless/
├── src/
│   ├── handler.py           # RunPod handler entry point
│   ├── model.py             # Model loading and management
│   ├── completion_chat.py   # Main generation logic
│   ├── utils.py             # Utility functions
│   └── persona_utils/       # Persona monitoring utilities
│       ├── __init__.py
│       ├── analysis.py
│       ├── conversation.py
│       ├── model_chatspace.py
│       ├── persona_data.py
│       └── spans_chatspace.py
├── vendor/
│   ├── chatspace/           # ChatSpace library
│   └── steerllm/            # SteerLLM library
├── data/
│   └── casperhansen/
│       └── llama-3.3-70b-instruct-awq/
│           ├── contrast_vectors.pt
│           └── pca/
│               └── roles_layer40-min.pt
├── Dockerfile
├── requirements.txt
└── README.md
```

## Steering Vector Format

Steering vectors should be provided in the following format:

```json
{
  "hook": "blocks.40.hook_resid_post",
  "strength": 1.0,
  "steering_vector": [...]
}
```

- `hook`: The hook point in the model. Supported formats:
  - `blocks.{layer}.hook_resid_post` - After residual connection
  - `blocks.{layer}.hook_resid_pre` - Before residual connection (layer-1 is used)
- `strength`: Per-vector strength coefficient
- `steering_vector`: Array of floats matching the model's hidden dimension (8192 for Llama 3.3 70B)

## Local Development

### Running Locally

```bash
cd src
python handler.py
```

### Testing with curl

```bash
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": [{"role": "user", "content": "Hello!"}],
      "types": ["DEFAULT"],
      "vectors": [],
      "n_completion_tokens": 100
    }
  }'
```

## Notes

- This is a streaming-only implementation; non-streaming requests are not supported
- The `assistant_axis` feature is always enabled, providing persona analysis for all responses
- Frequency penalty is accepted but ignored (vLLM limitation)
- For multi-GPU setups, adjust `TENSOR_PARALLEL_SIZE` accordingly

