## neuronpedia activations server

Minimal FastAPI server for raw residual stream extraction with plain PyTorch + `transformers`.

### Scope

- One endpoint: `POST /raw`

### Run

```bash
uv sync
uv run python start.py \
  --model_id meta-llama/Llama-3.1-8B-Instruct \
  --model_dtype bfloat16
```

Startup behavior:

- server eagerly downloads/loads the startup model during boot (no lazy first-request load)
- startup model is configured via `MODEL_ID` (or `--model_id`)
- optional `MODEL_DTYPE`/`DEVICE` (or `--model_dtype`/`--device`)

Optional auth:

- set `SECRET` and send `X-SECRET-KEY` header (same behavior as inference server)

### Example request

```bash
curl -X POST "http://localhost:5010/raw" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompts": ["The Eiffel Tower is in", "The Colosseum is in"],
    "hook_point": "residual_stream",
    "type": "final_output_token"
  }'
```

### Example response

```json
{
  "hook_point": "residual_stream",
  "type": "final_output_token",
  "dtype": "bfloat16",
  "device": "cuda",
  "results": [
    {
      "token_strings": ["<|begin_of_text|>", "The", "ĠEiffel", "ĠTower", "Ġis", "Ġin"],
      "token_ids": [128000, 791, 46943, 10780, 374, 304],
      "activations": [
        {
          "layer": 0,
          "token_indices": [5],
          "values": [[0.1377, -0.4219, 0.0038, ...]]
        },
        {
          "layer": 1,
          "token_indices": [5],
          "values": [[0.1084, -0.3906, 0.0271, ...]]
        },
        {
          "layer": 2,
          "token_indices": [5],
          "values": [[...]]
        }
      ]
    },
    {
      "token_strings": ["<|begin_of_text|>", "The", "ĠCol", "osse", "um", "Ġis", "Ġin"],
      "token_ids": [128000, 791, 8505, 564, 507, 374, 304],
      "activations": [
        {
          "layer": 0,
          "token_indices": [6],
          "values": [[...]]
        },
        {
          "layer": 1,
          "token_indices": [6],
          "values": [[...]]
        }
      ]
    }
  ]
}
```

### Request fields

- `model` (required): Hugging Face model id
- `prompts` (required): array of strings
- `hook_point` (optional): defaults to `residual_stream`
- `type` (optional): defaults to `final_output_token`

Defaults:

- max batch size defaults to `16` prompts (`MAX_BATCH_SIZE` to override)
- max prompt size defaults to `2048` tokens (`MAX_PROMPT_TOKENS` to override)

Other endpoints:

- `GET /health`
- `GET /busy` (returns `{"busy": true|false}` based on the request lock)
