#### neuronpedia 🧠🔍 graph server

This is the attribution graph generation server. `--attribution-engine` selects which algorithm builds the graph:

1. **circuit-tracer** (default) - Based on [circuit-tracer](https://github.com/safety-research/circuit-tracer) by Piotrowski & Hanna. Decomposes MLP layers using transcoders/CLTs.
2. **lm-saes-crm** - Based on [Language-Model-SAEs](https://github.com/OpenMOSS/Language-Model-SAEs) by OpenMOSS. Uses Complete Replacement Models (CRM) that decompose both MLP layers (transcoders) and attention layers (Lorsa), producing richer graphs with attention-circuit features.

- [Install](#install)
- [Config](#config)
- [Start Server](#start-server)
- [Start Server - CRM Backend (Lorsa + Transcoders)](#start-server---crm-backend-lorsa--transcoders)
- [Example Request - Forward Pass (Tokenize + Salient Logits)](#example-request---forward-pass-tokenize--salient-logits)
- [Example Request - Output Graph JSON Directly](#example-request---output-graph-json-directly)
- [Example Request - CRM Graph with Lorsa Features](#example-request---crm-graph-with-lorsa-features)
- [Example Request - Output Graph JSON to S3 with presigned URL](#example-request---output-graph-json-to-s3-with-presigned-url)
- [Example Request - Steering (Interventions) With Top Logits](#example-request---steering-interventions-with-top-logits)
- [Runpod Serverless](#runpod-serverless)

### Install

```
# Navigate to the graph app directory
cd apps/graph

# Install for the circuit-tracer attribution engine (the default)
uv sync --extra circuit-tracer

# Install for the CRM attribution engine (adds lm-saes/llamascopium and its dependencies)
uv sync --extra crm
```

Each attribution engine's dependencies are an opt-in extra, and you need one of them — a bare `uv sync` installs the server but neither attribution library. The two extras are independent because the two engines share no code: `--attribution-engine circuit-tracer` imports no `llamascopium`, and `--attribution-engine lm-saes-crm` imports no `circuit_tracer`. Installing both at once (`--extra circuit-tracer --extra crm`) works and is what the parity tests use.

### Config

Create an `.env` file with `SECRET` and `HF_TOKEN` (see `.env.example`)

- `SECRET` is the server secret that needs to be passed in the `x-secret-key` request header (see examples below)
- Make sure your `HF_TOKEN` has access to the [Gemma-2-2B model](https://huggingface.co/google/gemma-2-2b) on Huggingface.

### Start Server

```
# Make sure you are in the apps/graph directory

# Only run one of the following, depending on which model you want to run.

# Run with Gemma-2-2B model with the Gemmascope transcoders
uv run python start.py --model_id google/gemma-2-2b --transcoder_set gemma

# Run with Qwen3-4B model with transcoders trained by Anthropic Fellows
uv run python start.py --model_id Qwen/Qwen3-4B --transcoder_set mwhanna/qwen3-4b-transcoders

# Run with Gemma-2-2B model with CLTs trained by Anthropic Fellows
uv run python start.py --model_id google/gemma-2-2b --transcoder_set mntss/clt-gemma-2-2b-2.5M
```

#### Disk and memory

Transcoder sets are downloaded in full before the server starts, and they are much larger than the models: `mwhanna/qwen3-4b-transcoders` is 57 GiB and `mntss/clt-gemma-2-2b-2.5M` is 160 GiB, so budget disk accordingly (the pod definitions give those two 250 GB).

Decoder weights are loaded lazily by default, sliced out of those files on demand rather than held on the GPU. **Encoders are not**: `--lazy-encoder` does the same for them and is off by default, so on a wide set the resident encoders, not the model, are what fills the card. Per layer they are `width × d_model × 2` bytes, which is 0.78 GiB for `mwhanna/qwen3-4b-transcoders` (28 GiB over 36 layers) and 1.25 GiB for `mwhanna/gemma-scope-2-4b-it` at width 262k (42.5 GiB over 34 layers — more than a 48 GiB card holds, so that set needs the flag to start at all). With it on, measured on a 32 GiB card, qwen3-4b idles at 8.9 GiB and the 2.5M CLT at 6.3 GiB.

Neither flag reduces the download, only residency. Both are force-disabled with a warning for sets whose config declares `special_load_fn: gemma-scope-2`, whose key names the lazy path cannot read; to get lazy loading there, pre-cache with `circuit_tracer.utils.caching.save_transcoders_to_cache`. Note that this is a property of the config, not of the repo name — `mwhanna/gemma-scope-2-4b-it` declares no `special_load_fn`, so it loads through the generic path and honours both flags.

Peak memory is driven by the attribution's backward pass, not by what is resident, so the safe `batch_size` does not follow model size: on that same card qwen3-4b handles 48 but OOMs at 128 on a single 24.9 GiB allocation, while the much wider CLT is comfortable at 256.

### Start Server - CRM Backend (Lorsa + Transcoders)

The Complete Replacement Model (CRM) backend uses [lm-saes](https://github.com/OpenMOSS/Language-Model-SAEs) to generate graphs. These graphs include both transcoder features (MLP decomposition) and Lorsa features (attention decomposition), enabling complete circuit tracing as described in [Bridging the Attention Gap](https://interp.open-moss.com/posts/complete-replacement).

Requires `uv sync --extra crm` first.

Available checkpoints are hosted at [OpenMOSS-Team/Llama-Scope-2-Qwen3-1.7B](https://huggingface.co/OpenMOSS-Team/Llama-Scope-2-Qwen3-1.7B) with configurations: expansion `8x` or `32x`, top-k `k64`, `k128`, or `k256`.

```
# Qwen3-1.7B with 8x expansion, K=64 (smallest, fastest - good for testing)
uv run python start.py \
  --attribution-engine lm-saes-crm \
  --model_id Qwen/Qwen3-1.7B \
  --sae_repo OpenMOSS-Team/Llama-Scope-2-Qwen3-1.7B \
  --sae_expansion 8x \
  --sae_topk k64

# Qwen3-1.7B with 8x expansion, K=128 (higher quality graphs)
uv run python start.py \
  --attribution-engine lm-saes-crm \
  --model_id Qwen/Qwen3-1.7B \
  --sae_repo OpenMOSS-Team/Llama-Scope-2-Qwen3-1.7B \
  --sae_expansion 8x \
  --sae_topk k128
```

This loads 28 transcoder modules + 28 Lorsa modules (one per layer). The 8x/k64 config requires ~24GB VRAM; 32x configs require ~40GB+.

#### Model engine

`--model-engine` chooses what *executes* the model. It applies to either attribution engine and changes nothing about what the graph contains — only which architectures are reachable. `--attribution-engine`, by contrast, picks the algorithm and so decides the graph's contents. The two are independent:

- `interp_engine` (default) uses [`interp-engine`](https://github.com/decoderesearch/interp-engine) to hook a HuggingFace `AutoModelForCausalLM` in place, so it reaches any model `transformers` can load. Several tensors the attribution needs are not module boundaries on a HuggingFace model and are reconstructed instead; see `neuronpedia_graph/crm_interp_engine.py`.
- `transformerlens` converts the weights into TransformerLens' convention, so it only reaches architectures TransformerLens has a conversion for.
- `nnsight` is available for `circuit-tracer` only. llamascopium's attribution reaches the model through methods nnsight's proxies do not provide, so the CRM engine rejects it with an explicit error.

Values keep underscores (`interp_engine`, not `interp-engine`) because they are passed straight through to circuit-tracer's own `backend=` literal.

Under CRM both options run llamascopium's attribution itself, unmodified.

The two agree to a few parts per million on the adjacency matrix in `float32`, and select the same graph nodes, QK tracing included (`tests/test_crm_backend_parity.py`, opt in with `RUN_CRM_PARITY=1`) — which is what makes `interp_engine` the default. In `bfloat16` this attribution is not reproducible to better than ~0.3 relative *within* a single backend, so compare in `float32` if you are checking a change.

One case still needs `transformerlens`: QK tracing attributes to the model's output-projection biases as graph leaves, and `interp_engine` can only expose the replacement modules' own biases. On an architecture whose attention or MLP output projection carries a bias it refuses with an error naming the layers, rather than quietly dropping those source nodes.

The split falls along the GPT-2-to-Llama transition, since Llama dropped biases and everything after it followed. Surveyed with the same lookup the check uses:

| affected (QK tracing refuses) | unaffected |
| --- | --- |
| GPT-2, GPT-NeoX / Pythia, Starcoder2, GPT-J (MLP bias only) | Llama, Qwen2, Qwen3, Gemma 2, Gemma 3, Mistral, Phi-3, StableLM 2 |

This is currently theoretical: CRM needs Lorsa + transcoder checkpoints, and the only published set is [Llama-Scope-2 for Qwen3-1.7B](https://huggingface.co/OpenMOSS-Team/Llama-Scope-2-Qwen3-1.7B), which is unaffected. No affected architecture has CRM weights to run in the first place. Separately, OPT, BLOOM, Falcon and Phi-2 cannot use this backend at all — interp-engine has no attention/MLP resolution for them, so it fails at load with an explicit error rather than at QK tracing.

```
uv run python start.py \
  --attribution-engine lm-saes-crm \
  --model-engine transformerlens \
  --model_id Qwen/Qwen3-1.7B \
  --sae_repo OpenMOSS-Team/Llama-Scope-2-Qwen3-1.7B \
  --sae_expansion 8x \
  --sae_topk k64
```

### Example Request - Output Graph JSON Directly

This will run a graph generation for the prompt "1 2 " on Gemma-2-2B.

> Warning: This will be a large text response (2MB), so you may want to pipe it into a file by appending this at the end: ` > count12.json`.

```
curl -X POST http://localhost:5004/generate-graph \
  -H "Content-Type: application/json" \
  -H "x-secret-key: YOUR_SECRET" \
  -d '{
    "prompt": "1 2 ",
    "model_id": "google/gemma-2-2b",
    "batch_size": 48,
    "max_n_logits": 10,
    "desired_logit_prob": 0.95,
    "node_threshold": 0.8,
    "edge_threshold": 0.85,
    "slug_identifier": "count-1-2",
    "max_feature_nodes" : 5000
  }'
```

### Example Request - CRM Graph with Lorsa Features

This generates a Complete Replacement Model (CRM) attribution graph for an acronym completion prompt on Qwen3-1.7B. The output includes both `cross layer transcoder` nodes (MLP features) and `lorsa` nodes (attention features).

Start the server with `--attribution-engine lm-saes-crm` first (see above).

```
curl -X POST http://localhost:5004/generate-graph \
  -H "Content-Type: application/json" \
  -H "x-secret-key: YOUR_SECRET" \
  -d '{
    "prompt": "The National Digital Analytics Group (ND",
    "model_id": "Qwen/Qwen3-1.7B",
    "batch_size": 16,
    "max_n_logits": 10,
    "desired_logit_prob": 0.95,
    "node_threshold": 0.6,
    "edge_threshold": 0.8,
    "slug_identifier": "acronym-ndag",
    "max_feature_nodes": 3000,
    "enable_qk_tracing": true,
    "qk_top_fraction": 0.5,
    "qk_topk": 4
  }' > acronym-ndag.json
```

The output graph JSON will contain nodes with `feature_type` values including `"lorsa"` (attention features rendered as triangles in the UI) and `"lorsa error"` (attention reconstruction errors), in addition to the standard `"cross layer transcoder"`, `"mlp reconstruction error"`, `"embedding"`, and `"logit"` types.

The CRM backend also supports optional **QK tracing**, which performs a second-order, pair-wise attribution over the Q/K pathways of the top Lorsa heads. When enabled, target Lorsa nodes are annotated with a `qk_tracing_results` object:

```json
{
  "pair_wise_contributors": [[q_node_id, k_node_id, attribution], ...],
  "top_q_marginal_contributors": [[q_node_id, attribution], ...],
  "top_k_marginal_contributors": [[k_node_id, attribution], ...]
}
```

QK tracing parameters (all optional):

- `enable_qk_tracing` (bool, default `false`): Turn on QK tracing. Increases memory and is typically 2x–10x slower.
- `qk_top_fraction` (float, default `0.6`): Fraction of the highest-influence Lorsa heads (among those surviving pruning) that are further QK-traced. `1.0` means all of them.
- `qk_topk` (int, default `10`): Number of upstream contributors to keep per target Lorsa head, for each of pair-wise, Q-marginal, and K-marginal. Must satisfy `batch_size >= qk_topk`.

Other test prompts from the [CRM paper](https://interp.open-moss.com/posts/complete-replacement):

- `a="Craig"\nassert a[0]=='` (string indexing, expects `C`)
- `I always loved visiting Aunt Sally. Whenever I was feeling sad, Aunt` (induction, expects ` Sally`)
- `The capital of France is` (factual, expects ` Paris`)

### Example Request - Output Graph JSON to S3 with presigned URL

Since the graph JSONs can be large (up to 100MB+ sometimes!), on Neuronpedia we store the JSONs on S3. When the webapp calls the graph server for a graph generation, instead of returning this large file back to the webapp, we have the graph server upload it directly to S3. Then, when the user requests the graph on the webapp, it's downloaded from S3.

This adds two parameters to the request:

- `signed_url`(URL string): Signed S3 PUT request, which the graph server will use to directly upload the json file.
- `compress`(boolean): Whether or not to gzip the JSON before uploading.

To use upload graphs to S3 with this, you'll need to:

1. Create an S3 bucket with public read permissions.
2. Create an S3 access key+secret pair that has PUT permissions for that bucket.
3. Using those S3 access credentials, generate the `signed_url` with an S3 library.

- [Here is the Typescript code](https://github.com/hijohnnylin/neuronpedia/blob/58350119d64fc1089b007a7a29a9ce3686cf950d/apps/webapp/app/api/graph/generate/route.ts#L222-L243) for how we generate `signed_url` on Neuronpedia when a request for generating graphs comes in.
- To do this in Python, you can reference [examples like this](https://jimbobbennett.dev/blogs/get-put-s3-boto/).

Finally, the example command (it won't work unless you replace `S3_SIGNED_PUT_URL`):

```
curl -X POST http://localhost:5004/generate-graph \
  -H "Content-Type: application/json" \
  -H "x-secret-key: YOUR_SECRET" \
  -d '{
    "prompt": "1 2 ",
    "model_id": "google/gemma-2-2b",
    "batch_size": 48,
    "max_n_logits": 10,
    "desired_logit_prob": 0.95,
    "node_threshold": 0.8,
    "edge_threshold": 0.85,
    "slug_identifier": "count-1-2",
    "max_feature_nodes" : 5000,
    "signed_url": S3_SIGNED_PUT_URL
  }'
```

### Example Request - Steering (Interventions) With Top Logits

The following ablates a French feature and increases a Spanish feature by 350. It uses the activations from the specified prompt.

We steer this on three features/positions:

1. Layer 20, 1454 (French feature) - We ablate this. We tell the steering server that this feature is active on `token_active_position` 6 and we want to ablate it at `steer_position` 7.
2. Layer 20, 341 (Spanish feature) - We add 350 to the activations for this feature at the last token (position 7). We tell the steering server that this feature is active at position 7.
3. Layer 20, 341 (Same Spanish feature) - We add 350 to the activations for this feature. We set `steer_generated_tokens` to true so that it will steer all tokens that are generated. You cannot specify both a `steer_position` and a `steer_generated_tokens` in the same feature - hence why we split it into 2 features in the input body.

It returns the top logits at each position for both the steered and default completions.
The `top_k` field is the number of top logits to return per completion token.

Request

```
curl -X POST http://localhost:5004/steer \
  -H "Content-Type: application/json" \
  -H "x-secret-key: YOUR_SECRET" \
  -d '{
    "model_id": "google/gemma-2-2b",
    "prompt": "Fait: Michael Jordan joue au",
    "features": [
        {
          "layer": 20,
          "index": 1454,
          "token_active_position": 6,
          "steer_position": 6,
          "ablate": true
        },
        {
          "layer": 20,
          "index": 341,
          "token_active_position": 7,
          "steer_position": 7,
          "delta": 350
        },
        {
          "layer": 20,
          "index": 341,
          "token_active_position": 7,
          "steer_generated_tokens": true,
          "delta": 350
        }
    ],
    "n_tokens": 10,
    "top_k": 3,
    "temperature": 0,
    "freq_penalty": 0,
    "freeze_attention": false
  }'
```

Response (Arrays Truncated)

```
{
  "DEFAULT_GENERATION": "Fait: Michael Jordan joue au basket avec son fils, Jeffrey Jordan, à la",
  "STEERED_GENERATION": "Fait: Michael Jordan joue au baloncesto.\n\nFalso: Michael Jordan no juega",
  "DEFAULT_LOGITS_BY_TOKEN": [
    [
      "F",
      []
    ],
    [
      "ait",
      []
    ],
    [
      ":",
      []
    ],
    [
      " Michael",
      []
    ],
    [
      " Jordan",
      []
    ],
    [
      " joue",
      []
    ],
    [
      " au",
      [
        [
          " basket",
          0.547155499458313
        ],
        [
          " basketball",
          0.1174481138586998
        ],
        [
          " golf",
          0.06529955565929413
        ]
      ]
    ],
    [
      " basket",
      [
        [
          " avec",
          0.12210318446159363
        ],
        [
          " depuis",
          0.09880638122558594
        ],
        [
          "-",
          0.08087616413831711
        ]
      ]
    ]
  ],
  "STEERED_LOGITS_BY_TOKEN": [
    [
      "F",
      []
    ],
    [
      "ait",
      []
    ],
    [
      ":",
      []
    ],
    [
      " Michael",
      []
    ],
    [
      " Jordan",
      []
    ],
    [
      " joue",
      []
    ],
    [
      " au",
      [
        [
          " baloncesto",
          0.3005445897579193
        ],
        [
          " golf",
          0.19333133101463318
        ],
        [
          " basketball",
          0.08585426211357117
        ]
      ]
    ],
    [
      " baloncesto",
      [
        [
          ".",
          0.17341850697994232
        ],
        [
          " en",
          0.14220058917999268
        ],
        [
          ",",
          0.10603509098291397
        ]
      ]
    ]
  ]
}
```

### Example Request - Forward Pass (Tokenize + Salient Logits)

The `/forward-pass` endpoint tokenizes a prompt and returns the top predicted next tokens with their probabilities. This is used by the webapp to preview token counts and salient logits before generating a full graph.

```
curl -X POST http://localhost:5004/forward-pass \
  -H "Content-Type: application/json" \
  -H "x-secret-key: YOUR_SECRET" \
  -d '{
    "prompt": "The capital of France is",
    "max_n_logits": 5,
    "desired_logit_prob": 0.9
  }'
```

## Documentation / Usage (Swagger)

FastAPI has a built-in docs + endpoint tester. After running the server, to see interactive docs, go to [http://localhost:5004/docs](http://localhost:5004/docs)

Notes/Caveats:

- If you set a SECRET (not set by default) in your `.env` file, you'll need to add a `x-secret-key` header.

### Runpod Serverless

The `apps/graph/runpod` directory contains a [Runpod Serverless](https://docs.runpod.io/serverless/overview) worker that does the same as `apps/graph` - it just in a format the Runpod expects. It has its own `README.md`.
