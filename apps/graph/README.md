#### neuronpedia 🧠🔍 graph server

This is the attribution graph generation server based on [circuit-tracer](https://github.com/safety-research/circuit-tracer) by Piotrowski & Hanna. These endpoints are what's called when you click `+ New Graph` on the [Neuronpedia Circuit Tracer](https://www.neuronpedia.org/gemma-2-2b/graph) page.

- [Install](#install)
- [Config](#config)
- [Start Server](#start-server)
- [Example Request - Output Graph JSON Directly](#example-request---output-graph-json-directly)
- [Example Request - Output Graph JSON to S3 with presigned URL](#example-request---output-graph-json-to-s3-with-presigned-url)
- [Example Request - Steering (Interventions) With Top Logits](#example-request---steering-interventions-with-top-logits)
- [Runpod Serverless](#runpod-serverless)

### Install

```
# Navigate to the graph app directory
cd apps/graph

# Install dependencies using Poetry
poetry install
```

### Config

Create an `.env` file with `SECRET` and `HF_TOKEN` (see `.env.example`)

- `SECRET` is the server secret that needs to be passed in the `x-secret-key` request header (see examples below)
- Make sure your `HF_TOKEN` has access to the [Gemma-2-2B model](https://huggingface.co/google/gemma-2-2b) on Huggingface.

### Start Server

```
# Make sure you are in the apps/graph directory

# Only run one of the following, depending on which model you want to run.

# Run with Gemma-2-2B model with the Gemmascope transcoders
poetry run python start.py --model_id google/gemma-2-2b

# Run with Llama-3.2-1B model with transcoders trained by Anthropic Fellows
poetry run python start.py --model_id meta-llama/Llama-3.2-1B
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

### Runpod Serverless

The `apps/graph/runpod` directory contains a [Runpod Serverless](https://docs.runpod.io/serverless/overview) worker that does the same as `apps/graph` - it just in a format the Runpod expects. It has its own `README.md`.
