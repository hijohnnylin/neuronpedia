#### Neuronpedia 🧠🔍 Inference Server

- [What This Is](#what-this-is)
- [Interpretability Backend](#interpretability-backend)
- [Setup + Run](#setup--run)
  - [Customizing the Inference Server (Different Models, SAEs, etc)](#customizing-the-inference-server-different-models-saes-etc)
  - [Documentation / Usage (Swagger)](#documentation--usage-swagger)
  - [Loading Specific SAEs From SAELens](#loading-specific-saes-from-saelens)
  - [Running Without Any SAEs](#running-without-any-saes)
  - [Sizing `--vllm_gpu_memory_utilization`](#sizing---vllm_gpu_memory_utilization)
  - [Trading SAE VRAM for KV Cache (`--sae_gpu_budget_gib`)](#trading-sae-vram-for-kv-cache---sae_gpu_budget_gib)
  - [If the Backend Dies in FlashInfer](#if-the-backend-dies-in-flashinfer)
  - [Developing the Wire Format](#developing-the-wire-format)
- [Usage Examples](#usage-examples)
  - [Get Activations for a Single Feature and Prompt](#get-activations-for-a-single-feature-and-prompt)
  - [Get Activations From One or More Layers/Sources/SAEs for a Prompt](#get-activations-from-one-or-more-layerssourcessaes-for-a-prompt)
  - [Get Raw Residual Stream Vectors for a Prompt](#get-raw-residual-stream-vectors-for-a-prompt)
  - [Get Cosine Similarities](#get-cosine-similarities)
  - [Steering Chat Example gemma-2-2b-it (Returns Dog)](#steering-chat-example-gemma-2-2b-it-returns-dog)
  - [Steering Example gpt2-small res-jb](#steering-example-gpt2-small-res-jb)
  - [Jacobian Lens Example gemma-3-4b (Base, Raw Prompt)](#jacobian-lens-example-gemma-3-4b-base-raw-prompt)
  - [Jacobian Lens Example gemma-3-4b-it (Instruct, Chat)](#jacobian-lens-example-gemma-3-4b-it-instruct-chat)
- [Testing, Linting, and Formatting](#testing-linting-and-formatting)

## What This Is

Python server that supports Neuronpedia's inference capabilities - steering, testing activations, search via inference, etc.

It can be a standalone server. It does not require Neuronpedia to run.

It optionally uses [SAELens](https://github.com/jbloomAus/SAELens) to pre-load SAEs into memory and perform inference.

SAELens is not required. It can also just load models and perform inference using steering vectors that are passed into the request.

Every request and response body is a Pydantic model under `neuronpedia_inference/schemas/`. Those models are the wire format: `apps/inference/openapi.json` is generated from them, and the webapp's TypeScript types are generated from that. See [Developing the Wire Format](#developing-the-wire-format) below.

## Interpretability Backend

The server runs on **[`interp-engine`](https://github.com/decoderesearch/interp-engine)** — a standalone
raw-transformers interpretability core (plain HuggingFace models in eager PyTorch
with our own forward-hook capture/steering layer). It replaced the previous
TransformerLens + nnsight stack, so there is no `--nnsight` flag and no
TransformerLens dependency anymore. The engine is pulled in as a path dependency
(see `pyproject.toml`), and the vLLM backend ships via its optional extra
(`interp-engine[vllm]`).

The backend is **auto-selected at startup** (override with `--device`, or force with
`--force-vllm` / `--force-eager`; shard across GPUs with `--num-gpus N`):

- **CUDA + a vLLM-supported architecture** → the engine-owned **vLLM** backend
  (fast serving; capture, steering, generation, DFA, attention patterns via
  off-kernel recompute, and logit/Jacobian lens all run on it, N-way concurrent).
- **CUDA, arch not supported by vLLM** → **`EagerModel`** on CUDA.
- **No CUDA** → `EagerModel` on MPS (for fp16/fp32-native models) or CPU.

Because the engine handles every architecture in raw transformers, new/day-one
models (Gemma 3, gpt-oss, Qwen 3.x, …) load without special flags. `GET
/v1/capabilities` reports the loaded model, the active backend, and per-endpoint
support.

**Multi-GPU (single node):** `--num-gpus N` shards a model too large for one card
across `N` GPUs — vLLM uses tensor-parallelism (`tensor_parallel_size=N`); eager
`EagerModel` uses accelerate `device_map="auto"`. e.g. Llama-3.3-70B across 4× A40:
`--num-gpus 4` (see `local_scripts/pods.yaml`).

**GPU memory (vLLM backend):** `--vllm_gpu_memory_utilization 0.73` caps the
fraction of the card vLLM takes for weights + KV cache (default 0.9). It has to
leave room for the SAE cache, which the server allocates outside vLLM's
accounting — too high and startup dies in vLLM's warmup with an out-of-memory
error, after the KV cache size already looked fine.
`local_scripts/sae_memory.py` computes the value per model, and the
`VLLM_GPU_MEMORY_UTILIZATION` env var overrides the flag (as env vars do for
every `start.py` argument).

**Which vLLM backend a pod runs:** interp-engine splits vLLM into three, and two
env vars pick between them. They differ only in what the pod declares up front,
which is what decides whether it can keep CUDA graphs.

| Pod config | Engine backend | Speed | What it serves |
| --- | --- | --- | --- |
| neither var set | `vllm` | baseline | every endpoint; every capture point reachable |
| `STATIC_POINTS=...` | `vllm-static` | graph speed | the declared sites only; anything else 400s |
| `GENERATION_ONLY=true` | `vllm-generate` | graph speed | completions and tokenization only |

The two graph backends are fast for the same reason they are limited: replay
never calls the Python forward that capture and steering hooks live on, so the
sites have to be named before the graphs are captured. `STATIC_POINTS` names
them — `auto` for `resid_post` at every layer (`resid_streams` on a
hyper-connection trunk), `sae` for the SAE sites once those SAEs load, or an
explicit JSON list of `[name, layer]` pairs. Declare `attn` at a layer to keep
attention and DFA. Because the set is fixed at load, this is a routing decision:
a pod that did not declare a point cannot grow one, and says so in a 400 that
lists what it did declare.

`GENERATION_ONLY=true` is the empty case, worth up to +249% decode on a 1B model
and ~1% at 8B — small-model completion traffic and nothing else. The server
refuses to start if `SAE_SETS` is non-empty (an SAE read *is* a capture), if
`STATIC_POINTS` is also set (the two name different backends), or if it resolved
to eager. Every pod reports its real capability from `/v1/capabilities`
(`hooks_available`, `graph_replay`, `static_points`, `static_writes`), so a
router can send capture traffic where it will be served instead of retrying
400s. See the engine's `docs/PERFORMANCE.md` for the measurements.

`FREEZE_POINTS` was the pre-1.3 name for `STATIC_POINTS`, and `vllm-freeze` the
old name for `vllm-static`. A pod that still sets the old variable is refused at
startup rather than quietly served as hooked vLLM.

**CUDA-graph batch sizes:** graph replay is per token count. vLLM's default
captures every 8 tokens up to 256, which on DeepSeek-V4 is ~47 GiB and leaves no
KV. Both graph backends instead pass `1,2,4,8,16,32,64,128,256` through
interp-engine as `compilation_config.cudagraph_capture_sizes`. Override with
`--vllm_cudagraph_capture_sizes` or `VLLM_CUDAGRAPH_CAPTURE_SIZES`. A prompt
longer than the largest size still runs; that prefill is eager, decode still
graphs. Hooked vLLM ignores this.

> ⚠️ **Warning:** This is _draft_ documentation. We expect to either have better READMEs or use a hosted documentation website.

## Setup + Run

This loads the `openai-community/gpt2` model and the `res-jb` SAEs through SAELens:

```
uv sync
uv run python start.py
```

### Customizing the Inference Server (Different Models, SAEs, etc)

Open the `start.py` script to see the flags that Neuronpedia reads either from the arguments or from the environment variables.

### Documentation / Usage (Swagger)

FastAPI has a built-in docs + endpoint tester. After running the server, to see interactive docs, go to [http://localhost:5002/docs](http://localhost:5002/docs)

Notes/Caveats:

- Simple usage: Expand `/v1/activation/single`, click Try It Out, then click "Execute".
- If you set a SECRET (not set by default) in your `.env` file, you'll need to add a `x-secret-key` header.

### Loading Specific SAEs From SAELens

Example of loading model `google/gemma-2-2b` and a GemmaScope SAE using arguments

```
uv run python start.py \
  --model_id google/gemma-2-2b \
  --sae_sets gemmascope-res-16k \
  --model_dtype bfloat16 \
  --sae_dtype bfloat16
```

You'll notice we use the `model_id` and `sae_sets` flags to set which SAE to load from SAELens.

You can run the following to get the currently supported models and SAEs

```
uv run python start.py --list_models
```

The `model_id` is the Hugging Face repo id of the model (`openai-community/gpt2`, `google/gemma-2-2b`), which is what the weights load from. The SAELens directory keys some SAEs by Neuronpedia short id instead (`gpt2-small`); the server maps between the two with `np_model_to_hf.json`, so give it the Hugging Face id. `sae_sets` is the text after the layer number and hyphen in a Neuronpedia source ID - for example, if you have a Neuronpedia feature at URL `http://neuronpedia.org/gpt2-small/0-res-jb/123`, the `0-res-jb` is the source ID, and the `sae_sets` is `res-jb`.

You can also find Neuronpedia source IDs in the SAELens [pretrained SAEs YAML file](https://github.com/jbloomAus/SAELens/blob/main/sae_lens/pretrained_saes.yaml) or by clicking into models in the [Neuronpedia datasets exports](https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/) directory.

### Running Without Any SAEs

SAEs are optional. Pass `--no_saes` (or `SAE_SETS='[]'`) to start a server that loads only
the model:

```
uv run python start.py \
  --model_id meta-llama/Llama-3.1-8B-Instruct \
  --model_dtype bfloat16 \
  --no_saes \
  --token_limit 2048
```

Such a server skips the SAELens directory lookup entirely and serves everything that does
not need an SAE — `/v1/activation/raw`, `/v1/lens/prompt`, `/v1/steer/*` by vector,
`/v1/tokenize`. The SAE-backed endpoints reject their `source_set`, since none is loaded.

Note also that the `model` field in a request body is advisory. A server holds exactly one
model, so the field selects nothing and a mismatch is logged rather than rejected.

### Sizing `--vllm_gpu_memory_utilization`

vLLM reserves that fraction of the **whole card** at startup and knows nothing
about the SAE cache, which lives in the FastAPI server process — a different
process, allocated outside vLLM's pool and before the engine boots. So the two
add up and the card runs out, usually as a `torch.OutOfMemoryError` in vLLM's
warmup after "Available KV cache memory" already looked healthy.

The `VLLM_GPU_MEMORY_UTILIZATION` env var does the same job and still wins if it
is set (that is `start.py`'s convention for every setting), which is what
`local_scripts/pods.yaml` uses. Prefer the flag interactively: it survives
copy-paste and edits that an assignment before `uv run` does not. Note the
variable is ours, not vLLM's, so vLLM logs "Unknown vLLM environment variable
detected" for it either way — harmless, and the flag makes the ownership obvious.

These values are sized for a **48 GB card** (A40 / A6000, what `pods.yaml`
defaults to) against a `--sae_sets` trimmed to one set. They are higher than the
ones in `pods.yaml`, which have to fit every set a pod serves:

| Model                          | Trimmed SAE cache | `--vllm_gpu_memory_utilization` |
| ------------------------------ | ----------------- | ------------------------------- |
| `gpt2-small`                   | 1.8 GiB (fp32)    | 0.88                            |
| `gemma-2-2b` / `gemma-2-2b-it` | 3.7 GiB           | 0.84                            |
| `gemma-2-9b-it`                | 5.2 GiB           | 0.80                            |
| `gemma-3-270m`                 | 0.2 GiB           | default (0.9), no override      |
| `gemma-3-1b` / `-4b` / `-12b`  | under 1 GiB       | default (0.9), no override      |
| `qwen3.6-27b`                  | none              | needs an 80 GB card             |
| `llama3.1-8b`                  | 16.0 GiB          | 0.55                            |
| `llama3.1-8b-it`               | 14.0 GiB          | 0.60                            |
| `gpt-oss-20b`                  | 8.4 GiB           | 0.73                            |

On a different card, or with a wider `--sae_sets`, recompute rather than guess:

```bash
python local_scripts/sae_memory.py --args "--model_id openai-community/gpt2 --sae_sets res-jb --sae_dtype float32"
python local_scripts/sae_memory.py --vram-gib 23.5 --args "..."   # e.g. a 24 GB card
```

It reads SAE parameter counts from Hugging Face metadata (no weights
downloaded), applies `--sae_dtype`, and prints the arithmetic plus the resulting
KV cache headroom. `local_scripts/README.md` covers it.

There are two distinct failures here, and only the first is the one above. If
vLLM asks for more than is **free right now**, it refuses to start instead of
OOMing later:

```
ValueError: Free memory on device cuda:0 (27.81/31.26 GiB) on startup is less
than desired GPU memory utilization (0.9, 28.13 GiB)
```

The fraction is of the card's _total_, so anything already resident counts
against you — a desktop session holding 1 GiB is enough to make the default 0.9
unstartable on a 32 GB card, before any SAE is loaded. On a workstation, set the
value explicitly rather than reading it off the 48 GB table above. What worked on
a 32 GB 5090 with one SAE loaded: 0.80 for `gemma-3-270m` and `gemma-2-2b`, 0.75
for `gemma-2-2b-it` and `llama3.1-8b`, 0.70 for `llama3.1-8b-it`, 0.65 for
`gpt-oss-20b`.

### Trading SAE VRAM for KV Cache (`--sae_gpu_budget_gib`)

The table above is dominated by the SAE cache, and most of it is idle: a request
reads one source at a time. `--sae_gpu_budget_gib` (env: `SAE_GPU_BUDGET_GIB`)
moves the master copies to host RAM and keeps only that many GiB of them on the
card, evicting least-recently-used. A miss costs tens of milliseconds to stage
back in. Requests reserve residency before they run, so nothing is evicted
mid-encode; when the budget is too small for the traffic, requests queue instead
of failing.

```
--sae_gpu_budget_gib 8       # keep 8 GiB of SAEs on the GPU
--sae_gpu_budget_gib auto    # size it from what vLLM leaves behind
```

It is opt-in: unset means every SAE stays resident, as it always has. `auto` is
derived from `--vllm_gpu_memory_utilization` rather than measured, because vLLM
reserves its share later and in a child process. The startup log prints what it
resolved to, and `GET /v1/capabilities` reports live cache occupancy.

The host needs room for **all** the SAEs — paging relocates that memory, it does
not shrink it — and the utilization has to be raised afterwards, or the VRAM the
SAEs gave up just sits unused. `local_scripts/sae_memory.py` prints both numbers.
`--max_loaded_saes` goes inert here: it evicts by count, which only means
something while "loaded" means "on the GPU". `--sae_pinned_host_gib` caps how
much host RAM may be page-locked; leave it unset and the server measures what is
free, holding back a reserve for the engine process.

### If the Backend Dies in FlashInfer

vLLM's sampler uses FlashInfer, which JIT-compiles a CUDA extension on first
sample, so a workstation needs a **system CUDA toolkit** — not just the CUDA
libraries in the venv:

```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

Install one matching your driver, but install a **specific, new enough** one
(`apt install cuda-toolkit-13-1`) and not the unversioned `nvidia-cuda-toolkit`
metapackage — that one is CUDA 12.4 on Ubuntu 24.04, and it turns the error above
into a worse one:

```
RuntimeError: FlashInfer requires GPUs with sm75 or higher
```

on a card that is obviously past sm75. FlashInfer resolves its toolkit by
`which nvcc` **before** falling back to `/usr/local/cuda`, so `/usr/bin/nvcc`
(12.4) shadows a perfectly good `/usr/local/cuda` (13.1). sm120 needs CUDA >=
12.9, so `_normalize_cuda_arch` throws, `CompilationContext.__init__` swallows it
as `Failed to get device capability` — a warning, not an error — and leaves
`TARGET_CUDA_ARCHS` empty. The sm75 check then fails for a card that was never
detected, which is why the message points nowhere near the cause. Fix it by
naming the toolkit explicitly:

```bash
export CUDA_HOME=/usr/local/cuda   # or FLASHINFER_CUDA_ARCH_LIST=12.0
```

Two things that look like they should work but don't:

- The toolkit torch bundles at `.venv/.../nvidia/cu13` has a working `nvcc`, and
  pointing `CUDA_HOME` at it gets further before failing: flashinfer 0.6.13
  vendors its own CCCL headers, which reject that newer toolkit with
  `"CUDA compiler and CUDA toolkit headers are incompatible"`. Use the system one.
- `pip install flashinfer-cubin` does not help either. The sampler is a
  ninja-built extension, not one of the downloadable cubins.

The build also needs `ninja` on `PATH`. It is already a venv dependency, so
launch through `uv run` rather than `.venv/bin/python`, or the first compile
fails with `FileNotFoundError: 'ninja'`.

Compilation is a one-time ~30 s, cached in `~/.cache/flashinfer/<version>`. If
you would rather not sort out toolkits, `VLLM_USE_FLASHINFER_SAMPLER=0` falls
back to PyTorch's sampler; at temperature 0 both paths are argmax, so greedy
output is unaffected (this is what the test harness and both GPU CI workflows
do). Verified working on an RTX 5090 (sm_120) with `CUDA_HOME=/usr/local/cuda`
pointing at CUDA 13.1, alongside an apt CUDA 12.4 on `PATH`.

### Developing the Wire Format

The Pydantic models are the source of truth, so there is no spec to edit first and no client
package to publish. To add or change an endpoint:

1. Write the request/response models in `neuronpedia_inference/schemas/`, subclassing
   `BaseSchema`, and export them from that package's `__init__.py`. `BaseSchema` aliases to
   camelCase on the wire, so write snake_case field names and let the alias generator handle it.
2. Document the handler's response with `responses={200: {"model": YourResponse}}` — preferred
   over `response_model=`, which re-validates large payloads at runtime.
3. `make openapi` here (or `make inference-openapi` from the repo root) to rewrite the committed
   `openapi.json`.
4. `make webapp-openapi` from the repo root to regenerate `apps/webapp/lib/api/inference.d.ts`.

Both artifacts are committed and guarded: `tests/unit/test_openapi.py` fails if `openapi.json` is
stale, and `.github/workflows/openapi-drift.yml` fails if the TypeScript is. The full
cross-server contract, including the streaming exceptions, is in the "Cross-server APIs" section
of the repo-root [AGENTS.md](../../AGENTS.md).

## Usage Examples

All endpoints are documented in `openapi.json`, and served interactively at `/docs` once the
server is up. Some usage examples below:

### Get Activations for a Single Feature and Prompt

```bash
curl -X POST http://127.0.0.1:5002/v1/activation/single \
-H "Content-Type: application/json" \
-d '{
 "prompt": "this is about dogs!",
 "model": "gemma-2-2b",
 "source": "20-gemmascope-res-16k",
 "index": "12082"
}'
```

You'll get the following response

```json
{
  "activation": {
    "values": [0, 0, 0, 0, 70.5, 48.25],
    "max_value": 70.5,
    "max_value_index": 4,
    "dfa_values": null,
    "dfa_max_value": null,
    "dfa_target_index": null
  },
  "tokens": ["<bos>", "this", " is", " about", " dogs", "!"]
}
```

### Get Activations From One or More Layers/Sources/SAEs for a Prompt

This gets top features that were activated for layers 0, 2, 10, and 11 in source `res-jb` in `gpt2-small` for prompt `this is about dogs!`.
Append `| jq` to the end for formatted output.

```
curl -X POST http://127.0.0.1:5002/v1/activation/all \
-H "Content-Type: application/json" \
-d '{
 "prompt": "this is about dogs!",
 "model": "gpt2-small",
 "selected_sources": ["0-res-jb", "2-res-jb", "10-res-jb", "11-res-jb"],
 "source_set": "res-jb",
 "sort_by_token_indexes": [],
 "ignore_bos": true
}'
```

### Get Raw Residual Stream Vectors for a Prompt

`/v1/activation/raw` returns the residual stream (`resid_post`) at each prompt's **final
token**, with no SAE involved — useful for embedding prompts in the model's own basis. Omit
`layers` to get every layer.

```
curl -X POST http://127.0.0.1:5002/v1/activation/raw \
-H "Content-Type: application/json" \
-d '{
 "model": "gpt2-small",
 "prompts": ["The Eiffel Tower is in", "The Colosseum is in"],
 "layers": [0, 5, 11]
}'
```

Each result carries the prompt's tokens plus, per requested layer, the index of the token
read and its vector:

```json
{
  "hook_point": "residual_stream",
  "type": "final_output_token",
  "dtype": "float16",
  "device": "cpu",
  "results": [
    {
      "token_strings": ["<|endoftext|>", "The", " E", "iff", "el", " Tower", " is", " in"],
      "token_ids": [50256, 464, 412, 733, 417, 8765, 318, 287],
      "activations": [
        { "layer": 0, "token_indices": [7], "values": [[1.375, -0.2451, 0.7568, "..."]] },
        { "layer": 5, "token_indices": [7], "values": [["..."]] },
        { "layer": 11, "token_indices": [7], "values": [["..."]] }
      ]
    }
  ]
}
```

Up to 16 prompts per request, each bounded by the server's `activation_token_limit` (see
`/v1/capabilities`). On a 16-bit model the values are rounded to 4 decimals, which is all the
precision the checkpoint carries.

### Get Cosine Similarities

```bash
curl -X POST http://127.0.0.1:5002/v1/util/sae-topk-by-decoder-cossim \
  -H "Content-Type: application/json" \
  -d '{
    "feature": {
      "model": "gemma-2-2b",
      "source": "20-gemmascope-res-16k",
      "index": 12082
    },
    "model": "gemma-2-2b",
    "source": "20-gemmascope-res-16k",
    "num_results": 10
  }'
```

### Steering Chat Example gemma-2-2b-it (Returns Dog)

```
uv run python start.py \
  --model_id gemma-2-2b \
  --override_model_id gemma-2-2b-it \
  --sae_sets gemmascope-res-16k \
  --model_dtype bfloat16 \
  --sae_dtype bfloat16
```

```
curl -X POST http://127.0.0.1:5002/v1/steer/completion-chat \
  -H "Content-Type: application/json" \
  -d '{
     "prompt": [{
      "role": "user",
      "content": "Hi, what are you?"
     }],
     "model": "gemma-2-2b-it",
     "features": [
       {
         "model": "gemma-2-2b-it",
         "source": "20-gemmascope-res-16k",
         "index": 12082,
         "strength": 300
       }
     ],
     "types": [
       "STEERED",
       "DEFAULT"
     ],
     "n_completion_tokens": 16,
     "temperature": 0,
     "strength_multiplier": 1,
     "freq_penalty": 0,
     "seed": 16,
     "steer_special_tokens": true,
     "steer_method": "SIMPLE_ADDITIVE",
     "normalize_steering": false
   }'
```

### Steering Example gpt2-small res-jb

Dog feature

```bash
curl -X POST http://127.0.0.1:5002/v1/steer/completion \
  -H "Content-Type: application/json" \
  -d '{
     "prompt": "I often think about",
     "model": "gpt2-small",
     "features": [
       {
         "model": "gpt2-small",
         "source": "7-res-jb",
         "index": 5919,
         "strength": 27
       }
     ],
     "types": [
       "STEERED"
     ],
     "n_completion_tokens": 16,
     "temperature": 0.5,
     "strength_multiplier": 1.5,
     "freq_penalty": 1,
     "seed": 16,
     "steer_method": "SIMPLE_ADDITIVE",
     "normalize_steering": false
   }'
```

### Jacobian Lens Example gemma-3-4b (Base, Raw Prompt)

The `/v1/lens/prompt` endpoint returns a position × layer lens "slice" for a
prompt. `type` is an **array** of one or more of `JACOBIAN_LENS` (uses the fitted
lens loaded at startup) and `LOGIT_LENS` (no fitted lens needed). Requesting both
is essentially free — the model runs only once and the residuals are shared — and
the response returns one entry in `results` per requested type. For a **base**
model, pass a raw text `prompt`.

Start the server for the base model (`gemma-3-4b` → `google/gemma-3-4b-pt`). The
fitted Jacobian lens is downloaded from a Hugging Face model repo (default
`neuronpedia/jacobian-lens`) at startup based on the neuronpedia model id, from
`<np_model_id>/jlens/<dataset>/<slug>_jacobian_lens.pt` (falling back to
`<slug>_jacobian_lens_n1000.pt`, then the first `.pt` in that directory).

You can override the lens loading with flags (each has a matching env var):

- `--jlens_hf_repo <repo>` (`JLENS_HF_REPO`): HF model repo to download from.
- `--jlens_hf_path <path/to/file.pt>` (`JLENS_HF_PATH`): exact path within the
  repo to the lens `.pt`, used verbatim instead of deriving it.
- `--jlens_dataset <name>` (`JLENS_DATASET`): dataset folder the lens was fit on
  (default `Salesforce-wikitext`).
- `--jlens_source <abs/path>` (`JLENS_SOURCE`): load a local lens directory
  instead of downloading.
- `--neuronpedia_model_id <id>` (`NEURONPEDIA_MODEL_ID`): explicit neuronpedia
  model id, only needed when `np_model_to_hf.json` is not at the repo root.
- `--jlens_skip` (`JLENS_SKIP`): start without a fitted lens (LOGIT_LENS still
  works).
- `--jlens_gpu_budget_gib <n|auto|off>` (`JLENS_GPU_BUDGET_GIB`): GPU memory the
  lens may keep its per-layer `J_bar` in. Defaults to `auto`, which measures what
  is left of the card once the model and SAEs are up. A read-out transports
  through every fitted layer on every batch, so raise this rather than leave the
  lens short, and check the startup log for which of these it took:
  - **the whole lens fits** — on vLLM it is uploaded into the worker, and the
    read-out then transports and unembeds there, so residuals never cross the
    process boundary. This is several times faster than the alternative and is
    what the endpoint is sized for.
  - **it does not fit** — the lens stays in the server process, which holds as
    many layers on the GPU as the budget allows and re-copies the rest from host
    memory per batch. Correct, but the residuals make the round trip to the
    worker and back, and the startup log warns.

```
uv run python start.py \
  --model_id google/gemma-3-4b-pt \
  --model_dtype bfloat16 \
  --sae_dtype bfloat16
```

```bash
curl -X POST http://127.0.0.1:5002/v1/lens/prompt \
  -H "Content-Type: application/json" \
  -d '{
     "model": "google/gemma-3-4b-pt",
     "type": ["JACOBIAN_LENS", "LOGIT_LENS"],
     "prompt": "The Eiffel Tower is located in the city of",
     "top_n": 10,
     "layer_stride": 1,
     "include_final_layer": true
   }'
```

The response is shaped `{ "request_info": {...}, "results": [...] }`, with one
item in `results` for each requested type. Shared context (tokens, vocab
fragment, params) lives in `request_info`.

### Jacobian Lens Example gemma-3-4b-it (Instruct, Chat)

For an **instruct** model, pass a `chat` conversation instead of `prompt` (the
tokenizer's chat template is applied automatically). Here we load the instruct
model via `--override_model_id`:

```
uv run python start.py \
  --model_id gemma-3-4b \
  --override_model_id gemma-3-4b-it \
  --model_dtype bfloat16 \
  --sae_dtype bfloat16
```

```bash
curl -X POST http://127.0.0.1:5002/v1/lens/prompt \
  -H "Content-Type: application/json" \
  -d '{
     "model": "gemma-3-4b-it",
     "type": ["JACOBIAN_LENS"],
     "chat": [
       { "role": "user", "content": "What is the capital of France?" }
     ],
     "top_n": 10,
     "layer_stride": 1,
     "include_final_layer": true
   }'
```

## Testing, Linting, and Formatting

This project uses [pytest](https://docs.pytest.org/en/stable/) for testing, [pyright](https://github.com/microsoft/pyright) for type-checking, and [Ruff](https://docs.astral.sh/ruff/) for formatting and linting.

If you add new code, it would be greatly appreciated if you could add tests in the `tests` directory. You can run the tests with:

```bash
make test
```

Before commiting, make sure you format the code with:

```bash
make format
```

Finally, run all CI checks locally with:

```bash
make check-ci
```

If these pass, you're good to go! Open a pull request with your changes.
