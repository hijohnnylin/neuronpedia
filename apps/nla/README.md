# NLA Inference Server

FastAPI server for **NLA (Natural Language Autoencoder)** inference. Translates activation vectors from language models into natural language descriptions and back.

An NLA pair is two fine-tuned LMs:

|                   | Direction     | What it does                                                                       |
| ----------------- | ------------- | ---------------------------------------------------------------------------------- |
| **Verbalizer**    | vector → text | Injects vector as a 1-token embedding, then autoregressively decodes a description |
| **Reconstructor** | text → vector | Truncated LM + linear head, reconstructs the original vector from the description  |

The round-trip MSE (reconstructed vs original) measures how well the description captured the vector's content.

A third model — the **source model** — is the base LM whose activations are being interpreted. The server loads it for `/extract` and `/explain`, and (optionally) for `/completion`.

## Quick start

```bash
cd apps/nla
uv sync

# Default: loads verbalizer + reconstructor + truncated FP8-quantized source on a single GPU
uv run server.py

# With auth
SECRET=my-secret uv run server.py

# Verbalizer only (skip reconstructor and source)
NLA_RECONSTRUCTOR_MODEL="" NLA_SOURCE_MODEL="" uv run server.py
```

Server starts on **port 5009**. The verbalizer model loads via `sgl.Engine` in-process — no separate SGLang server needed.

CLI args override env vars:

```bash
uv run server.py --port 8080 --max-concurrent 32 --mem-fraction 0.45 \
  --truncate-source --fp8-source
```

| Flag                     | Default   | Description                                                                                                                                                                 |
| ------------------------ | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--host`                 | `0.0.0.0` | Bind address                                                                                                                                                                |
| `--port`                 | `5009`    | Port                                                                                                                                                                        |
| `--max-concurrent`       | `24`      | Max concurrent SGLang generations per `/explain`/`/describe` request                                                                                                        |
| `--mem-fraction`         | `0.38`    | GPU memory fraction for sglang (verbalizer weights + KV cache)                                                                                                              |
| `--tp-size`              | `1`       | Tensor parallelism degree for the sglang verbalizer. Prefer per-model GPU pinning over TP on PCIe-only boxes — see [Multi-GPU](#multi-gpu).                                 |
| `--verbalizer-device`    | _auto_    | GPU for the sglang verbalizer (e.g. `cuda:0`). Auto: `cuda:0`.                                                                                                              |
| `--reconstructor-device` | _auto_    | Device for the HF reconstructor (e.g. `cuda:1`). Auto: `cuda:1` on multi-GPU boxes, else `cuda:0`.                                                                          |
| `--source-device`        | _auto_    | Device for the HF source model (e.g. `cuda:1`). Auto: `cuda:1` on multi-GPU boxes, else `cuda:0`.                                                                           |
| `--truncate-source`      | `True`    | Drop source-model layers past extraction layer + lm_head + final norm. Saves ~25% of source VRAM. **Disables `/completion`** — use an external API for completions when on. |
| `--no-truncate-source`   |           | Keep full source model (enables `/completion`)                                                                                                                              |
| `--fp8-verbalizer`       | `False`   | Sglang's runtime FP8 (loads bf16 from disk, converts on GPU). **Hopper+ only** (sm_90+); has a bf16 load-time peak — see [Memory tuning](#memory-tuning--quantization).     |
| `--fp8-source`           | `False`   | FP8 weight-only source model via `torchao`. Works on Ampere+.                                                                                                               |
| `--fp8-reconstructor`    | `False`   | FP8 weight-only reconstructor backbone via `torchao`. Works on Ampere+.                                                                                                     |
| `--kv-cache-dtype`       | _unset_   | sglang KV-cache dtype, e.g. `fp8_e5m2` / `fp8_e4m3`. Halves KV-pool bytes/token. See [Verbalizer decode performance](#verbalizer-decode-performance).                       |
| `--cuda-graph-max-bs`    | _unset_   | Max batch size for sglang's CUDA-graph capture. Match `--max-concurrent`. See [Verbalizer decode performance](#verbalizer-decode-performance).                              |
| `--torch-compile`        | `False`   | `torch.compile` decode kernels. ~10–20% faster decode; **adds ~1–3 min to boot**. See [Verbalizer decode performance](#verbalizer-decode-performance).                      |

## Endpoints

All endpoints require an `X-SECRET-KEY` header if `SECRET` is set.

### `GET /`

Health check. The response reports the active configuration so callers can detect quantization / truncation state without scraping logs.

```bash
curl http://localhost:5009/
```

```json
{
  "status": "ok",
  "verbalizer_model": "kitft/nla-qwen2.5-7b-actor-step4200",
  "d_model": 3584,
  "verbalizer_quantization": null,
  "verbalizer_kv_cache_dtype": null,
  "verbalizer_cuda_graph_max_bs": null,
  "verbalizer_torch_compile": false,
  "verbalizer_device": "cuda:0",
  "reconstructor_available": true,
  "reconstructor_fp8": false,
  "reconstructor_int4": false,
  "reconstructor_device": "cuda:0",
  "source_model": "Qwen/Qwen2.5-7B-Instruct",
  "extraction_layer": 20,
  "source_truncated": true,
  "source_fp8": true,
  "source_device": "cuda:0",
  "completion_available": false,
  "num_cuda_devices": 1
}
```

`completion_available: false` indicates the source model is loaded in truncated mode — `/completion` will return 503 with a clear message.

### `POST /explain`

One-shot: extract activations from the source model and describe them. Combines `/extract` + `/describe` in a single call.

- **`positions` omitted or `[]`** — describes all token positions
- **`positions: [-1]`** — last token only
- **`positions: [0, -1]`** — first and last tokens (Python-style negative indexing)
- **`stream: true`** — stream results as SSE events (one per token position)

```bash
curl -X POST http://localhost:5009/explain \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The capital of France is Paris",
    "positions": [-1],
    "temperature": 0.7
  }'
```

```json
{
  "layer_index": 20,
  "results": [
    {
      "token": "Paris",
      "token_id": 12366,
      "position": 6,
      "l2_norm": 42.3,
      "description": "Tokens related to European capital cities and geography",
      "mse": 0.25,
      "cosine_similarity": 0.87,
      "generated": false,
      "fragment_index": 0,
      "fragment_count": 1
    }
  ]
}
```

Up to `NLA_MAX_CONCURRENT_EXPLAINS` (default `1`) `/explain` requests run in parallel; the (N+1)th request gets HTTP 429 immediately (fail-fast, no queueing). Across all in-flight `/explain` and `/describe` calls, total verbalizer fan-out is bounded server-wide by `NLA_MAX_CONCURRENT`, so sglang's KV-pool occupancy stays capped regardless of how many requests are running. The verbalizer early-stops on `</explanation>` to avoid spending tokens past the close tag.

#### Streaming

With `stream: true`, results are sent as Server-Sent Events (SSE). The first event contains metadata, then each result is sent as it completes, followed by a `[DONE]` sentinel.

```bash
curl --no-buffer -X POST http://localhost:5009/explain \
  -H "Content-Type: application/json" \
  -d '{"text": "The capital of France is Paris", "stream": true}'
```

```
data: {"layer_index": 20, "total": 7, "prompt_length": 7}

data: {"position": 0, "text": "...", "done": false}    # streaming partial verbalizer text

data: {"token":"The","token_id":785,"position":0,"l2_norm":42.3,"description":"...","mse":0.25,"cosine_similarity":0.87,"generated":false,"fragment_index":0,"fragment_count":1}

...

data: [DONE]
```

### `POST /describe`

Describe one or more activation vectors in natural language. Returns MSE confidence scores if the reconstructor is loaded. Set `stream: true` for SSE output.

```bash
curl -X POST http://localhost:5009/describe \
  -H "Content-Type: application/json" \
  -d '{
    "activations": [[0.1, -0.2, ...]],
    "temperature": 0.7,
    "max_new_tokens": 200
  }'
```

Each vector must be length `d_model` (3584 for Qwen 2.5 7B).

```json
{
  "results": [
    {
      "description": "Tokens related to cooking recipes and food preparation",
      "mse": 0.23,
      "cosine_similarity": 0.88
    }
  ]
}
```

- `mse`: Reconstruction MSE, range [0, 4]. ~0.2 = good, ~1 = mediocre, 2 = orthogonal. `null` if no reconstructor.
- `cosine_similarity`: Cosine between original and reconstructed vector. `null` if no reconstructor.

### `POST /score`

Score an existing text description against the original activation vector. Requires the reconstructor model.

```bash
curl -X POST http://localhost:5009/score \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Tokens related to cooking recipes",
    "activation": [0.1, -0.2, ...]
  }'
```

```json
{ "mse": 0.31, "cosine_similarity": 0.85 }
```

### `POST /compare`

Describe the difference between two activation vectors. Computes `a - b`, then describes the resulting direction.

```bash
curl -X POST http://localhost:5009/compare \
  -H "Content-Type: application/json" \
  -d '{
    "activation_a": [0.1, -0.2, ...],
    "activation_b": [0.3, 0.1, ...],
    "temperature": 0.7,
    "max_new_tokens": 200
  }'
```

```json
{
  "description": "Shift from formal to informal register",
  "diff_norm": 142.5,
  "mse": 0.45,
  "cosine_similarity": 0.78
}
```

### `POST /extract`

Extract per-token activation vectors from the source model. Use this to get raw activations for `/describe`.

```bash
curl -X POST http://localhost:5009/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "The capital of France is Paris"}'
```

```json
{
  "layer_index": 20,
  "tokens": [
    {
      "token": "The",
      "token_id": 785,
      "position": 0,
      "activation": [0.1, -0.2, ...],
      "l2_norm": 42.3
    }
  ]
}
```

### `POST /tokenize`

Tokenize text using the source model's tokenizer. Returns per-position metadata including byte-fragment grouping for multi-byte glyphs (CJK, emojis, accents) — clients can use `fragment_index` / `fragment_count` to render multi-token glyphs as a single visual unit.

```bash
curl -X POST http://localhost:5009/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello 世界"}'
```

```json
{
  "tokens": [
    {
      "token": "Hello",
      "token_id": 9707,
      "position": 0,
      "fragment_index": 0,
      "fragment_count": 1
    },
    {
      "token": " ",
      "token_id": 220,
      "position": 1,
      "fragment_index": 0,
      "fragment_count": 1
    },
    {
      "token": "世界",
      "token_id": 99489,
      "position": 2,
      "fragment_index": 0,
      "fragment_count": 2
    },
    {
      "token": "世界",
      "token_id": 99245,
      "position": 3,
      "fragment_index": 1,
      "fragment_count": 2
    }
  ],
  "prompt_length": 4,
  "text": "Hello 世界"
}
```

### `POST /completion`

Generate a continuation for the prompt using the source model. **Requires `--no-truncate-source`** (or `NLA_TRUNCATE_SOURCE=0`) — the truncated source model has no `lm_head` and can't sample tokens. With truncation on, this endpoint returns 503 and tells you how to enable it.

```bash
curl -X POST http://localhost:5009/completion \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The capital of France is",
    "completion_tokens": 16,
    "temperature": 0.7,
    "stream": false
  }'
```

When truncate-source is on (default), most users offload `/completion` to an external API like OpenRouter and keep the local source model in extraction-only mode — that's what enables the FP8/truncate VRAM savings.

## Configuration

All configuration is via environment variables. Defaults are for the Qwen 2.5 7B NLA checkpoints (`actor` = verbalizer, `critic` = reconstructor; older naming kept for HF Hub compatibility).

### Models

| Variable                  | Default                                | Description                                                                         |
| ------------------------- | -------------------------------------- | ----------------------------------------------------------------------------------- |
| `NLA_VERBALIZER_MODEL`    | `kitft/nla-qwen2.5-7b-actor-step4200`  | HF hub ID or local path to verbalizer checkpoint                                    |
| `NLA_RECONSTRUCTOR_MODEL` | `kitft/nla-qwen2.5-7b-critic-step4200` | HF hub ID or local path to reconstructor checkpoint. Set to empty string to disable |
| `NLA_SOURCE_MODEL`        | `Qwen/Qwen2.5-7B-Instruct`             | Base model for activation extraction. Set to empty string to disable                |

### Memory tuning & quantization

These flags control how much VRAM the server consumes. The default config (truncated source + bf16 verbalizer + bf16 reconstructor) fits a 7B NLA + 7B source on a single 48 GB A40 with `mem_fraction=0.38`.

| Variable / Flag                                 | Default   | What it does                                                                                                                                                                                                                      |
| ----------------------------------------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `NLA_TRUNCATE_SOURCE` / `--truncate-source`     | `1` (on)  | Drop source-model layers past `extraction_layer` + `lm_head` + final norm. Saves ~25% of source-model VRAM (~4 GB on Qwen 7B at layer 20). **Disables `/completion`.**                                                            |
| `NLA_FP8_VERBALIZER` / `--fp8-verbalizer`       | `0` (off) | Sglang's runtime FP8 quantization (loads bf16 weights, converts to fp8 on GPU). Saves ~50% on verbalizer weights post-load, BUT **peak GPU usage during load is the bf16 size** — for a bf16-on-disk 70B model that's ~141 GB and won't fit a single 96 GB GPU. **Hopper+ only** (sm_90+); fails on Ampere/Ada with `type fp8e4nv not supported in this architecture`. **Pre-quantized checkpoints (compressed-tensors / AWQ / GPTQ) don't need this flag** — sglang auto-detects the recipe from `config.json`. See [Pre-quantized checkpoints](#pre-quantized-checkpoints-build-at-rest-skip-runtime-quant). |
| `NLA_FP8_SOURCE` / `--fp8-source`               | `0` (off) | FP8 weight-only source model via `torchao` (`Float8WeightOnlyConfig`). Saves ~50% on the kept source layers. Works on Ampere/Ada/Hopper/Blackwell.                                                                                |
| `NLA_FP8_RECONSTRUCTOR` / `--fp8-reconstructor` | `0` (off) | FP8 weight-only reconstructor backbone via `torchao`. Saves ~50% on kept reconstructor layers. The `value_head` stays in bf16 to preserve output-projection numerics.                                                             |

Recommended dtype combos (single GPU):

| GPU                          | VRAM      | Recommended config                                                                                                                               |
| ---------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| A40 / A100 (Ampere)          | 40-80 GB  | `--truncate-source --fp8-source` (FP8 verbalizer not available)                                                                                  |
| RTX Pro 6000 Blackwell       | 96 GB     | `--truncate-source --fp8-source --fp8-verbalizer --fp8-reconstructor`                                                                            |
| H100 / H200 / B200 (Hopper+) | 80-192 GB | `--truncate-source --fp8-source` for headroom; add `--fp8-verbalizer --fp8-reconstructor` if scaling to bigger models (e.g. 27B+ on H100 80 GB). |

`--fp8-source` and `--fp8-reconstructor` require `torchao` (already in the project deps). On Ampere/Ada, FP8 is weight-only (storage win, no compute speedup). On Hopper+ you also get the FP8 tensor-core compute path.

### Pre-quantized checkpoints (build at rest, skip runtime quant)

The `--fp8-*` flags above quantize **every boot**, which (1) costs ~10–60 s of CPU/GPU time per restart, (2) holds a transient bf16 copy of the weights during conversion, and (3) leans on `torchao`, which sglang 0.5.x cannot consume. For production deployments — especially the verbalizer, which sglang loads — bake the quantization into the checkpoint once and ship the pre-quantized artifact. Three sibling PEP 723 scripts produce **`compressed-tensors`-format** checkpoints (canonical `*.safetensors`, no pickle, no torchao) loadable by both HF transformers and sglang:

| Script                                       | Target                                                                                                              | Default scheme | Notes                                                                                                                                                  |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `build_compressed_tensors_verbalizer.py`     | sglang verbalizer                                                                                                   | `FP8_DYNAMIC`  | Eliminates the bf16 load-time transient (sglang loads fp8 buffers directly). Required for fitting Llama-3.3-70B-NLA-av on a single 96 GB GPU.            |
| `build_compressed_tensors_reconstructor.py`  | HF reconstructor backbone                                                                                           | `W8A16` (FP8 weight-only) | Bake FP8 or FP4 in. `--quant fp8` -> `W8A16` (low-drift, recommended). `--quant fp4` -> `NVFP4A16` (FP4 weight-only, Blackwell sm_100+ for sglang). `value_head.safetensors` is preserved verbatim in bf16. |
| `build_compressed_tensors_source.py`         | HF source model (`google/gemma-3-27b-it`)                                                                           | `W8A16` (FP8 weight-only) | Vision-stripped, layer-truncated, re-saved as `Gemma3ForCausalLM`, FP8-quantized. Replaces the full `--truncate-source --fp8-source` runtime path with a single pre-built artifact. |

All three are `uv run`-able standalone (PEP 723 inline-metadata scripts isolate them from the main project venv, which is necessary because `llmcompressor`'s deps and `sglang`'s pinned deps don't overlap cleanly):

```bash
# Verbalizer (FP8 for sglang)
HF_TOKEN=hf_xxx uv run build_compressed_tensors_verbalizer.py \
    --verbalizer-model kitft/Llama-3.3-70B-NLA-av --upload

# Reconstructor (FP8 weight-only, lowest activation drift)
HF_TOKEN=hf_xxx uv run build_compressed_tensors_reconstructor.py \
    --reconstructor-model kitft/nla-gemma3-27b-ar --quant fp8 --upload

# Reconstructor (FP4 weight-only, NVFP4A16; Blackwell-only for sglang loading)
HF_TOKEN=hf_xxx uv run build_compressed_tensors_reconstructor.py \
    --reconstructor-model kitft/nla-gemma3-27b-ar --quant fp4 --upload

# Source (Gemma-3-27B-IT, vision-stripped + truncated to layer 41 + FP8)
HF_TOKEN=hf_xxx uv run build_compressed_tensors_source.py \
    --source-model google/gemma-3-27b-it --layer-index 41 --upload
```

Once you've published the pre-quantized artifacts, point the server at them and **disable the runtime-quant flags**:

```bash
NLA_VERBALIZER_MODEL=<your-org>/Llama-3.3-70B-NLA-av-FP8 \
NLA_RECONSTRUCTOR_MODEL=<your-org>/nla-gemma3-27b-ar-FP8 \
NLA_SOURCE_MODEL=<your-org>/gemma-3-27b-it-FP8-trunc41 \
NLA_OVERRIDE_EXTRACTION_LAYER=41 \
NLA_TRUNCATE_SOURCE=0 \
NLA_FP8_SOURCE=0 \
NLA_FP8_RECONSTRUCTOR=0 \
NLA_INT4_RECONSTRUCTOR=0 \
uv run server.py
```

That's it — no per-model quantization flag is needed. All three loaders auto-detect `quantization_config.quant_method` from `config.json`:
- HF transformers does this for the source + reconstructor via `CompressedTensorsHfQuantizer`.
- Sglang does the same for the verbalizer — its config-auto-detect reads the same `quant_method` field and routes to the matching scheme (`compressed-tensors`, `awq`, `gptq`, …).

`server.py` no longer exposes a flag to pin the verbalizer quantization mode explicitly — `NLA_FP8_VERBALIZER` (and `--fp8-verbalizer`) is purely for sglang's **runtime** bf16->fp8 conversion path; pre-quantized checkpoints don't go through that path. If you need to override sglang's auto-detect (rare; mostly only useful if a specific sglang version mis-routes a config), edit the `quantization=` argument passed to `sgl.Engine` in `init_models()` directly.

Boot time drops from ~30–90 s of runtime quantization to whatever HF + sglang need to mmap the safetensors shards — typically 5–15 s.

> **Why not the older `build_quantized_models.py`?** That script uses `torchao.AffineQuantizedTensor` + `safe_serialization=False` (pickled `pytorch_model-*.bin` shards). The torchao path:
> - is not loadable by sglang at any version (`"torchao"` is not in sglang's quantization registry);
> - has surfaced HF/torchao version-mismatch loading errors in this stack;
> - cannot be safely loaded with `safetensors`-only loaders.
>
> The `compressed-tensors` scripts above are the canonical replacement. `build_quantized_models.py` is kept around for backward compatibility with existing checkpoints but new builds should use the `compressed-tensors` scripts.

### Verbalizer decode performance

Once the model and quantization are fixed, three sglang knobs trade boot time / memory for verbalizer decode throughput. They're all optional and independent — each one can be toggled without touching the others.

| Variable / Flag                             | Default | What it does                                                                                                                                                                                                                                              | Accuracy cost                                                                                                                                            |
| ------------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `NLA_KV_CACHE_DTYPE` / `--kv-cache-dtype`   | _unset_ | sglang KV-cache dtype, e.g. `fp8_e5m2` or `fp8_e4m3`. Halves KV-pool bytes-per-token vs bf16 — frees ~1.5–3 GB on a 27B verbalizer at batch 32 that you can spend on more `NLA_MAX_CONCURRENT` or just headroom. Native FP8 compute on Hopper+/Blackwell. | **Small but real.** ~0.2–0.8% on standard chat-evals; for the verbalizer this can show up as slightly more unclosed `</explanation>` tags. Validate against your reconstructor MSE/cosine before shipping. `fp8_e4m3` is usually marginally better quality than `fp8_e5m2`. |
| `NLA_CUDA_GRAPH_MAX_BS` / `--cuda-graph-max-bs` | _unset_ | Max batch size for sglang's CUDA-graph capture. sglang's default ceiling (8 or 32 depending on version) caps the captured fast path; if `NLA_MAX_CONCURRENT` is higher, peak fan-out drops into eager-mode kernel launches. Bump to match `NLA_MAX_CONCURRENT` to keep peak decode on the captured path (typically +5–15% on decode at high concurrency). | **None.** CUDA graphs are record-and-replay — bit-identical to eager mode.                                                                              |
| `NLA_TORCH_COMPILE` / `--torch-compile`     | `0`     | Ask sglang to `torch.compile` the decode kernels. ~10–20% extra on decode after warmup. Free at runtime; no steady-state VRAM cost.                                                                                                                       | **Effectively none** in practice — kernel fusion can shift FP-add order at the last bit, drowned out by sampling noise at any non-zero temperature.       |

**Boot-time cost of `--torch-compile`:** ~1–3 minutes on cold boot (sglang compiles kernels across multiple captured batch shapes plus runs warmup). Approximate scale: ~45 s for a 7B verbalizer, ~2 min for a 27B. Set `TORCHINDUCTOR_CACHE_DIR` to a persistent path (e.g. `/root/.cache/torchinductor`) to amortize across restarts.

Stacking all three on a memory-tight 27B FP8 setup (e.g. RTX Pro 6000 Blackwell, 90+ GB used) typically yields **1.3–1.5× throughput on `/describe` and `/explain`** at batch 32. They compose linearly — if any one regresses or OOMs at boot, drop just that flag. Recommended order to enable, easiest first:

1. `NLA_CUDA_GRAPH_MAX_BS=<your NLA_MAX_CONCURRENT>` — free, no-risk.
2. `NLA_TORCH_COMPILE=1` — free at runtime, costs minutes at boot once `TORCHINDUCTOR_CACHE_DIR` is set.
3. `NLA_KV_CACHE_DTYPE=fp8_e4m3` — A/B against your reconstructor scores; revert to `fp8_e5m2` or unset if quality drops.

### Multi-GPU

When more than one CUDA device is visible, the server **pins each model to one GPU** rather than tensor-parallelizing any single model. This is strictly better than TP on PCIe-only boxes (no NVLink), where NCCL all-reduce after every transformer block eats the TP speedup; per-model pinning skips NCCL entirely and gets free pipeline parallelism (the verbalizer fan-out for one `/explain` can overlap with the source extract for the next).

Auto layout (no env vars required):

| GPU      | What runs there                           |
| -------- | ----------------------------------------- |
| `cuda:0` | sglang verbalizer (biggest VRAM consumer) |
| `cuda:1` | HF source + HF reconstructor              |

Override per model:

```bash
NLA_VERBALIZER_DEVICE=cuda:0 \
NLA_SOURCE_DEVICE=cuda:1 \
NLA_RECONSTRUCTOR_DEVICE=cuda:1 \
uv run server.py --truncate-source --fp8-source --fp8-verbalizer --int4-reconstructor
```

Or via CLI flags: `--verbalizer-device cuda:0 --source-device cuda:1 --reconstructor-device cuda:1`.

The `verbalizer_device`, `reconstructor_device`, and `source_device` fields in `GET /` confirm the actual placement at runtime.

#### Critical: verbalizer load-time bf16 transient

`--fp8-verbalizer` tells sglang to use FP8 at runtime, but the **on-disk format of the checkpoint** still determines whether the load itself fits in memory. Sglang's FP8 path (`sglang/srt/layers/quantization/fp8.py:create_weights`) checks `is_checkpoint_fp8_serialized`:

- **FP8-serialized on disk** (e.g. `RedHatAI/*-FP8` `compressed-tensors` checkpoints): sglang allocates fp8 weight buffers directly. Peak GPU usage during load ≈ **final fp8 size (~70 GB for Llama-3.3-70B)**.
- **BF16 on disk** (e.g. `kitft/Llama-3.3-70B-NLA-av`, `meta-llama/Llama-3.3-70B-Instruct`, most fine-tuned HF checkpoints): sglang allocates **bf16** weight buffers per Linear, loads bf16 weights, then quantizes to fp8 via `process_weights_after_loading`. Peak GPU usage during load ≈ **bf16 size (~141 GB for Llama-3.3-70B)** — does NOT fit on a single 96 GB GPU regardless of `mem_fraction`.

The OOM looks like this and is **not fixable by tuning `mem_fraction`** — the failure is in `Fp8LinearMethod.create_weights` allocating bf16 staging tensors before the post-load quantization pass:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 896.00 MiB.
GPU 0 has a total capacity of 94.97 GiB of which 291.88 MiB is free.
... this process has 92.18 GiB memory in use.
```

If you hit this, you have two options:

1. **Pre-quantize the verbalizer to FP8 on disk in `compressed-tensors` format** (via [`build_compressed_tensors_verbalizer.py`](#pre-quantized-checkpoints-build-at-rest-skip-runtime-quant), which wraps [`llmcompressor`](https://github.com/vllm-project/llm-compressor)). Sglang loads it directly with no bf16 transient — auto-detected from `config.json`'s `quantization_config.quant_method`, no per-model quant flag required.
2. **Use `--tp-size 2`** to split the bf16 transient across both GPUs at load (Layout D below). Each GPU peaks at ~70 GB bf16 instead of ~141 GB on one. Final per-GPU verb shard is ~35 GB FP8.

> **`build_quantized_models.py --target verbalizer-fp8` is NOT a fix here.** That script produces a `torchao` (`AffineQuantizedTensor`-backed) FP8 checkpoint, which sglang 0.5.x cannot load — `"torchao"` is not in sglang's `QUANTIZATION_METHODS` registry, and a config.json with `quant_method="torchao"` will crash sglang at engine init with `ValueError: Invalid quantization method`. Use `build_compressed_tensors_verbalizer.py` (llmcompressor + compressed-tensors) for sglang-compatible FP8 — see [Pre-quantized checkpoints](#pre-quantized-checkpoints-build-at-rest-skip-runtime-quant).

For Llama-3.3-70B specifically, until you've built a `compressed-tensors` FP8 verbalizer, **Layout A is theoretical**; Layout D is what actually loads.

Recommended layouts on 2× RTX Pro 6000 Blackwell (96 GB each, PCIe Gen5, no NVLink) for **Llama-3.3-70B**:

#### Layout A — FP8 verbalizer + FP8 source + INT4 reconstructor (canonical, **requires FP8-serialized verb checkpoint**)

> Only viable if the verbalizer checkpoint is FP8-serialized on disk (see [load-time transient note above](#critical-verbalizer-load-time-bf16-transient)). For a bf16-on-disk verbalizer, use Layout D instead.

| Model         | Device   | Quantization              | Approx. VRAM             |
| ------------- | -------- | ------------------------- | ------------------------ |
| Verbalizer    | `cuda:0` | FP8 (sglang)              | ~70 GB weights + KV pool |
| Source        | `cuda:1` | FP8 (torchao), truncated  | ~47 GB                   |
| Reconstructor | `cuda:1` | INT4 (torchao)            | ~26 GB                   |

```bash
NLA_SOURCE_MODEL=meta-llama/Llama-3.3-70B-Instruct \
NLA_OVERRIDE_EXTRACTION_LAYER=<your-layer> \
NLA_MEM_FRACTION=0.85 \
NLA_MAX_CONCURRENT=32 \
uv run server.py \
  --truncate-source --fp8-source --fp8-verbalizer --int4-reconstructor
# uses auto layout — verbalizer→cuda:0, source+reconstructor→cuda:1
```

Per-GPU totals: `cuda:0` ≈ ~70 GB weights + ~10 GB KV pool + ~3 GB sglang/CUDA overhead ≈ **~83 GB** at `mem_fraction=0.85` (≈13 GB free); `cuda:1` ≈ 47 + 26 + ~3 GB CUDA context ≈ **~76 GB** (≈20 GB free).

#### Layout B — FP8 verbalizer + AWQ source + INT4 reconstructor (max headroom)

| Model         | Device   | Quantization             | Approx. VRAM |
| ------------- | -------- | ------------------------ | ------------ |
| Verbalizer    | `cuda:0` | FP8 (sglang)             | ~70 GB       |
| Source        | `cuda:1` | AWQ INT4 (HF), truncated | ~27 GB       |
| Reconstructor | `cuda:1` | INT4 (torchao)           | ~26 GB       |

```bash
NLA_SOURCE_MODEL=casperhansen/llama-3.3-70b-instruct-awq \
NLA_OVERRIDE_EXTRACTION_LAYER=<your-layer> \
NLA_MEM_FRACTION=0.85 \
NLA_MAX_CONCURRENT=32 \
uv run server.py \
  --truncate-source --fp8-verbalizer --int4-reconstructor
# (no --fp8-source — the AWQ source ships pre-quantized)
```

Per-GPU totals: `cuda:0` ≈ ~83 GB at `mem_fraction=0.85`; `cuda:1` ≈ 27 + 26 + ~3 GB ≈ **~56 GB** (≈40 GB free — bump `NLA_SOURCE_MAX_CONCURRENT` to take advantage).

#### Layout C — explicit pin (e.g. swap which GPU holds what)

If your GPU 0 is busy with another workload, or you prefer the verbalizer on GPU 1:

```bash
NLA_VERBALIZER_DEVICE=cuda:1 \
NLA_RECONSTRUCTOR_DEVICE=cuda:0 \
NLA_SOURCE_DEVICE=cuda:0 \
uv run server.py \
  --truncate-source --fp8-source --fp8-verbalizer --int4-reconstructor
```

Or equivalently via CLI: `--verbalizer-device cuda:1 --reconstructor-device cuda:0 --source-device cuda:0`.

#### Layout D — TP=2 verbalizer with HF colocation (**works for bf16-on-disk verb checkpoints**)

This is the practical layout for an unmodified bf16 verbalizer checkpoint on 2× 96 GB. Sglang shards the bf16 weights across both GPUs at load time, sidestepping the ~141 GB bf16 transient peak. Each GPU ends up with ~35 GB FP8 verb shard + one HF model. Expect ~0–30% verbalizer throughput hit vs. single-GPU FP8 due to PCIe-bound NCCL all-reduce — accept it as the cost of fitting bf16 weights on this hardware.

| Model         | Device              | Quantization                    | Approx. VRAM           |
| ------------- | ------------------- | ------------------------------- | ---------------------- |
| Verbalizer    | `cuda:0` + `cuda:1` | FP8 (sglang, TP=2)              | ~35 GB / GPU + KV pool |
| Reconstructor | `cuda:0`            | INT4 (torchao)                  | ~26 GB                 |
| Source        | `cuda:1`            | FP8 (torchao), truncated        | ~47 GB                 |

```bash
NLA_SOURCE_MODEL=meta-llama/Llama-3.3-70B-Instruct \
NLA_OVERRIDE_EXTRACTION_LAYER=<your-layer> \
NLA_MEM_FRACTION=0.45 \
NLA_RECONSTRUCTOR_DEVICE=cuda:0 \
NLA_SOURCE_DEVICE=cuda:1 \
NLA_MAX_CONCURRENT=32 \
uv run server.py \
  --tp-size 2 \
  --truncate-source --fp8-source --fp8-verbalizer --int4-reconstructor
```

Per-GPU totals at `mem_fraction=0.45`:
- `cuda:0`: ~35 GB verb shard + ~26 GB INT4 reconstructor + ~6 GB KV pool slice + ~3 GB sglang/CUDA overhead ≈ **~70 GB** (≈26 GB free)
- `cuda:1`: ~35 GB verb shard + ~47 GB FP8 source + ~6 GB KV pool slice + ~3 GB sglang/CUDA overhead ≈ **~91 GB** (≈5 GB free, **tight** — monitor with `nvidia-smi`)

If `cuda:1` OOMs on source load, swap to AWQ source for ~20 GB headroom:

```bash
NLA_SOURCE_MODEL=casperhansen/llama-3.3-70b-instruct-awq \
# drop --fp8-source — the AWQ checkpoint ships pre-quantized
NLA_MEM_FRACTION=0.65   # source is now only 27 GB on cuda:1, so sglang can have more pool
```

With the AWQ source variant, both GPUs end up around 70 GB used and you get ~30 GB of KV pool combined.

#### Layout E — pre-quantized FP8 verbalizer (`compressed-tensors`), single GPU

If you've produced an FP8-serialized verbalizer in `compressed-tensors` format (via `llmcompressor`), sglang loads it directly with no bf16 transient. This is the production-grade alternative to Layout D — the verbalizer fits on one GPU, no NCCL during inference, no PCIe TP overhead.

| Model         | Device   | Quantization                | Approx. VRAM             |
| ------------- | -------- | --------------------------- | ------------------------ |
| Verbalizer    | `cuda:0` | FP8 (`compressed-tensors`)  | ~70 GB weights + KV pool |
| Reconstructor | `cuda:1` | INT4 (torchao)              | ~26 GB                   |
| Source        | `cuda:1` | FP8 (torchao), truncated    | ~47 GB                   |

```bash
# Verbalizer must be a compressed-tensors FP8 checkpoint (e.g. produced by
# `build_compressed_tensors_verbalizer.py` from kitft/Llama-3.3-70B-NLA-av).
NLA_VERBALIZER_MODEL=<your-compressed-tensors-fp8-checkpoint> \
NLA_SOURCE_MODEL=meta-llama/Llama-3.3-70B-Instruct \
NLA_OVERRIDE_EXTRACTION_LAYER=53 \
NLA_MEM_FRACTION=0.85 \
NLA_MAX_CONCURRENT=32 \
uv run server.py \
  --truncate-source --fp8-source --int4-reconstructor
# (--fp8-verbalizer is NOT used; the checkpoint is already fp8 — sglang
#  auto-detects compressed-tensors from config.json's quantization_config.)
```

Per-GPU totals: same as Layout A — `cuda:0` ≈ ~83 GB at `mem_fraction=0.85`; `cuda:1` ≈ ~76 GB. No NCCL, full per-token throughput on the verbalizer.

#### Picking `NLA_MEM_FRACTION`

**What it actually controls.** `mem_fraction_static` is the **fraction of each GPU's total memory** that sglang will use for **its own weights + KV cache pool**. From `sglang/srt/model_executor/model_runner_kv_cache_mixin.py:138`:

```
KV pool budget = available_gpu_memory_after_weights
                 − total_gpu_memory × (1 − mem_fraction_static)
```

Equivalently: sglang uses up to `mem_fraction × total` per GPU for `(weights + KV pool)`, and **leaves `(1 − mem_fraction) × total` free** for everything else (CUDA graphs, activations, **and any other model you've put on the same GPU**). It does NOT throttle weight loading itself — weights load until they fit (or OOM), then whatever budget is left becomes the KV pool. If the post-weights leftover is negative, you get `Not enough memory. Please try to increase --mem-fraction-static.`

The right mem_fraction depends on whether sglang has the GPU to itself or shares it with an HF model.

##### Case 1 — verbalizer alone on its GPU (Layouts A / B / C)

Weights are fixed (~70 GB FP8 for Llama-3.3-70B), so `mem_fraction` effectively sizes the KV pool:

```
KV pool ≈ mem_fraction × 96 GB − weights (~70 GB) − sglang overhead (~3 GB)
```

KV bytes per token for Llama-3.3-70B GQA: `80 layers × 8 KV heads × 128 head_dim × 2 (K+V) × 2 bytes (bf16) ≈ 320 KiB/token`. For `NLA_MAX_CONCURRENT=N` parallel verbalizer streams of up to 300 tokens each (≈100 prompt + 200 gen), worst-case pool occupancy is `N × 300 × 320 KiB`.

| `NLA_MAX_CONCURRENT` | Worst-case KV needed | Recommended `NLA_MEM_FRACTION` (96 GB GPU) | Resulting pool | Free VRAM |
| -------------------- | -------------------- | ------------------------------------------ | -------------- | --------- |
| 8                    | ~0.8 GB              | `0.80`                                     | ~4 GB          | ~19 GB    |
| 16                   | ~1.6 GB              | `0.82`                                     | ~6 GB          | ~17 GB    |
| 32                   | ~3.0 GB              | `0.85`                                     | ~9 GB          | ~14 GB    |
| 64                   | ~6.0 GB              | `0.88`                                     | ~12 GB         | ~12 GB    |
| 128                  | ~12 GB               | `0.93`                                     | ~17 GB         | ~7 GB     |

Heuristic: pick a pool ≈ **2–3× the worst-case need** so sglang has slack for fragmentation, prefix variability, and bursty arrivals.

##### Case 2 — verbalizer (TP=N) shares GPUs with HF models (Layout D)

Sglang doesn't know about HF allocations. You have to leave room for them inside the `(1 − mem_fraction)` reservation. Per GPU:

```
mem_fraction × 96 GB  ≤  96 GB − HF_size_on_this_gpu − ctx_overhead (~3 GB) − HF_safety (~2 GB)
```

Use the **most-loaded GPU** to pick a single `mem_fraction` that satisfies all GPUs simultaneously. For Llama-3.3-70B + TP=2:

| HF model on a GPU              | Required free per GPU | Max `mem_fraction` | Resulting per-GPU sglang budget |
| ------------------------------ | --------------------- | ------------------ | ------------------------------- |
| INT4 reconstructor (~26 GB)    | ~31 GB                | ~`0.68`            | ~65 GB                          |
| FP8 source truncated (~47 GB)  | ~52 GB                | ~`0.45`            | ~43 GB                          |
| AWQ source truncated (~27 GB)  | ~32 GB                | ~`0.66`            | ~63 GB                          |

So the practical recipes for Llama-3.3-70B TP=2:

- FP8 source on one GPU → **`mem_fraction=0.45`** (constrained by the FP8 source GPU).
- AWQ source on one GPU → **`mem_fraction=0.65`** (both GPUs balanced ~30 GB HF).

##### Sanity checking at runtime

If startup logs show `init_kv_pool` errors or `Not enough memory. Please try to increase --mem-fraction-static`, weights didn't fit — either lower the colocated HF model size or, on Case 1, raise `mem_fraction` (you have headroom). If you OOM during HF model load on a colocated GPU (your sglang process is fine but HF process can't get its weights in), **lower** `mem_fraction` so sglang reserves more free space.

`GET /` reports the actual placement of each model so you can confirm. Use `benchmark/sweep.py` to find the empirical optimum once load fits.

#### What NOT to do

| Anti-pattern                                                | Why it breaks                                                                                                                                                                                                                                                                                            |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Verbalizer + reconstructor both on `cuda:0` (FP8 verb)      | ~70 GB verb + ~26 GB recon ≈ 96 GB before KV pool — OOMs unless `NLA_MEM_FRACTION` is cut to ~0.10                                                                                                                                                                                                       |
| `--fp8-verbalizer` on a bf16-on-disk checkpoint, single GPU | Sglang allocates bf16 staging buffers per Linear; peak is ~141 GB (full bf16) before post-load fp8 conversion, doesn't fit a 96 GB GPU. Either use TP=2 (Layout D) or pre-quantize the verbalizer to `compressed-tensors` FP8 with `build_compressed_tensors_verbalizer.py`; sglang loads pre-quantized checkpoints directly via auto-detect from `config.json` with no flag and no bf16 transient (Layout E). `build_quantized_models.py --target verbalizer-fp8` does NOT help — its torchao output is not loadable by sglang 0.5.x. |
| Source on one GPU, reconstructor on the other (Layouts A/B) | Adds a per-`/explain` cross-GPU `score()`; marginal latency hit, wastes the colocation win                                                                                                                                                                                                               |
| `--tp-size 2` "for performance" on PCIe-only boxes          | NCCL all-reduce over PCIe dominates per-layer cost; pure-throughput win is small (~0–30%) and often negative under contention. **TP=2 is justified for fitting the bf16 load transient (Layout D), not for speed.** Once you have a pre-quantized FP8 verbalizer, TP=1 + per-model pinning is preferred. |

`NLA_TP_SIZE > 1` is still wired through to sglang for the verbalizer only, but is **not recommended** unless you have NVLink-class GPUs (H100 NVL, H200, B200) where NCCL latency is sub-millisecond. On PCIe boxes the per-layer all-reduce overhead typically wipes out the TP speedup.

### Engine / runtime

| Variable                    | Default  | Description                                                                                                                                                                                                        |
| --------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `SECRET`                    | _(none)_ | Auth secret — requests must include `X-SECRET-KEY` header                                                                                                                                                          |
| `NLA_TP_SIZE`               | `1`      | Tensor parallelism degree for the sglang verbalizer. The HF source/reconstructor are NOT tensor-parallel — only sglang honors this. Prefer per-model GPU pinning (see [Multi-GPU](#multi-gpu)) on PCIe-only boxes. |
| `NLA_VERBALIZER_DEVICE`     | _auto_   | GPU placement for the sglang verbalizer (e.g. `cuda:0`). Passed to sglang as `base_gpu_id`. Auto: `cuda:0`.                                                                                                        |
| `NLA_RECONSTRUCTOR_DEVICE`  | _auto_   | Device for the HF reconstructor (e.g. `cuda:1`, `cpu`). Auto: `cuda:1` on multi-GPU boxes, else `cuda:0`.                                                                                                          |
| `NLA_SOURCE_DEVICE`         | _auto_   | Device for the HF source model (e.g. `cuda:1`, `cpu`). Auto: `cuda:1` on multi-GPU boxes, else `cuda:0`.                                                                                                           |
| `NLA_MEM_FRACTION`          | `0.38`   | GPU memory fraction for sglang (verbalizer + KV cache). Tune per hardware via `benchmark/sweep.py`.                                                                                                                |
| `NLA_MAX_CONCURRENT`        | `24`     | Max concurrent verbalizer generations per `/explain`/`/describe` request. Bounds sglang KV pool occupancy.                                                                                                         |
| `NLA_SOURCE_MAX_CONCURRENT` | `4`      | Max concurrent source-model GPU ops (`/extract`, `/explain` extract phase, `/completion`). Lower if you OOM under load.                                                                                            |
| `NLA_RECONSTRUCTION_BATCH_SIZE` | `4` | How many reconstructor `score()` calls to coalesce into one forward pass during streaming `/describe` and `/explain`. The batcher fires as soon as this many streams have produced an explanation, OR as soon as no more submissions are pending (partial flush) — early finishers never wait for ALL streams to complete. Set to `1` to disable batching. |

Default device selection: each model is auto-pinned to a single GPU (CUDA → MPS → CPU). On multi-GPU boxes the verbalizer goes to `cuda:0` and the source + reconstructor share `cuda:1` — see [Multi-GPU](#multi-gpu).

### NLA metadata overrides

These are normally loaded from `nla_meta.yaml` in the checkpoint directory and should NOT be set unless you need to override the sidecar (e.g. testing a checkpoint without the metadata file).

| Variable                                     | Description                                                     |
| -------------------------------------------- | --------------------------------------------------------------- |
| `NLA_OVERRIDE_D_MODEL`                       | Model hidden size (e.g. 3584 for Qwen 7B)                       |
| `NLA_OVERRIDE_INJECTION_SCALE`               | L2-norm vectors get rescaled to before injection (e.g. 150)     |
| `NLA_OVERRIDE_INJECTION_CHAR`                | Injection marker character (e.g. `㈎` U+320E for Qwen)          |
| `NLA_OVERRIDE_INJECTION_TOKEN_ID`            | Token ID for the injection char                                 |
| `NLA_OVERRIDE_INJECTION_LEFT_NEIGHBOR`       | Token ID immediately left of the injection position             |
| `NLA_OVERRIDE_INJECTION_RIGHT_NEIGHBOR`      | Token ID immediately right of the injection position            |
| `NLA_OVERRIDE_VERBALIZER_PROMPT_TEMPLATE`    | Verbalizer prompt template; must contain `{injection_char}`     |
| `NLA_OVERRIDE_RECONSTRUCTOR_PROMPT_TEMPLATE` | Reconstructor prompt template; must contain `{explanation}`     |
| `NLA_OVERRIDE_EXTRACTION_LAYER`              | Which layer to extract activations from (~2/3 depth typically)  |
| `NLA_OVERRIDE_MSE_SCALE`                     | Reconstructor MSE normalization scale (default `sqrt(d_model)`) |

## Example: Gemma-3-12B

```bash
NLA_VERBALIZER_MODEL=kitft/nla-gemma3-12b-actor \
NLA_RECONSTRUCTOR_MODEL=kitft/nla-gemma3-12b-critic \
NLA_SOURCE_MODEL=google/gemma-3-12b-it \
NLA_OVERRIDE_EXTRACTION_LAYER=32 \
NLA_MEM_FRACTION=0.31 \
HF_TOKEN=hf_... \
uv run server.py --truncate-source --fp8-source
```

Note: Gemma-3 requires `HF_TOKEN` set (gated repo). The embedding scale (`sqrt(hidden) ≈ 62` for Gemma-3-12B's hidden=3840) is handled automatically.

## Benchmarking

The `benchmark/` subfolder has empirical tools to find the right `NLA_MEM_FRACTION` and `NLA_MAX_CONCURRENT` for your specific GPU + model + dtype combo, optimized for the `/explain` workload. See [`benchmark/README.md`](benchmark/README.md) for details.

Quick fast path (probe an already-running server, no model reloads):

```bash
SECRET=mysecret uv run python benchmark/probe.py \
    --concurrencies 1,2,4,8,16,32,64 \
    --max-context-size 4096 \
    --output report.json
```

Sweep multiple `mem_fraction` values (does require a server reboot per value):

```bash
SECRET=mysecret HF_HOME=/workspace/.cache/huggingface \
uv run python benchmark/sweep.py \
    --mem-fractions 0.35,0.45,0.55 \
    --output-dir reports/A40-fp8-source/ \
    -- \
    --truncate-source --fp8-source --max-concurrent 96 \
    --probe-concurrencies 1,4,16,32,64,96
```

Reports include per-concurrency latency, peak VRAM, and a recommendation for safe `NLA_MAX_CONCURRENT` and `NLA_MEM_FRACTION` headroom.

## Architecture notes

- **Verbalizer**: runs via `sgl.Engine` in-process — no separate SGLang server. The server uses `engine.async_generate()` to avoid event-loop conflicts between uvicorn's uvloop and SGLang's internal async/ZMQ. Radix cache is disabled because activation injection feeds `input_embeds` directly (radix cache keys on token IDs, which don't exist for embed-injected requests). The embedding table is loaded separately from safetensors (~300 MB for Qwen 7B) for the injection step. Verbalizer sampling sets `stop=["</explanation>"]` so generation terminates as soon as the explanation block closes.
- **Reconstructor**: standard PyTorch model, no SGLang. Single forward pass, not autoregressive. Truncated to `extraction_layer + 1` layers (lm_head + final norm replaced with `Identity`) plus a learned `value_head`. With `--fp8-reconstructor`, the backbone is FP8 weight-only via `torchao`; the `value_head` stays in bf16 to preserve output numerics.
- **Source model**: HuggingFace transformers. With `--truncate-source` (default), layers past `extraction_layer`, `lm_head`, and final norm are replaced with `Identity` and dropped — saves ~25% of source-model VRAM but disables `/completion`. Activations are captured via a forward hook on the target layer (thread-local capture buffer so concurrent forwards don't collide). With `--fp8-source`, the kept layers are FP8 weight-only via `torchao`.
- **Concurrency**: four primitives.
  - `_source_semaphore` (`SOURCE_MAX_CONCURRENT`): real allocation gate — bounds simultaneous source-model GPU ops (each consumes transient KV / activation memory dynamically).
  - Per-request `Semaphore(MAX_CONCURRENT)`: pool-subscription gate inside `/describe` and `/explain`. Bounds occupancy of sglang's pre-allocated KV cache pool, doesn't itself allocate VRAM.
  - `_explain_semaphore` (`NLA_MAX_CONCURRENT_EXPLAINS`, default 1): bounds simultaneous in-flight `/explain` calls. Default 1 preserves the legacy single-request behavior; raise to permit parallel `/explain`s. The (N+1)th request gets HTTP 429 immediately rather than queueing. Note that the source-extract phase serializes on the GPU under high concurrency, so wall-time variance grows with `NLA_MAX_CONCURRENT_EXPLAINS × extract_time / verbalizer_time`. Raise `NLA_SOURCE_MAX_CONCURRENT` along with this to keep the extract gate from becoming the new bottleneck.
  - `_verbalizer_semaphore` (`NLA_MAX_CONCURRENT`): server-wide bound on in-flight sglang streams. Shared by every `/explain` and `/describe` fan-out, so KV-pool occupancy stays at most `NLA_MAX_CONCURRENT` regardless of how many requests are running in parallel.
  - Per-request `_ReconstructionBatcher` (`NLA_RECONSTRUCTION_BATCH_SIZE`): coalesces fan-out `reconstructor.score()` calls inside `/describe` and `/explain` into one batched forward pass. Fires as soon as the configured number of streams (default 4) have produced an explanation; partial batches flush automatically once no more submissions are pending. Earlier finishers wait for batch peers but never for the whole fan-out.
- **Single GPU**: with default `--truncate-source --fp8-source` settings, all three models fit on a 48 GB A40 for Qwen 7B.
- **Multi-GPU**: each model is pinned to one GPU (no tensor parallelism by default). See [Multi-GPU](#multi-gpu) — the recommended layout for 2× 96 GB PCIe boxes is verbalizer on `cuda:0`, source + reconstructor on `cuda:1`.

## Using `nla_inference.py` directly

The inference module can be used standalone without the server:

```python
from nla_inference import NLAClient, NLAReconstructor, SourceModel, make_nla_config
import numpy as np

# Extract activations from the source model
source = SourceModel(
    "Qwen/Qwen2.5-7B-Instruct",
    layer_index=20,
    device="cuda",
    truncate=True,    # drop layers past 20 + lm_head + final norm
    fp8=True,         # torchao Float8WeightOnly
)
tokens = source.extract("The capital of France is Paris")
v = np.array(tokens[0]["activation"], dtype=np.float32)

# Describe the activation
client = NLAClient(
    "kitft/nla-qwen2.5-7b-actor-step4200",
    embed_device="cuda",
    mem_fraction_static=0.45,
    quantization="fp8",   # Hopper+ only; omit/None on Ampere
)
description = client.generate(v)

# Reconstructor scoring
reconstructor = NLAReconstructor(
    "kitft/nla-qwen2.5-7b-critic-step4200",
    device="cuda",
    fp8=True,             # FP8 backbone via torchao; value_head stays bf16
)
mse, cos = reconstructor.score(description, v)

# Clean up the sglang engine subprocess
client.shutdown()
```

`SourceModel(..., truncate=True)` fails-loud on `/completion` semantics — `model.generate()` will produce garbage from the `Identity` lm_head — so don't use a truncated `SourceModel` for completion in custom code. Use `truncate=False` if you need both extraction and generation.

## Troubleshooting

### `sgl_kernel` import error / `libnuma.so.1` not found

If you see an error like:

```
ImportError: Could not load any common_ops library!
...
libnuma.so.1: cannot open shared object file: No such file or directory
```

This is a missing system library, not a version mismatch. `sgl_kernel` ships binaries for multiple GPU architectures (sm80, sm86, sm89, sm90, etc.) but needs `libnuma` to load them. Install it:

```bash
# Ubuntu/Debian
apt-get install -y libnuma-dev

# RHEL/CentOS
yum install -y numactl-devel
```

If you're building a Docker image, add this before installing Python dependencies.

### `type fp8e4nv not supported in this architecture` when using `--fp8-verbalizer`

```
ValueError: type fp8e4nv not supported in this architecture.
The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')
```

sglang's FP8 verbalizer path uses a Triton kernel (`_per_token_group_quant_8bit`) hardcoded to E4M3 FP8, which only compiles on Hopper+ (sm_90 and up). Your GPU is Ampere or Ada and doesn't support that variant.

Workarounds:

- Drop `--fp8-verbalizer`. The verbalizer stays in bf16 (~14 GB at 7B). Combine with `--truncate-source --fp8-source --fp8-reconstructor` for ~10 GB savings on the other two models.
- Move to Hopper+ hardware (H100, H200, RTX Pro 6000 Blackwell, B200, etc.).

### `/completion` returns 503 with "loaded in truncated mode"

Expected — the default config uses `--truncate-source` to save VRAM, which drops the `lm_head` and post-extraction layers. To enable `/completion`, either:

- Restart with `--no-truncate-source` (or `NLA_TRUNCATE_SOURCE=0`) — costs ~4 GB of source-model VRAM at 7B.
- Or call an external completion API (OpenRouter etc.) and keep the local server in extraction-only mode.

### `build_quantized_models.py` checkpoint fails to load

If a checkpoint produced by the legacy `build_quantized_models.py` script (FP8/INT4 reconstructor, FP8 source) fails to load — symptoms include `KeyError` in HF transformers' weight remapping, `Invalid quantization method: torchao` from sglang, or pickle-deserialization errors against the `pytorch_model-*.bin` shards — that is the torchao path showing its age. Don't try to patch around it.

Rebuild the same artifact with the `compressed-tensors` scripts (see [Pre-quantized checkpoints](#pre-quantized-checkpoints-build-at-rest-skip-runtime-quant)) — `build_compressed_tensors_reconstructor.py` for reconstructor FP8/FP4, `build_compressed_tensors_source.py` for the truncated FP8 source. The output is canonical `*.safetensors` with an embedded `compressed-tensors` config that HF transformers and sglang both load natively, with no torchao dep at runtime.

### Server OOMs at high concurrency

- Run `benchmark/sweep.py` to find the right `NLA_MEM_FRACTION` × `NLA_MAX_CONCURRENT` combination empirically.
- The recommended formula: pick `NLA_MAX_CONCURRENT` ≤ 0.85 × the highest concurrency that succeeded in the benchmark.
- If sglang OOMs at startup before any request lands, `NLA_MEM_FRACTION` is too high — sglang took too much VRAM and the source/reconstructor couldn't fit. Lower it.
- If `/explain` errors out with `RemoteProtocolError` mid-request, sglang's KV pool ran out — either lower `NLA_MAX_CONCURRENT` or raise `NLA_MEM_FRACTION` to give sglang more pool budget.
