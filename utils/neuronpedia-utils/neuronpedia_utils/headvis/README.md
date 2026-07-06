# HeadVis pipeline for Neuronpedia

Offline pipeline that computes attention-head metrics and HeadVis-style
sequence samples for a Hugging Face causal LM, writes the data tree the
[HeadVis frontend](https://github.com/anthropics/headvis) consumes, and
imports per-head metrics and per-sequence rows into Neuronpedia's
Postgres.

Based on
["Visualizing Attention Heads"](https://transformer-circuits.pub/2026/headvis/index.html)
by Luger, Kamath et al (Anthropic 2026).

## What's in this folder

| file | purpose |
|---|---|
| `compute-head-metrics.py` | Main pipeline. One model + one dataset → one HeadVis tree. |
| `run-all-head-metrics.py` | Batch wrapper. Iterates every model in `np_model_to_hf.json`. |
| `load-head-metrics.py` | Imports `scatter_data.json` + per-head aggregates into `ModelHeadMetrics`. |
| `load-head-sequences.py` | Imports per-head sampled sequences into `ModelHeadSequence`. |
| `model_head_metrics.py` | Dataclass mirroring `ModelHeadMetrics`. |
| `model_head_sequence.py` | Dataclass mirroring `ModelHeadSequence`. |
| `../np_model_to_hf.json` | Map from Neuronpedia model id → Hugging Face model name. Shared with `jlens/` (lives in the parent `neuronpedia_utils/` dir). |
| `pyproject.toml` / `uv.lock` | Standalone uv environment. |

## Setup

```bash
cd neuronpedia/utils/neuronpedia-utils/neuronpedia_utils/headvis
uv sync
```

For GPU use, `torch>=2.8` is pinned to the CUDA 12.8 wheels via
`[tool.uv.sources]` in `pyproject.toml`.

## Output tree

Each run is keyed by the Neuronpedia model id (looked up from
`np_model_to_hf.json` by reverse-mapping the `--model-name` Hugging Face
name) and the sanitized dataset name:

```
<exports-dir>/<np_model_id>/headvis/<sanitized-dataset>/
├── config.json                 # frontend config + nested pipeline_config
├── scatter_data.json           # flat per-head metrics (one row per head)
├── server_config.json          # {} → static-only mode
├── attributions/
│   └── manifest.json           # {}
└── heads/
    └── L{layer}H{head}.json    # one file per head
```

`<exports-dir>` defaults to `../exports` (sibling of this folder), so a
default run lands at:

```
neuronpedia_utils/exports/<np_model_id>/headvis/<sanitized-dataset>/
```

If the Hugging Face model name passed via `--model-name` is not in
`np_model_to_hf.json` the script hard-errors. Add the mapping there
before running.

`heads/L{l}H{h}.json` contains:

- `sequences` — sampled sequences with sparse-COO attention, partitioned
  into `n_intervals` activation buckets.
- `qk_distance_histogram` — log-spaced bins of `|q-k|` over the kept COO
  entries, weighted by attention mass.
- `top_query_tokens`, `top_key_tokens` — token-string-keyed marginal
  attention (top 50 each).
- `histogram` — per-head `max_activation` distribution across **all**
  processed sequences (not just sampled).
- `statistics` — `n_samples`, `mean_max_activation`.

The frontend lazy-loads one head JSON at a time, so each file stays
small (~1 MB at default settings) regardless of how many heads the model
has. For Llama 3.3 70B (80 × 64 = 5120 heads) the tree is ~5 GB total
but no individual file exceeds a couple MB.

The `pipeline_config` block embedded in `config.json` carries the full
provenance of the run so the loaders and downstream tooling have a
single source of truth.

## Run a single model

```bash
uv run python compute-head-metrics.py \
  --model-name google/gemma-3-1b-pt \
  --dataset-name monology/pile-uncopyrighted \
  --n-sequences 16384 \
  --batch-size 16
```

This writes to:
`<exports-dir>/gemma-3-1b/headvis/monology-pile-uncopyrighted/`

Useful knobs:

| flag | default | what it does |
|---|---|---|
| `--exports-dir` | `../exports` | Root exports directory. |
| `--np-model-map` | `../np_model_to_hf.json` | Path to the NP-id ↔ HF-name map (shared with `jlens/`). |
| `--n-sequences` | 2000 | Number of valid sequences to process. |
| `--seq-len` | 512 | Tokenized sequence length (truncated). |
| `--batch-size` | 4 | Sequences per forward pass. |
| `--dtype` | bfloat16 | Model dtype (auto/float32/float16/bfloat16). |
| `--attn-implementation` | eager | Required for `output_attentions=True`. |
| `--n-intervals` | 5 | Activation deciles for stratified sampling. |
| `--samples-per-interval` | 3 | Reservoir-sampled sequences per non-top interval. |
| `--samples-per-top-interval` | 10 | Top-K sequences kept in the highest-activation interval. |
| `--sparse-topk-per-row` | 8 | Keep at most this many key positions per query row in the sparse COO. |
| `--sparse-threshold` | 0.005 | Drop COO entries below this attention value. |
| `--sample-seed` | 0 | Reservoir RNG seed. |
| `--induction-attention-threshold` | 0.01 | Zero induction values below this before summing. |
| `--hf-cache-dir` | (tempdir) | Where Hugging Face downloads land. Deleted on exit unless `--keep-hf-cache`. |
| `--print-summary` | off | Print top heads per metric after writing. |

## Sampling design

The frontend's "Sequences" panel partitions sampled sequences into
activation deciles (1 = lowest, N = highest). Sampling is streaming so
it scales to large head counts without holding all sequences in RAM:

1. **Warmup** (first `min(2000, max(50, n_sequences // 10))` valid
   sequences). Per `(layer, head, sequence)` we record only
   `max_activation`. No COO is computed in this phase.
2. **Boundary computation.** Equal-frequency quantile boundaries are
   derived per `(layer, head)` from the warmup buffer.
3. **Streaming sampling.** Each subsequent sequence routes into its
   bucket. The top bucket uses a deterministic top-K min-heap (capacity
   `samples_per_top_interval`); other buckets use Algorithm-R reservoir
   sampling (capacity `samples_per_interval`, seeded by
   `--sample-seed`).
4. `max_activation` skips position 0 in both row and column to avoid
   BOS-sink dominating the score (matches the frontend's
   `getRowMax` / `findMax` in `src/lib/sparse.js`).

### Trade-offs to know about

- **Warmup discard.** Sequences observed during warmup contribute to
  boundary estimation and the full-dataset `histogram`, but their COO
  isn't computed, so they don't enter the sample slots. With
  `n_sequences=16384` this loses ~1.6k sequences from the sample pool —
  ~10% of the dataset.
- **Per-row top-K.** The COO keeps at most `sparse-topk-per-row` keys
  per query position, then drops anything below `sparse-threshold`.
  Causal attention is highly peaked, so top-8 typically captures >95%
  of each row's mass.
- **Tokens.** `tokenizer.decode([id])` per token id so spaces and
  punctuation render correctly in the visualizer.

## Scatter metrics

Six metrics per head are written to `scatter_data.json` (and to the
`heads/` JSONs via the histogram/statistics blocks):

| metric | meaning |
|---|---|
| `self_attention_score` | Mean attention to the diagonal. |
| `prev_token_score` | Mean attention to the immediately preceding token. |
| `pattern_entropy` | Mean Shannon entropy of each row of the attention matrix. |
| `qk_distance` | Mean `|q − k|` weighted by attention mass. |
| `qk_distance_variance` | Variance of the same. |
| `induction_score` | Mean attention to the position after a prior occurrence of the same token. |

## Run every model in `np_model_to_hf.json`

```bash
uv run python run-all-head-metrics.py \
  --dataset-name monology/pile-uncopyrighted \
  --n-sequences 16384 \
  --batch-size 8
```

Models whose run directory already has a complete tree (`config.json`,
`scatter_data.json`, and a non-empty `heads/`) are skipped. Sampling
flags are passed through to each subprocess; pass them at the runner
level for one consistent setting across every model.

`--dry-run` prints the planned commands without launching anything.

## Database schema

Two tables in `apps/webapp/prisma/schema.prisma`:

### `ModelHeadMetrics`

One row per `(modelId, datasetName, run-config, layer, headIndex)`.

- Six scalar metrics: `selfAttentionScore`, `prevTokenScore`,
  `patternEntropy`, `qkDistance`, `qkDistanceVariance`, `inductionScore`.
- Five JSON aggregates from each `heads/L{l}H{h}.json`:
  `qkDistanceHistogram`, `topQueryTokens`, `topKeyTokens`,
  `activationHistogram`, `headStatistics`.
- Unique on
  `(modelId, datasetName, nSequences, seqLen, dtype, attnImplementation, layer, headIndex)`.
  Re-imports UPSERT in place.

### `ModelHeadSequence`

One row per sampled sequence per `(model, dataset, run-config, layer,
head)`. No uniqueness constraint — the import path errors out if any
rows already exist for a given run, so duplicates can't sneak in.

- `interval Int`, `maxActivation Float`.
- `tokens String[]`, `attentionIndices Int[]`, `attentionValues Float[]`
  as native Postgres arrays.
- Indexes: `(modelId, layer, headIndex)`, `(modelId)`, and the safety
  pre-check index `(modelId, datasetName, nSequences, dtype,
  attnImplementation)`.

The flat-COO encoding follows the on-disk format: `idx = q * seq_len + k`
where `seq_len` is the per-sequence length (the number of entries in
`tokens`). We don't store `seq_len` as a column; the frontend decodes
`q`/`k` using `tokens.length`, which is exactly that stride.

## Import head-level metrics into Postgres

```bash
uv run python load-head-metrics.py \
  ../exports \
  --dry-run
```

Set `DATABASE_URL` (or `DATABASE_NAME` /`DATABASE_USERNAME` /
`DATABASE_PASSWORD` /`DATABASE_HOST` /`DATABASE_PORT`) via `.env` to
point at your Neuronpedia Postgres. The loader walks
`<exports-dir>/<np_id>/headvis/<dataset>/` for each run, reads
`config.json` + `scatter_data.json` + every `heads/L*H*.json`, and
upserts `ModelHeadMetrics`.

The conflict key is
`(modelId, datasetName, nSequences, seqLen, dtype, attnImplementation, layer, headIndex)`.
Re-running with the same parameters refreshes the metric values and the
JSON aggregate columns in place.

## Import sampled sequences into Postgres

```bash
uv run python load-head-sequences.py \
  ../exports \
  --dry-run
```

Iterates the same exports tree, reads each `heads/L*H*.json`, and
INSERTs one row per sampled sequence into `ModelHeadSequence`.

There's no UPSERT and no unique constraint. Before any inserts the
loader runs a pre-check per run: if `ModelHeadSequence` already has any
rows matching the run's
`(modelId, datasetName, nSequences, dtype, attnImplementation)`,
the import aborts with an error pointing you at the affected run
directory. (`seqLen` is not part of this key; `ModelHeadSequence` doesn't
store it — per-sequence length is just `len(tokens)`.) To re-import,
delete those rows first.

## Hardware notes

For most ≤7B-class models, a single 24–80 GB GPU is plenty. Larger
models stress GPU memory mostly because `attn_implementation=eager` is
required to surface attention weights — FlashAttention/SDPA can't be
used with `output_attentions=True`.

Per-layer attention storage at bf16, B=1, S=512:
`B × H × S × S × 2 bytes`. For 64 heads × 512 × 512 = 32 MB per layer;
all layers stay resident through the `output_attentions=True` forward.

### Llama 3.3 70B reference

- 80 layers, 64 query heads, GQA (8 KV).
- Weights: ~141 GB bf16.
- Attention matrices held resident: ~2.5 GB.
- Activations + workspace: ~5–10 GB.
- **Total: ~150–160 GB.**

| setup | fit | notes |
|---|---|---|
| 1× H200 141GB | yes (just) | cleanest; no multi-GPU code change |
| 2× H100 80GB (NVLink) | comfortable | recommended cost/perf sweet spot |
| 4× A100 80GB / H100 80GB | comfortable | very roomy |
| 8× A100 80GB | comfortable | drop-in for existing 8-GPU nodes |
| 2× A100 80GB | tight | may OOM with eager attention; force B=1 and S≤512 |
| 4× A100 40GB | tight | 160 GB total; possible OOM under load |
| 1× A100/H100 80GB | no | weights alone don't fit |

For multi-GPU, swap `model.to(device)` in `run_head_metrics` for
`device_map="auto"` (in `model_kwargs`) — `accelerate` is already a
declared dependency. Inputs should then be moved to `model.device`
rather than the resolved CLI device.

Reduction levers if memory is tight:

- `--seq-len 256` halves the per-layer attention footprint (most
  impactful single change).
- `--sparse-topk-per-row 4` halves per-record COO size on disk.
- `--batch-size 1` is already the right setting for 70B; raising it
  multiplies attention storage linearly.

### Throughput estimates (n_sequences=16384, S=512, B=1)

| hardware | per-batch | total run |
|---|---|---|
| 1× H200 141GB | ~1.2 s | ~5.5 h |
| 2× H100 80GB (NVLink) | ~1.5 s | ~7 h |
| 8× A100 80GB | ~1.7 s | ~7.5 h |
| 4× A100 80GB | ~2.1 s | ~9.5 h |

The per-batch CPU-side sampler work scales with head count
(~5120 heads on 70B); on slower CPUs it can match the GPU forward time.
If that becomes the bottleneck, vectorize the per-`(batch, head)` loop
in `_build_sparse_records_for_layer`.

## Disk and DB footprint

Default settings (5 intervals × 3 reservoir samples + 1 × 10 top samples,
top-8 per row, S=512, threshold 0.005):

| model | heads | tree size | DB rows (`ModelHeadSequence`) |
|---|---|---|---|
| Gemma 3 1B (26 × 4) | 104 | ~100 MB | ~2.3K |
| Llama 3.1 8B (32 × 32) | 1024 | ~1 GB | ~22K |
| Llama 3.3 70B (80 × 64) | 5120 | ~5 GB | ~112K |

`ModelHeadMetrics` always has exactly one row per head (104 / 1024 /
5120 respectively).

## Frontend

Once a tree is on disk, point any HeadVis-frontend `data/` symlink at
`<exports-dir>/<np_model_id>/headvis/<dataset>/` and serve `dist/`
statically. See the upstream README for the build flow.
