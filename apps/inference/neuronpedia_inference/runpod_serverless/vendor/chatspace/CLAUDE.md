# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**chatspace** is a dataset embedding toolkit for downloading datasets from various sources, sampling randomly, and embedding them with either OpenAI Embeddings API or local SentenceTransformer models. All embeddings and metadata are persisted to `/workspace` in columnar formats (Parquet) for later analysis.

## Storage Conventions

All large disk writes must go to `/workspace`:
- Raw datasets: `/workspace/datasets/raw/{source}/{dataset}/{version_or_split}/...`
- Processed datasets: `/workspace/datasets/processed/{dataset}/{version_or_split}/...`
- Embeddings output: `/workspace/embeddings/{model}/{dataset}/shard-<index>.parquet`
- Manifests / indexes: `/workspace/indexes/{model}/{dataset}/manifest.json`
- Caches / temp: `/workspace/cache/...`
- Logs: `/workspace/logs/...`

## Commands

### Running the CLI

The package is installed as a CLI command `chatspace` using uv:

```bash
# Download and describe a dataset (lightweight metadata only)
uv run chatspace download-dataset --name <dataset> [--split <split>] [--source huggingface]

# Embed dataset with OpenAI API (requires OPENAI_API_KEY in .env)
uv run chatspace embed-dataset --dataset <name> [--split <split>] [--n <count>|--p <fraction>] [--execute]

# Embed dataset with local SentenceTransformer model
uv run chatspace embed-hf --dataset <name> [--subset <config>] [--model <model>] [options...]
```

### Common Development Commands

```bash
# Run fast unit tests (skips slow integration tests)
make test

# Run all tests (including slow integration tests)
make test-all

# Run ONLY the slow integration tests
make test-integration

# Run with uv directly
uv run python main.py

# Run a specific example script
bash runs/fineweb-10BT.sh
```

### Tool Preferences

**IMPORTANT**: When using bash commands for file operations, prefer modern tools:

- **Search files by pattern**: Use `fd` instead of `find`
  - `fd "pattern" /path` (fast, intuitive syntax, respects .gitignore)
  - NOT `find /path -name "pattern"` (slow, verbose syntax)

- **Search file contents**: Use `rg` (ripgrep) instead of `grep`
  - `rg "pattern" /path` (fast, colored output, respects .gitignore)
  - NOT `grep -r "pattern" /path` (slow, can hang on large directories)

- **Why**: `rg` and `fd` are orders of magnitude faster and won't hang on large directories like `/workspace` or external repos. They also have saner defaults (skip .git, skip binary files, colored output).

## Architecture

### Entry Points

- `main.py`: Simple entry point that lazy-imports `chatspace.cli:main`
- `chatspace/cli.py`: CLI argument parser and command handlers
  - `handle_download_dataset`: Scaffolds dataset metadata under `/workspace`
  - `handle_embed_dataset`: OpenAI embeddings pipeline (dry-run by default, requires `--execute`)
  - `handle_embed_hf`: Local SentenceTransformer embeddings pipeline

### Core Modules

- `chatspace/hf_embed/`: Full-featured embedding pipeline for SentenceTransformer models (modular package)
  - `config.py`: `SentenceTransformerConfig` dataclass with validation
  - `dataset.py`: Dataset loading, conversation extraction, row streaming
  - `model.py`: `_ModelRunner` class for tokenization, inference, warmup, and compilation
  - `bucketing.py`: Token bucketing (`_BucketBuffer`), batch sizing, padding logic
  - `pipeline.py`: Main orchestration (`run_sentence_transformer`), threading, encoder loop
  - `writer.py`: `_ShardWriter` for Parquet I/O, manifest generation
  - `metrics.py`: `PipelineStats`, `StageTimings`, `PipelineMetrics` for performance tracking
  - `utils.py`: Pure utility functions (paths, checksums, git SHA, ISO timestamps)

  **Key features:**
  - Streaming dataset loader with deterministic sampling
  - Token-based bucketing (power-of-2 sizes from `bucket_min_tokens` to `bucket_max_tokens`)
  - Adaptive batch sizing based on `tokens_per_batch` (overrides `batch_size` when set)
  - Multi-threaded pipeline: loader → encoder (main thread) → writer
  - Optional `torch.compile` with per-bucket compilation and warmup
  - Detailed stage timings (busy/idle time for loader, encoder, writer)
  - Parquet shards with checksums, norms, and metadata
  - Manifest generation with shard stats, git SHA, tool version

- `chatspace/env.py`: Environment variable utilities
  - `load_environment()`: Loads `.env` without overriding existing vars
  - `get_env(name, default, required)`: Safe retrieval with validation

### Pipeline Stages

The `embed-hf` command uses a three-stage pipeline:

1. **Loader** (background thread): Streams dataset rows and enqueues them
2. **Encoder** (main thread): Tokenizes, buckets by sequence length, batches, encodes, and enqueues embedded batches
3. **Writer** (background thread): Accumulates rows and writes Parquet shards when `rows_per_shard` threshold is reached

### Key Configuration

- **Token batching**: Use `--tokens-per-batch` to control batch size by total token count (e.g., 131072) instead of fixed sequence count
- **Bucketing**: Sequences are padded to the next power-of-2 length between `bucket_min_tokens` (default 128) and `bucket_max_tokens` (default 32768)
- **Compilation**: Pass `--compile-model` to enable `torch.compile` with per-bucket caching
- **Warmup**: When compilation is enabled, all bucket sizes are warmed up before the main pipeline runs

## Environment Variables

Required for OpenAI embeddings:
- `OPENAI_API_KEY`: API key for OpenAI embeddings

Optional:
- `OPENAI_BASE_URL`: Override base URL (e.g., Azure or gateway)
- `OPENAI_EMBED_MODEL`: Default model name (default: `text-embedding-3-small`)
- `OPENAI_TIMEOUT`: Request timeout in seconds

## Data Model

Each embedding row includes:
- `id`: Stable identifier for the sample
- `source`: Dataset source (e.g., "huggingface")
- `dataset`: Dataset name
- `split`: Optional split (train/test/validation)
- `text`: Input string used for embedding
- `metadata`: JSON object with provenance and field info
- `embedding`: Float vector (fixed dimensionality per model)
- `model`: Embedding model name
- `created_at`: ISO timestamp (UTC)
- `run_id`: Run identifier for tracking

## Reproducibility

Each run records:
- Git commit SHA (if available)
- Tool version (`chatspace.__version__`)
- CLI arguments in `run_config`
- Shard-level checksums (SHA256) in manifest
- Sampling seed and parameters
- Pipeline stage timings and utilization metrics

## Testing Guidelines

- Add or update tests in `tests/` that cover new code paths; mirror naming (`test_<module>.py`) and use `pytest` fixtures
- Validate concurrency changes with `python test_multiprocessing.py`
- Include small `uv run chatspace embed-hf --max-rows` dry runs when touching dataset or writer logic
- Guard against regressions by checking embedding dimension, norm bounds, and manifest integrity
- **IMPORTANT**: Always run tests with timeouts - bugs can cause hangs and GPU memory recovery is tricky

### vLLM Steering Tests

- `tests/test_vllm_comprehensive_integration.py`: End-to-end integration test covering:
  - Batch generation with chat formatting (10 prompts, 40 tokens each)
  - Multi-method steering (additive, projection cap, ablation on multiple layers)
  - Hidden state capture during prefill AND decode
  - HuggingFace parity validation (cosine similarity ~1.0, MAE <0.02)
  - Concurrent generation with temporal overlap verification
  - Capture isolation (concurrent requests don't mix)
  - Per-request steering (each request uses independent steering configuration)
- Run this test when modifying steering logic, capture mechanisms, or concurrency handling
- Expected runtime: ~20-25 seconds with CUDA available

## Coding Style

- Follow PEP 8 with 4-space indentation, descriptive snake_case for functions, UpperCamelCase for classes
- Prefer type hints and module-level docstrings; mirror existing tone in `chatspace/hf_embed/pipeline.py`
- Route diagnostics through `logging` module; avoid bare `print` except in CLI entry points
- Keep shard and manifest writers immutable: create new files rather than mutating outputs in-place

## Async/Await Patterns and Pitfalls

**CRITICAL**: When migrating sync APIs to async, preserve the concurrency model!

### The Sequential Loop Antipattern ❌

```python
# BAD: Processes prompts sequentially (destroys throughput)
async def generate(self, prompts: list[str], ...):
    results = []
    for prompt in prompts:
        async for output in self._engine.generate(prompt, ...):
            final_output = output
        results.append(final_output)
    return results
```

This forces each request to complete before the next starts, preventing vLLM from batching them on the GPU. With 32 requests, this can cause 20-30x throughput loss.

### Correct: Concurrent Processing ✅

```python
# GOOD: Processes prompts concurrently (maximum throughput)
async def generate(self, prompts: list[str], ...):
    async def process_one(prompt: str):
        async for output in self._engine.generate(prompt, ...):
            final_output = output
        return final_output

    tasks = [process_one(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    return results
```

This allows vLLM's async engine to batch all requests together for maximum GPU utilization.

### Why This Happens

- **Sync APIs** (like `vllm.LLM.generate(prompts)`) handle batching internally
- **Async APIs** (like `AsyncLLMEngine.generate(prompt)`) require explicit concurrent task launching
- Simply adding `async`/`await` keywords without `asyncio.gather()` makes things sequential!

### When This Matters

- **Batched calls**: `model.generate([p1, p2, p3])` - MUST use gather internally
- **Individual calls**: `asyncio.gather(model.generate(p1), ...)` - Already concurrent at call site
- Always preserve the batched API's efficiency when migrating to async

**Historical note**: VLLMSteerModel had this bug from Oct 2025 to Nov 2025, causing 97% throughput loss. The sync→async migration preserved functionality but broke the concurrency model. See JOURNAL.md 2025-11-19 entry for details.

## Commit Guidelines

- Use concise, imperative commit subjects (e.g., "Fix steering vector training pipeline")
- **DO NOT commit with --amend unless explicitly asked**
- PRs should describe motivation, summarize changes, list validation commands, and link issues

## vLLM Steering Runtime Notes

- `chatspace/vllm_steering/runtime.py` monkey-patches decoder layer `forward` to add steering vectors
- **Requires eager execution**: Always launch `VLLMSteerModel` with `enforce_eager=True` (default)
- CUDA-graph capture breaks steering because compiled graphs ignore the Python-side patch
- Running via `uv run` keeps repo root on `sys.path`, so `sitecustomize.py` patch triggers automatically
- Use `scripts/steering_smoke.py` for quick verification of steering behavior
- **Qwen decoder layer fusion**: vLLM fuses RMSNorm with skip connection, returns `(mlp_delta, residual_before_mlp)` - must add `delta + residual` to mirror HuggingFace captures

### Concurrency and Per-Request Steering

**Per-Request Steering Model:**
- Steering configuration is passed per-request via the `steering_spec` parameter to `generate()`
- No global state means no locking required - all requests are independent
- Different requests in the same batch can use different steering configurations (heterogeneous batching)
- **Concurrent generation**: Multiple requests can run simultaneously without coordination
- Workers maintain per-request steering state keyed by request ID
- Steering state is automatically cleaned up when request completes
- **Migration note**: The old global API (`set_layer_vector()`, etc.) was removed in favor of per-request specs

### Hidden State Capture Behavior

**Capture API Structure:**
- vLLM captures return a **single concatenated tensor** per layer containing all processed tokens
- Format: `captures[layer_idx][0]["hidden"]` with shape `[seq_len, hidden_size]`
- This tensor includes both prefill and decode tokens in sequence order
- To extract specific tokens, slice the tensor by position (e.g., `captures[2][0]["hidden"][prompt_len:]` for decode-only)

**Critical: Autoregressive Generation Length**
- Captured tensors have length `prompt_tokens + (generated_tokens - 1)`, NOT `prompt_tokens + generated_tokens`
- **Why**: In autoregressive generation, the final sampled token is never processed through the model
  1. Prefill: Process all prompt tokens
  2. Decode iterations 1..(N-1): Each iteration processes a token and produces logits for the next
  3. Final iteration N: Sample from logits only - the Nth token never flows through the model
- **Example**: 15-token prompt generating 10 tokens
  - Output text: 25 tokens total
  - Captured hidden states: 24 tokens (15 prefill + 9 decode)
  - Missing: The 10th generated token (sampled but never processed)
- **When validating**: Always use `expected_len = prompt_len + (generated_len - 1)`
- This is universal LLM behavior, not vLLM-specific

**Capture Isolation:**
- Concurrent requests with capture enabled maintain proper per-request isolation
- Each request's captures are tracked independently via request IDs
- Validated in `tests/test_vllm_comprehensive_integration.py` with temporal overlap verification

### Zero-Copy Shared Memory Activation Extraction

**Performance Optimization:**
- Traditional activation fetch serializes tensors via `.numpy().tobytes()`, taking ~12.3s for 8GB of captures
- Shared memory IPC eliminates serialization overhead, achieving ~3.8ms for the same 8GB (3176x speedup)
- Shared memory is always enabled for activation captures (no bytes fallback)

**How It Works:**
1. Worker views tensor as uint8 bytes and copies to shared memory: `tensor.view(torch.uint8).numpy()` → `shm.buf`
2. Worker returns metadata only (shm_name, shape, dtype, nbytes) instead of raw bytes
3. Client memory-maps shared memory and reconstructs tensor: `np.ndarray(..., buffer=shm.buf)` → `torch.frombuffer()` → `.view(dtype)`
4. Client releases shared memory when done via explicit cleanup or context manager

**Note:** Using uint8 view for all dtypes (including bfloat16) eliminates the dependency on ml-dtypes and works with numpy>=1.20.

**Usage Patterns:**

```python
# Recommended: Async context manager (automatic cleanup)
results, handles = await model.generate(
    prompts,
    sampling_params,
    capture_hidden_states=True,
    layers=[5, 10, 15],
)

async with handles[0] as handle:
    captures = handle.captures
    # Process captures...
# Shared memory automatically released on exit

# Alternative: Explicit cleanup
handle = handles[0]
captures = handle.captures
# Process captures...
await handle.close()  # Explicit release
```

**Configuration:**

Environment variables control shared memory behavior:
- `CHATSPACE_SHM_TTL=600` - Worker-side timeout in seconds for stale segment cleanup (default: 10 minutes)
- `CHATSPACE_MAX_SHM_GB=128` - Maximum total shared memory usage in GB (default: 128GB)

**Safety Features:**

Three layers of cleanup prevent memory leaks:
1. **Primary**: Async context manager (`async with handle:`) releases shared memory on exit
2. **Backup**: `weakref.finalize()` callback fires if handle is garbage collected without cleanup
   - Emits ResourceWarning if handle held shared memory but was never accessed
3. **Failsafe**: Worker-side TTL (default 10 minutes) + background thread scanning every 60 seconds
   - Automatically unlinks stale segments that weren't explicitly released

**Performance Characteristics:**
- Best for large batches with many layers (e.g., 32 requests × 64 layers = 2048 tensors)
- No size threshold - all activation captures use shared memory
- If shared memory limit is reached, a RuntimeError is raised (fail-fast, no silent fallback)

**Troubleshooting:**
- If you see ResourceWarning: "CaptureHandle held N shared memory regions but was never accessed"
  - Use `async with handle:` context manager or call `await handle.close()` explicitly
- Check worker logs for "TTL expired" warnings if shared memory isn't being released promptly
- If you get RuntimeError about shared memory limit, increase `CHATSPACE_MAX_SHM_GB` or reduce batch size
- Monitor `/dev/shm` usage if concerned about memory consumption

## Code Cleanup and Technical Debt Management

**Lessons learned from 2025-11-07 cleanup effort** (removed 347+ lines of dead code and defensive patterns):

### What Works Well

1. **Phased approach with tests between phases**
   - Phase 1: Critical fixes (bare excepts, silent failures, hot-path optimizations)
   - Phase 2: Dead code removal (hook variants, unused abstractions, timed functions)
   - Commit after each phase passes tests - this creates safe rollback points

2. **Parallel subagents for mechanical edits**
   - Use separate agents for unrelated files (e.g., runtime.py vs hf_embed/)
   - Reduces context bloat and speeds up execution
   - Each agent can focus on one area deeply

3. **Conservative Phase 2-style deletions are safe**
   - Removing code with zero call sites: hook variants (185 lines), timed extraction (43 lines), unused ABCs (50 lines)
   - grep/Glob to verify zero usage before deleting
   - These cleanups have high confidence and low risk

### What to Avoid

1. **"Simplification" refactors are high-risk**
   - Phase 3 broke captures system with subtle control flow bugs
   - Flattening nested conditionals changed early-return behavior
   - Inlining helper functions changed how env vars were parsed
   - **Rule**: Avoid refactors that touch hot paths or complex conditional logic unless there's a specific bug to fix

2. **Early returns change semantics**
   - Adding `return None` inside an `if isinstance(output, (tuple, list)):` block prevented fallthrough to dict/object checks
   - Original code used flat `if` statements that naturally fell through to final `return None`
   - **Rule**: When "flattening" code, preserve exact control flow - don't add early returns unless they were there originally

3. **Environment variable parsing is brittle**
   - Inlining `_env_flag` and `_get_env_int` helper functions introduced subtle bugs
   - Python ternary expressions need careful handling of None checks
   - **Rule**: Leave working environment parsing code alone unless it's causing actual problems

### How to Avoid This Cleanup in the Future

1. **Write targeted code from the start**
   - Don't create hook variant systems "for future profiling" - add them when actually needed
   - Don't build abstractions (ABCs, inheritance) until you have 2+ implementations
   - Don't create "timed" versions of functions - use profiler when needed

2. **Delete as you go**
   - When removing a backend (e.g., HuggingFace), immediately delete all related code
   - When an experiment doesn't pan out (StepContext), delete it before committing
   - Don't leave "commented out" or "maybe useful later" code

3. **Trust type systems over runtime checks**
   - If function signature says `metadata: dict[str, Any]` (not optional), don't check `if metadata is not None`
   - If you're checking None repeatedly, the type hint is probably wrong
   - Fix the type hint instead of adding defensive checks

4. **Avoid defensive patterns in hot paths**
   - Don't check `isinstance(cap_config, _ProjectionCapConfig)` on every forward pass - setters guarantee type
   - Don't call `.to(device=dest.device, dtype=dest.dtype)` if tensor is already correct device/dtype
   - Profile first, then optimize - don't guess

5. **Keep it simple**
   - One hook implementation is enough (not 5 variants)
   - Direct imports are clearer than single-line wrapper functions
   - Flat conditionals are often clearer than nested ones (but don't force it)

### Key Takeaway

**Dead code removal (Phase 2) is safe and valuable. "Simplification" refactors (Phase 3) are risky and often not worth it unless fixing a specific bug.**

When in doubt, delete unused code aggressively, but leave working code alone.

## Journaling Practices

- Scratch notes and active investigations go in `TEMP_JOURNAL.md` (gitignored)
- Capture UTC timestamp with `date -u` before editing logs
- Once work stabilizes, move key findings to canonical `JOURNAL.md`
- Note tmux sessions, long-running jobs, and `/workspace` artifact paths for resumability
- **IMPORTANT**: Implementation summaries, feature documentation, and completion notes should be added to `JOURNAL.md`, NOT as separate markdown files in the repo root. Keep the repo clean by consolidating documentation in the journal.

## Benchmark Journal

Store benchmark results in `bench_journal/` with dated markdown files:
- Naming: `YYYY-MM-DD-description.md` (e.g., `2025-11-29-steering-idle-overhead.md`)
- Include: objective, hardware, model, workload parameters, results tables, conclusions
- Reference related GitHub issues
- Keep methodology reproducible (include test commands/scripts used)

### Exploratory Benchmark Scripts

When exploring optimization approaches:

1. **Keep in `scripts/`**: Final benchmark scripts that validate the chosen approach
2. **Archive in `scripts/archive/`**: Exploratory scripts that document alternative approaches tested
3. **Log results in `JOURNAL.md`**: Key findings, performance numbers, and which approaches won/lost
4. **Delete**: Intermediate scripts that are superseded and don't add documentary value

This pattern preserves the "why" (journal), the "how we got there" (archive), and the "final answer" (scripts), while avoiding clutter from throwaway experiments.