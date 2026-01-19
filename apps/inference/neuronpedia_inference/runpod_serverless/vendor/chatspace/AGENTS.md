# Repository Guidelines

## Project Structure & Module Organization
- Core library lives in `chatspace/`, with the multiprocessing embed pipeline under `chatspace/hf_embed/` (config, dataset loaders, pipeline orchestration, writers).
- CLI entry points sit in `chatspace/cli.py`; helper scripts for dataset prep and analysis live in `scripts/` and reproducible runs in `runs/`.
- Tests target public APIs in `tests/`; standalone smoke checks such as `test_multiprocessing.py` remain at the repo root.
- Use `/workspace/datasets`, `/workspace/embeddings`, and `/workspace/indexes` for any sizable artifacts; keep the repo tree clean of large outputs.

## Build, Test, and Development Commands
- `uv sync` — install and lock project dependencies (Python 3.11 required).
- `uv run chatspace embed-hf --dataset <org/name> --model <model> --max-rows 1000` — run the embedding pipeline end-to-end with deterministic row limits.
- `uv run pytest` — execute the unit test suite in `tests/` with isolated virtualenv support.
- `uv run python test_multiprocessing.py` — quick regression for the multiprocessing controller without downloading datasets.
- ALWAYS run python via `uv run python`, DO NOT run plain `python`, system python will not have requisite deps installed

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, descriptive snake_case for functions, and UpperCamelCase for classes.
- Prefer type hints and module-level docstrings; mirror existing docstring tone in `chatspace/hf_embed/pipeline.py`.
- Route diagnostics through the standard `logging` module; avoid bare `print` except in CLI entry points.
- Keep shard and manifest writers immutable: create new files rather than mutating prior outputs in-place.

## Testing Guidelines
- Add or update tests in `tests/` that cover new code paths; mirror naming (`test_<module>.py`) and use `pytest` fixtures.
- Validate concurrency changes with `python test_multiprocessing.py`; include small `uv run chatspace embed-hf --max-rows` dry runs when touching dataset or writer logic.
- Guard against regressions by checking embedding dimension, norm bounds, and manifest integrity in tests whenever feasible.
- ALWAYS run tests with some kind of timeout - bugs can cause the session to hang and recollecting GPU memory from stuck processes can be quite tricky

## Commit & Pull Request Guidelines
- Use concise, imperative commit subjects similar to `Fix steering vector training pipeline and data loader`; squash noisy intermediate commits before review.
- PRs should describe motivation, summarize functional changes, list validation commands (e.g., `uv run pytest`), and link issues or run IDs.
- Include screenshots or log excerpts when modifying visualization notebooks or CLI surfaces; note any `/workspace` artifacts that reviewers can reproduce.
- DO NOT commit with --amend unless explicitly asked to

## Data & Storage Practices
- Respect the storage policy: stream downloads into `/workspace/datasets/raw/...` and write processed shards to `/workspace/embeddings/{model}/{dataset}`.
- Record deterministic sampling parameters and shard stats in manifests under `/workspace/indexes/{model}/{dataset}/manifest.json` for resumable runs.

## Run Logs & Journaling
- Before editing the engineering log, capture the current UTC timestamp with `date -u` so new entries use an accurate datestamp.
- Record material debugging and training updates in `JOURNAL.md`, summarizing the change, duration, and key outputs or checkpoints.
- Note any tmux sessions, long-running jobs, or `/workspace` artifact paths so the next agent can resume without guesswork.

## Steering Runtime Notes
- `chatspace/vllm_steering/runtime.py` monkey-patches the decoder layer `forward` to add steering vectors. This requires eager execution—always launch `VLLMSteerModel` with `enforce_eager=True` (default) or the compiled graph path will skip the patch entirely.
- CUDA-graph capture still breaks steering because the compiled kernels ignore the Python-side addition. Leave graphs disabled until we upstream a fused/graph-safe injector.
- Running via `uv run …` keeps the repo root on `sys.path`, so the worker-side `sitecustomize.py` patch installer triggers automatically before model load.
- Use `scripts/steering_smoke.py` (`uv run python scripts/steering_smoke.py --layer 16 --scale 100000`) for quick verification; expect steered logprobs to diverge only under eager execution.
- Scratch notes belong in `TEMP_JOURNAL.md`; the file is gitignored, so keep the canonical log in `JOURNAL.md` once work stabilizes.
- Log anything interesting / surprising to TEMP_JOURNAL.md
- vLLM’s Qwen decoder layers fuse the RMSNorm with the skip connection and return `(mlp_delta, residual_before_mlp)`; to mirror HuggingFace captures you must add `delta + residual` when extracting the hidden state.
