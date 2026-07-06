# Based on HeadVis by Luger, Kamath et al (Anthropic 2026)
# https://transformer-circuits.pub/2026/headvis/index.html
# https://github.com/anthropics/headvis
#
# Usage Example:
# uv run python run-all-head-metrics.py \
#   --dataset-name monology/pile-uncopyrighted \
#   --n-sequences 16384 \
#   --batch-size 8

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
# Repo base: headvis -> neuronpedia_utils -> neuronpedia-utils -> utils -> root
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_EXPORTS_DIR = SCRIPT_DIR.parent / "exports"
DEFAULT_MODEL_MAP_PATH = REPO_ROOT / "np_model_to_hf.json"

DEFAULT_N_INTERVALS = 5
DEFAULT_SAMPLES_PER_INTERVAL = 3
DEFAULT_SAMPLES_PER_TOP_INTERVAL = 10
DEFAULT_SPARSE_TOPK_PER_ROW = 8
DEFAULT_SPARSE_THRESHOLD = 0.005
DEFAULT_SAMPLE_SEED = 0


def sanitize_filename_part(value: str) -> str:
    value = value.replace("/", "-")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run compute-head-metrics.py for every Hugging Face model listed in "
            "np_model_to_hf.json, skipping outputs that already exist."
        )
    )
    parser.add_argument(
        "--models-json",
        default=str(DEFAULT_MODEL_MAP_PATH),
        help="Path to JSON mapping Neuronpedia model names to Hugging Face model names.",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Hugging Face dataset name to pass through to compute-head-metrics.py.",
    )
    parser.add_argument(
        "--n-sequences",
        type=int,
        required=True,
        help="Number of sequences to process per model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Batch size to use per model.",
    )
    parser.add_argument(
        "--exports-dir",
        default=str(DEFAULT_EXPORTS_DIR),
        help=(
            "Root exports directory. Each run is written to "
            "<exports-dir>/<np_model_id>/headvis/<dataset>/."
        ),
    )
    parser.add_argument(
        "--metrics-script",
        default=str(SCRIPT_DIR / "compute-head-metrics.py"),
        help="Path to compute-head-metrics.py.",
    )
    parser.add_argument(
        "--n-intervals",
        type=int,
        default=DEFAULT_N_INTERVALS,
        help="Number of activation intervals for stratified sequence sampling.",
    )
    parser.add_argument(
        "--samples-per-interval",
        type=int,
        default=DEFAULT_SAMPLES_PER_INTERVAL,
        help="Reservoir-sampled sequences per non-top interval.",
    )
    parser.add_argument(
        "--samples-per-top-interval",
        type=int,
        default=DEFAULT_SAMPLES_PER_TOP_INTERVAL,
        help="Top-K sequences (by max attention) kept in the highest-activation interval.",
    )
    parser.add_argument(
        "--sparse-topk-per-row",
        type=int,
        default=DEFAULT_SPARSE_TOPK_PER_ROW,
        help="Keep at most this many key positions per query row in the sparse COO.",
    )
    parser.add_argument(
        "--sparse-threshold",
        type=float,
        default=DEFAULT_SPARSE_THRESHOLD,
        help="Drop COO entries whose attention value is below this threshold.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help="Random seed for reservoir sampling within non-top intervals.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without launching metrics jobs.",
    )
    args = parser.parse_args()

    if args.n_sequences < 1:
        parser.error("--n-sequences must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.n_intervals < 1:
        parser.error("--n-intervals must be at least 1")
    if args.samples_per_interval < 1:
        parser.error("--samples-per-interval must be at least 1")
    if args.samples_per_top_interval < 1:
        parser.error("--samples-per-top-interval must be at least 1")
    if args.sparse_topk_per_row < 1:
        parser.error("--sparse-topk-per-row must be at least 1")
    if not (0.0 <= args.sparse_threshold < 1.0):
        parser.error("--sparse-threshold must be in [0, 1)")

    return args


def load_model_map(models_json_path: Path) -> list[tuple[str, str]]:
    """Return a list of (np_model_id, hf_model_name) tuples, deduped on hf_model_name."""
    with open(models_json_path) as f:
        model_map: dict[str, Any] = json.load(f)

    if not all(isinstance(v, str) for v in model_map.values()):
        raise ValueError(f"All values in {models_json_path} must be strings.")
    if not all(isinstance(k, str) for k in model_map.keys()):
        raise ValueError(f"All keys in {models_json_path} must be strings.")

    seen_hf: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for np_id, hf_name in model_map.items():
        if hf_name in seen_hf:
            continue
        seen_hf.add(hf_name)
        pairs.append((np_id, hf_name))
    return pairs


def run_dir_for_model(exports_dir: Path, np_model_id: str, dataset_name: str) -> Path:
    return exports_dir / np_model_id / "headvis" / sanitize_filename_part(dataset_name)


def run_tree_is_complete(run_dir: Path) -> bool:
    if not run_dir.is_dir():
        return False
    if not (run_dir / "config.json").is_file():
        return False
    if not (run_dir / "scatter_data.json").is_file():
        return False
    heads_dir = run_dir / "heads"
    if not heads_dir.is_dir():
        return False
    try:
        next(heads_dir.glob("L*H*.json"))
    except StopIteration:
        return False
    return True


def main() -> None:
    args = parse_args()
    models_json_path = Path(args.models_json).expanduser().resolve()
    metrics_script_path = Path(args.metrics_script).expanduser().resolve()
    exports_dir = Path(args.exports_dir).expanduser()
    if not exports_dir.is_absolute():
        exports_dir = SCRIPT_DIR / exports_dir
    exports_dir = exports_dir.resolve()
    exports_dir.mkdir(parents=True, exist_ok=True)

    model_pairs = load_model_map(models_json_path)
    print(f"Loaded {len(model_pairs)} unique models from {models_json_path}")

    for index, (np_model_id, hf_model_name) in enumerate(model_pairs, start=1):
        run_dir = run_dir_for_model(exports_dir, np_model_id, args.dataset_name)
        prefix = f"[{index}/{len(model_pairs)}] {np_model_id} ({hf_model_name})"
        if run_tree_is_complete(run_dir):
            print(f"{prefix}: skipping, complete HeadVis tree already at {run_dir}")
            continue

        command = [
            sys.executable,
            str(metrics_script_path),
            "--model-name",
            hf_model_name,
            "--dataset-name",
            args.dataset_name,
            "--n-sequences",
            str(args.n_sequences),
            "--batch-size",
            str(args.batch_size),
            "--exports-dir",
            str(exports_dir),
            "--np-model-map",
            str(models_json_path),
            "--hf-cache-dir",
            "/tmp/headvis-hf-cache",
            "--n-intervals",
            str(args.n_intervals),
            "--samples-per-interval",
            str(args.samples_per_interval),
            "--samples-per-top-interval",
            str(args.samples_per_top_interval),
            "--sparse-topk-per-row",
            str(args.sparse_topk_per_row),
            "--sparse-threshold",
            str(args.sparse_threshold),
            "--sample-seed",
            str(args.sample_seed),
        ]

        print(f"{prefix}: running")
        print(" ".join(command))
        if args.dry_run:
            continue

        env = os.environ.copy()
        subprocess.run(command, cwd=SCRIPT_DIR, env=env, check=True)

    print("Done.")


if __name__ == "__main__":
    main()
