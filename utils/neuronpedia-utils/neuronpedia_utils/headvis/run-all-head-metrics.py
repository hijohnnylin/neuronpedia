# Based on HeadVis by Luger, Kamath et al (Anthropic 2026)
# https://transformer-circuits.pub/2026/headvis/index.html
# https://github.com/anthropics/headvis
#
# Usage Example:
# poetry run python run-all-head-metrics.py \
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


DEFAULT_OUTPUT_DIR = "head-metrics"


def sanitize_filename_part(value: str) -> str:
    value = value.replace("/", "-")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Run compute-head-metrics.py for every Hugging Face model listed in "
            "np_model_to_hf.json, skipping outputs that already exist."
        )
    )
    parser.add_argument(
        "--models-json",
        default=str(script_dir / "np_model_to_hf.json"),
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
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where head metric JSON files are written.",
    )
    parser.add_argument(
        "--metrics-script",
        default=str(script_dir / "compute-head-metrics.py"),
        help="Path to compute-head-metrics.py.",
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

    return args


def load_model_names(models_json_path: Path) -> list[str]:
    with open(models_json_path) as f:
        model_map: dict[str, Any] = json.load(f)

    hf_model_names = list(model_map.values())
    if not all(isinstance(model_name, str) for model_name in hf_model_names):
        raise ValueError(f"All values in {models_json_path} must be strings.")

    return list(dict.fromkeys(hf_model_names))


def output_path_for_model(
    output_dir: Path, model_name: str, dataset_name: str
) -> Path:
    return output_dir / (
        f"{sanitize_filename_part(model_name)}-"
        f"{sanitize_filename_part(dataset_name)}.json"
    )


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    models_json_path = Path(args.models_json).expanduser().resolve()
    metrics_script_path = Path(args.metrics_script).expanduser().resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = load_model_names(models_json_path)
    print(f"Loaded {len(model_names)} unique model names from {models_json_path}")

    for index, model_name in enumerate(model_names, start=1):
        output_path = output_path_for_model(output_dir, model_name, args.dataset_name)
        prefix = f"[{index}/{len(model_names)}] {model_name}"
        if output_path.exists():
            print(f"{prefix}: skipping, output exists at {output_path}")
            continue

        command = [
            sys.executable,
            str(metrics_script_path),
            "--model-name",
            model_name,
            "--dataset-name",
            args.dataset_name,
            "--n-sequences",
            str(args.n_sequences),
            "--batch-size",
            str(args.batch_size),
            "--output-dir",
            str(output_dir),
            "--hf-cache-dir",
            "/tmp/headvis-hf-cache",
        ]

        print(f"{prefix}: running")
        print(" ".join(command))
        if args.dry_run:
            continue

        env = os.environ.copy()
        subprocess.run(command, cwd=script_dir, env=env, check=True)

    print("Done.")


if __name__ == "__main__":
    main()
