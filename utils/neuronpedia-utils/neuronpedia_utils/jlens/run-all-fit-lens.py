# Based on the Jacobian lens ("jlens") reference implementation by Anthropic PBC.
# Companion code for the "Verbalizable Workspace" paper.
# https://github.com/anthropics/jlens
# SPDX-License-Identifier: Apache-2.0
#
# Fits a Jacobian lens for every Hugging Face model listed in
# np_model_to_hf.json, writing each run to
#   <exports-dir>/<np_model_id>/jlens/<dataset>/
# alongside a config.yaml describing exactly how the lens was generated.
#
# Usage Example:
# uv run python run-all-fit-lens.py \
#   --dataset-name Salesforce/wikitext \
#   --n-prompts 1000 \
#   --stop-at-delta 2e-3

import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
# Repo base: jlens -> neuronpedia_utils -> neuronpedia-utils -> utils -> root
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_EXPORTS_DIR = SCRIPT_DIR.parent / "exports"
DEFAULT_MODEL_MAP_PATH = REPO_ROOT / "np_model_to_hf.json"
DEFAULT_FIT_SCRIPT = SCRIPT_DIR / "fit_lens.py"
DEFAULT_HF_CACHE_DIR = "/tmp/jlens-hf-cache"

# Fit defaults. These are also what gets recorded into config.yaml, so the
# generated lens is fully reproducible from the file alone.
DEFAULT_DATASET_NAME = "Salesforce/wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-103-raw-v1"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_TEXT_FIELD = "text"
DEFAULT_MAX_CHARS = 2000
DEFAULT_N_PROMPTS = 1000
DEFAULT_DIM_BATCH = 8
DEFAULT_MAX_SEQ_LEN = 128
DEFAULT_DTYPE = "bfloat16"
DEFAULT_DEVICE_MAP = "cuda"
DEFAULT_STOP_AT_DELTA = 2e-3
DEFAULT_MIN_PROMPTS = 100
DEFAULT_STOP_WINDOW = 10
DEFAULT_LEVELS = "1e-2,5e-3,1e-3"

ATTRIBUTION = (
    "Jacobian lens ('jlens') by Anthropic PBC — companion code for the "
    "'Verbalizable Workspace' paper (https://github.com/anthropics/jlens), "
    "Apache-2.0. Fit via Neuronpedia run-all-fit-lens.py."
)


def sanitize_filename_part(value: str) -> str:
    value = value.replace("/", "-")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a Jacobian lens for every Hugging Face model in "
            "np_model_to_hf.json, skipping runs that already exist."
        )
    )
    parser.add_argument("--models-json", default=str(DEFAULT_MODEL_MAP_PATH))
    parser.add_argument("--fit-script", default=str(DEFAULT_FIT_SCRIPT))
    parser.add_argument("--exports-dir", default=str(DEFAULT_EXPORTS_DIR))
    parser.add_argument("--hf-cache-dir", default=DEFAULT_HF_CACHE_DIR)

    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--dataset-config",
        default=DEFAULT_DATASET_CONFIG,
        help="dataset config/subset; pass 'none' for datasets without one",
    )
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--text-field", default=DEFAULT_TEXT_FIELD)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)

    parser.add_argument("--n-prompts", type=int, default=DEFAULT_N_PROMPTS)
    parser.add_argument("--dim-batch", type=int, default=DEFAULT_DIM_BATCH)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument(
        "--dtype", default=DEFAULT_DTYPE, choices=("bfloat16", "float16", "float32")
    )
    parser.add_argument(
        "--device-map",
        default=DEFAULT_DEVICE_MAP,
        help="'cuda' for single GPU, 'auto' to shard large models across all GPUs",
    )
    parser.add_argument("--stop-at-delta", type=float, default=DEFAULT_STOP_AT_DELTA)
    parser.add_argument("--min-prompts", type=int, default=DEFAULT_MIN_PROMPTS)
    parser.add_argument("--stop-window", type=int, default=DEFAULT_STOP_WINDOW)
    parser.add_argument("--levels", default=DEFAULT_LEVELS)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--keep-hf-cache",
        action="store_true",
        help="keep downloaded weights in --hf-cache-dir (default: delete after each model)",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="print what would run without fitting."
    )
    args = parser.parse_args()

    if args.n_prompts < 1:
        parser.error("--n-prompts must be at least 1")
    if args.dim_batch < 1:
        parser.error("--dim-batch must be at least 1")
    if args.min_prompts < 1:
        parser.error("--min-prompts must be at least 1")
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


def query_gpus() -> list[dict[str, Any]]:
    """Per-GPU name and VRAM (total/free, GB) via nvidia-smi; [] if unavailable."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    gpus: list[dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        index, name, total_mib, free_mib = parts
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "total_gb": round(float(total_mib) / 1024, 1),
                "free_gb": round(float(free_mib) / 1024, 1),
            }
        )
    return gpus


def run_dir_for_model(exports_dir: Path, np_model_id: str, dataset_name: str) -> Path:
    return exports_dir / np_model_id / "jlens" / sanitize_filename_part(dataset_name)


def run_is_complete(run_dir: Path) -> bool:
    if not run_dir.is_dir():
        return False
    if not (run_dir / "config.yaml").is_file():
        return False
    return bool(glob.glob(str(run_dir / "*_jacobian_lens.pt")))


def build_fit_command(
    args: argparse.Namespace, fit_script: Path, hf_model_name: str, run_dir: Path
) -> list[str]:
    command = [
        sys.executable,
        str(fit_script),
        hf_model_name,
        "--out_dir",
        str(run_dir),
        "--dataset",
        args.dataset_name,
        "--dataset_config",
        args.dataset_config,
        "--dataset_split",
        args.dataset_split,
        "--text_field",
        args.text_field,
        "--max_chars",
        str(args.max_chars),
        "--n_prompts",
        str(args.n_prompts),
        "--dim_batch",
        str(args.dim_batch),
        "--max_seq_len",
        str(args.max_seq_len),
        "--dtype",
        args.dtype,
        "--device_map",
        args.device_map,
        "--min_prompts",
        str(args.min_prompts),
        "--stop_window",
        str(args.stop_window),
        "--levels",
        args.levels,
        "--hf_cache_dir",
        args.hf_cache_dir,
    ]
    if args.target_layer is not None:
        command += ["--target_layer", str(args.target_layer)]
    if args.stop_at_delta is not None:
        command += ["--stop_at_delta", repr(args.stop_at_delta)]
    if args.no_compile:
        command.append("--no_compile")
    if args.trust_remote_code:
        command.append("--trust_remote_code")
    if args.keep_hf_cache:
        command.append("--keep_hf_cache")
    return command


def read_results(run_dir: Path) -> dict[str, Any]:
    """Pull prompts-fitted and final Δmean from the convergence CSV, if present."""
    matches = glob.glob(str(run_dir / "*_convergence.csv"))
    if not matches:
        return {}
    last_row: dict[str, str] | None = None
    with open(matches[0], newline="") as f:
        for last_row in csv.DictReader(f):
            pass
    if last_row is None:
        return {}
    return {
        "prompts_fitted": int(last_row["n_done"]),
        "final_identity_distance": float(last_row["identity_distance"]),
        "final_mean_rel_change": float(last_row["mean_rel_change"]),
    }


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _yaml_dump(obj: dict[str, Any], indent: int = 0) -> str:
    """Minimal YAML emitter for nested dicts of scalars (no external deps)."""
    pad = "  " * indent
    lines: list[str] = []
    for key, value in obj.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.append(_yaml_dump(value, indent + 1))
        else:
            lines.append(f"{pad}{key}: {_yaml_scalar(value)}")
    return "\n".join(line for line in lines if line)


def write_config_yaml(
    run_dir: Path,
    *,
    np_model_id: str,
    hf_model_name: str,
    args: argparse.Namespace,
    command: list[str],
    gpus: list[dict[str, Any]],
    results: dict[str, Any],
) -> None:
    """Write config.yaml: a comment header (GPUs/command/attribution) + the
    full config used to generate the lens (including default arguments)."""
    header = ["# Jacobian lens fit — generated by Neuronpedia run-all-fit-lens.py"]
    header.append(f"# {ATTRIBUTION}")
    header.append("#")
    header.append("# GPU VRAM available when this lens was fit:")
    if gpus:
        for gpu in gpus:
            header.append(
                f"#   GPU {gpu['index']} ({gpu['name']}): "
                f"{gpu['free_gb']:.1f} GB free / {gpu['total_gb']:.1f} GB total"
            )
    else:
        header.append("#   (nvidia-smi unavailable — VRAM not recorded)")
    header.append("#")
    header.append("# Exact command used:")
    header.append(f"#   {shlex.join(command)}")
    header.append("#")
    header.append(f"# Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}")
    header.append("")

    config_config = (
        None
        if args.dataset_config.lower() in ("none", "", "null")
        else args.dataset_config
    )
    body: dict[str, Any] = {
        "np_model_id": np_model_id,
        "hf_model_name": hf_model_name,
        "dataset": {
            "name": args.dataset_name,
            "config": config_config,
            "split": args.dataset_split,
            "text_field": args.text_field,
            "max_chars": args.max_chars,
        },
        "fit": {
            "n_prompts": args.n_prompts,
            "dim_batch": args.dim_batch,
            "max_seq_len": args.max_seq_len,
            "target_layer": args.target_layer,
            "dtype": args.dtype,
            "device_map": args.device_map,
            "compile": not args.no_compile,
            "trust_remote_code": args.trust_remote_code,
            "stop_at_delta": args.stop_at_delta,
            "min_prompts": args.min_prompts,
            "stop_window": args.stop_window,
            "levels": args.levels,
        },
        "gpus": {
            f"gpu_{gpu['index']}": {
                "name": gpu["name"],
                "free_gb": gpu["free_gb"],
                "total_gb": gpu["total_gb"],
            }
            for gpu in gpus
        }
        or None,
        "results": results or None,
        "command": shlex.join(command),
        "attribution": ATTRIBUTION,
    }
    with open(run_dir / "config.yaml", "w") as f:
        f.write("\n".join(header))
        f.write(_yaml_dump(body))
        f.write("\n")


def main() -> None:
    args = parse_args()
    models_json_path = Path(args.models_json).expanduser().resolve()
    fit_script = Path(args.fit_script).expanduser().resolve()
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
        if run_is_complete(run_dir):
            print(f"{prefix}: skipping, complete lens already at {run_dir}")
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        command = build_fit_command(args, fit_script, hf_model_name, run_dir)
        gpus = query_gpus()

        print(f"{prefix}: running")
        print("  " + shlex.join(command))
        if args.dry_run:
            continue

        try:
            subprocess.run(command, cwd=SCRIPT_DIR, env=os.environ.copy(), check=True)
        except subprocess.CalledProcessError as exc:
            print(
                f"{prefix}: FAILED (exit {exc.returncode}); leaving run dir for inspection"
            )
            continue

        results = read_results(run_dir)
        write_config_yaml(
            run_dir,
            np_model_id=np_model_id,
            hf_model_name=hf_model_name,
            args=args,
            command=command,
            gpus=gpus,
            results=results,
        )
        print(f"{prefix}: done -> {run_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
