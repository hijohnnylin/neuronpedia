"""Lightweight steering training runner with optional model reuse."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time
from pathlib import Path
from typing import Any

import torch

from ..constants import STEERING_RUN_ROOT
from ..utils import ensure_dir, iso_now, sanitize_component
from . import runs as run_utils
from .constants import ERROR_FILENAME, SUMMARY_FILENAME
from .model import SteeringVectorConfig
from .train import add_training_arguments, build_model, prepare_tokenizer, run_training

DEFAULT_RUN_ROOT = STEERING_RUN_ROOT


def add_job_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT, help="Base directory for steering runs")
    parser.add_argument("--run-id", default=None, help="Optional run identifier (defaults to current UTC timestamp)")
    parser.add_argument("--attempt", type=int, default=1, help="Attempt index (>=1) to separate retries")
    parser.add_argument(
        "--skip-if-committed",
        action="store_true",
        help="Skip datasets that already have a successful summary under the run root",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log planned actions without invoking training",
    )
    parser.add_argument(
        "--reuse-base-model",
        action="store_true",
        help="Load the base model/tokenizer once and reuse across datasets processed sequentially",
    )
    parser.add_argument(
        "--dataset-stride",
        type=int,
        default=None,
        help="Stride to select datasets from the provided list (combine with --dataset-offset)",
    )
    parser.add_argument(
        "--dataset-offset",
        type=int,
        default=None,
        help="Offset when applying --dataset-stride (defaults to 0)",
    )


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if hasattr(obj, "item"):
        try:
            return _json_ready(obj.item())
        except Exception:
            return str(obj)
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)


def _gpu_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "torch_cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if not torch.cuda.is_available():
        return snapshot
    snapshot["device_count"] = torch.cuda.device_count()
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append({"index": index, "name": props.name, "total_memory": props.total_memory})
    snapshot["devices"] = devices
    snapshot["visible_device_indices"] = [device["index"] for device in devices]
    return snapshot


def _resolve_output_paths(
    run_root: Path,
    model_name: str,
    dataset_name: str,
    run_id: str,
    attempt: int,
) -> tuple[Path, Path]:
    model_slug = sanitize_component(model_name)
    dataset_slug = sanitize_component(dataset_name)
    dataset_root = run_root / model_slug / dataset_slug
    attempt_dir = dataset_root / run_id / f"try{attempt:02d}"
    return dataset_root, attempt_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)


def _prepare_train_args(args: argparse.Namespace, output_dir: Path, dataset_name: str) -> argparse.Namespace:
    job_only = {
        "run_root",
        "run_id",
        "attempt",
        "skip_if_committed",
        "dry_run",
        "reuse_base_model",
        "dataset_stride",
        "dataset_offset",
    }
    payload = {k: v for k, v in vars(args).items() if k not in job_only}
    payload["output_dir"] = output_dir
    payload["datasets"] = [dataset_name]
    return argparse.Namespace(**payload)


def _apply_dataset_stride(
    datasets: list[str],
    stride: int | None,
    offset: int,
) -> list[str]:
    if stride is None or stride <= 1:
        return datasets
    return [name for idx, name in enumerate(datasets) if idx % stride == offset]


def _run_single_dataset(
    args: argparse.Namespace,
    dataset_name: str,
    run_id: str,
    shared_model,
    shared_tokenizer,
) -> None:
    dataset_root, output_dir = _resolve_output_paths(args.run_root, args.model, dataset_name, run_id, args.attempt)

    if args.skip_if_committed and run_utils.has_successful_run(args.run_root, dataset_name):
        print(f"Skipping {dataset_name}: existing summary detected under {dataset_root}")
        return

    if args.dry_run:
        print(f"[dry-run] Would train {dataset_name} -> {output_dir}")
        return

    ensure_dir(output_dir)

    train_args = _prepare_train_args(args, output_dir, dataset_name)
    shared_model_arg = shared_model if args.reuse_base_model else None
    shared_tokenizer_arg = shared_tokenizer if args.reuse_base_model else None

    gpu_info = _gpu_snapshot()
    host = socket.gethostname()
    started_at = iso_now()
    print(
        f"Training steering vector: dataset={dataset_name} model={args.model} output={output_dir} host={host}"
    )

    start_time = time.monotonic()
    try:
        summary = run_training(
            train_args,
            model=shared_model_arg,
            tokenizer=shared_tokenizer_arg,
            reset_vector=True,
        )
    except Exception as exc:
        duration = time.monotonic() - start_time
        error_payload = {
            "status": "failed",
            "dataset": dataset_name,
            "model": args.model,
            "run_id": run_id,
            "attempt": args.attempt,
            "started_at": started_at,
            "ended_at": iso_now(),
            "duration_seconds": duration,
            "exception": repr(exc),
            "gpu": gpu_info,
            "host": host,
            "pid": os.getpid(),
        }
        _write_json(output_dir / ERROR_FILENAME, error_payload)
        raise

    duration = time.monotonic() - start_time
    summary_payload = {
        "status": "success",
        "dataset": dataset_name,
        "model": args.model,
        "run_id": run_id,
        "attempt": args.attempt,
        "started_at": started_at,
        "ended_at": iso_now(),
        "duration_seconds": duration,
        "summary": summary,
        "gpu": gpu_info,
        "host": host,
        "pid": os.getpid(),
    }
    _write_json(output_dir / SUMMARY_FILENAME, summary_payload)
    print(f"Completed steering training for {dataset_name}. Summary written to {output_dir / SUMMARY_FILENAME}")


def _validate_datasets(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    datasets = list(args.datasets)
    if not datasets:
        parser.error("--datasets requires at least one dataset name")
    multi = len(datasets) > 1
    if multi and args.output_dir is not None:
        parser.error("--output-dir cannot be combined with multiple datasets; allow runner to manage directories")
    if multi and not args.reuse_base_model:
        parser.error("Multiple datasets require --reuse-base-model so the base model is reused in-process")
    return datasets


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    add_job_arguments(parser)
    args = parser.parse_args(argv)

    if args.attempt < 1:
        parser.error("--attempt must be >= 1")

    if args.dataset_stride is not None and args.dataset_stride < 1:
        parser.error("--dataset-stride must be >= 1")

    dataset_offset = args.dataset_offset if args.dataset_offset is not None else 0
    if dataset_offset < 0:
        parser.error("--dataset-offset must be >= 0")

    datasets = _validate_datasets(args, parser)
    if args.dataset_stride is not None and dataset_offset >= args.dataset_stride:
        parser.error("--dataset-offset must be < --dataset-stride")

    datasets = _apply_dataset_stride(datasets, args.dataset_stride, dataset_offset)
    if not datasets:
        print(
            f"No datasets matched this worker (stride={args.dataset_stride}, offset={dataset_offset}). Nothing to do."
        )
        return

    args.datasets = datasets
    run_id = args.run_id or sanitize_component(iso_now(), lowercase=False)

    shared_model = None
    shared_tokenizer = None
    if args.reuse_base_model and not args.dry_run:
        shared_model, shared_tokenizer = _load_shared_components(args)

    for dataset in datasets:
        _run_single_dataset(args, dataset, run_id, shared_model, shared_tokenizer)


def _load_shared_components(args: argparse.Namespace):
    tokenizer = prepare_tokenizer(args.model)
    cfg = SteeringVectorConfig(
        model_name=args.model,
        target_layer=args.target_layer,
        init_scale=args.init_scale,
    )
    model = build_model(cfg, args.device_map)
    return model, tokenizer


if __name__ == "__main__":
    main()
