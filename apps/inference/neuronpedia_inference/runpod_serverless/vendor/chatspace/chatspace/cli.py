from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from .env import load_environment, get_env
from .utils import ensure_dir, iso_now


def handle_download_dataset(args: argparse.Namespace) -> None:
    """Resolve a dataset and persist lightweight metadata under /workspace.

    This scaffolds the /workspace layout without forcing a full download yet.
    """
    from datasets import load_dataset  # import here to keep CLI startup fast

    dataset_name: str = args.name
    split: str | None = args.split
    source: str = args.source

    # Try to resolve the dataset. This may populate the HF cache, but we avoid
    # copying the full dataset here. We'll persist a small descriptor instead.
    ds = load_dataset(dataset_name, split=split) if split else load_dataset(dataset_name)

    base_dir = Path("/workspace/datasets/raw") / source / dataset_name.replace("/", "__") / (split or "all")
    ensure_dir(base_dir)

    descriptor: Dict[str, Any] = {
        "dataset": dataset_name,
        "source": source,
        "split": split,
        "created_at": iso_now(),
        "num_rows": int(getattr(ds, "num_rows", 0)) if split else None,
        "features": getattr(getattr(ds, "features", None), "keys", lambda: [])(),
        "note": "Lightweight descriptor; full materialization deferred to processing steps.",
    }

    with (base_dir / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(descriptor, f, indent=2)

    print(f"Wrote descriptor: {base_dir / 'dataset.json'}")


def handle_embed_dataset(args: argparse.Namespace) -> None:
    """Scaffold for embedding; currently dry-run unless --execute is passed."""
    dataset_name: str = args.dataset
    split: str | None = args.split
    n: int | None = args.n
    p: float | None = args.p
    seed: int = args.seed
    model: str = args.model or get_env("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    execute: bool = args.execute
    text_field: str = args.text_field

    # Paths
    emb_dir = Path("/workspace/embeddings") / model / dataset_name.replace("/", "__")
    idx_dir = Path("/workspace/indexes") / model / dataset_name.replace("/", "__")
    ensure_dir(emb_dir)
    ensure_dir(idx_dir)

    plan = {
        "dataset": dataset_name,
        "split": split,
        "n": n,
        "p": p,
        "seed": seed,
        "model": model,
        "execute": execute,
        "text_field": text_field,
        "emb_dir": str(emb_dir),
        "idx_dir": str(idx_dir),
    }
    print("Embedding plan:\n" + json.dumps(plan, indent=2))

    if not execute:
        print("Dry-run: pass --execute to perform embeddings.")
        return

    # Minimal execution path: sample a few rows and embed them
    from datasets import load_dataset
    from openai import OpenAI

    ds = load_dataset(dataset_name, split=split or "train")
    total = ds.num_rows
    if n is None and p is None:
        n = min(100, total)
    if p is not None and n is None:
        n = max(1, int(total * p))
    n = min(n or total, total)

    rng = __import__("random")
    rng.seed(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    indices = indices[:n]

    texts = []
    ids = []
    for i in indices:
        row = ds[i]
        text = str(row.get(text_field, "")).strip()
        if not text:
            continue
        texts.append(text)
        ids.append(str(row.get("id", i)))

    if not texts:
        print("No texts found with the provided text field; aborting.")
        return

    client = OpenAI(api_key=get_env("OPENAI_API_KEY", required=True))
    batch_size = args.batch_size
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        resp = client.embeddings.create(model=model, input=batch)
        embeddings.extend([d.embedding for d in resp.data])

    # Write a single shard in Parquet
    import pyarrow as pa
    import pyarrow.parquet as pq

    created_at = iso_now()
    shard_path = emb_dir / "shard-00000.parquet"

    num_rows = len(embeddings)
    metadata_json = [json.dumps({"text_field": text_field}) for _ in range(num_rows)]
    table = pa.table(
        {
            "id": pa.array(ids, type=pa.string()),
            "source": pa.array(["huggingface"] * num_rows, type=pa.string()),
            "dataset": pa.array([dataset_name] * num_rows, type=pa.string()),
            "split": pa.array([split or "train"] * num_rows, type=pa.string()),
            "text": pa.array(texts, type=pa.string()),
            "metadata": pa.array(metadata_json, type=pa.string()),
            "embedding": pa.array(embeddings, type=pa.list_(pa.float32())),
            "model": pa.array([model] * num_rows, type=pa.string()),
            "created_at": pa.array([created_at] * num_rows, type=pa.string()),
        }
    )
    pq.write_table(table, shard_path)

    manifest = {
        "model": model,
        "dataset": dataset_name,
        "created_at": created_at,
        "shards": [
            {
                "path": str(shard_path),
                "rows": len(embeddings),
                "format": "parquet",
            }
        ],
        "sampling": {"seed": seed, "n": n, "p": p},
    }
    with (idx_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote shard: {shard_path}")
    print(f"Wrote manifest: {idx_dir / 'manifest.json'}")


def handle_embed_hf(args: argparse.Namespace) -> None:
    """Embed a dataset using a local/hosted SentenceTransformer model."""
    # Lazy import to keep CLI startup fast
    from .hf_embed import SentenceTransformerConfig, run_sentence_transformer

    cfg = SentenceTransformerConfig(
        dataset=args.dataset,
        subset=args.subset,
        split=args.split,
        text_field=args.text_field,
        model_name=args.model,
        batch_size=args.batch_size,
        rows_per_shard=args.rows_per_shard,
        max_rows=args.max_rows,
        output_root=Path(args.output_root),
        dtype=args.dtype,
        device=args.device,
        attention_impl=args.attention,
        prefetch_batches=args.prefetch_batches,
        progress=not args.no_progress,
        run_id=args.run_id,
        resume=args.resume,
        bucket_min_tokens=args.bucket_min_tokens,
        bucket_max_tokens=args.bucket_max_tokens,
        tokens_per_batch=args.tokens_per_batch,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
        extract_first_assistant=args.extract_first_assistant,
    )

    run_sentence_transformer(cfg)


def handle_reduce_embeddings(args: argparse.Namespace) -> None:
    """Compute dimensionality reduction for embeddings."""
    from .reduce_embeddings import (
        load_embeddings_from_shards,
        compute_reduction,
        save_reduced_embeddings
    )
    import numpy as np

    # Load embeddings from all specified directories
    all_embeddings = []
    all_texts = []
    all_datasets = []

    for emb_dir_str in args.embedding_dirs:
        emb_dir = Path(emb_dir_str)
        embeddings, texts, datasets = load_embeddings_from_shards(
            emb_dir,
            max_samples=args.max_samples,
            sample_seed=args.sample_seed
        )
        all_embeddings.append(embeddings)
        all_texts.extend(texts)
        all_datasets.extend(datasets)

    # Concatenate all embeddings
    embeddings_combined = np.vstack(all_embeddings)
    print(f"Total embeddings: {len(embeddings_combined)}")
    print(f"Datasets: {set(all_datasets)}")

    # Compute reduction
    reduced = compute_reduction(
        embeddings_combined,
        method=args.method,
        n_components=args.n_components,
        seed=args.seed,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        perplexity=args.perplexity
    )

    # Save
    metadata = {
        "method": args.method,
        "n_components": args.n_components,
        "n_samples": len(reduced),
        "datasets": list(set(all_datasets)),
        "embedding_dirs": args.embedding_dirs,
        "seed": args.seed,
        "sample_seed": args.sample_seed,
        "max_samples_per_dataset": args.max_samples,
    }

    save_reduced_embeddings(
        Path(args.output),
        reduced,
        all_texts,
        all_datasets,
        metadata
    )


def handle_visualize_embeddings(args: argparse.Namespace) -> None:
    """Launch interactive web viewer for reduced embeddings."""
    from .visualize_embeddings import load_reduced_embeddings, create_app

    df, metadata = load_reduced_embeddings(Path(args.input))
    app = create_app(df, metadata)

    print(f"Starting server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


def handle_steering_train_forward(args: argparse.Namespace) -> None:
    """Forward subcommand arguments to the steering job runner."""
    from .steering import job as steering_job

    argv = list(args.job_args)
    if argv and argv[0] == "--":
        argv = argv[1:]
    steering_job.main(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chatspace", description="Dataset download and embedding toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download-dataset", help="Resolve and describe a dataset")
    p_dl.add_argument("--name", required=True, help="Dataset name, e.g., 'openwebtext' or 'ai2_arc' or 'org/ds'")
    p_dl.add_argument("--split", default=None, help="Dataset split (e.g., train)")
    p_dl.add_argument("--source", default="huggingface", help="Dataset source label")
    p_dl.set_defaults(func=handle_download_dataset)

    p_emb = sub.add_parser("embed-dataset", help="Sample and embed a dataset")
    p_emb.add_argument("--dataset", required=True, help="Dataset name, e.g., 'cnn_dailymail' or 'org/ds'")
    p_emb.add_argument("--split", default=None, help="Dataset split (defaults to train)")
    p_emb.add_argument("--n", type=int, default=None, help="Number of samples to embed")
    p_emb.add_argument("--p", type=float, default=None, help="Fraction of samples to embed")
    p_emb.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic sampling")
    p_emb.add_argument("--model", default=None, help="Embedding model (defaults to OPENAI_EMBED_MODEL or text-embedding-3-small)")
    p_emb.add_argument("--text-field", default="text", help="Field name containing text")
    p_emb.add_argument("--batch-size", type=int, default=128, help="Embedding request batch size")
    p_emb.add_argument("--execute", action="store_true", help="Perform embeddings (omit for dry-run)")
    p_emb.set_defaults(func=handle_embed_dataset)

    p_hf = sub.add_parser("embed-hf", help="Embed a dataset with a SentenceTransformer model")
    p_hf.add_argument("--dataset", required=True, help="Dataset name, e.g., 'HuggingFaceFW/fineweb'")
    p_hf.add_argument("--subset", default=None, help="Dataset config/subset name (e.g., 'sample-10BT')")
    p_hf.add_argument("--split", default="train", help="Dataset split (default: train)")
    p_hf.add_argument("--text-field", default="text", help="Field containing text to embed")
    p_hf.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B", help="SentenceTransformer model name")
    p_hf.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding")
    p_hf.add_argument("--rows-per-shard", type=int, default=8192, help="Rows per Parquet shard")
    p_hf.add_argument("--max-rows", type=int, default=None, help="Optional cap on processed rows (testing)")
    p_hf.add_argument("--output-root", default="/workspace", help="Base output directory")
    p_hf.add_argument("--dtype", default="bfloat16", help="Model dtype hint")
    p_hf.add_argument("--device", default=None, help="Device hint, e.g. 'cuda', 'auto'")
    p_hf.add_argument(
        "--attention",
        default="flash_attention_2",
        help="Attention implementation hint (install extra 'flash-attn' to enable flash_attention_2)",
    )
    p_hf.add_argument("--prefetch-batches", type=int, default=4, help="Number of batches to prefetch from the loader")
    p_hf.add_argument("--bucket-min-tokens", type=int, default=128, help="Minimum token count for bucket sizing")
    p_hf.add_argument("--bucket-max-tokens", type=int, default=32768, help="Maximum token count for bucket sizing")
    p_hf.add_argument("--tokens-per-batch", type=int, default=None, help="Target number of tokens per batch (overrides --batch-size when set)")
    p_hf.add_argument("--compile-model", action="store_true", help="Enable torch.compile for model forward pass")
    p_hf.add_argument("--compile-mode", default="default", help="torch.compile mode (default: 'default')")
    p_hf.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    p_hf.add_argument("--run-id", default=None, help="Optional run identifier")
    p_hf.add_argument("--resume", action="store_true", help="Resume if shards already exist (not yet implemented)")
    p_hf.add_argument("--extract-first-assistant", action="store_true", help="Extract first assistant response from conversation field")
    p_hf.set_defaults(func=handle_embed_hf)

    p_reduce = sub.add_parser("reduce-embeddings", help="Compute dimensionality reduction (UMAP/t-SNE) for embeddings")
    p_reduce.add_argument("--embedding-dirs", nargs="+", required=True, help="Directories containing embedding shards (can specify multiple)")
    p_reduce.add_argument("--output", required=True, help="Output path for reduced embeddings (parquet)")
    p_reduce.add_argument("--method", choices=["umap", "tsne"], default="umap", help="Dimensionality reduction method")
    p_reduce.add_argument("--n-components", type=int, default=2, help="Output dimensionality (2 or 3)")
    p_reduce.add_argument("--max-samples", type=int, default=None, help="Maximum samples per dataset to load")
    p_reduce.add_argument("--sample-seed", type=int, default=42, help="Random seed for sampling")
    p_reduce.add_argument("--seed", type=int, default=42, help="Random seed for reduction algorithm")
    p_reduce.add_argument("--n-neighbors", type=int, default=15, help="UMAP: number of neighbors")
    p_reduce.add_argument("--min-dist", type=float, default=0.1, help="UMAP: minimum distance")
    p_reduce.add_argument("--metric", default="cosine", help="Distance metric")
    p_reduce.add_argument("--perplexity", type=float, default=30, help="t-SNE: perplexity parameter")
    p_reduce.set_defaults(func=handle_reduce_embeddings)

    p_viz = sub.add_parser("visualize-embeddings", help="Launch interactive web viewer for reduced embeddings")
    p_viz.add_argument("--input", required=True, help="Path to reduced embeddings parquet file")
    p_viz.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    p_viz.add_argument("--port", type=int, default=8050, help="Port to bind to")
    p_viz.add_argument("--debug", action="store_true", help="Enable debug mode")
    p_viz.set_defaults(func=handle_visualize_embeddings)

    p_steer = sub.add_parser(
        "steering-train",
        help="Run a steering-vector training job with filesystem coordination",
    )
    p_steer.add_argument(
        "job_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to chatspace.steering.job",
    )
    p_steer.set_defaults(func=handle_steering_train_forward)

    return parser


def main(argv: list[str] | None = None) -> None:
    load_environment()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
