"""
Dimensionality reduction for embeddings.

Computes UMAP or t-SNE projections of high-dimensional embeddings
and saves them to disk for interactive visualization.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from umap import UMAP
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


def load_embeddings_from_shards(
    embedding_dir: Path,
    max_samples: int | None = None,
    sample_seed: int = 42
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Load embeddings from parquet shards.

    Returns:
        embeddings: (N, D) array of embeddings
        texts: List of text strings
        datasets: List of dataset identifiers for each sample
    """
    shard_files = sorted(embedding_dir.glob("*.parquet"))
    if not shard_files:
        raise ValueError(f"No parquet files found in {embedding_dir}")

    logger.info(f"Found {len(shard_files)} shards in {embedding_dir}")

    embeddings_list = []
    texts_list = []
    datasets_list = []

    # Extract dataset name from path
    # /workspace/embeddings/{model}/{dataset}/{split}/shard-*.parquet
    dataset_name = embedding_dir.parent.name

    for shard_file in tqdm(shard_files, desc="Loading shards"):
        table = pq.read_table(shard_file, columns=["embedding", "text"])

        # Convert to numpy
        embeddings = np.array(table["embedding"].to_pylist())
        texts = table["text"].to_pylist()

        embeddings_list.append(embeddings)
        texts_list.extend(texts)
        datasets_list.extend([dataset_name] * len(texts))

        if max_samples and sum(len(e) for e in embeddings_list) >= max_samples:
            break

    embeddings = np.vstack(embeddings_list)

    # Random sample if needed
    if max_samples and len(embeddings) > max_samples:
        rng = np.random.RandomState(sample_seed)
        indices = rng.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        texts_list = [texts_list[i] for i in indices]
        datasets_list = [datasets_list[i] for i in indices]

    logger.info(f"Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    return embeddings, texts_list, datasets_list


def compute_reduction(
    embeddings: np.ndarray,
    method: Literal["umap", "tsne"] = "umap",
    n_components: int = 2,
    seed: int = 42,
    **kwargs
) -> np.ndarray:
    """
    Compute dimensionality reduction.

    Args:
        embeddings: (N, D) array
        method: "umap" or "tsne"
        n_components: Output dimensionality (typically 2 or 3)
        seed: Random seed
        **kwargs: Additional parameters for UMAP or t-SNE

    Returns:
        reduced: (N, n_components) array
    """
    logger.info(f"Computing {method.upper()} reduction to {n_components}D...")

    if method == "umap":
        reducer = UMAP(
            n_components=n_components,
            random_state=seed,
            n_neighbors=kwargs.get("n_neighbors", 15),
            min_dist=kwargs.get("min_dist", 0.1),
            metric=kwargs.get("metric", "cosine"),
            verbose=True
        )
    elif method == "tsne":
        reducer = TSNE(
            n_components=n_components,
            random_state=seed,
            perplexity=kwargs.get("perplexity", 30),
            metric=kwargs.get("metric", "cosine"),
            verbose=1,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    reduced = reducer.fit_transform(embeddings)
    logger.info(f"Reduction complete: {reduced.shape}")

    return reduced


def save_reduced_embeddings(
    output_path: Path,
    reduced: np.ndarray,
    texts: list[str],
    datasets: list[str],
    metadata: dict
):
    """Save reduced embeddings to parquet with metadata."""

    # Truncate very long texts for hover display
    texts_truncated = [t[:500] + "..." if len(t) > 500 else t for t in texts]

    # Build pyarrow table
    table = pa.table({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "z": reduced[:, 2] if reduced.shape[1] > 2 else [0.0] * len(reduced),
        "text": texts_truncated,
        "dataset": datasets,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)

    # Save metadata
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved reduced embeddings to {output_path}")
    logger.info(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute dimensionality reduction for embeddings"
    )
    parser.add_argument(
        "--embedding-dirs",
        nargs="+",
        required=True,
        help="Directories containing embedding shards (can specify multiple)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for reduced embeddings (parquet)"
    )
    parser.add_argument(
        "--method",
        choices=["umap", "tsne"],
        default="umap",
        help="Dimensionality reduction method"
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Output dimensionality (2 or 3)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per dataset to load"
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reduction algorithm"
    )
    # UMAP params
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", default="cosine")
    # t-SNE params
    parser.add_argument("--perplexity", type=float, default=30)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

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
    logger.info(f"Total embeddings: {len(embeddings_combined)}")
    logger.info(f"Datasets: {set(all_datasets)}")

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


if __name__ == "__main__":
    main()
