"""Build the neuron act range table from the live dashboard records in R2.

Each layer_XX.json.gz is one gzip member per neuron, located by layer_XX.idx (n+1 uint64
offsets). `stats` and `quantiles` sit near the start of a record, so only its first 2 KiB is
decompressed. Output keys:

  max              [L, N]     float32  stats.max (0 when missing)
  min              [L, N]     float32  stats.min (0 when missing)
  quantile_levels  [Q]        float64  quantiles.q, as fractions
  quantiles        [L, N, Q]  float32  quantiles.v (activation at each level)
  n_tokens         [1]        int64    tokens in the dashboard corpus (meta.json)

Usage: build_act_range_from_records.py OUT_DIR MODEL...  (needs R2_* in the environment)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import boto3
import numpy as np
from safetensors.numpy import save_file

BUCKET = "neuronpedia"
LAYERS = {
    "llama3.3-70b-it": 80,
    "qwen3-32b": 64,
    "gemma-4-31b-it": 60,
    "qwen3-8b": 36,
    "qwen3.5-4b": 32,
    "gemma-4-12b-it": 48,
    "gemma-4-12b": 48,
    "llama3.1-8b-it": 32,
    "qwen3-14b": 40,
}
CHUNK = 128 * 2**20
NUM = rb"(null|NaN|[-+0-9.eE]+)"
STATS = re.compile(rb'"stats":\s*\{\s*"max":\s*' + NUM + rb',\s*"min":\s*' + NUM)
QUANTS = re.compile(rb'"quantiles":\s*\{\s*"q":\s*\[([^\]]*)\],\s*"v":\s*\[([^\]]*)\]')


def client():
    """An R2 S3 client from the environment."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_S3_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def num(b: bytes) -> float:
    """A JSON number, with null/NaN as NaN."""
    b = b.strip()
    return float("nan") if b in (b"null", b"NaN") else float(b)


def layer_stats(model: str, layer: int):
    """(layer, max, min, levels, quantiles) from every record of one layer."""
    s3 = client()
    base = f"neuron-dashboards/v2/{model}/neurons/layer_{layer:02d}"
    offs = np.frombuffer(
        s3.get_object(Bucket=BUCKET, Key=f"{base}.idx")["Body"].read(), dtype="<u8"
    )
    n = len(offs) - 1
    hi = np.full(n, np.nan, dtype=np.float32)
    lo = np.full(n, np.nan, dtype=np.float32)
    quants = None
    levels = None
    i = 0
    while i < n:
        j = max(int(np.searchsorted(offs, offs[i] + CHUNK, side="right")) - 1, i + 1)
        start, end = int(offs[i]), int(offs[j])
        buf = s3.get_object(
            Bucket=BUCKET, Key=f"{base}.json.gz", Range=f"bytes={start}-{end - 1}"
        )["Body"].read()
        for k in range(i, j):
            a, b = int(offs[k]) - start, int(offs[k + 1]) - start
            head = zlib.decompressobj(31).decompress(buf[a:b], 2048)
            ms, mq = STATS.search(head), QUANTS.search(head)
            if ms is None or mq is None:
                raise RuntimeError(
                    f"{base} neuron {k}: no stats/quantiles in {head[:300]!r}"
                )
            hi[k], lo[k] = num(ms.group(1)), num(ms.group(2))
            q = [num(x) for x in mq.group(1).split(b",")]
            v = [num(x) for x in mq.group(2).split(b",")]
            if levels is None:
                levels = np.array(q, dtype=np.float64)
                quants = np.full((n, len(q)), np.nan, dtype=np.float32)
            elif len(q) != len(levels) or not np.allclose(q, levels):
                raise RuntimeError(
                    f"{base} neuron {k}: quantile levels {q} differ from {levels.tolist()}"
                )
            quants[k] = v
        i = j
    return layer, hi, lo, levels, quants


def build(out_dir: str, model: str) -> None:
    """Write out_dir/model.safetensors from all layers, 12 at a time."""
    n_layers = LAYERS[model]
    n_tokens = json.loads(
        client()
        .get_object(Bucket=BUCKET, Key=f"neuron-dashboards/v2/{model}/meta.json")[
            "Body"
        ]
        .read()
    )["n_tokens"]
    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(12) as pool:
        for layer, hi, lo, levels, quants in pool.map(
            layer_stats, [model] * n_layers, range(n_layers)
        ):
            results[layer] = (hi, lo, levels, quants)
            print(
                f"{model} L{layer:02d} done ({time.time() - t0:.0f}s): nan max {int(np.isnan(hi).sum())} nan quantiles {int(np.isnan(quants).sum())}",
                flush=True,
            )
    levels = results[0][2]
    for li in range(n_layers):
        assert np.allclose(results[li][2], levels), f"layer {li} quantile levels differ"
    hi = np.nan_to_num(np.stack([results[li][0] for li in range(n_layers)]), nan=0.0)
    lo = np.nan_to_num(np.stack([results[li][1] for li in range(n_layers)]), nan=0.0)
    quants = np.nan_to_num(
        np.stack([results[li][3] for li in range(n_layers)]), nan=0.0
    )
    path = f"{out_dir}/{model}.safetensors"
    save_file(
        {
            "max": hi,
            "min": lo,
            "quantile_levels": levels,
            "quantiles": quants,
            "n_tokens": np.array([n_tokens], dtype=np.int64),
        },
        path,
    )
    print(
        f"wrote {path}: max {hi.shape}, quantiles {quants.shape}, levels {levels.tolist()}, n_tokens {n_tokens}, {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    os.makedirs(sys.argv[1], exist_ok=True)
    for m in sys.argv[2:]:
        build(sys.argv[1], m)
