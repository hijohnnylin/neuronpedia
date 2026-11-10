"""Convert the v2 neuron dashboards in R2 to Neuronpedia export files.

Each layer is one source of the "neurons" source set, like an SAE source in the exports:

  OUT/<model>/neurons/<layer>/
    model.jsonl, sourceset.jsonl, source.jsonl
    features/batch-N.jsonl.gz      one Neuron row per neuron, BATCH neurons per file
    activations/batch-N.jsonl.gz   the Activation rows of the same neurons

The rows are the ones recordToNeuronDashboard (apps/webapp/lib/utils/neuron-dashboards.ts) makes
in the browser: top examples, then the quantile bands, then the bottom examples. Echo examples,
gate and up values and the log histogram have no column, so they stay in the dashboards only.

A layer is written to <layer>.partial/ and renamed when complete, so a stopped run resumes.

Usage: convert_to_export.py MODEL [--layers 0,1] [--out /data/exports] [--workers 8]
  (needs R2_* in the environment)
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import mmap
import os
import shutil
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime

import boto3
import numpy as np
import orjson

BUCKET = "neuronpedia"
PREFIX = "neuron-dashboards/v2"
DASHBOARDS_URL = "https://dashboards.neuronpedia.org/v2"
SOURCE_SET = "neurons"
FORMAT_VERSION = 1
BATCH = 128
HIST_BINS = 50
LOG_BINS = 64
LOG_MIN = -3
LOG_DECADES = 6
META = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_meta.json")


def client():
    """Return an S3 client for R2."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_S3_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def log(msg: str) -> None:
    """Print a timestamped line."""
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def hist_bin_range(k: int, sigma: float) -> tuple[float, float]:
    """Magnitude range [lo, hi) of log bin k; the last bin has no upper edge."""

    def edge(i: int) -> float:
        return sigma * 10 ** (LOG_MIN + (i * LOG_DECADES) / LOG_BINS)

    if k == 0:
        return 0.0, edge(0)
    if k > LOG_BINS:
        return edge(LOG_BINS), math.inf
    return edge(k - 1), edge(k)


def linear_histogram(
    rec: dict, num_bins: int = HIST_BINS
) -> tuple[list[float], list[int]]:
    """Equal-width bins from min to max with gate states summed, as linearHistogram in the webapp."""
    hist, stats = rec["hist"], rec["stats"]
    lo = min(stats["min"], 0)
    hi = max(stats["max"], 0)
    sigma = hist["sigma"]
    if not (hi > lo) or not (sigma > 0):
        return [], []
    width = (hi - lo) / num_bins
    heights = [0.0] * num_bins

    def bin_of(x: float) -> int:
        return min(num_bins - 1, max(0, math.floor((x - lo) / width)))

    for sign in (1, -1):
        limit = hi if sign > 0 else -lo
        side = ("open_pos", "closed_pos") if sign > 0 else ("open_neg", "closed_neg")
        counts = np.add(hist[side[0]], hist[side[1]]).tolist()
        for k, count in enumerate(counts):
            if not (count > 0):
                continue
            bin_lo, raw_hi = hist_bin_range(k, sigma)
            bin_hi = min(raw_hi, limit)
            if not (bin_hi > bin_lo):
                heights[bin_of(sign * limit)] += count
                continue
            for j in range(num_bins):
                a = lo + j * width
                b = a + width
                mag_lo = max(a, 0) if sign > 0 else max(-b, 0)
                mag_hi = b if sign > 0 else -a
                ov_lo = max(mag_lo, bin_lo)
                ov_hi = min(mag_hi, bin_hi)
                if ov_hi > ov_lo:
                    if bin_lo == 0:
                        frac = (ov_hi - ov_lo) / (bin_hi - bin_lo)
                    else:
                        frac = math.log(ov_hi / ov_lo) / math.log(bin_hi / bin_lo)
                    heights[j] += count * frac
    values = [lo + (j + 0.5) * width for j in range(num_bins)]
    # Math.round rounds halves up; Python's round() rounds them to even.
    return values, [math.floor(h + 0.5) for h in heights]


def neuron_row(rec: dict, model: str, layer: str, created: str, creator: str) -> dict:
    """The Neuron row of one record."""
    hist = rec["hist"]
    total = rec["stats"]["n_hist_tokens"]
    positive = sum(hist["open_pos"]) + sum(hist["closed_pos"])
    values, heights = linear_histogram(rec)
    return {
        "modelId": model,
        "layer": layer,
        "index": str(rec["neuron"]),
        "creatorId": creator,
        "createdAt": created,
        "maxActApprox": rec["stats"]["max"],
        "hasVector": False,
        "vector": [],
        "vectorLabel": None,
        "vectorDefaultSteerStrength": None,
        "hookName": None,
        "topkCosSimIndices": [],
        "topkCosSimValues": [],
        "neuron_alignment_indices": [],
        "neuron_alignment_values": [],
        "neuron_alignment_l1": [],
        "correlated_neurons_indices": [],
        "correlated_neurons_pearson": [],
        "correlated_neurons_l1": [],
        "correlated_features_indices": [],
        "correlated_features_pearson": [],
        "correlated_features_l1": [],
        "neg_str": [t for _, t, _ in rec["logits"]["neg"]],
        "neg_values": [v for _, _, v in rec["logits"]["neg"]],
        "pos_str": [t for _, t, _ in rec["logits"]["pos"]],
        "pos_values": [v for _, _, v in rec["logits"]["pos"]],
        "frac_nonzero": positive / total if total > 0 else 0,
        "freq_hist_data_bar_heights": heights,
        "freq_hist_data_bar_values": values,
        "logits_hist_data_bar_heights": [],
        "logits_hist_data_bar_values": [],
        "decoder_weights_dist": [],
        "sourceSetName": None,
    }


def activation_rows(
    rec: dict, vocab: list[str], model: str, layer: str, created: str, creator: str
) -> list[dict]:
    """The Activation rows of one record: top, then bands, then bottom."""
    index = str(rec["neuron"])
    prefix = f"{model}-{layer}-{index}"
    n_vocab = len(vocab)
    rows = []

    def add(
        ex: dict,
        row_id: str,
        bin_min: float,
        bin_max: float,
        bin_contains: float | None,
    ) -> None:
        values = [v if v is not None else 0 for v in ex["acts"]]
        peak = ex["pos"] - ex["start"]
        rows.append(
            {
                "id": row_id,
                "modelId": model,
                "layer": layer,
                "index": index,
                "creatorId": creator,
                "createdAt": created,
                "tokens": [vocab[t] if t < n_vocab else f"<{t}>" for t in ex["tokens"]],
                "values": values,
                # The peak token's value: window values have 3 significant digits, the peak has 4.
                "maxValue": ex["act"],
                "maxValueTokenIndex": peak,
                "minValue": min(min(values, default=ex["act"]), ex["act"]),
                "dataIndex": str(ex["doc"]),
                "dataSource": "bottom" if ex["act"] < 0 else "top",
                "binMin": bin_min,
                "binMax": bin_max,
                "binContains": bin_contains,
                "qualifyingTokenIndex": peak,
                "dfaValues": [],
                "dfaTargetIndex": None,
                "dfaMaxValue": None,
                "lossValues": [],
                "logitContributions": None,
            }
        )

    for i, ex in enumerate(rec["top"]):
        add(ex, f"{prefix}-top-{i}", -1, rec["stats"]["max"], None)
    for b, band in enumerate(rec["bands"]):
        for i, ex in enumerate(band["examples"]):
            add(
                ex,
                f"{prefix}-band{b}-{i}",
                band["lo"],
                band["hi"],
                band["q_hi"] - band["q_lo"],
            )
    lowest = min(rec["bands"], key=lambda band: band["q_lo"], default=None)
    for i, ex in enumerate(rec["bottom"]):
        add(
            ex,
            f"{prefix}-bottom-{i}",
            rec["stats"]["min"],
            lowest["lo"] if lowest else ex["act"],
            None,
        )
    return rows


def write_gz(path: str, rows: list[dict]) -> None:
    """Write rows as gzip JSONL, level 5 like the other exports."""
    with open(path, "wb") as f:
        f.write(
            gzip.compress(
                b"".join(orjson.dumps(r) + b"\n" for r in rows), compresslevel=5
            )
        )


def load_meta() -> dict:
    """Read export_meta.json."""
    with open(META) as f:
        return json.load(f)


def meta_rows(model: str, layer: int, created: str, creator: str) -> dict[str, dict]:
    """The model, source set and source rows of one layer."""
    meta = load_meta()
    m = meta["models"][model]
    row = dict(m["model"])
    row.setdefault("createdAt", created)
    row.setdefault("updatedAt", created)
    sourceset = {
        "modelId": model,
        "name": SOURCE_SET,
        "creatorId": creator,
        "creatorName": "Neuronpedia",
        "creatorEmail": None,
        "type": "Neurons",
        "urls": [],
        "releaseName": None,
        "visibility": "PUBLIC",
        "hasDashboards": True,
        "defaultOfModelId": None,
        "defaultRange": 2,
        "defaultShowBreaks": True,
        "showCorrelated": False,
        "showDfa": False,
        "showHeadAttribution": False,
        "showUmap": False,
        "graphEnabled": False,
        "defaultGraphModelId": None,
        "hasGraphs": False,
        "similarityMatrixEnabled": False,
        "externalDashboardsUrl": f"{DASHBOARDS_URL}/{model}",
        "createdAt": created,
    }
    sourceset.update(m["sourceset"])
    source = {
        "id": str(layer),
        "modelId": model,
        "setName": SOURCE_SET,
        "creatorId": creator,
        "visibility": "PUBLIC",
        "hasDashboards": True,
        "hasUmap": False,
        "hasUmapClusters": False,
        "hasUmapLogSparsity": False,
        "dataset": meta["dataset"],
        "num_prompts": None,
        "num_tokens_in_prompt": None,
        "notes": None,
        "saelensConfig": None,
        "saelensRelease": None,
        "saelensSaeId": None,
        "hfRepoId": None,
        "hfFolderId": None,
        "defaultOfModelId": None,
        "cosSimMatchModelId": None,
        "cosSimMatchSourceId": None,
        "createdAt": created,
    }
    source.update(m["source"])
    return {"model": row, "sourceset": sourceset, "source": source}


def convert_layer(
    model: str, layer: int, out: str, vocab: list[str], created: str
) -> str:
    """Download one layer's records and write its export source directory."""
    root = os.path.join(out, model, SOURCE_SET)
    final = os.path.join(root, str(layer))
    if os.path.isdir(final):
        return f"layer {layer}: exists, skipped"
    work = final + ".partial"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(os.path.join(work, "features"))
    os.makedirs(os.path.join(work, "activations"))
    t0 = time.time()
    c = client()
    stem = f"{PREFIX}/{model}/neurons/layer_{layer:02d}"
    gz_path = os.path.join(work, "records.json.gz")
    c.download_file(BUCKET, f"{stem}.json.gz", gz_path)
    idx = np.frombuffer(
        c.get_object(Bucket=BUCKET, Key=f"{stem}.idx")["Body"].read(), dtype="<u8"
    )
    n = len(idx) - 1

    creator = load_meta()["models"][model]["model"]["creatorId"]
    for name, row in meta_rows(model, layer, created, creator).items():
        with open(os.path.join(work, f"{name}.jsonl"), "wb") as f:
            f.write(orjson.dumps(row) + b"\n")

    layer_id = str(layer)
    n_rows = 0
    with (
        open(gz_path, "rb") as f,
        mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data,
    ):
        for b, lo in enumerate(range(0, n, BATCH)):
            neurons, acts = [], []
            for i in range(lo, min(lo + BATCH, n)):
                rec = json.loads(
                    zlib.decompressobj(31).decompress(data[idx[i] : idx[i + 1]])
                )
                if (
                    rec["v"] != FORMAT_VERSION
                    or rec["layer"] != layer
                    or rec["neuron"] != i
                ):
                    raise ValueError(
                        f"{model} layer {layer}: record {i} is v{rec['v']} {rec['layer']}/{rec['neuron']}"
                    )
                neurons.append(neuron_row(rec, model, layer_id, created, creator))
                acts.extend(
                    activation_rows(rec, vocab, model, layer_id, created, creator)
                )
            write_gz(os.path.join(work, "features", f"batch-{b}.jsonl.gz"), neurons)
            write_gz(os.path.join(work, "activations", f"batch-{b}.jsonl.gz"), acts)
            n_rows += len(acts)
    os.remove(gz_path)
    os.rename(work, final)
    return f"layer {layer}: {n} neurons, {n_rows} activations, {time.time() - t0:.0f}s"


def main() -> None:
    """Convert one model's dashboards, one layer per worker."""
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--layers", default="", help="comma-separated; all layers if empty")
    ap.add_argument("--out", default="/data/exports")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    c = client()
    meta = json.loads(
        c.get_object(Bucket=BUCKET, Key=f"{PREFIX}/{args.model}/meta.json")[
            "Body"
        ].read()
    )
    vocab = json.loads(
        c.get_object(Bucket=BUCKET, Key=f"{PREFIX}/{args.model}/corpus/vocab.json")[
            "Body"
        ].read()
    )
    n_layers = meta["layers"]
    layers = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        else list(range(n_layers))
    )
    created = datetime.now(UTC).replace(tzinfo=None).isoformat()
    log(
        f"{args.model}: {len(layers)} of {n_layers} layers to {args.out}/{args.model}/{SOURCE_SET}"
    )
    with ProcessPoolExecutor(args.workers) as ex:
        futures = {
            ex.submit(convert_layer, args.model, li, args.out, vocab, created): li
            for li in layers
        }
        for fut in as_completed(futures):
            log(f"{args.model} {fut.result()}")
    log(f"{args.model}: DONE")


if __name__ == "__main__":
    main()
