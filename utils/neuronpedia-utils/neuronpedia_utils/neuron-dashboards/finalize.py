"""Turn raw per-layer arrays into per-neuron JSON records and upload all outputs to R2.

Per layer, the records are separate gzip members written one after another into
neurons/layer_XX.json.gz. neurons/layer_XX.index.json holds N + 1 byte offsets, so neuron n is
bytes offsets[n] .. offsets[n+1]-1 (one HTTP Range request, then gunzip).

R2 layout under s3://neuronpedia/neuron-dashboards/v2/:
  corpus/docs.parquet, corpus/stats.json
  <model>/corpus/{tokens.npy, offsets.npy, vocab.json}
  <model>/meta.json, <model>/raw/layer_XX.npz, <model>/neurons/layer_XX.{json.gz,index.json}
"""

import argparse
import gzip
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context

import boto3
import numpy as np
import orjson
from boto3.s3.transfer import TransferConfig

PREFIX = "neuron-dashboards/v2"
BUCKET = "neuronpedia"
G = {}


def log(*a):
    """Print with a timestamp."""
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def r2():
    """S3 client for R2."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_S3_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


TCFG = TransferConfig(
    multipart_threshold=64 * 2**20, multipart_chunksize=64 * 2**20, max_concurrency=16
)


def upload(client, path, key):
    """Upload one file to BUCKET/PREFIX/key."""
    client.upload_file(path, BUCKET, f"{PREFIX}/{key}", Config=TCFG)


def sig(x):
    """Round each value to 3 significant digits (NaN becomes None) and return a list."""
    return [
        float(f"{v:.3g}") if v == v else None for v in x.astype(np.float64).tolist()
    ]


def r3(v):
    """Round a scalar to about 4 significant digits."""
    v = float(v)
    if v == 0 or not np.isfinite(v):
        return v if np.isfinite(v) else None
    return round(v, int(max(0, 3 - np.floor(np.log10(abs(v))))))


def examples(prefix, n, k_axis=None):
    """Build example dicts for neuron n from list arrays named prefix_*."""
    a = G["arr"]
    toks, offs = G["toks"], G["offs"]
    lo = G["win_lo"]
    if k_axis is None:
        get = lambda key: a[f"{prefix}_{key}"][n]  # noqa: E731
    else:
        get = lambda key: a[f"{prefix}_{key}"][n, k_axis]  # noqa: E731
    doc, pos, act, pg, pu, win, gb = (
        get("doc"),
        get("pos"),
        get("act"),
        get("pg"),
        get("pu"),
        get("win"),
        get("gbits"),
    )
    out = []
    for j in range(len(doc)):
        d = int(doc[j])
        if d < 0:
            continue
        p = int(pos[j])
        dlen = int(offs[d + 1] - offs[d])
        s = max(0, p + lo)
        e = min(dlen, p + lo + len(win[j]))
        i0, i1 = s - (p + lo), e - (p + lo)
        bits = int(gb[j]) & ((1 << 64) - 1)
        closed = "".join("1" if (bits >> i) & 1 else "0" for i in range(i0, i1))
        out.append(
            {
                "doc": d,
                "pos": p,
                "act": r3(act[j]),
                "gate": r3(pg[j]),
                "up": r3(pu[j]),
                "start": s,
                "tokens": toks[offs[d] + s : offs[d] + e].tolist(),
                "acts": sig(win[j][i0:i1]),
                "gate_closed": closed,
            }
        )
    return out


def record(n):
    """Full JSON record for neuron n of the current layer."""
    a, meta = G["arr"], G["meta"]
    cnt = float(a["stat_count"])
    mean = a["stat_sum"][n] / cnt
    var = max(a["stat_sumsq"][n] / cnt - mean * mean, 0.0)
    ab_o, ab_c = a["stat_abs_open"][n], a["stat_abs_closed"][n]
    hb = meta["hist_bins"]["per_sign"]
    h = a["hist"][n]
    vocab = G["vocab"]
    bands = []
    for i, (ql, qh) in enumerate(meta["bands"]):
        bands.append(
            {
                "q_lo": ql,
                "q_hi": qh,
                "lo": r3(a["band_edges"][n, i]),
                "hi": r3(a["band_edges"][n, len(meta["bands"]) + i]),
                "examples": examples("bands", n, i),
            }
        )
    rec = {
        "v": 1,
        "model": meta["model"],
        "layer": G["layer"],
        "neuron": n,
        "stats": {
            "max": r3(max(a["top_act"][n, 0], a["etop_act"][n, 0])),
            "min": r3(min(a["bot_act"][n, 0], a["ebot_act"][n, 0])),
            "mean": r3(mean),
            "std": r3(np.sqrt(var)),
            "sigma_calib": r3(a["sigma"][n]),
            "frac_gate_open": r3(a["stat_n_open"][n] / cnt),
            "echo_abs_share": r3(ab_c / max(ab_o + ab_c, 1e-30)),
            "n_hist_tokens": int(cnt),
        },
        "quantiles": {
            "q": meta["report_quantiles"],
            "v": [r3(x) for x in a["quantiles_main"][n]],
        },
        "hist": {
            "sigma": r3(a["sigma"][n]),
            "open_pos": h[0:hb].tolist(),
            "open_neg": h[hb : 2 * hb].tolist(),
            "closed_pos": h[2 * hb : 3 * hb].tolist(),
            "closed_neg": h[3 * hb : 4 * hb].tolist(),
        },
        "top": examples("top", n),
        "bottom": examples("bot", n),
        "echo_top": examples("etop", n),
        "echo_bottom": examples("ebot", n),
        "bands": bands,
        "logits": {
            "pos": [
                [int(t), vocab[t], r3(v)]
                for t, v in zip(a["logit_pos_ids"][n], a["logit_pos_vals"][n])
            ],
            "neg": [
                [int(t), vocab[t], r3(v)]
                for t, v in zip(a["logit_neg_ids"][n], a["logit_neg_vals"][n])
            ],
        },
    }
    return rec


def work(rng):
    """Gzip the records for neurons in rng; return the list of members."""
    return [
        gzip.compress(orjson.dumps(record(n), option=orjson.OPT_SERIALIZE_NUMPY), 6)
        for n in range(*rng)
    ]


FORMAT_DOC = {
    "neuron": "act = act_fn(gate) * up, the input to down_proj. gate_open means gate > 0.",
    "lists": {
        "top": "gate-open examples, largest act, one per document",
        "bottom": "gate-open examples, most negative act",
        "echo_top": "gate-closed examples (echoes), largest act",
        "echo_bottom": "gate-closed examples, most negative act",
        "bands": "random examples with act in [lo, hi), band edges are calibration quantiles q_lo, q_hi",
    },
    "example": {
        "doc": "row in corpus/docs.parquet",
        "pos": "peak token position in the document (model tokens)",
        "act/gate/up": "values at the peak token",
        "start": "document position of tokens[0] and acts[0]",
        "tokens": "model token ids (see <model>/corpus/vocab.json)",
        "acts": "activation at each window token",
        "gate_closed": "one char per window token, '1' means gate <= 0",
    },
    "hist": "counts on every hist_stride-th token. Bin 0: |act|/sigma < 1e-3; bins 1..64 log-spaced to 1e3; bin 65 above",
    "logits": "[token id, token, value] for W_U (final_norm * post_ffn_norm * down_proj[:, n]); norms as scales only",
    "excluded": "position 0 of each document is never counted",
}


def main():
    """Finalize one model's run directory, or upload the corpus."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="dash.py output directory")
    ap.add_argument("--corpus", default="/workspace/corpus")
    ap.add_argument("--model")
    ap.add_argument("--layers", default="")
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--upload-corpus", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    client = None if args.no_upload else r2()

    if args.upload_corpus:
        for f in ("docs.parquet", "stats.json"):
            upload(client, f"{args.corpus}/{f}", f"corpus/{f}")
        log("corpus uploaded")
        if not args.run:
            return

    meta = json.load(open(f"{args.run}/meta.json"))
    model = meta["model"]
    meta["win_lo"] = meta["window"][0]
    meta["format"] = FORMAT_DOC
    os.makedirs(f"{args.run}/neurons", exist_ok=True)
    G.update(
        meta=meta,
        win_lo=meta["window"][0],
        toks=np.load(f"{args.corpus}/{model}/tokens.npy", mmap_mode="r"),
        offs=np.load(f"{args.corpus}/{model}/offsets.npy"),
        vocab=json.load(open(f"{args.corpus}/{model}/vocab.json")),
    )
    pool_up = ThreadPoolExecutor(4)
    futs = []
    if client:
        json.dump(meta, open(f"{args.run}/meta_full.json", "w"), indent=1)
        futs.append(
            pool_up.submit(
                upload, client, f"{args.run}/meta_full.json", f"{model}/meta.json"
            )
        )
        for f in ("tokens.npy", "offsets.npy", "vocab.json"):
            futs.append(
                pool_up.submit(
                    upload, client, f"{args.corpus}/{model}/{f}", f"{model}/corpus/{f}"
                )
            )

    layers = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        else list(range(meta["layers"]))
    )
    ctx = get_context("fork")
    t0 = time.time()
    for li in layers:
        raw = f"{args.run}/raw/layer_{li:02d}.npz"
        with np.load(raw) as z:
            G["arr"] = {k: z[k] for k in z.files}
        G["layer"] = li
        N = meta["neurons_per_layer"]
        step = max(1, N // (args.workers * 4))
        ranges = [(i, min(N, i + step)) for i in range(0, N, step)]
        with ctx.Pool(args.workers) as pool:
            parts = pool.map(work, ranges)
        offsets = [0]
        shard = f"{args.run}/neurons/layer_{li:02d}.json.gz"
        with open(shard, "wb") as f:
            for part in parts:
                for m in part:
                    f.write(m)
                    offsets.append(offsets[-1] + len(m))
        idx = f"{args.run}/neurons/layer_{li:02d}.index.json"
        json.dump({"layer": li, "offsets": offsets}, open(idx, "w"))
        log(
            f"layer {li}: {offsets[-1] / 2**20:.0f} MiB, {(time.time() - t0) / 60:.1f} min"
        )
        if client:
            futs.append(
                pool_up.submit(
                    upload, client, shard, f"{model}/neurons/layer_{li:02d}.json.gz"
                )
            )
            futs.append(
                pool_up.submit(
                    upload, client, idx, f"{model}/neurons/layer_{li:02d}.index.json"
                )
            )
            futs.append(
                pool_up.submit(upload, client, raw, f"{model}/raw/layer_{li:02d}.npz")
            )
    for fu in futs:
        fu.result()
    log("DONE")


if __name__ == "__main__":
    main()
