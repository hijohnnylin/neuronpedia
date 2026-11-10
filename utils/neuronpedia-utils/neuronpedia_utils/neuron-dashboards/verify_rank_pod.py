"""Check a pod's rankByPercentile order against rarity recomputed from the dashboard records.

Usage: verify_rank_pod.py POD_URL MODEL LAYERS PROMPT SORT_INDEXES HF_TOKENIZER [N]
"""

import gzip
import json
import math
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor

import requests
from transformers import AutoTokenizer

pod, model, layers_arg, prompt, sort_arg, tok_id = sys.argv[1:7]
N = int(sys.argv[7]) if len(sys.argv) > 7 else 15
SORT = [int(x) for x in sort_arg.split(",")] if sort_arg else []
TOK = AutoTokenizer.from_pretrained(tok_id)
DASH = f"https://dashboards.neuronpedia.org/v2/{model}"
N_TOKENS = requests.get(f"{DASH}/meta.json", timeout=30).json()["n_tokens"]

r = requests.post(
    f"{pod}/v1/activation/all",
    headers={"x-secret-key": os.environ["INFERENCE_SECRET"]},
    timeout=300,
    json={
        "prompt": prompt,
        "model": model,
        "sourceSet": "neurons",
        "selectedSources": layers_arg.split(","),
        "sortByTokenIndexes": SORT,
        "ignoreBos": True,
        "numResults": N,
        "rankByPercentile": True,
    },
)
assert r.status_code == 200, (r.status_code, r.text[:300])
res = r.json()
print(
    "tokens:",
    res["tokens"],
    "| sorting on",
    [res["tokens"][i] for i in SORT] or "max over tokens",
)


def record(f):
    """The dashboard record of one returned neuron."""
    layer, n = int(f["source"]), f["index"]
    s, e = struct.unpack(
        "<QQ",
        requests.get(
            f"{DASH}/neurons/layer_{layer:02d}.idx",
            headers={"Range": f"bytes={8 * n}-{8 * n + 15}"},
            timeout=30,
        ).content,
    )
    return json.loads(
        gzip.decompress(
            requests.get(
                f"{DASH}/neurons/layer_{layer:02d}.json.gz",
                headers={"Range": f"bytes={s}-{e - 1}"},
                timeout=30,
            ).content
        )
    )


def tail(knots, rs, a):
    """Interpolate rarity along one tail's knots (without the layer floor)."""
    knots = list(knots)
    for i in range(1, len(knots)):
        knots[i] = max(knots[i], knots[i - 1])
    if a <= knots[0]:
        return 0.0
    if a > knots[-1]:
        return rs[-1] + math.log10(a / knots[-1])
    for i in range(1, len(knots)):
        if a <= knots[i]:
            span = knots[i] - knots[i - 1]
            return rs[i - 1] + ((a - knots[i - 1]) / span if span > 0 else 1.0) * (
                rs[i] - rs[i - 1]
            )
    return rs[-1]


def rarity(rec, highest, lowest):
    """Two-sided rarity and whether the low side wins, as the server computes it."""
    qs = list(zip(rec["quantiles"]["q"], rec["quantiles"]["v"], strict=True))
    top = math.log10(N_TOKENS)
    high = tail(
        [v for p, v in qs if p >= 0.5] + [rec["stats"]["max"]],
        [-math.log10(1 - p) for p, _ in qs if p >= 0.5] + [top],
        highest,
    )
    low = tail(
        [-v for p, v in reversed(qs) if p <= 0.5] + [-rec["stats"]["min"]],
        [-math.log10(p) for p, _ in reversed(qs) if p <= 0.5] + [top],
        -lowest,
    )
    return max(high, low), low > high


with ThreadPoolExecutor(16) as pool:
    recs = list(pool.map(record, res["activations"]))
prev, bad, side_bad = math.inf, 0, 0
for f, rec in zip(res["activations"], recs, strict=True):
    vals = [f["values"][i] for i in SORT] if SORT else f["values"]
    rr, low = rarity(rec, max(vals), min(vals))
    peak = min(vals) if low else max(vals)
    side_flag = "" if low == bool(f.get("rankedByLowest")) else "  <-- SIDE DIFFERS"
    side_bad += bool(side_flag)
    # Context of the dashboard example on the side that matched.
    ex = rec["bottom"][0] if low else rec["top"][0]
    sign = -1 if low else 1
    i = max(
        range(len(ex["acts"])),
        key=lambda j: sign * ex["acts"][j] if ex["acts"][j] is not None else -1e9,
    )
    ctx = (
        TOK.decode(ex["tokens"][max(0, i - 6) : i])
        + f"[{TOK.decode([ex['tokens'][i]])}]"
    ).replace("\n", " ")[-45:]
    flag = "" if rr <= prev + 1e-3 else "  <-- OUT OF ORDER"
    bad += bool(flag)
    prev = min(prev, rr)
    extreme = rec["stats"]["min"] if low else rec["stats"]["max"]
    print(
        f"  L{int(f['source']):02d}/{f['index']:<6} {'LOW ' if low else 'high'} rarity {rr:4.2f}  value {peak:8.3f}/{extreme:<9.3f} {ctx}{flag}{side_flag}"
    )
print("ORDER OK" if bad == 0 else f"{bad} rows out of order")
print("SIDES OK" if side_bad == 0 else f"{side_bad} rows with a different side")
