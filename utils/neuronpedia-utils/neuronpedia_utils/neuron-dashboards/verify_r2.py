"""Check that every output file of one model is in R2 with the same size as on disk."""

import json
import os
import sys

from finalize import BUCKET, PREFIX, r2

model, run, corpus = sys.argv[1], sys.argv[2], sys.argv[3]
L = json.load(open(f"{run}/meta.json"))["layers"]
want = {
    f"{model}/corpus/{f}": f"{corpus}/{model}/{f}"
    for f in ("tokens.npy", "offsets.npy", "vocab.json")
}
want[f"{model}/meta.json"] = f"{run}/meta_full.json"
for li in range(L):
    want[f"{model}/raw/layer_{li:02d}.npz"] = f"{run}/raw/layer_{li:02d}.npz"
    want[f"{model}/neurons/layer_{li:02d}.json.gz"] = (
        f"{run}/neurons/layer_{li:02d}.json.gz"
    )
    want[f"{model}/neurons/layer_{li:02d}.index.json"] = (
        f"{run}/neurons/layer_{li:02d}.index.json"
    )

c = r2()
have = {}
for page in c.get_paginator("list_objects_v2").paginate(
    Bucket=BUCKET, Prefix=f"{PREFIX}/{model}/"
):
    for o in page.get("Contents", []):
        have[o["Key"][len(PREFIX) + 1 :]] = o["Size"]
bad = [k for k, p in want.items() if have.get(k) != os.path.getsize(p)]
total = sum(have.values())
print(
    f"{model}: {len(want) - len(bad)}/{len(want)} files match, {total / 2**30:.1f} GiB in R2"
)
for k in bad[:20]:
    print("MISSING OR WRONG SIZE:", k)
sys.exit(1 if bad else 0)
