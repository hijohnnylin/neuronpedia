"""Check a neuron act range table against random dashboard records: validate_act_range.py TABLE MODEL N_LAYERS D_MLP [N]."""

import gzip
import json
import random
import struct
import sys
from concurrent.futures import ThreadPoolExecutor

import requests
from safetensors.numpy import load_file

table_path, model, n_layers, d_mlp = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    int(sys.argv[4]),
)
n = int(sys.argv[5]) if len(sys.argv) > 5 else 60
DASH = f"https://dashboards.neuronpedia.org/v2/{model}/neurons"
t = load_file(table_path)
assert t["max"].shape == (n_layers, d_mlp), t["max"].shape
rng = random.Random(0)
picks = [(rng.randrange(n_layers), rng.randrange(d_mlp)) for _ in range(n)]


def record_stats(pick):
    """stats.max and stats.min of one dashboard record."""
    layer, idx = pick
    raw = requests.get(
        f"{DASH}/layer_{layer:02d}.idx",
        headers={"Range": f"bytes={8 * idx}-{8 * idx + 15}"},
        timeout=30,
    ).content
    s, e = struct.unpack("<QQ", raw)
    rec = json.loads(
        gzip.decompress(
            requests.get(
                f"{DASH}/layer_{layer:02d}.json.gz",
                headers={"Range": f"bytes={s}-{e - 1}"},
                timeout=30,
            ).content
        )
    )
    return rec["stats"]["max"], rec["stats"]["min"]


with ThreadPoolExecutor(16) as pool:
    stats = list(pool.map(record_stats, picks))
bad = []
for (layer, idx), (mx, mn) in zip(picks, stats, strict=True):
    tm, tn = float(t["max"][layer, idx]), float(t["min"][layer, idx])
    if abs(tm - mx) > 1e-3 * max(1.0, abs(mx)) or abs(tn - mn) > 1e-3 * max(
        1.0, abs(mn)
    ):
        bad.append((layer, idx, tm, mx, tn, mn))
print(f"{model}: {n - len(bad)}/{n} match the dashboard stats.max and stats.min")
ratios = sorted(
    float(t["max"][l, i]) / mx
    for (l, i), (mx, _) in zip(picks, stats, strict=True)
    if mx and mx > 0
)
q = lambda f: ratios[min(len(ratios) - 1, int(f * len(ratios)))]  # noqa: E731
print(
    f"  table max / record max: min {ratios[0]:.3f} p1 {q(0.01):.3f} p5 {q(0.05):.3f} median {q(0.5):.3f} max {ratios[-1]:.3f}"
)
for b in bad[:10]:
    print(
        "  mismatch L%d/%d table max %.4g record %.4g | table min %.4g record %.4g" % b
    )
