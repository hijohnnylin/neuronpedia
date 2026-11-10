"""Check one dash.py run: known neurons and internal consistency."""

import json
import sys

import numpy as np

run, corpus, model = sys.argv[1], sys.argv[2], sys.argv[3]
checks = [tuple(map(int, x.split("/"))) for x in sys.argv[4:]]
toks = np.load(f"{corpus}/{model}/tokens.npy", mmap_mode="r")
offs = np.load(f"{corpus}/{model}/offsets.npy")
vocab = json.load(open(f"{corpus}/{model}/vocab.json"))
meta = json.load(open(f"{run}/meta.json"))
lo = meta["window"][0]


def ctx(doc, pos, before=12, after=3):
    """Text around a position with the peak token in brackets."""
    s, e = offs[doc], offs[doc + 1]
    ids = toks[s:e]
    left = "".join(vocab[t] for t in ids[max(0, pos - before) : pos])
    right = "".join(vocab[t] for t in ids[pos + 1 : pos + 1 + after])
    return (left + "[[" + vocab[ids[pos]] + "]]" + right).replace("\n", "\\n")


for layer, n in checks:
    z = np.load(f"{run}/raw/layer_{layer:02d}.npz")
    print(
        f"\n===== L{layer}/N{n}  sigma {z['sigma'][n]:.3g}  frac_open {z['stat_n_open'][n] / z['stat_count']:.2f}"
    )
    print(
        "quantiles_main",
        dict(zip(meta["report_quantiles"], np.round(z["quantiles_main"][n], 3))),
    )
    for name in ("top", "bot", "etop", "ebot"):
        print(f"-- {name}")
        for j in range(min(6, z[f"{name}_doc"].shape[1])):
            d = z[f"{name}_doc"][n, j]
            if d < 0:
                continue
            p = z[f"{name}_pos"][n, j]
            print(
                f"  a={z[f'{name}_act'][n, j]:8.3f} g={z[f'{name}_pg'][n, j]:7.2f} u={z[f'{name}_pu'][n, j]:7.2f}  {ctx(d, p)}"
            )
    for b in range(len(meta["bands"])):
        acts = z["bands_act"][n, b][z["bands_doc"][n, b] >= 0]
        print(
            f"  band {meta['bands'][b]} edges {z['band_edges'][n, b]:.3g}..{z['band_edges'][n, b + 7]:.3g} acts {np.round(acts, 3)}"
        )
    print("  logits+", [vocab[t] for t in z["logit_pos_ids"][n][:8]])
    print("  logits-", [vocab[t] for t in z["logit_neg_ids"][n][:8]])

# consistency over one layer
z = np.load(f"{run}/raw/layer_{checks[0][0]:02d}.npz")
c = -lo
for name, closed in (("top", 0), ("bot", 0), ("etop", 1), ("ebot", 1)):
    ok = z[f"{name}_doc"] >= 0
    bit = (z[f"{name}_gbits"] >> c) & 1
    pg = z[f"{name}_pg"]
    peak_win = z[f"{name}_win"][:, :, c].astype(np.float32)
    print(
        f"{name}: filled {ok.mean():.3f}  bit==closed {(bit[ok] == closed).mean():.4f}  "
        f"gate sign ok {((pg[ok] <= 0) == bool(closed)).mean():.4f}  "
        f"win[peak]==act {np.isclose(peak_win[ok], z[f'{name}_act'][ok], rtol=2e-3, atol=1e-3).mean():.4f}"
    )
s = z["top_act"]
print(
    "top sorted desc:",
    bool((np.diff(s, axis=1)[z["top_doc"][:, 1:] >= 0] <= 1e-6).all()),
)
print("hist total == count:", np.unique(z["hist"].sum(1)), z["stat_count"])
print("band fill:", np.round((z["bands_doc"] >= 0).mean(axis=(0, 2)), 3))
inb = (z["bands_act"] >= z["band_edges"][:, :7, None] - 1e-2) & (
    z["bands_act"] < z["band_edges"][:, 7:, None] + 1e-2
)
print("band acts inside edges:", np.round(inb[z["bands_doc"] >= 0].mean(), 4))
