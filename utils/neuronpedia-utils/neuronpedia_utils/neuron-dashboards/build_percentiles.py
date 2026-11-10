"""Build the per-neuron |activation| percentile table for neuron ablation.

Reads the dashboard's raw/layer_XX.npz files and writes one .safetensors file that the
inference server loads with NEURON_PERCENTILES_PATH:

  levels         [Q]          percent, ascending, 0-100
  abs_quantiles  [L, N, Q]    |act| at each level, float16

The source is the dashboard histogram: counts of |act|/sigma in 66 log bins, per sign and
gate state, on every hist_stride-th token. Sign and gate state are added together here.
In a bin, the value is interpolated linearly. The top bin ends at the largest |act| seen
on any token (from the top and bottom example lists), and every level is capped there.

Usage:
  python build_percentiles.py RAW_DIR OUT.safetensors
  python build_percentiles.py --self-test
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

HB = 66  # magnitude bins per (sign, gate) group; must match dash.py
LEVELS = np.array(
    list(range(0, 21))  # every 1% in the low range, where ablation sweeps run
    + list(range(25, 96, 5))
    + [96, 97, 98, 99, 99.5, 99.9, 100],
    dtype=np.float64,
)


def mag_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper |act|/sigma bound of each magnitude bin, as dash.py bins them."""
    e = 10.0 ** (-3.0 + np.arange(65) * 6 / 64)
    lo = np.concatenate([[0.0], e])
    hi = np.concatenate([[1e-3], e[1:], [np.inf]])
    return lo, hi


def abs_quantiles(
    hist: np.ndarray, sigma: np.ndarray, max_abs: np.ndarray, levels: np.ndarray
) -> np.ndarray:
    """|act| at each level per neuron. hist [N, 4*HB]; sigma, max_abs [N]. Returns [N, Q] float64."""
    h = hist.astype(np.float64)
    counts = h[:, :HB] + h[:, HB : 2 * HB] + h[:, 2 * HB : 3 * HB] + h[:, 3 * HB :]
    s = sigma.astype(np.float64)
    top = np.where(s > 0, max_abs / np.where(s > 0, s, 1), 0.0)
    lo, hi = mag_bounds()
    lo_full = np.broadcast_to(lo, counts.shape)
    hi_full = np.broadcast_to(hi, counts.shape).copy()
    hi_full[:, -1] = np.maximum(top, 1e3)

    cdf = counts.cumsum(1)
    tot = cdf[:, -1]
    rows = np.arange(len(counts))
    out = np.empty((len(counts), len(levels)))
    for qi, level in enumerate(levels):
        tgt = level / 100.0 * tot
        j = np.minimum((cdf < tgt[:, None]).sum(1), HB - 1)
        prev = np.where(j > 0, cdf[rows, np.maximum(j - 1, 0)], 0.0)
        cnt = np.maximum(counts[rows, j], 1.0)
        f = np.clip((tgt - prev) / cnt, 0.0, 1.0)
        rel = lo_full[rows, j] + f * (hi_full[rows, j] - lo_full[rows, j])
        out[:, qi] = np.minimum(rel, top) * s
    out[tot == 0] = 0.0
    return out


def max_abs_of(npz) -> np.ndarray:
    """Largest |act| per neuron over all tokens, from the four example lists."""
    cols = [np.abs(npz[f"{name}_act"][:, 0]) for name in ("top", "bot", "etop", "ebot")]
    return np.max(np.stack(cols, 1), 1)


def build(raw_dir: str, out_path: str) -> None:
    """Write the table for every raw/layer_XX.npz in raw_dir."""
    from safetensors.numpy import save_file

    paths = sorted(glob.glob(os.path.join(raw_dir, "layer_*.npz")))
    if not paths:
        raise SystemExit(f"no layer_*.npz in {raw_dir}")
    layers = []
    for i, p in enumerate(paths):
        assert p.endswith(f"layer_{i:02d}.npz"), (
            f"layer files are not contiguous at {p}"
        )
        with np.load(p) as z:
            q = abs_quantiles(z["hist"], z["sigma"], max_abs_of(z), LEVELS)
        layers.append(np.minimum(q, np.finfo(np.float16).max).astype(np.float16))
        p10, p50 = (int(np.searchsorted(LEVELS, v)) for v in (10, 50))
        print(
            f"layer {i:02d}: median p10 {np.median(q[:, p10]):.4g}  median p50 {np.median(q[:, p50]):.4g}",
            flush=True,
        )
    table = np.stack(layers)
    save_file({"levels": LEVELS.astype(np.float32), "abs_quantiles": table}, out_path)
    print(
        f"wrote {out_path}: abs_quantiles {table.shape} float16, {os.path.getsize(out_path) / 1e6:.0f} MB"
    )


def self_test() -> None:
    """Histogram synthetic activations the way dash.py does and compare with exact quantiles."""
    import dash
    import torch

    rng = np.random.default_rng(0)
    n_tok, n = 200_000, 6
    scales = np.array([0.01, 0.1, 1.0, 3.0, 0.5, 2.0])
    a = rng.standard_t(3, size=(n_tok, n)) * scales
    a[:, 4] *= rng.random(n_tok) < 0.1  # mostly-zero neuron
    g = rng.standard_normal((n_tok, n))
    sigma = np.sqrt((a**2).mean(0))
    k = dash.hist_index(
        torch.tensor(a, dtype=torch.float32),
        torch.tensor(g),
        torch.tensor(sigma, dtype=torch.float32),
    )
    hist = np.zeros((n, 4 * HB), dtype=np.int64)
    for col in range(n):
        hist[col] = np.bincount(k[:, col].numpy(), minlength=4 * HB)
    got = abs_quantiles(hist, sigma, np.abs(a).max(0), LEVELS)
    want = np.quantile(np.abs(a), LEVELS / 100.0, axis=0).T
    # Log bins are 24% wide, so allow 12% relative error, or a tiny absolute one near zero.
    ok = np.isclose(got, want, rtol=0.12, atol=1e-3 * sigma[:, None])
    assert ok.all(), (
        f"{(~ok).sum()} cells off; worst {np.max(np.abs(got - want) / np.maximum(want, 1e-9)):.3f}"
    )
    assert np.allclose(got[:, -1], np.abs(a).max(0)) and (got[:, 0] == 0).all()
    print("self-test passed:", got.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        if not (args.raw_dir and args.out):
            ap.error("RAW_DIR and OUT are required")
        build(args.raw_dir, args.out)
