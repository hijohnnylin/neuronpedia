"""Per-neuron corpus activation distribution, for ranking search-all results by percentile.

Raw MLP neuron activations are not comparable across neurons: scales differ by orders of
magnitude, and a dense neuron reaches half its max on ordinary text while a sparse one almost
never fires. With `rankByPercentile`, search-all ranks each neuron by how rare its activation is
for that neuron on the dashboard corpus. Both tails count, because a gated MLP neuron can be
strongly negative. High side: rarity = -log10(1 - p), so the 99th percentile is 2, the 99.999th
is 5 and the corpus max is log10(n_tokens). Low side: rarity = -log10(p), so the 1st percentile
is 2 and the corpus min is log10(n_tokens).

The table is a `.safetensors` file loaded once from `NEURON_ACT_RANGE_PATH`:

  max              [n_layers, d_mlp]     largest activation on the corpus (dashboard `stats.max`)
  min              [n_layers, d_mlp]     smallest activation on the corpus (dashboard `stats.min`)
  quantile_levels  [Q]                   ascending fractions, 0-1 (dashboard `quantiles.q`)
  quantiles        [n_layers, d_mlp, Q]  activation at each level (dashboard `quantiles.v`)
  n_tokens         [1]                   tokens in the corpus
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch

# A neuron whose corpus extreme is near zero would turn any small activation into a rare one. Its
# knots are raised to this share of its layer's median extreme on that side.
MIN_SCALE_FRACTION_OF_LAYER_MEDIAN = 0.05

NO_TABLE_MESSAGE = (
    "rankByPercentile needs a neuron activation range table, and this server was started without NEURON_ACT_RANGE_PATH."
)

_REQUIRED_KEYS = ("max", "min", "quantile_levels", "quantiles", "n_tokens")


class NeuronActRangeError(ValueError):
    """The table is missing or does not fit the request. The message is safe to return as a 400."""


@dataclass(frozen=True)
class TailCurve:
    """Piecewise-linear map from distance into one tail to rarity, per neuron.

    Knot k of neuron n in layer l is at distance ``knots[l, n, k]`` and rarity ``rarity[k]``.
    The distance is the activation on the high side and minus the activation on the low side.
    Knots run from the median to the corpus extreme and never decrease along k.
    """

    knots: torch.Tensor  # [n_layers, d_mlp, K] float32
    rarity: torch.Tensor  # [K] float32


@dataclass(frozen=True)
class NeuronRarityTable:
    """The high and the low tail of each neuron's corpus distribution."""

    high: TailCurve
    low: TailCurve


def _tail_curve(levels: torch.Tensor, quantiles: torch.Tensor, extreme: torch.Tensor, n_tokens: int) -> TailCurve:
    """One tail's curve. ``levels`` run from the median out, ``extreme`` is the corpus max (or -min)."""
    tail_p = torch.minimum(levels, 1.0 - levels)
    knot_rarity = [-math.log10(float(p)) for p in tail_p] + [math.log10(n_tokens)]
    knots = torch.cat((quantiles, extreme.unsqueeze(-1)), dim=-1)
    knots = torch.cummax(knots, dim=-1).values
    for li in range(knots.shape[0]):
        positive = extreme[li][extreme[li] > 0]
        median = float(positive.median()) if positive.numel() > 0 else 1.0
        knots[li] = knots[li].clamp(min=median * MIN_SCALE_FRACTION_OF_LAYER_MEDIAN)
    return TailCurve(knots=knots.contiguous(), rarity=torch.tensor(knot_rarity, dtype=torch.float32))


@lru_cache(maxsize=2)
def load_neuron_rarity_table(path: str) -> NeuronRarityTable:
    """Read the table and build both tail curves, on the CPU."""
    from safetensors import SafetensorError
    from safetensors.torch import load_file

    try:
        tensors = load_file(path)
    except (OSError, SafetensorError) as e:
        raise NeuronActRangeError(f"Cannot read the neuron activation range table {path}: {e}") from e
    missing = [k for k in _REQUIRED_KEYS if k not in tensors]
    if missing:
        raise NeuronActRangeError(f"Neuron activation range table {path} has no {missing}")
    table_max = tensors["max"].float()
    table_min = tensors["min"].float()
    quantiles = tensors["quantiles"].float()
    levels = tensors["quantile_levels"].double()
    if (
        table_max.ndim != 2
        or table_min.shape != table_max.shape
        or quantiles.shape != (*table_max.shape, levels.shape[0])
    ):
        raise NeuronActRangeError(
            f"Neuron activation range table {path}: max is {tuple(table_max.shape)}, min "
            f"{tuple(table_min.shape)} and quantiles {tuple(quantiles.shape)}, expected [n_layers, d_mlp] "
            f"twice and [n_layers, d_mlp, {levels.shape[0]}]"
        )
    n_tokens = int(tensors["n_tokens"].reshape(-1)[0])
    upper = levels >= 0.5
    lower = torch.flip(torch.nonzero(levels <= 0.5).squeeze(1), dims=(0,))
    return NeuronRarityTable(
        high=_tail_curve(levels[upper], quantiles[:, :, upper], table_max, n_tokens),
        low=_tail_curve(levels[lower], -quantiles[:, :, lower], -table_min, n_tokens),
    )


def _tail_rarity(curve: TailCurve, layer: int, distance: torch.Tensor) -> torch.Tensor:
    """Rarity at ``distance`` into one tail, per neuron of a layer.

    0 at or below the first knot, log10(n_tokens) at the corpus extreme, and log10(d / extreme)
    more beyond it.
    """
    knots = curve.knots[layer].to(distance.device)
    rarity = curve.rarity.to(distance.device)
    d = distance.to(torch.float32)
    n_knots = knots.shape[1]
    hi = torch.searchsorted(knots, d.unsqueeze(1)).squeeze(1).clamp(max=n_knots - 1)
    lo = (hi - 1).clamp(min=0)
    k_lo = knots.gather(1, lo.unsqueeze(1)).squeeze(1)
    k_hi = knots.gather(1, hi.unsqueeze(1)).squeeze(1)
    span = k_hi - k_lo
    t = torch.where(span > 0, (d - k_lo) / span.clamp(min=1e-30), torch.ones_like(d)).clamp(0.0, 1.0)
    out = rarity[lo] + t * (rarity[hi] - rarity[lo])
    out = torch.where(d <= knots[:, 0], torch.zeros_like(d), out)
    top = knots[:, -1]
    return torch.where(d > top, rarity[-1] + torch.log10(d / top), out)


def neuron_rarity(
    path: str | None, layer: int, highest: torch.Tensor, lowest: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rarity of each neuron of a layer, and whether its low side is the rarer one.

    ``highest`` and ``lowest`` are each neuron's highest and lowest activation, ``[d_mlp]``. The
    rarity is the larger of the two tails' rarities, float32 on their device. Raises
    NeuronActRangeError when it cannot be served.
    """
    if not path:
        raise NeuronActRangeError(NO_TABLE_MESSAGE)
    table = load_neuron_rarity_table(path)
    n_layers, d_mlp = table.high.knots.shape[:2]
    if not 0 <= layer < n_layers:
        raise NeuronActRangeError(
            f"The neuron activation range table has {n_layers} layers; layer {layer} is out of range"
        )
    if d_mlp != highest.shape[0] or d_mlp != lowest.shape[0]:
        raise NeuronActRangeError(
            f"The neuron activation range table has {d_mlp} neurons per layer; this model has {highest.shape[0]}"
        )
    high = _tail_rarity(table.high, layer, highest)
    low = _tail_rarity(table.low, layer, -lowest)
    low_is_rarer = low > high
    return torch.where(low_is_rarer, low, high), low_is_rarer
