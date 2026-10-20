"""Per-neuron corpus activation distribution, for ranking search-all results by percentile.

Raw MLP neuron activations are not comparable across neurons: scales differ by orders of
magnitude, and a dense neuron reaches half its max on ordinary text while a sparse one almost
never fires. With `rankByPercentile`, search-all ranks each neuron by how rare its activation is
for that neuron on the dashboard corpus: rarity = -log10(1 - p), so the 99th percentile is 2,
the 99.999th is 5 and the corpus max is log10(n_tokens).

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

# A neuron whose corpus max is near zero would turn any small activation into a rare one. Its
# knots are raised to this share of its layer's median positive max.
MIN_SCALE_FRACTION_OF_LAYER_MEDIAN = 0.05

NO_TABLE_MESSAGE = (
    "rankByPercentile needs a neuron activation range table, and this server was started without NEURON_ACT_RANGE_PATH."
)

_REQUIRED_KEYS = ("max", "quantile_levels", "quantiles", "n_tokens")


class NeuronActRangeError(ValueError):
    """The table is missing or does not fit the request. The message is safe to return as a 400."""


@dataclass(frozen=True)
class NeuronRarityTable:
    """Piecewise-linear map from activation to rarity, per neuron.

    Knot k of neuron n in layer l is at activation ``knots[l, n, k]`` and rarity ``rarity[k]``.
    Knots run from the median to the corpus max and never decrease along k.
    """

    knots: torch.Tensor  # [n_layers, d_mlp, K] float32
    rarity: torch.Tensor  # [K] float32


@lru_cache(maxsize=2)
def load_neuron_rarity_table(path: str) -> NeuronRarityTable:
    """Read the table and build its knots, on the CPU."""
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
    quantiles = tensors["quantiles"].float()
    levels = tensors["quantile_levels"].double()
    if table_max.ndim != 2 or quantiles.shape != (*table_max.shape, levels.shape[0]):
        raise NeuronActRangeError(
            f"Neuron activation range table {path}: max is {tuple(table_max.shape)} and quantiles "
            f"{tuple(quantiles.shape)}, expected [n_layers, d_mlp] and [n_layers, d_mlp, {levels.shape[0]}]"
        )
    n_tokens = int(tensors["n_tokens"].reshape(-1)[0])
    upper = levels >= 0.5
    knot_rarity = [-math.log10(1.0 - float(p)) for p in levels[upper]] + [math.log10(n_tokens)]
    knots = torch.cat((quantiles[:, :, upper], table_max.unsqueeze(-1)), dim=-1)
    knots = torch.cummax(knots, dim=-1).values
    for li in range(knots.shape[0]):
        positive = table_max[li][table_max[li] > 0]
        median = float(positive.median()) if positive.numel() > 0 else 1.0
        knots[li] = knots[li].clamp(min=median * MIN_SCALE_FRACTION_OF_LAYER_MEDIAN)
    return NeuronRarityTable(knots=knots.contiguous(), rarity=torch.tensor(knot_rarity, dtype=torch.float32))


def neuron_rarity(path: str | None, layer: int, activations: torch.Tensor) -> torch.Tensor:
    """Rarity of one activation per neuron of a layer, ``[d_mlp]`` float32 on its device.

    0 at or below the median, log10(n_tokens) at the corpus max, and log10(a / max) more above
    it. Raises NeuronActRangeError when it cannot be served.
    """
    if not path:
        raise NeuronActRangeError(NO_TABLE_MESSAGE)
    table = load_neuron_rarity_table(path)
    if not 0 <= layer < table.knots.shape[0]:
        raise NeuronActRangeError(
            f"The neuron activation range table has {table.knots.shape[0]} layers; layer {layer} is out of range"
        )
    if table.knots.shape[1] != activations.shape[0]:
        raise NeuronActRangeError(
            f"The neuron activation range table has {table.knots.shape[1]} neurons per layer; "
            f"this model has {activations.shape[0]}"
        )
    knots = table.knots[layer].to(activations.device)
    rarity = table.rarity.to(activations.device)
    a = activations.to(torch.float32)
    n_knots = knots.shape[1]
    hi = torch.searchsorted(knots, a.unsqueeze(1)).squeeze(1).clamp(max=n_knots - 1)
    lo = (hi - 1).clamp(min=0)
    k_lo = knots.gather(1, lo.unsqueeze(1)).squeeze(1)
    k_hi = knots.gather(1, hi.unsqueeze(1)).squeeze(1)
    span = k_hi - k_lo
    t = torch.where(span > 0, (a - k_lo) / span.clamp(min=1e-30), torch.ones_like(a)).clamp(0.0, 1.0)
    out = rarity[lo] + t * (rarity[hi] - rarity[lo])
    out = torch.where(a <= knots[:, 0], torch.zeros_like(a), out)
    top = knots[:, -1]
    return torch.where(a > top, rarity[-1] + torch.log10(a / top), out)
