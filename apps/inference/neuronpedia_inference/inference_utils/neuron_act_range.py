"""Per-neuron corpus activation range, for ranking search-all results by fraction of max.

Raw MLP neuron activations are not comparable across neurons: scales differ by orders of
magnitude between neurons and between layers. With `normalizeByMax`, search-all divides each
neuron's sort key by the largest activation that neuron reached on a text corpus, so 1.0 means
"as strong as its strongest dashboard example".

The table is a `.safetensors` file loaded once from `NEURON_ACT_RANGE_PATH`:

  max  [n_layers, d_mlp]  largest activation on the corpus (the dashboard `stats.max`)
  min  [n_layers, d_mlp]  smallest activation on the corpus (the dashboard `stats.min`)
"""

from __future__ import annotations

from functools import lru_cache

import torch

# A neuron whose corpus max is near zero would turn any small activation on the prompt into a
# huge fraction. Its divisor is raised to this share of its layer's median positive max.
MIN_SCALE_FRACTION_OF_LAYER_MEDIAN = 0.05

NO_TABLE_MESSAGE = (
    "normalizeByMax needs a neuron activation range table, and this server was started without NEURON_ACT_RANGE_PATH."
)


class NeuronActRangeError(ValueError):
    """The table is missing or does not fit the request. The message is safe to return as a 400."""


@lru_cache(maxsize=2)
def load_neuron_max_scales(path: str) -> torch.Tensor:
    """Return the divisor per neuron, ``[n_layers, d_mlp]`` float32 on the CPU, always > 0."""
    from safetensors import SafetensorError
    from safetensors.torch import load_file

    try:
        tensors = load_file(path)
    except (OSError, SafetensorError) as e:
        raise NeuronActRangeError(f"Cannot read the neuron activation range table {path}: {e}") from e
    if "max" not in tensors:
        raise NeuronActRangeError(f"Neuron activation range table {path} has no 'max'")
    table = tensors["max"].float()
    if table.ndim != 2:
        raise NeuronActRangeError(
            f"Neuron activation range table {path}: max is {tuple(table.shape)}, expected [n_layers, d_mlp]"
        )
    scales = torch.empty_like(table)
    for li in range(table.shape[0]):
        row = table[li]
        positive = row[row > 0]
        median = float(positive.median()) if positive.numel() > 0 else 1.0
        scales[li] = row.clamp(min=median * MIN_SCALE_FRACTION_OF_LAYER_MEDIAN)
    return scales


def neuron_max_scale(path: str | None, layer: int, d_mlp: int) -> torch.Tensor:
    """The ``[d_mlp]`` divisor for one layer. Raises NeuronActRangeError when it cannot be served."""
    if not path:
        raise NeuronActRangeError(NO_TABLE_MESSAGE)
    scales = load_neuron_max_scales(path)
    if not 0 <= layer < scales.shape[0]:
        raise NeuronActRangeError(
            f"The neuron activation range table has {scales.shape[0]} layers; layer {layer} is out of range"
        )
    if scales.shape[1] != d_mlp:
        raise NeuronActRangeError(
            f"The neuron activation range table has {scales.shape[1]} neurons per layer; this model has {d_mlp}"
        )
    return scales[layer]
