# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Forward-hook utilities for capturing the residual stream.

These are framework-agnostic: they work with any ``nn.Module`` whose blocks
return the residual stream as their (first) output. They live here, separate
from the model loading code, so that fitting and inference can share them
without depending on HuggingFace.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import torch
from torch import nn


class ActivationRecorder:
    """Captures residual-stream tensors at the given block indices.

    On entering the ``with`` block, a forward hook is registered on each
    requested block; on the next forward pass the block's output tensor is
    stored in :attr:`activations`, keyed by block index. The stored tensors are
    live references to the autograd graph (not detached), so they can be passed
    straight to :func:`torch.autograd.grad` as ``inputs`` — this is what makes
    the batched-backward Jacobian trick in :func:`jlens.fitting.jacobian_for_prompt`
    work.

    Hooks are registered in :meth:`__enter__` and removed in :meth:`__exit__`,
    so a recorder constructed but never entered does not touch the model::

        with ActivationRecorder(model.layers, at=[5, 10, 20]) as recorder:
            model.forward(input_ids)
            h5 = recorder.activations[5]

    Args:
        blocks: The sequence of residual blocks (e.g. ``model.layers``).
        at: Block indices to record at. Duplicates are de-duplicated.
        start_graph_at: If given, the hook for this index marks the captured
            tensor with ``requires_grad_(True)`` before any downstream block
            sees it. When the model's parameters all have ``requires_grad=False``
            (so no autograd graph would otherwise be built), this makes the
            captured residual the leaf that roots the graph — everything from
            this block onward is differentiable, everything before it builds no
            graph at all. :func:`jlens.fitting.jacobian_for_prompt` uses this so
            the retained graph spans only ``min(source_layers)..target_layer``.
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        at: Iterable[int],
        *,
        start_graph_at: int | None = None,
    ) -> None:
        self._blocks = blocks
        self._indices = sorted(set(at))
        self._start_graph_at = start_graph_at
        if start_graph_at is not None and start_graph_at not in self._indices:
            self._indices = sorted({*self._indices, start_graph_at})
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, index: int) -> Callable[..., None]:
        is_graph_root = index == self._start_graph_at

        def hook(module: nn.Module, inputs, output) -> None:
            # Some HF blocks return a tuple (hidden, present_kv, ...).
            tensor = output if torch.is_tensor(output) else output[0]
            if is_graph_root:
                tensor.requires_grad_(True)
            self.activations[index] = tensor

        return hook

    def __enter__(self) -> ActivationRecorder:
        for index in self._indices:
            self._handles.append(self._blocks[index].register_forward_hook(self._make_hook(index)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
