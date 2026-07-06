# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""The model interface the lens needs — independent of any model library.

Everything in :mod:`jlens.fitting` and :mod:`jlens.lens` is typed against
:class:`LensModel`, so any model can be plugged in by implementing these six
members. :func:`jlens.hf.from_hf` is the HuggingFace adapter; the test suite's
``TinyDecoder`` is a 30-line from-scratch example.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import torch
from torch import nn


class LensModel(Protocol):
    """What :func:`jlens.fitting.fit` and :class:`jlens.lens.JacobianLens` need
    from a model.

    Attributes:
        n_layers: Number of residual blocks.
        d_model: Residual-stream width.
        layers: The residual blocks, indexable by integer — what
            :class:`~jlens.hooks.ActivationRecorder` hooks.
    """

    n_layers: int
    d_model: int
    layers: Sequence[nn.Module]

    def encode(self, text: str, *, max_length: int = ...) -> torch.Tensor:
        """Tokenize ``text`` to ``input_ids`` of shape ``[1, seq_len]`` on the
        model's input device."""
        ...

    def forward(self, input_ids: torch.Tensor) -> Any:
        """Run the residual stack on ``input_ids`` (no LM head). Must build an
        autograd graph through :attr:`layers` when grad is enabled."""
        ...

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """Map a residual-stream tensor ``[..., d_model]`` to logits
        ``[..., vocab_size]`` (final norm + LM head)."""
        ...
