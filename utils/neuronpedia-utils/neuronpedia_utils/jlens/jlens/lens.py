# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""The Jacobian lens: applying a fitted ``J̄`` to read out early-layer residuals.

A :class:`JacobianLens` holds the per-layer average Jacobians ``J̄_ℓ`` produced by
:func:`jlens.fitting.fit`. Applying it is a single matrix multiply per layer::

    lens_ℓ(h)  =  unembed( J̄_ℓ @ h )

where ``h`` is the residual at layer ``ℓ`` and ``unembed`` is the model's own
final-norm + LM-head. The class is independent of the model that was used to fit
it — pass any :class:`~jlens.protocol.LensModel` to :meth:`JacobianLens.apply` to
do the forward + readout in one call, or use :meth:`JacobianLens.transport` if
you already have residuals.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel


class JacobianLens:
    """A fitted Jacobian lens — per-layer ``J̄_ℓ`` matrices and the readout method.

    Construct directly with the Jacobian dict (as returned by
    :func:`jlens.fitting.fit`), or load from disk with :meth:`load`.

    Attributes:
        jacobians: ``{layer_index: Tensor[d_model, d_model]}``. Each ``J̄_ℓ`` maps
            the residual at layer ``ℓ`` into the final-layer basis.
        source_layers: Sorted list of fitted layer indices (the keys of
            :attr:`jacobians`).
        n_prompts: Number of prompts the lens was averaged over.
        d_model: Residual-stream width.
    """

    def __init__(
        self,
        jacobians: dict[int, torch.Tensor],
        *,
        n_prompts: int,
        d_model: int,
    ) -> None:
        self.jacobians = {layer: J.float() for layer, J in jacobians.items()}
        self.source_layers = sorted(self.jacobians)
        self.n_prompts = n_prompts
        self.d_model = d_model

    def __repr__(self) -> str:
        return (
            f"JacobianLens(d_model={self.d_model}, n_prompts={self.n_prompts}, "
            f"source_layers=[{self.source_layers[0]}..{self.source_layers[-1]}] "
            f"({len(self.source_layers)} layers))"
        )

    def save(self, path: str) -> None:
        """Save to ``path`` (``torch.save``; Jacobians stored as fp16)."""
        torch.save(
            {
                "J": {layer: J.to(torch.float16) for layer, J in self.jacobians.items()},
                "n_prompts": self.n_prompts,
                "source_layers": self.source_layers,
                "d_model": self.d_model,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> JacobianLens:
        """Load a lens previously written by :meth:`save`."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "J" not in checkpoint:
            raise ValueError(
                f"{path} is not a JacobianLens file "
                f"(found keys {sorted(checkpoint)!r}; a fit() checkpoint?)"
            )
        return cls(
            jacobians=checkpoint["J"],
            n_prompts=checkpoint["n_prompts"],
            d_model=checkpoint["d_model"],
        )

    @classmethod
    def merge(cls, lenses: Sequence[JacobianLens]) -> JacobianLens:
        """Combine lenses fitted on disjoint prompt subsets into one.

        This is how sharded fits are aggregated: each machine runs
        :func:`jlens.fitting.fit` on its slice of the prompt corpus, saves the
        result, and one machine merges. The merged ``J̄_ℓ`` is the
        ``n_prompts``-weighted mean of the inputs.

        Args:
            lenses: Lenses to merge. Must agree on ``source_layers`` and
                ``d_model``.

        Returns:
            A new :class:`JacobianLens` averaging over the union of prompts.

        Raises:
            ValueError: If ``lenses`` is empty or the inputs disagree on shape.
        """
        if not lenses:
            raise ValueError("merge() needs at least one lens")
        first = lenses[0]
        for other in lenses[1:]:
            if other.source_layers != first.source_layers or other.d_model != first.d_model:
                raise ValueError("lenses disagree on source_layers / d_model")
        n_total = sum(lens.n_prompts for lens in lenses)
        merged: dict[int, torch.Tensor] = {}
        for layer in first.source_layers:
            weighted_sum = sum(lens.jacobians[layer] * lens.n_prompts for lens in lenses)
            merged[layer] = weighted_sum / n_total
        return cls(jacobians=merged, n_prompts=n_total, d_model=first.d_model)

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """Map a residual at ``layer`` into the final-layer basis: ``J̄_ℓ @ h``.

        Args:
            residual: Tensor of shape ``[..., d_model]``.
            layer: Source layer index (must be in :attr:`source_layers`).

        Returns:
            Tensor of the same shape, in the final-layer basis.
        """
        J_bar = self.jacobians[layer].to(residual.device)
        return residual @ J_bar.T

    @torch.no_grad()
    def apply(
        self,
        model: LensModel,
        prompt: str,
        *,
        layers: Sequence[int] | None = None,
        position: int = -1,
        max_seq_len: int = 512,
        use_jacobian: bool = True,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Run ``model`` on ``prompt`` and return lens logits at ``position``.

        Args:
            model: The model to read out from.
            prompt: Input text.
            layers: Layers to read out at. Defaults to all of
                :attr:`source_layers`. Must be a subset of
                :attr:`source_layers`.
            position: Token position to read out (Python indexing into the
                sequence; ``-1`` is the last token).
            max_seq_len: Truncate the prompt to this many tokens.
            use_jacobian: If ``False``, skip the ``J̄`` transport — this is the
                vanilla logit-lens baseline.

        Returns:
            A triple ``(lens_logits, model_logits, input_ids)``. ``lens_logits``
            maps each requested layer to a ``[vocab_size]`` tensor;
            ``model_logits`` is the model's actual final-layer logits at
            ``position`` (same shape).

        Raises:
            ValueError: If any requested layer is not in :attr:`source_layers`.
        """
        if layers is None:
            layers = self.source_layers
        unknown = set(layers) - set(self.source_layers)
        if use_jacobian and unknown:
            raise ValueError(
                f"layers {sorted(unknown)} not in source_layers; "
                f"fitted layers are {self.source_layers}"
            )
        final_layer = model.n_layers - 1
        record_at = sorted(set(layers) | {final_layer})

        input_ids = model.encode(prompt, max_length=max_seq_len)
        with ActivationRecorder(model.layers, at=record_at) as recorder:
            model.forward(input_ids)
            activations = {i: recorder.activations[i].detach() for i in record_at}

        lens_logits: dict[int, torch.Tensor] = {}
        for layer in layers:
            residual = activations[layer][0, position].float()
            if use_jacobian:
                residual = self.transport(residual, layer)
            lens_logits[layer] = model.unembed(residual.unsqueeze(0)).squeeze(0).float().cpu()

        final_residual = activations[final_layer][0, position].float()
        model_logits = model.unembed(final_residual.unsqueeze(0)).squeeze(0).float().cpu()
        return lens_logits, model_logits, input_ids
