# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Fitting the Jacobian lens.

The Jacobian lens reads out an early-layer residual ``h_ℓ`` by linearly
transporting it into the final-layer basis with the average input–output
Jacobian, then decoding with the model's own unembedding::

    lens_ℓ(h)  =  unembed( J̄_ℓ @ h )

The estimator implemented here (see :func:`jacobian_for_prompt`) sums the
cotangent over **all valid target positions at once** and then averages the
resulting gradient over **source positions** — so for each source position the
contribution is ``Σ_{p' ≥ p} ∂h_final[p', :] / ∂h_ℓ[p, :]``, the sum over later
target positions of the cross-position Jacobian. This is the reduction used in
the paper; it makes a single backward pass yield one full row of ``J_ℓ`` per
batch element, and empirically the cross-position bleed is small (the
same-position term dominates). A reader implementing the per-position
``∂h_final[p] / ∂h_ℓ[p]`` and averaging over ``p`` separately will get a
slightly different ``J̄`` — both work as a lens.

On Qwen 3.6-27B (64 layers, ``d_model=5120``) this is ~5 minutes per prompt on
2× H100; the paper figures use 1000 prompts. Host-RAM footprint is roughly
``2 × n_source_layers × d_model² × 4`` bytes (running sum + per-prompt scratch)
— ~13 GB for the full 64-layer 27B fit; restrict ``source_layers`` to reduce
it. To shard across machines, run :func:`fit` on each machine with its slice of
the prompt list and merge with :meth:`jlens.lens.JacobianLens.merge`.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FitProgress:
    """Per-prompt progress emitted by :func:`fit` (via ``metrics_callback``).

    The convergence fields answer "how much is the lens still moving?" — i.e.
    when do extra prompts stop mattering. ``mean_rel_change`` is the relative
    Frobenius change of the running-mean Jacobian contributed by *this* prompt,
    averaged over the fitted layers::

        δ_n = mean_ℓ ‖J̄_ℓ(n) − J̄_ℓ(n−1)‖_F / ‖J̄_ℓ(n)‖_F

    For an i.i.d. running mean this decays like ``~1/n``; watch for the prompt
    count where it flattens below a small threshold (e.g. ~1e-3). That is the
    point where the corpus has "levelled off" and more prompts buy little.

    Attributes:
        prompt_idx: Index of the prompt just processed (0-based).
        n_prompts: Total number of prompts in this run.
        n_done: Number of prompts successfully accumulated so far.
        seq_len: Tokenized length of this prompt.
        n_valid_positions: Source positions averaged over for this prompt.
        elapsed_s: Wall-clock seconds spent on this prompt.
        identity_distance: ``‖J̄_late − I‖_F / √d`` for the highest fitted layer.
        mean_rel_change: Convergence metric above. ``nan`` for the first prompt
            (no previous mean to compare against).
    """

    prompt_idx: int
    n_prompts: int
    n_done: int
    seq_len: int
    n_valid_positions: int
    elapsed_s: float
    identity_distance: float
    mean_rel_change: float

#: Positions before this index are excluded from the Jacobian average — early
#: positions act as attention sinks and have atypical residual statistics.
SKIP_FIRST_N_POSITIONS = 16


def valid_position_mask(seq_len: int, *, skip_first: int = SKIP_FIRST_N_POSITIONS) -> torch.Tensor:
    """Boolean mask over sequence positions to include in the Jacobian average.

    Early positions are dominated by attention-sink behaviour and the final
    position has no next-token target, so both are excluded.

    Args:
        seq_len: Length of the tokenized prompt.
        skip_first: Number of leading positions to exclude.

    Returns:
        Boolean tensor of shape ``[seq_len]``.

    Raises:
        ValueError: If the prompt is too short to leave any valid positions.
    """
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[skip_first : seq_len - 1] = True
    if mask.sum() == 0:
        raise ValueError(f"prompt too short: seq_len={seq_len}, need > {skip_first + 1} tokens")
    return mask


def jacobian_for_prompt(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """Compute the per-layer Jacobian estimator ``J_ℓ`` for one prompt.

    The trick that makes this tractable: run one forward pass on the prompt
    replicated ``dim_batch`` times along the batch axis, retain the autograd
    graph, then run ``ceil(d_model / dim_batch)`` backward passes on it. In each
    backward, batch element ``b`` carries a one-hot cotangent (the upstream
    gradient passed as ``grad_outputs``) at output dimension ``dim_start + b``,
    set at *every* valid target position simultaneously. The gradient at source
    position ``p`` is therefore ``Σ_{p' ≥ p} ∂h_target[p', dim_start+b] / ∂h_ℓ[p, :]``;
    the result we keep is the mean over source positions ``p`` — see the module
    docstring for how this relates to a strict per-position Jacobian.

    Args:
        model: The model to compute Jacobians for.
        prompt: Input text.
        source_layers: Layer indices ``ℓ`` to compute ``J_ℓ`` at.
        target_layer: Layer to take gradients with respect to. Defaults to the
            final layer (``n_layers - 1``). For models with heavy late-layer
            scaling (e.g. Gemma 4's ``layer_scalar``), targeting ``n_layers - 2``
            can give a better-conditioned ``J̄``.
        dim_batch: Output dimensions computed per backward pass. Higher uses
            more GPU memory (the prompt is replicated this many times in the
            forward); per-prompt time is nearly invariant since the total
            backward FLOPs are the same however they're sliced.
        max_seq_len: Truncate the prompt to this many tokens.
        skip_first: Leading positions to exclude (attention sinks). See
            :func:`valid_position_mask`.

    Returns:
        A triple ``(jacobians, seq_len, n_valid_positions)``. ``jacobians`` maps
        each source layer to a ``[d_model, d_model]`` fp32 CPU tensor — the
        position-averaged Jacobian for this prompt.
    """
    n_layers, d_model = model.n_layers, model.d_model
    if target_layer is None:
        target_layer = n_layers - 1
    if max(source_layers) >= target_layer:
        raise ValueError(
            f"source_layers must all be < target_layer={target_layer}; got max={max(source_layers)}"
        )

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    n_valid_positions = int(position_mask.sum())

    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in source_layers
    }
    n_passes = math.ceil(d_model / dim_batch)

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        # One forward on the prompt replicated dim_batch times. The retained
        # graph is reused for every backward pass below.
        replicated_ids = input_ids.expand(dim_batch, -1)
        model.forward(replicated_ids)
        target_activation = recorder.activations[target_layer]  # [dim_batch, seq_len, d_model]
        source_activations = [recorder.activations[layer] for layer in source_layers]

        valid_positions = position_mask.nonzero(as_tuple=True)[0].to(target_activation.device)
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)

        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n_dims_this_pass = min(dim_batch, d_model - dim_start)
            # Batch element b gets a one-hot cotangent at output dim
            # (dim_start + b), summed over every valid target position. The
            # gradient at each source layer (mean over source positions) is row
            # (dim_start + b) of J_ℓ as defined in the module docstring.
            cotangent.zero_()
            cotangent[
                batch_indices[:n_dims_this_pass, None],
                valid_positions[None, :],
                dim_start + batch_indices[:n_dims_this_pass, None],
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=(pass_idx < n_passes - 1),
            )
            for layer, grad in zip(source_layers, grads, strict=True):
                # grad: [dim_batch, seq_len, d_model] on whatever device this
                # layer lives on; mean over the valid positions → dim_batch rows.
                positions_on_device = valid_positions.to(grad.device, non_blocking=True)
                rows = grad[:n_dims_this_pass, positions_on_device, :].float().mean(dim=1)
                jacobians[layer][dim_start : dim_start + n_dims_this_pass, :] = rows.cpu()
            del grads
            if pass_idx % 100 == 0 or pass_idx == n_passes - 1:
                logger.debug(
                    "    pass %d/%d (dims %d–%d)",
                    pass_idx + 1,
                    n_passes,
                    dim_start,
                    dim_start + n_dims_this_pass,
                )

    return jacobians, seq_len, n_valid_positions


def _atomic_save(obj: object, path: str) -> None:
    """``torch.save`` to a temp file then ``os.replace`` — never leaves a
    half-written checkpoint."""
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def fit(
    model: LensModel,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    checkpoint_path: str | None = None,
    resume: bool = True,
    metrics_callback: Callable[[FitProgress], bool | None] | None = None,
) -> JacobianLens:
    """Fit ``J̄_ℓ`` over a list of prompts and return a :class:`JacobianLens`.

    The per-prompt Jacobians from :func:`jacobian_for_prompt` are accumulated as
    a running mean. If ``checkpoint_path`` is set, the running sum is saved
    after every prompt (atomic write) and resumed from on restart.

    Args:
        model: The model to fit on.
        prompts: Text prompts to average over. The paper uses ~1000 paragraphs
            of generic web text; 100–200 is enough for a recognisable lens.
        source_layers: Layers to fit ``J̄_ℓ`` at. Defaults to every layer below
            ``target_layer``.
        target_layer: See :func:`jacobian_for_prompt`. Defaults to the final
            layer; negative values index from the end.
        dim_batch: See :func:`jacobian_for_prompt`.
        max_seq_len: Truncate each prompt to this many tokens.
        skip_first: See :func:`jacobian_for_prompt`.
        checkpoint_path: If set, write a resumable checkpoint here after every
            prompt.
        resume: If ``True`` and ``checkpoint_path`` exists, resume from it.
        metrics_callback: If set, called after every successfully processed
            prompt with a :class:`FitProgress`. Use it to log/plot the
            convergence metric and decide how many prompts are enough. If it
            returns a truthy value, fitting stops early (a checkpoint is still
            written first) — e.g. to halt once the lens has converged.

    Returns:
        The fitted :class:`JacobianLens`.
    """
    n_layers, d_model = model.n_layers, model.d_model
    if target_layer is None:
        target_layer = n_layers - 1
    elif target_layer < 0:
        target_layer = n_layers + target_layer
    if source_layers is None:
        source_layers = list(range(target_layer))
    else:
        source_layers = sorted(set(source_layers))
    if max(source_layers) >= target_layer:
        raise ValueError(
            f"source_layers must all be < target_layer={target_layer}; got max={max(source_layers)}"
        )

    logger.info(
        "fit: n_layers=%d d_model=%d, fitting %d source layers (target=L%d) on %d prompts",
        n_layers,
        d_model,
        len(source_layers),
        target_layer,
        len(prompts),
    )

    # Running state: sum of per-prompt Jacobians, success count, and the list
    # index to resume from. ``next_idx`` is tracked separately from ``n_done``
    # so a too-short prompt that was skipped is not re-processed on resume.
    jacobian_sum: dict[int, torch.Tensor]
    n_done: int
    next_idx: int
    if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if set(state["source_layers"]) != set(source_layers):
            raise ValueError(
                f"checkpoint at {checkpoint_path} was fitted with source_layers="
                f"{state['source_layers']}, not {source_layers} — pass resume=False to discard it"
            )
        jacobian_sum, n_done, next_idx = state["jacobian_sum"], state["n_done"], state["next_idx"]
        logger.info("  resuming from checkpoint: %d/%d prompts processed", next_idx, len(prompts))
    else:
        jacobian_sum = {
            layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in source_layers
        }
        n_done = 0
        next_idx = 0

    def write_checkpoint() -> None:
        if checkpoint_path is not None:
            _atomic_save(
                {
                    "jacobian_sum": jacobian_sum,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    "source_layers": source_layers,
                },
                checkpoint_path,
            )

    late_layer = max(source_layers)
    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < next_idx:
            continue
        start_time = time.perf_counter()
        try:
            per_prompt_J, seq_len, n_valid = jacobian_for_prompt(
                model,
                prompt,
                source_layers,
                target_layer=target_layer,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
            )
        except ValueError as exc:
            logger.warning("  skipping prompt %d: %s", prompt_idx, exc)
            next_idx = prompt_idx + 1
            write_checkpoint()
            continue
        # Relative change of the running mean contributed by this prompt,
        # averaged over fitted layers — the "is it still moving?" signal.
        # Welford: J̄(m) − J̄(m−1) = (Xₘ − J̄(m−1)) / m, so we never need a copy
        # of the previous mean. ``m`` is the prompt count after this one.
        m = n_done + 1
        rel_changes: list[float] = []
        for layer in source_layers:
            X = per_prompt_J[layer]
            if n_done >= 1:
                prev_mean = jacobian_sum[layer] / n_done
                step = (X - prev_mean).norm().item() / m
                new_mean_norm = ((jacobian_sum[layer] + X) / m).norm().item()
                if new_mean_norm > 0.0:
                    rel_changes.append(step / new_mean_norm)
            jacobian_sum[layer] += X
        n_done += 1
        next_idx = prompt_idx + 1
        mean_rel_change = (
            sum(rel_changes) / len(rel_changes) if rel_changes else float("nan")
        )

        identity_distance = (
            (jacobian_sum[late_layer] / n_done) - torch.eye(d_model)
        ).norm().item() / math.sqrt(d_model)
        elapsed_s = time.perf_counter() - start_time
        logger.info(
            "  prompt %d/%d  seq_len=%d n_valid=%d  %.0fs  ||J̄_%d − I||_F/√d=%.3f  Δmean=%.2e",
            prompt_idx + 1,
            len(prompts),
            seq_len,
            n_valid,
            elapsed_s,
            late_layer,
            identity_distance,
            mean_rel_change,
        )
        stop_requested = False
        if metrics_callback is not None:
            stop_requested = bool(
                metrics_callback(
                    FitProgress(
                        prompt_idx=prompt_idx,
                        n_prompts=len(prompts),
                        n_done=n_done,
                        seq_len=seq_len,
                        n_valid_positions=n_valid,
                        elapsed_s=elapsed_s,
                        identity_distance=identity_distance,
                        mean_rel_change=mean_rel_change,
                    )
                )
            )
        write_checkpoint()
        if stop_requested:
            logger.info("  metrics_callback requested early stop after %d prompts", n_done)
            break

    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")
    jacobian_mean = {layer: jacobian_sum[layer] / n_done for layer in source_layers}
    logger.info("fit: done, %d prompts", n_done)
    return JacobianLens(jacobians=jacobian_mean, n_prompts=n_done, d_model=d_model)
