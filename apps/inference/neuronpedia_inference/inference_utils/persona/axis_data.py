"""Readout axes - one fitted direction, and what turns an activation into a number.

An axis is a named direction plus everything needed to read it: the layer it was fitted at, how
the activation is scaled first, and the affine constants that put the result on a readable scale.

**This server ships none.** Every axis arrives with the request, either inline or as a reference
to a published artifact, and :mod:`.axis_request` is what turns one into the :class:`AxisAsset`
below. The caller owns the catalogue, so adding an axis is a row in their database rather than a
release here, and this process holds no opinion about which axes exist for which model.

An artifact is a folder holding ``axis.safetensors`` (three vectors) and ``axis.yaml`` (the
scalars and the rendering conditions the fit assumed), which :func:`load_axis` reads. One axis is
one direction: a fit with several components is several artifacts, which is what lets each of
them name its own layer.

An axis is conventionally named ``<author>_<name>``, and the manifest states the author in a
field of its own. Two groups fitting the same trait is the expected case rather than a collision:
``mit_empathy`` and some later ``lu_empathy`` are different measurements. Nothing here enforces
that convention, because the id belongs to whoever sent the request.

A projection is::

    raw   = ((x - scaler_mean) [L2-normalized] - pca_mean) . direction
    d     = raw - center
    value = d / scale_pos  if d >= 0  else  d / scale_neg

and is never clipped -- a value outside [-1, 1] is a real reading, not something to pin to the
boundary.

One divisor per pole, because a fitted axis is rarely symmetric about its centre. A single
divisor has to be the larger of the two spreads, which squeezes the tighter pole towards zero:
``mit_erudite``'s negative tail is 5.8x its positive spread, so under one divisor its
"sophisticated" pole could not read past +0.21 and looked flat whatever the prompt. The map is
continuous at ``center``, where both branches are zero, and monotone, so it preserves rank order
exactly. A manifest carrying a lone ``scale`` means both poles share it, which is the spelling
that predates this and is what ``lu_assistant-axis`` still ships.

**Two readings of the same projection, and both are reported.** The value above is a ratio to
the p99 landmark of the fitting corpus, so it exceeds 1 for roughly 2% of turns by construction
-- accurate, and unreadable on a gauge a person expects to stop at 100%. An asset may therefore
also carry per-pole quantile tables, and :func:`percentile` turns the same projection into the
share of the corpus it is past, which cannot leave [-1, 1]. That is what a display shows;
the ratio is the measurement, and it stays unclipped because how far past the corpus a turn
sits is the signal that an axis is being read off distribution. Dropping either loses something
the other cannot say.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
import yaml
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "axis.yaml"
TENSORS_FILENAME = "axis.safetensors"

# The L2 denominator floor. Fixed here rather than per asset: the converter rejects an asset
# fitted with a different value, so this is the only one in play.
L2_EPS = 1e-12

TENSOR_NAMES = ("direction", "scaler_mean", "pca_mean")

# The poles a manifest predating ``pole_positive`` / ``pole_negative`` encodes in its display
# title, as ``- simplistic <-> + sophisticated``. Every asset shipped in this tree is written that
# way, and a title was the only place the two names existed; parsing it is how they reach the pole
# fields until those assets are replaced. New manifests declare the poles instead, and this goes
# with the last old one. The arrow is spelled as escapes because a variation selector in a .py
# file breaks GitHub's CodeQL extractor.
TITLE_POLES = re.compile(r"^\s*-\s*(.+?)\s*\u2194\ufe0f?\s*\+\s*(.+?)\s*$")

# The quantile tables a percentile is read off. Optional, and all three or none: an asset fitted
# before them ships only the divisors, and reports no percentile rather than a made-up one.
QUANTILE_TENSOR_NAMES = ("quantile_levels", "quantiles_pos", "quantiles_neg")

Normalize = Literal["l2", "none"]


@dataclass(frozen=True)
class RenderConditions:
    """How a conversation must be templated for an axis's numbers to mean anything.

    A projection onto a fitted direction only holds if inference renders the conversation the
    way it was rendered during fitting. These conditions are applied to the prompt *before*
    generation, so they change the text the user sees -- which is why two axes that disagree
    here cannot both be served in one request.

    **What this cannot say: that a fit assumed a non-empty system turn.** The ``mit_*`` axes were
    fitted with a persona-describing system prompt on every conversation, and the page serving
    them sends none, so they are read off distribution today. ``blank_system_prompt: false`` is
    not the same statement -- it only means "do not blank the caller's prompt". Nor is this a
    template divergence to detect after the fact: Llama 3.1 emits its system block with the
    knowledge-cutoff preamble either way, so "no system message" and "empty system message"
    render byte for byte identically, and ``date_string`` takes effect regardless. It is a
    distribution shift, and closing it means a ``requires_system_prompt`` condition here plus a
    400 beside the disagreement check in ``_agreed_render_conditions`` -- which is a decision
    about what the caller sends, not a bug to fix in the loader.
    """

    # The fit used an empty system turn, so a caller-supplied system prompt has to be blanked.
    # (On Llama 3 a non-empty system turn also makes the template inject its knowledge-cutoff
    # preamble, which was not there during fitting.)
    blank_system_prompt: bool = False
    # Extra keyword arguments for the chat template. Llama 3.1 injects the current date into the
    # system block, so a fit on that model pins `date_string` or drifts off distribution as the
    # calendar moves.
    template_kwargs: dict[str, str] = field(default_factory=dict)

    def key(self) -> tuple:
        """A hashable form, for checking that a set of axes agrees."""
        return (self.blank_system_prompt, tuple(sorted(self.template_kwargs.items())))

    def describe(self) -> str:
        parts = [f"blank_system_prompt={self.blank_system_prompt}"]
        parts += [f"{name}={value}" for name, value in sorted(self.template_kwargs.items())]
        return ", ".join(parts)


@dataclass(frozen=True)
class AxisAsset:
    """One readout axis, loaded and ready to project with."""

    id: str
    # Who fitted this axis: the ``<author>_`` prefix of the id, restated as data so a caller
    # attributing a reading does not have to split the id on an underscore.
    author: str
    title: str
    layer: int
    normalize: Normalize
    center: float
    # One divisor per pole of the axis; see the projection formula in the module docstring.
    scale_pos: float
    scale_neg: float
    render: RenderConditions
    direction: torch.Tensor
    scaler_mean: torch.Tensor
    pca_mean: torch.Tensor
    caveat: str | None = None
    # The quantile tables, present together or not at all; see :func:`percentile`. Absent on any
    # asset fitted before them, which is why every consumer treats a percentile as optional.
    quantile_levels: torch.Tensor | None = None
    quantiles_pos: torch.Tensor | None = None
    quantiles_neg: torch.Tensor | None = None
    # What each end of the axis means. Optional because an axis is a direction first: one supplied
    # with a request may name no poles at all, and reports none rather than inventing them.
    pole_positive: str | None = None
    pole_negative: str | None = None
    pole_positive_description: str | None = None
    pole_negative_description: str | None = None
    # The commit an axis fetched from a published artifact was read at. None for anything local.
    source_revision: str | None = None

    @property
    def hidden_size(self) -> int:
        return int(self.direction.shape[0])

    @property
    def has_percentile(self) -> bool:
        """Whether this axis can report a bounded reading as well as the unbounded ratio."""
        return self.quantile_levels is not None


def _read_manifest(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping, got {type(raw).__name__}")
    return raw


def _read_render(raw: Any, path: str) -> RenderConditions:
    if raw is None:
        return RenderConditions()
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: 'render' must be a mapping, got {type(raw).__name__}")
    template_kwargs = raw.get("template_kwargs") or {}
    if not isinstance(template_kwargs, dict):
        raise ValueError(f"{path}: 'render.template_kwargs' must be a mapping")
    return RenderConditions(
        blank_system_prompt=bool(raw.get("blank_system_prompt", False)),
        template_kwargs={str(key): str(value) for key, value in template_kwargs.items()},
    )


def _read_scales(raw: dict[str, Any], path: str) -> tuple[float, float]:
    """The two pole divisors, ``(scale_pos, scale_neg)``.

    A lone ``scale`` means both poles share a divisor. That is the spelling every asset used
    before the poles were separated, so it is honoured rather than migrated: an asset written
    against it keeps loading and keeps reporting the numbers it always did.
    """
    shared = raw.get("scale")
    default = 1.0 if shared is None else float(shared)
    scales = {name: float(raw.get(name, default)) for name in ("scale_pos", "scale_neg")}
    for name, value in scales.items():
        # Dividing by it would report inf for every turn on that side of the centre.
        if value == 0.0:
            raise ValueError(f"{path}: {name!r} must be non-zero")
    return scales["scale_pos"], scales["scale_neg"]


def _read_quantiles(
    tensors: dict[str, torch.Tensor], path: str
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """The quantile tables, or three ``None`` for an asset that ships none.

    All three or nothing: two of them cannot be interpreted without the third, so a partial
    set is a malformed asset rather than something to work around.
    """
    present = [name for name in QUANTILE_TENSOR_NAMES if name in tensors]
    if not present:
        return None, None, None
    if len(present) != len(QUANTILE_TENSOR_NAMES):
        missing = [name for name in QUANTILE_TENSOR_NAMES if name not in tensors]
        raise ValueError(f"{path}: quantile tables are incomplete, missing {missing}")

    levels, table_pos, table_neg = (tensors[name].float() for name in QUANTILE_TENSOR_NAMES)
    shapes = {name: tuple(tensors[name].shape) for name in QUANTILE_TENSOR_NAMES}
    if len(set(shapes.values())) != 1 or len(shapes["quantile_levels"]) != 1:
        raise ValueError(f"{path}: expected three matching 1-D quantile tables, got {shapes}")
    if levels.shape[0] < 2:
        raise ValueError(f"{path}: 'quantile_levels' needs at least two entries")
    # An unsorted table makes the interpolation non-monotone, which reorders readings rather
    # than merely misplacing them -- the one failure a bounded scale must not have.
    for name, table in (("quantile_levels", levels), ("quantiles_pos", table_pos), ("quantiles_neg", table_neg)):
        if bool((table.diff() < 0).any()):
            raise ValueError(f"{path}: {name!r} must be nondecreasing")
    return levels, table_pos, table_neg


def _read_poles(raw: dict[str, Any], title: str) -> tuple[str | None, str | None]:
    """``(pole_positive, pole_negative)``: from their own keys, or parsed out of the title.

    Neither spelling is required. An axis is a direction first, and one that names no poles
    reports none rather than having names invented for it.
    """
    declared = {name: str(raw.get(name) or "").strip() or None for name in ("pole_positive", "pole_negative")}
    if declared["pole_positive"] or declared["pole_negative"]:
        return declared["pole_positive"], declared["pole_negative"]
    match = TITLE_POLES.match(title)
    return (match.group(2), match.group(1)) if match else (None, None)


def load_axis(axis_id: str, axis_dir: str) -> AxisAsset:
    """Read one axis directory. Raises on anything malformed; the caller decides what that means.

    ``axis_id`` is the name the reading is reported under, and is the caller's to choose. It is
    not checked against the manifest's ``author``: the two agree by convention in a published
    artifact, but the id belongs to the request rather than to the folder it came from.
    """
    manifest_path = os.path.join(axis_dir, MANIFEST_FILENAME)
    tensors_path = os.path.join(axis_dir, TENSORS_FILENAME)
    raw = _read_manifest(manifest_path)

    author = str(raw.get("author") or "").strip()
    if not author:
        raise ValueError(f"{manifest_path}: 'author' is required and must be non-empty")

    title = str(raw.get("title") or "").strip()
    pole_positive, pole_negative = _read_poles(raw, title)
    if not title:
        # A manifest that names its poles has said everything a title says, in the two fields a
        # reader wants them in. Synthesized here rather than made optional on the asset, so that
        # `title` stays a plain string for the callers that display one.
        if not (pole_positive and pole_negative):
            raise ValueError(f"{manifest_path}: 'title', or both 'pole_positive' and 'pole_negative', is required")
        title = f"- {pole_negative} \u2194\ufe0f + {pole_positive}"
    if "layer" not in raw:
        raise ValueError(f"{manifest_path}: 'layer' is required")
    layer = int(raw["layer"])
    normalize = str(raw.get("normalize", "none"))
    if normalize not in ("l2", "none"):
        raise ValueError(f"{manifest_path}: 'normalize' must be 'l2' or 'none', got {normalize!r}")
    scale_pos, scale_neg = _read_scales(raw, manifest_path)

    tensors = load_file(tensors_path)
    missing = [name for name in TENSOR_NAMES if name not in tensors]
    if missing:
        raise ValueError(f"{tensors_path}: missing tensor(s) {missing}")
    shapes = {name: tuple(tensors[name].shape) for name in TENSOR_NAMES}
    if len(set(shapes.values())) != 1 or len(shapes["direction"]) != 1:
        raise ValueError(f"{tensors_path}: expected three matching 1-D tensors, got {shapes}")

    levels, table_pos, table_neg = _read_quantiles(tensors, tensors_path)

    caveat = raw.get("caveat")
    descriptions = {
        name: str(raw.get(name) or "").strip() or None
        for name in ("pole_positive_description", "pole_negative_description")
    }
    return AxisAsset(
        id=axis_id,
        author=author,
        title=title,
        layer=layer,
        normalize=normalize,  # type: ignore[arg-type]  # narrowed by the check above
        center=float(raw.get("center", 0.0)),
        scale_pos=scale_pos,
        scale_neg=scale_neg,
        render=_read_render(raw.get("render"), manifest_path),
        direction=tensors["direction"].float(),
        scaler_mean=tensors["scaler_mean"].float(),
        pca_mean=tensors["pca_mean"].float(),
        caveat=str(caveat) if caveat else None,
        quantile_levels=levels,
        quantiles_pos=table_pos,
        quantiles_neg=table_neg,
        pole_positive=pole_positive,
        pole_negative=pole_negative,
        pole_positive_description=descriptions["pole_positive_description"],
        pole_negative_description=descriptions["pole_negative_description"],
    )


def calibrate(raw: torch.Tensor, axis: AxisAsset) -> torch.Tensor:
    """A raw projection -> the value that gets reported, each pole by its own divisor.

    Both branches are computed and one selected, which is cheaper than indexing for the handful
    of turns in a request and keeps the two spellings of the formula in one line. Continuous at
    ``center``, where the numerator is zero on both sides, and deliberately unclipped.
    """
    d = raw - axis.center
    return torch.where(d >= 0, d / axis.scale_pos, d / axis.scale_neg)


def percentile(raw: torch.Tensor, axis: AxisAsset) -> torch.Tensor | None:
    """A raw projection -> where it falls in the fitting corpus, in [-1, 1]. ``None`` without tables.

    Where :func:`calibrate` gives a ratio to the p99 landmark -- a real measurement that passes
    1 for about 2% of turns by construction -- this gives the share of the fitting corpus a
    reading is past, on its own pole. Bounded because a share cannot exceed 1: interpolation
    holds the table's end values, so a turn beyond everything the fit saw reads exactly 1.0 and
    can read no more. That is the number a display can show; "102%" reads as a broken gauge
    however defensible it is.

    Monotone, so it preserves rank order exactly, and zero at ``center`` on both branches, so
    the two poles meet without a step. What it gives up is resolution at the ends, where every
    reading past the corpus collapses onto 1.0 -- which is why the ratio is reported alongside
    it rather than replaced by it.
    """
    if axis.quantile_levels is None or axis.quantiles_pos is None or axis.quantiles_neg is None:
        return None
    levels = axis.quantile_levels.to(torch.float64)
    d = (raw - axis.center).to(torch.float64)
    forward = _interpolate(d, axis.quantiles_pos.to(torch.float64), levels)
    backward = -_interpolate(-d, axis.quantiles_neg.to(torch.float64), levels)
    return torch.where(d >= 0, forward, backward).to(raw.dtype)


def _interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """``numpy.interp`` for tensors: piecewise linear through ``(xp, fp)``, flat outside.

    Torch has no ``interp``, and holding the end values outside the table is the whole source
    of the bound, so it is written out rather than approximated by a clamp on the input.
    """
    index = torch.clamp(torch.searchsorted(xp, x, right=True), 1, xp.shape[0] - 1)
    lo, hi = xp[index - 1], xp[index]
    # A flat segment would divide by zero; the two ends of it agree anyway, so either is right.
    width = torch.where(hi > lo, hi - lo, torch.ones_like(hi))
    slope = (fp[index] - fp[index - 1]) / width
    return torch.clamp(fp[index - 1] + (x - lo) * slope, fp[0], fp[-1])


def project_axis(mean_acts_per_turn: torch.Tensor | list[torch.Tensor], axis: AxisAsset) -> np.ndarray:
    """Project per-turn activations onto one axis -> ``[n_turns]`` calibrated values.

    Mirrors the arithmetic the pickled sklearn assets performed, minus sklearn: the scaler's
    centering (and optional L2 normalization), then the PCA's own centering, then the dot
    product, then the per-pole calibration.
    """
    return calibrate(_raw_projection(mean_acts_per_turn, axis), axis).numpy()


def project_axis_with_percentile(
    mean_acts_per_turn: torch.Tensor | list[torch.Tensor], axis: AxisAsset
) -> tuple[np.ndarray, np.ndarray | None]:
    """Both readings of one projection: ``(value, percentile)``, the latter ``None`` without tables.

    One function because the two share the expensive half -- the projection itself -- and
    because they are meant to travel together. Reporting the percentile alone would lose how far
    past the corpus a turn sits, which is the signal that an axis is read off distribution.
    """
    raw = _raw_projection(mean_acts_per_turn, axis)
    ranked = percentile(raw, axis)
    return calibrate(raw, axis).numpy(), None if ranked is None else ranked.numpy()


def _raw_projection(mean_acts_per_turn: torch.Tensor | list[torch.Tensor], axis: AxisAsset) -> torch.Tensor:
    """The uncalibrated projection: ``[n_turns]`` values in the axis's own units."""
    acts = torch.stack(mean_acts_per_turn) if isinstance(mean_acts_per_turn, list) else mean_acts_per_turn
    x = acts.float().cpu() - axis.scaler_mean
    if axis.normalize == "l2":
        norms = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
        x = x / torch.clamp(norms, min=L2_EPS)
    return (x - axis.pca_mean) @ axis.direction
