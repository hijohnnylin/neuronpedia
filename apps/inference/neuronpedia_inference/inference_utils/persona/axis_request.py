"""Readout axes supplied with a request. This is how every axis reaches this server.

Axes used to be shipped on disk and named by id, which meant a fit only became measurable once
someone deployed it. Now the caller sends the direction: inline, or as a pointer to a published
artifact. Both arrive as an :class:`NPAxis` and leave as an :class:`AxisAsset`, so everything
downstream -- the render-conditions agreement, the capture pass, the projection, the readout --
is the same either way.

What this module is, then, is the validation that a build step used to do. Nothing has vetted a
request-supplied axis before it arrives, and the questions only the serving model can answer --
is this direction the right width, is that layer a layer this model has -- can only be answered
here. Each failure names what was wrong with which axis, because a direction of the wrong length
is a mistake a caller can fix and a silent 500 from a matmul is not.

The distinction the status codes draw: a payload this model cannot measure is the caller's (400),
and a Hub that will not answer is ours (502). Refusing the whole request rather than dropping the
bad axis, because a caller who named an axis and got a response without it has to notice the
absence to learn anything.
"""

import asyncio
import dataclasses
import logging
import os
from functools import lru_cache

import torch

from neuronpedia_inference.inference_utils.persona.axis_data import (
    MANIFEST_FILENAME,
    TENSORS_FILENAME,
    AxisAsset,
    Normalize,
    RenderConditions,
    load_axis,
)
from neuronpedia_inference.schemas import NPAxis, NPAxisNormalize, NPAxisSource

logger = logging.getLogger(__name__)

# Attribution for an axis whose sender did not claim it. Not derived from the id prefix: the
# convention that an id reads ``<author>_<name>`` holds for what this server ships, and reading a
# caller's id as if it did would attribute a reading to whoever their prefix happens to name.
DEFAULT_AUTHOR = "custom"

# How many axes one request may supply. The ceiling that matters is distinct layers rather than
# axes -- a readout costs one capture pass per layer, whatever the axis count at that layer -- but
# a cap on layers alone would let a caller send megabytes of directions to be told no. This bounds
# both at once, well above the handful a panel can display.
MAX_CUSTOM_AXES = 32

# Longest quantile table accepted. Two orders of magnitude past the 101 points a percentile per
# integer needs; past that the table is not a calibration curve, it is a payload.
MAX_QUANTILE_LEVELS = 4096

# Artifacts held parsed, keyed by the commit they were read at. Small because it exists to keep a
# conversation's turns from re-reading one file, not to be a mirror of the Hub.
HF_CACHE_SIZE = 32


class AxisRequestError(Exception):
    """A request-supplied axis that cannot be measured, with the status the endpoint should send."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _vector(values: list[float] | None, *, hidden_size: int, axis_id: str, field: str) -> torch.Tensor:
    """One request vector as a float tensor, checked for width and for being finite.

    An absent or empty vector becomes zeros, which is what makes the two mean subtractions
    optional: subtracting zero is the identity, so an axis that never had a scaler ships nothing
    rather than a vector of zeros the caller had to write out.
    """
    if not values:
        return torch.zeros(hidden_size)
    if len(values) != hidden_size:
        raise AxisRequestError(
            f"axis {axis_id!r}: {field} has {len(values)} entries, but this model is {hidden_size}-dimensional"
        )
    tensor = torch.tensor(values, dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all()):
        # NaN propagates through the projection into every turn's value, so the readout comes back
        # populated and meaningless. Cheaper to refuse than to explain later.
        raise AxisRequestError(f"axis {axis_id!r}: {field} contains a non-finite value")
    return tensor


def _quantiles(payload: NPAxis) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """``(levels, pos, neg)`` for an axis that sent tables, or three ``None`` for one that did not.

    The two tables come together or not at all -- one of them is not a calibration -- while the
    levels default to evenly spaced, which is what a table of per-percentile landmarks means.
    """
    pos, neg = payload.quantiles_pos, payload.quantiles_neg
    if pos is None and neg is None:
        if payload.quantile_levels is not None:
            raise AxisRequestError(f"axis {payload.id!r}: quantileLevels was sent without any quantile table")
        return None, None, None
    if pos is None or neg is None:
        missing = "quantilesPos" if pos is None else "quantilesNeg"
        raise AxisRequestError(
            f"axis {payload.id!r}: {missing} is missing; a percentile is read off both poles' tables"
        )
    if len(pos) != len(neg):
        raise AxisRequestError(
            f"axis {payload.id!r}: quantilesPos has {len(pos)} entries and quantilesNeg has {len(neg)}"
        )
    if len(pos) < 2:
        raise AxisRequestError(f"axis {payload.id!r}: a quantile table needs at least two entries")
    if len(pos) > MAX_QUANTILE_LEVELS:
        raise AxisRequestError(f"axis {payload.id!r}: quantile tables are limited to {MAX_QUANTILE_LEVELS} entries")

    levels = payload.quantile_levels
    if levels is None:
        levels = torch.linspace(0.0, 1.0, len(pos)).tolist()
    elif len(levels) != len(pos):
        raise AxisRequestError(
            f"axis {payload.id!r}: quantileLevels has {len(levels)} entries and the tables have {len(pos)}"
        )

    tables = {"quantileLevels": levels, "quantilesPos": pos, "quantilesNeg": neg}
    tensors: dict[str, torch.Tensor] = {}
    for name, values in tables.items():
        tensor = torch.tensor(values, dtype=torch.float32)
        if not bool(torch.isfinite(tensor).all()):
            raise AxisRequestError(f"axis {payload.id!r}: {name} contains a non-finite value")
        # Interpolating an unsorted table is non-monotone, which reorders readings rather than
        # merely misplacing them -- the one failure a bounded scale must not have.
        if bool((tensor.diff() < 0).any()):
            raise AxisRequestError(f"axis {payload.id!r}: {name} must be nondecreasing")
        tensors[name] = tensor
    return tensors["quantileLevels"], tensors["quantilesPos"], tensors["quantilesNeg"]


def asset_from_payload(payload: NPAxis, *, hidden_size: int, n_layers: int) -> AxisAsset:
    """An inline axis definition -> the asset the projection code takes.

    Raises :class:`AxisRequestError` for anything this model cannot measure.
    """
    if payload.direction is None or payload.layer is None:
        # The request model already refuses this; restated so the function holds on its own.
        raise AxisRequestError(f"axis {payload.id!r}: direction and layer are required without a source")
    if not 0 <= payload.layer < n_layers:
        raise AxisRequestError(
            f"axis {payload.id!r}: layer {payload.layer} is out of range for this model, which has "
            f"{n_layers} layers (0-{n_layers - 1})"
        )

    direction = _vector(payload.direction, hidden_size=hidden_size, axis_id=payload.id, field="direction")
    if float(torch.linalg.vector_norm(direction)) == 0.0:
        # Every turn would read exactly `center`, which looks like a model with no variation on
        # this trait rather than like an axis that was never sent.
        raise AxisRequestError(f"axis {payload.id!r}: direction is all zeros")
    for name, value in (("scalePos", payload.scale_pos), ("scaleNeg", payload.scale_neg)):
        if value == 0.0:
            raise AxisRequestError(
                f"axis {payload.id!r}: {name} may not be zero; every value on that pole would be inf"
            )

    levels, quantiles_pos, quantiles_neg = _quantiles(payload)
    normalize: Normalize = "l2" if payload.normalize is NPAxisNormalize.L2 else "none"
    return AxisAsset(
        id=payload.id,
        author=payload.author or DEFAULT_AUTHOR,
        # Nothing in this path parses a title, and nothing should start: the poles are their own
        # fields now. This is here for the displays that still read one.
        title=payload.display_name or payload.id,
        layer=payload.layer,
        normalize=normalize,
        center=payload.center,
        scale_pos=payload.scale_pos,
        scale_neg=payload.scale_neg,
        render=RenderConditions(
            blank_system_prompt=payload.render.blank_system_prompt,
            template_kwargs=dict(payload.render.template_kwargs),
        ),
        direction=direction,
        scaler_mean=_vector(payload.pre_norm_mean, hidden_size=hidden_size, axis_id=payload.id, field="preNormMean"),
        pca_mean=_vector(payload.post_norm_mean, hidden_size=hidden_size, axis_id=payload.id, field="postNormMean"),
        caveat=payload.caveat,
        quantile_levels=levels,
        quantiles_pos=quantiles_pos,
        quantiles_neg=quantiles_neg,
        pole_positive=payload.pole_positive,
        pole_negative=payload.pole_negative,
        pole_positive_description=payload.pole_positive_description,
        pole_negative_description=payload.pole_negative_description,
    )


def _resolve_revision(source: NPAxisSource) -> str:
    """The commit ``source`` names, so what gets cached and reported is one immutable artifact.

    Asked every time rather than cached: a branch name is a moving target, and a process that
    resolved it once at startup would serve last week's axis under this week's name for as long as
    it stays up. One request to the Hub is cheap next to a forward pass, and it is also where a
    missing or gated repo is reported as such rather than as a missing file.
    """
    from huggingface_hub import HfApi

    try:
        info = HfApi().repo_info(repo_id=source.hf_repo_id, revision=source.revision, repo_type="model")
    except Exception as exc:  # noqa: BLE001  # the Hub's exception tree is wide; the message is what matters
        raise AxisRequestError(
            f"Could not read {source.hf_repo_id} at revision {source.revision or 'default'}: {exc}",
            status_code=502,
        ) from exc
    if not info.sha:
        raise AxisRequestError(f"{source.hf_repo_id} reported no commit for revision {source.revision or 'default'}")
    return info.sha


@lru_cache(maxsize=HF_CACHE_SIZE)
def _artifact_at(repo_id: str, folder: str, revision: str) -> AxisAsset:
    """Download and parse one published artifact. Cached, keyed by the commit it was read at.

    Safe to share across requests because an :class:`AxisAsset` is frozen and the projection only
    reads it, and safe to cache at all only because the key is a commit rather than a branch.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    prefix = folder.strip("/")
    paths: dict[str, str] = {}
    for name in (MANIFEST_FILENAME, TENSORS_FILENAME):
        filename = f"{prefix}/{name}" if prefix else name
        try:
            paths[name] = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, repo_type="model")
        except EntryNotFoundError as exc:
            # The commit resolved, so the repo is there and this is a folder that is not an axis.
            raise AxisRequestError(f"{repo_id}/{prefix} at {revision} has no {name}") from exc
        except Exception as exc:  # noqa: BLE001
            raise AxisRequestError(
                f"Could not download {filename} from {repo_id} at {revision}: {exc}", status_code=502
            ) from exc

    # Both files land in the same snapshot directory, which is the shape `load_axis` reads.
    directory = os.path.dirname(paths[MANIFEST_FILENAME])
    try:
        asset = load_axis(prefix.rsplit("/", 1)[-1] or repo_id, directory)
    except Exception as exc:  # noqa: BLE001
        raise AxisRequestError(f"{repo_id}/{prefix} at {revision} is not a valid axis: {exc}") from exc
    return dataclasses.replace(asset, source_revision=revision)


async def _asset_from_source(payload: NPAxis, *, hidden_size: int, n_layers: int) -> AxisAsset:
    """A published artifact -> the asset the projection code takes, under the caller's id."""
    source = payload.source
    assert source is not None  # the caller checked; this keeps the type narrow
    if ".." in source.hf_folder:
        raise AxisRequestError(f"axis {payload.id!r}: hfFolder may not contain '..'")

    revision = await asyncio.to_thread(_resolve_revision, source)
    asset = await asyncio.to_thread(_artifact_at, source.hf_repo_id, source.hf_folder, revision)

    if asset.hidden_size != hidden_size:
        raise AxisRequestError(
            f"axis {payload.id!r}: {source.hf_repo_id}/{source.hf_folder} is {asset.hidden_size}-dimensional, "
            f"but this model is {hidden_size}-dimensional"
        )
    if not 0 <= asset.layer < n_layers:
        raise AxisRequestError(
            f"axis {payload.id!r}: {source.hf_repo_id}/{source.hf_folder} was fitted at layer {asset.layer}, "
            f"which this model does not have ({n_layers} layers)"
        )
    return dataclasses.replace(asset, id=payload.id)


async def resolve_request_axes(
    payloads: list[NPAxis],
    *,
    hidden_size: int,
    n_layers: int,
) -> list[AxisAsset]:
    """Every axis sent with a request, as assets, in the order they were sent.

    A duplicate id is refused rather than resolved: two readouts under one id leaves a caller
    unable to say which fit produced which numbers, and last-one-wins would answer silently for
    the wrong one.
    """
    if len(payloads) > MAX_CUSTOM_AXES:
        raise AxisRequestError(f"At most {MAX_CUSTOM_AXES} axes may be supplied with one request, got {len(payloads)}")

    seen: set[str] = set()
    assets: list[AxisAsset] = []
    for payload in payloads:
        if payload.id in seen:
            raise AxisRequestError(f"Duplicate readout axis id {payload.id!r} in this request")
        seen.add(payload.id)
        if payload.source is None:
            assets.append(asset_from_payload(payload, hidden_size=hidden_size, n_layers=n_layers))
        else:
            assets.append(await _asset_from_source(payload, hidden_size=hidden_size, n_layers=n_layers))
            logger.info(
                f"Resolved readout axis '{payload.id}' from {payload.source.hf_repo_id}/{payload.source.hf_folder} "
                f"at {assets[-1].source_revision}"
            )
    return assets
