"""Vectors supplied with a request. This is how every vector reaches this server.

They used to be shipped on disk and named by id, which meant a fit only became measurable once
someone deployed it. Now the caller sends the direction: inline, or as a pointer to a published
artifact. Both arrive as an :class:`NPVectorRead` and leave as a :class:`VectorAsset`, so
everything downstream -- the render-conditions agreement, the capture pass, the projection, the
readout -- is the same either way.

What this module is, then, is the validation that a build step used to do. Nothing has vetted a
request-supplied vector before it arrives, and the questions only the serving model can answer --
is this direction the right width, is that layer a layer this model has -- can only be answered
here. Each failure names what was wrong with which vector, because a direction of the wrong length
is a mistake a caller can fix and a silent 500 from a matmul is not.

The distinction the status codes draw: a payload this model cannot measure is the caller's (400),
and a Hub that will not answer is ours (502). Refusing the whole request rather than dropping the
bad vector, because a caller who named one and got a response without it has to notice the
absence to learn anything.
"""

import asyncio
import dataclasses
import logging
import os
from functools import lru_cache

import torch

from neuronpedia_inference.inference_utils.vectors.vector_data import (
    MANIFEST_FILENAME,
    TENSORS_FILENAME,
    CaptureSite,
    Normalize,
    Pooling,
    ReadSpec,
    RenderConditions,
    TokenSelection,
    VectorAsset,
    load_vector,
)
from neuronpedia_inference.schemas import (
    NPCaptureSite,
    NPNormalize,
    NPPooling,
    NPTokenSelection,
    NPVectorRead,
    NPVectorSource,
)

logger = logging.getLogger(__name__)

# Attribution for a vector sent inline. The request has no `author` field: a caller who holds the
# catalogue knows who fitted the row it came from, and labelling a reading is their job rather than
# a round trip through this server. Not derived from the id prefix either -- the convention that an
# id reads ``<author>_<name>`` holds for published artifacts, and reading a caller's id as if it did
# would attribute a reading to whoever their prefix happens to name.
DEFAULT_AUTHOR = "custom"

# How many vectors one request may read. The ceiling that matters is distinct captures rather than
# vectors -- a readout costs one capture pass per point, whatever the number of directions projected
# onto it -- but a cap on captures alone would let a caller send megabytes of directions to be told
# no. This bounds both at once, well above the handful a panel can display.
MAX_READS = 32

# Longest quantile table accepted. Two orders of magnitude past the 101 points a percentile per
# integer needs; past that the table is not a calibration curve, it is a payload.
MAX_QUANTILE_LEVELS = 4096

# Artifacts held parsed, keyed by the commit they were read at. Small because it exists to keep a
# conversation's turns from re-reading one file, not to be a mirror of the Hub.
HF_CACHE_SIZE = 32


class VectorRequestError(Exception):
    """A request-supplied vector that cannot be read, with the status the endpoint should send."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _tensor(values: list[float] | None, *, hidden_size: int, vector_id: str, field: str) -> torch.Tensor:
    """One list of floats from the request as a float tensor, checked for width and for being finite.

    An absent or empty list becomes zeros, which is what makes the two mean subtractions optional:
    subtracting zero is the identity, so a fit that never had a scaler ships nothing rather than a
    vector of zeros the caller had to write out.
    """
    if not values:
        return torch.zeros(hidden_size)
    if len(values) != hidden_size:
        raise VectorRequestError(
            f"vector {vector_id!r}: {field} has {len(values)} entries, but this model is {hidden_size}-dimensional"
        )
    tensor = torch.tensor(values, dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all()):
        # NaN propagates through the projection into every reading, so the readout comes back
        # populated and meaningless. Cheaper to refuse than to explain later.
        raise VectorRequestError(f"vector {vector_id!r}: {field} contains a non-finite value")
    return tensor


def _quantiles(payload: NPVectorRead) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """``(levels, pos, neg)`` for a read that sent tables, or three ``None`` for one that did not.

    The two tables come together or not at all -- one of them is not a calibration -- while the
    levels default to evenly spaced, which is what a table of per-percentile landmarks means.
    """
    pos, neg = payload.quantiles_pos, payload.quantiles_neg
    if pos is None and neg is None:
        if payload.quantile_levels is not None:
            raise VectorRequestError(f"vector {payload.id!r}: quantileLevels was sent without any quantile table")
        return None, None, None
    if pos is None or neg is None:
        missing = "quantilesPos" if pos is None else "quantilesNeg"
        raise VectorRequestError(f"vector {payload.id!r}: {missing} is missing; a percentile is read off both tables")
    if len(pos) != len(neg):
        raise VectorRequestError(
            f"vector {payload.id!r}: quantilesPos has {len(pos)} entries and quantilesNeg has {len(neg)}"
        )
    if len(pos) < 2:
        raise VectorRequestError(f"vector {payload.id!r}: a quantile table needs at least two entries")
    if len(pos) > MAX_QUANTILE_LEVELS:
        raise VectorRequestError(f"vector {payload.id!r}: quantile tables are limited to {MAX_QUANTILE_LEVELS} entries")

    levels = payload.quantile_levels
    if levels is None:
        levels = torch.linspace(0.0, 1.0, len(pos)).tolist()
    elif len(levels) != len(pos):
        raise VectorRequestError(
            f"vector {payload.id!r}: quantileLevels has {len(levels)} entries and the tables have {len(pos)}"
        )

    tables = {"quantileLevels": levels, "quantilesPos": pos, "quantilesNeg": neg}
    tensors: dict[str, torch.Tensor] = {}
    for name, values in tables.items():
        tensor = torch.tensor(values, dtype=torch.float32)
        if not bool(torch.isfinite(tensor).all()):
            raise VectorRequestError(f"vector {payload.id!r}: {name} contains a non-finite value")
        # Interpolating an unsorted table is non-monotone, which reorders readings rather than
        # merely misplacing them -- the one failure a bounded scale must not have.
        if bool((tensor.diff() < 0).any()):
            raise VectorRequestError(f"vector {payload.id!r}: {name} must be nondecreasing")
        tensors[name] = tensor
    return tensors["quantileLevels"], tensors["quantilesPos"], tensors["quantilesNeg"]


def _site(value: NPCaptureSite) -> CaptureSite:
    match value:
        case NPCaptureSite.RESID_POST:
            return "resid_post"


def _tokens(value: NPTokenSelection) -> TokenSelection:
    match value:
        case NPTokenSelection.ASSISTANT_TURNS:
            return "assistant_turns"
        case NPTokenSelection.ALL_TURNS:
            return "all_turns"


def _pool(value: NPPooling) -> Pooling:
    match value:
        case NPPooling.MEAN:
            return "mean"
        case NPPooling.LAST:
            return "last"
        case NPPooling.MAX:
            return "max"


def _read_spec(payload: NPVectorRead) -> ReadSpec:
    """The wire's read spec as the internal one. Absent means the defaults every fit so far used.

    Three exhaustive matches rather than three lookup tables, so that a member added to a wire enum
    without a capture implementation behind it fails `pyright` here -- the fall-through becomes
    reachable and the declared return type is unsatisfied -- rather than reaching a caller as a
    reading taken some other way.
    """
    if payload.read is None:
        return ReadSpec()
    return ReadSpec(
        site=_site(payload.read.site),
        tokens=_tokens(payload.read.tokens),
        pool=_pool(payload.read.pool),
    )


def asset_from_payload(payload: NPVectorRead, *, hidden_size: int, n_layers: int) -> VectorAsset:
    """An inline vector definition -> the asset the projection code takes.

    Carries no labels, because the request has none to give: an inline read is a direction and the
    arithmetic to apply it, and what the resulting number is called belongs to whoever holds the
    catalogue. So `author` is the placeholder below and `title` is the caller's own id, which is the
    most this side can honestly say. A `source`-fetched artifact is the other case -- there inference
    is the party that read the labels, and :func:`_asset_from_source` keeps them.

    Raises :class:`VectorRequestError` for anything this model cannot read.
    """
    if payload.direction is None or payload.layer is None:
        # The request model already refuses this; restated so the function holds on its own.
        raise VectorRequestError(f"vector {payload.id!r}: direction and layer are required without a source")
    if not 0 <= payload.layer < n_layers:
        raise VectorRequestError(
            f"vector {payload.id!r}: layer {payload.layer} is out of range for this model, which has "
            f"{n_layers} layers (0-{n_layers - 1})"
        )

    direction = _tensor(payload.direction, hidden_size=hidden_size, vector_id=payload.id, field="direction")
    if float(torch.linalg.vector_norm(direction)) == 0.0:
        # Every reading would come back as exactly `center`, which looks like a model with no
        # variation on whatever this measures rather than like a vector that was never sent.
        raise VectorRequestError(f"vector {payload.id!r}: direction is all zeros")
    for name, value in (("scalePos", payload.scale_pos), ("scaleNeg", payload.scale_neg)):
        if value == 0.0:
            raise VectorRequestError(
                f"vector {payload.id!r}: {name} may not be zero; every value on that side would be inf"
            )

    levels, quantiles_pos, quantiles_neg = _quantiles(payload)
    normalize: Normalize = "l2" if payload.normalize is NPNormalize.L2 else "none"
    return VectorAsset(
        id=payload.id,
        author=DEFAULT_AUTHOR,
        title=payload.id,
        layer=payload.layer,
        normalize=normalize,
        center=payload.center,
        scale_pos=payload.scale_pos,
        scale_neg=payload.scale_neg,
        render=RenderConditions(
            blank_system_prompt=payload.render.blank_system_prompt,
            template_kwargs=dict(payload.render.template_kwargs),
        ),
        read=_read_spec(payload),
        direction=direction,
        scaler_mean=_tensor(payload.pre_norm_mean, hidden_size=hidden_size, vector_id=payload.id, field="preNormMean"),
        pca_mean=_tensor(payload.post_norm_mean, hidden_size=hidden_size, vector_id=payload.id, field="postNormMean"),
        quantile_levels=levels,
        quantiles_pos=quantiles_pos,
        quantiles_neg=quantiles_neg,
    )


def _resolve_revision(source: NPVectorSource) -> str:
    """The commit ``source`` names, so what gets cached and reported is one immutable artifact.

    Asked every time rather than cached: a branch name is a moving target, and a process that
    resolved it once at startup would serve last week's fit under this week's name for as long as
    it stays up. One request to the Hub is cheap next to a forward pass, and it is also where a
    missing or gated repo is reported as such rather than as a missing file.
    """
    from huggingface_hub import HfApi

    try:
        info = HfApi().repo_info(repo_id=source.hf_repo_id, revision=source.revision, repo_type="model")
    except Exception as exc:  # noqa: BLE001  # the Hub's exception tree is wide; that it failed is what matters
        # The Hub's own text is logged rather than returned. It carries request urls and cache
        # paths from this pod, and the caller can act on none of it -- what they need is which
        # repo and revision this server could not read, which the message states.
        logger.warning(f"Could not read {source.hf_repo_id} at {source.revision or 'default'}", exc_info=True)
        raise VectorRequestError(
            f"Could not read {source.hf_repo_id} at revision {source.revision or 'default'}",
            status_code=502,
        ) from exc
    if not info.sha:
        raise VectorRequestError(f"{source.hf_repo_id} reported no commit for revision {source.revision or 'default'}")
    return info.sha


@lru_cache(maxsize=HF_CACHE_SIZE)
def _artifact_at(repo_id: str, folder: str, revision: str) -> VectorAsset:
    """Download and parse one published artifact. Cached, keyed by the commit it was read at.

    Safe to share across requests because an :class:`VectorAsset` is frozen and the projection only
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
            # The commit resolved, so the repo is there and this is a folder that holds no artifact.
            raise VectorRequestError(f"{repo_id}/{prefix} at {revision} has no {name}") from exc
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not download {filename} from {repo_id} at {revision}", exc_info=True)
            raise VectorRequestError(
                f"Could not download {filename} from {repo_id} at {revision}", status_code=502
            ) from exc

    # Both files land in the same snapshot directory, which is the shape `load_vector` reads.
    directory = os.path.dirname(paths[MANIFEST_FILENAME])
    try:
        asset = load_vector(prefix.rsplit("/", 1)[-1] or repo_id, directory)
    except Exception as exc:  # noqa: BLE001
        # `load_vector` names the file it rejected, which here is this pod's Hub cache path -- so
        # the reason goes to the log and the caller is told which artifact was bad. They own the
        # artifact and can validate it against `load_vector` directly; the path is ours.
        logger.warning(f"{repo_id}/{prefix} at {revision} is not a valid vector", exc_info=True)
        raise VectorRequestError(f"{repo_id}/{prefix} at {revision} is not a valid vector") from exc
    return dataclasses.replace(asset, source_revision=revision)


async def _asset_from_source(payload: NPVectorRead, *, hidden_size: int, n_layers: int) -> VectorAsset:
    """A published artifact -> the asset the projection code takes, under the caller's id."""
    source = payload.source
    assert source is not None  # the caller checked; this keeps the type narrow
    if ".." in source.hf_folder:
        raise VectorRequestError(f"vector {payload.id!r}: hfFolder may not contain '..'")

    revision = await asyncio.to_thread(_resolve_revision, source)
    asset = await asyncio.to_thread(_artifact_at, source.hf_repo_id, source.hf_folder, revision)

    if asset.hidden_size != hidden_size:
        raise VectorRequestError(
            f"vector {payload.id!r}: {source.hf_repo_id}/{source.hf_folder} is {asset.hidden_size}-dimensional, "
            f"but this model is {hidden_size}-dimensional"
        )
    if not 0 <= asset.layer < n_layers:
        raise VectorRequestError(
            f"vector {payload.id!r}: {source.hf_repo_id}/{source.hf_folder} was fitted at layer {asset.layer}, "
            f"which this model does not have ({n_layers} layers)"
        )
    return dataclasses.replace(asset, id=payload.id)


async def resolve_request_reads(
    payloads: list[NPVectorRead],
    *,
    hidden_size: int,
    n_layers: int,
) -> list[VectorAsset]:
    """Every vector sent with a request, as assets, in the order they were sent.

    A duplicate id is refused rather than resolved: two readouts under one id leaves a caller
    unable to say which fit produced which numbers, and last-one-wins would answer silently for
    the wrong one.
    """
    if len(payloads) > MAX_READS:
        raise VectorRequestError(f"At most {MAX_READS} vectors may be read in one request, got {len(payloads)}")

    seen: set[str] = set()
    assets: list[VectorAsset] = []
    for payload in payloads:
        if payload.id in seen:
            raise VectorRequestError(f"Duplicate vector id {payload.id!r} in this request")
        seen.add(payload.id)
        if payload.source is None:
            assets.append(asset_from_payload(payload, hidden_size=hidden_size, n_layers=n_layers))
        else:
            assets.append(await _asset_from_source(payload, hidden_size=hidden_size, n_layers=n_layers))
            logger.info(
                f"Resolved vector '{payload.id}' from {payload.source.hf_repo_id}/{payload.source.hf_folder} "
                f"at {assets[-1].source_revision}"
            )
    return assets
