"""Which activation a fitted lens decodes, and how to capture it on the served model.

A ``J_bar`` is a ``[d_model, d_model]`` matrix and says nothing about what it multiplies. On a
conventional trunk that is harmless -- there is one residual per layer and every lens is fitted on it
-- so the read-out hard-coded ``resid_post`` for as long as every served model had one. A
hyper-connection trunk (DeepSeek-V4's ``hc_mult = 4``) breaks that in the worst available way: the
trunk carries ``[tokens, n_streams, d_model]``, and the stream mean, the stream sum and any single
stream are all ``d_model``-wide, so a lens fitted on one of them loads, runs, and reads out
confidently in the wrong space. Nothing about the shapes distinguishes them and no assertion can.

So the artifact says. ``provenance`` in the ``.pt`` carries ``capture_point`` (where the fit hooked
the model) and ``stream_reduce`` (how it collapsed the stack), written by
``utils/.../jlens/convert-external-lens.py`` for a lens fitted elsewhere and by the fitter for one of
ours. This module turns that declaration into an :class:`~interp_engine.Address` and a reduction for
the model in front of us, and refuses when a multi-stream trunk is served a lens that declares
nothing -- which is the case the whole file exists for.

The same declaration decides where an *intervention* lands, not only where the read-out is taken.
A steer, an ablation or a swap is derived from the lens and has to be written to the tensor the
read-out will show, or the two disagree with nothing to say so -- see :attr:`LensResidualSpec.write_stream`.

``capture_point`` is deliberately NOT an engine address. It names the boundary architecturally, and
which address serves that boundary depends on the trunk: ``block_output`` is ``resid_post`` on a
conventional model and ``resid_streams`` on a hyper-connection one, because the engine refuses
``resid_post`` outright on the latter (see ``interp_engine.residual_basis``). Keeping the artifact in
architectural terms also means a published lens is readable by a consumer that does not have this
engine, and survives a canonical rename in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from interp_engine import STREAM_REDUCTIONS, Address, ResidualBasis, reduce_streams

# Where a fit hooked the model, and the address that boundary corresponds to as
# ``(single-stream trunk, hyper-connection trunk)``.
#
# The multi-stream column is not a translation of the first, it is the point that means the same
# thing on a trunk shaped differently: ``attn_in`` is ``resid_pre`` where contributions are added,
# and ``attn_stream_collapse`` -- the ``d_model`` vector attention actually reads -- where the block
# collapses the streams before its sublayer. ``attn_out``/``mlp_out`` are a sublayer's own output,
# which is ``d_model``-wide however the trunk then carries it, so they are the same on both and take
# no reduction.
#
# DUPLICATED as `CAPTURE_POINTS` in utils/neuronpedia-utils/neuronpedia_utils/jlens/
# convert-external-lens.py, which is a separate uv project and cannot import this. Change both or
# neither: a value it can write and this cannot read is a lens that loads and will not serve.
CAPTURE_POINT_ADDRESSES: dict[str, tuple[str, str]] = {
    "block_output": ("resid_post", "resid_streams"),
    "attn_out": ("attn_out", "attn_out"),
    "mlp_out": ("mlp_out", "mlp_out"),
    "attn_in": ("resid_pre", "attn_stream_collapse"),
    "mlp_in": ("resid_mid", "mlp_stream_collapse"),
}

# What a lens with no declaration means. Every lens fitted before `provenance` carried these fields
# -- which is all of ours -- was fitted on the block output of a single-stream trunk, because that is
# the only thing `fit_lens.py` has ever hooked and the only kind of model the endpoint has served.
# Assuming it on a MULTI-stream trunk is the one thing that must not happen, and `resolve` refuses
# there rather than falling back here.
DEFAULT_CAPTURE_POINT = "block_output"


class LensSpaceUnknown(ValueError):
    """A lens must declare where it was fitted to be served on this model, and does not.

    Its own type because the endpoint turns it into a 400 with the message intact: the fix is to
    re-run the converter over the artifact, which is a thing the operator can do, and not something
    to report as a generic load failure.
    """


@dataclass(frozen=True)
class LensResidualSpec:
    """The activation a lens decodes: a capture point plus how to collapse a stream stack."""

    capture_point: str = DEFAULT_CAPTURE_POINT
    stream_reduce: str = "none"
    stream_index: int | None = None

    def __post_init__(self) -> None:
        if self.capture_point not in CAPTURE_POINT_ADDRESSES:
            raise LensSpaceUnknown(
                f"unknown lens capture point {self.capture_point!r} (expected one of {sorted(CAPTURE_POINT_ADDRESSES)})"
            )
        if self.stream_reduce not in STREAM_REDUCTIONS:
            raise LensSpaceUnknown(
                f"unknown lens stream reduction {self.stream_reduce!r} (expected one of {sorted(STREAM_REDUCTIONS)})"
            )
        if (self.stream_reduce == "select") != (self.stream_index is not None):
            raise LensSpaceUnknown(
                f"stream_reduce={self.stream_reduce!r} and stream_index={self.stream_index!r} disagree: "
                "an index is required with 'select' and meaningless otherwise"
            )

    @property
    def reduces(self) -> bool:
        """Whether serving this lens collapses a stream axis, i.e. the trunk is multi-stream."""
        return self.stream_reduce != "none"

    @property
    def write_stream(self) -> int | None:
        """Which stream an intervention writes, or None to write the stack as a whole.

        ``select`` names one ``d_model`` vector of the stack, so an intervention derived from that
        lens has exactly one stream to land in. ``mean`` and ``sum`` name a mixture, which no single
        stream carries, and the write goes to every stream at once -- correct rather than
        approximate for the two ops that are linear in the residual, since removing each stream's
        component along a direction removes the mixture's, and swapping each stream's projection
        swaps the mixture's. An additive steer is the one that is only an analogue: it measures its
        magnitude against each stream's own norm rather than against the mixture's, so a given
        strength is the same fraction of the tensor being written either way.

        ``none`` is a conventional trunk, which has no stream axis for a write to name.
        """
        return self.stream_index if self.stream_reduce == "select" else None

    def point_name(self, n_streams: int) -> str:
        """The engine point name for this capture point on a trunk carrying ``n_streams`` streams."""
        single, multi = CAPTURE_POINT_ADDRESSES[self.capture_point]
        return single if n_streams <= 1 else multi

    def address(self, layer: int, n_streams: int) -> Address:
        return Address(self.point_name(n_streams), int(layer))

    def validate(self, basis: ResidualBasis) -> None:
        """Refuse now if this declaration and the served trunk disagree.

        Called once per request, before anything runs, because the disagreement it catches has no
        later symptom: a lens fitted on the stream mean and read out with no reduction produces a
        full read-out of plausible tokens.
        """
        basis.require_stream_reduction(
            self.stream_reduce,
            self.stream_index,
            point=self.point_name(basis.n_streams),
        )

    def reduce(self, tensor: torch.Tensor, n_streams: int) -> torch.Tensor:
        """Collapse a capture to the ``[..., d_model]`` vector the lens decodes.

        For the paths that hold captures in this process -- the eager backend, and vLLM's
        residual-shipping iterator. vLLM's fused read-out reduces inside the worker instead, from the
        same declaration passed down in the read-out spec, so both arms apply the same reduction to
        the same tensor rather than one of them being the definition.
        """
        return reduce_streams(
            tensor,
            self.stream_reduce,
            index=self.stream_index,
            n_streams=n_streams if self.reduces else None,
        )

    def describe(self) -> str:
        """One line for a startup log, since which space a lens reads is worth saying out loud."""
        index = "" if self.stream_index is None else f"[{self.stream_index}]"
        return f"{self.capture_point} reduced by {self.stream_reduce}{index}"


# What every lens on a conventional trunk decodes, whether or not its file says so. The read-out
# functions default to it, so a signature carrying this argument means what it meant before there was
# a choice, and the ~40 existing artifacts keep loading untouched.
BLOCK_OUTPUT = LensResidualSpec()


def block_output_point(model: Any) -> str:
    """The engine point that carries the block output on the trunk ``model`` serves.

    For asking whether an endpoint CAN run, ahead of any lens being resolved -- which is why it reads
    ``residual_basis`` defensively and assumes one stream when the object does not say. The gate it
    feeds (``engine_adapter.assert_residual_available``) is duck-typed throughout for the same
    reason: a capability question should not require a fully built model to ask, and the stubs that
    ask it are testing something else entirely.
    """
    basis = getattr(model, "residual_basis", None)
    return BLOCK_OUTPUT.point_name(int(getattr(basis, "n_streams", 1) or 1))


def from_provenance(provenance: Any) -> LensResidualSpec | None:
    """Read the residual declaration out of a lens file's ``provenance``, or None if absent.

    Absent is the ordinary case for a lens fitted before these fields existed, and the caller decides
    what that means -- harmless on a single-stream trunk, fatal on a multi-stream one. A PRESENT but
    unusable declaration raises instead: it was written by something, so a silent fallback would
    serve a space the artifact explicitly denies.
    """
    if not isinstance(provenance, dict):
        return None
    capture_point = provenance.get("capture_point")
    stream_reduce = provenance.get("stream_reduce")
    if capture_point is None and stream_reduce is None:
        return None
    index = provenance.get("stream_index")
    return LensResidualSpec(
        capture_point=str(capture_point or DEFAULT_CAPTURE_POINT),
        stream_reduce=str(stream_reduce or "none"),
        stream_index=None if index is None else int(index),
    )


def resolve_residual_spec(declared: LensResidualSpec | None, basis: ResidualBasis) -> LensResidualSpec:
    """The spec to serve this request with, given what the loaded lens declares (if anything).

    On a conventional trunk an absent declaration is filled in, because there is only one thing it
    could have meant and roughly forty existing artifacts mean it.

    On a hyper-connection trunk it is refused, and this is the point of the module. Every candidate
    reduction has the same shape and the same ``d_model``, so serving a guess produces a complete,
    confident read-out in a space the lens was not fitted in -- there is no downstream check that
    fails and no artifact of the output that looks wrong. The fix is a line in the file, so the error
    says so.
    """
    if declared is not None:
        declared.validate(basis)
        return declared
    if basis.n_streams <= 1:
        BLOCK_OUTPUT.validate(basis)
        return BLOCK_OUTPUT
    raise LensSpaceUnknown(
        f"This model carries {basis.n_streams} parallel residual streams, so a capture at every "
        "layer is a stack rather than one residual, and a lens read-out has to say which d_model "
        "vector of that stack it decodes -- the stream mean, the sum and each individual stream are "
        "all d_model-wide and indistinguishable once captured. The loaded lens declares nothing, and "
        "guessing would read out fluent tokens from the wrong space. Record it "
        "in the file's provenance with utils/neuronpedia-utils/neuronpedia_utils/jlens/"
        "convert-external-lens.py (--capture-point / --stream-reduce)."
    )
