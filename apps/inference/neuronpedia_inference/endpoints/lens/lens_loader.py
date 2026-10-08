"""Loading and storing the fitted Jacobian lens for the served model.

A Jacobian lens is a set of per-layer matrices ``J_bar_l`` (shape
``[d_model, d_model]``) fitted offline by ``utils/.../jlens`` and saved as a
single ``*_jacobian_lens.pt``. Applying it is a single matmul per layer, so we do
not depend on the ``jlens`` package here: we just load the tensors and keep a
small standalone holder (:class:`LoadedJacobianLens`).

At server startup we resolve the neuronpedia model id, then load the lens either
from a local override directory (``--JLENS_SOURCE``) or by downloading it from a
Hugging Face model repo (default ``neuronpedia/jacobian-lens``) at
``<np_model_id>/jlens/<dataset>/<slug>_jacobian_lens.pt``. Loading is best-effort:
a failure never crashes startup, it just makes JACOBIAN_LENS requests return an
error (LOGIT_LENS does not need a lens).

A J++ Lens (JPP_LENS) has the same form and is loaded the same way, as a second
:class:`LensKind` with its own store, HF path and engine set. It loads only when
``JPP_LENS`` is set.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch

from neuronpedia_inference.endpoints.lens.residual_spec import LensResidualSpec, from_provenance

logger = logging.getLogger(__name__)

# Where downloaded lenses are cached on disk between restarts.
_DOWNLOAD_CACHE_DIR = os.environ.get(
    "JLENS_CACHE_DIR",
    "/tmp/neuronpedia-jlens-cache",  # noqa: S108
)


def _repo_root_np_model_to_hf() -> Path | None:
    """Locate ``np_model_to_hf.json`` at the repo root, if present.

    The file lives at the workspace root (the user copies it there). From this
    module the root is five parents up:
    ``apps/inference/neuronpedia_inference/endpoints/lens/lens_loader.py``.
    """
    candidate = Path(__file__).resolve().parents[5] / "np_model_to_hf.json"
    return candidate if candidate.exists() else None


def _load_np_to_hf_mapping() -> dict[str, str] | None:
    path = _repo_root_np_model_to_hf()
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def _slug(hf_model_id: str) -> str:
    """Filesystem-safe stem used by jlens for the lens filename.

    Mirrors ``fit_lens.py::_slug`` so we can construct the exact HF path.
    """
    base = hf_model_id.rstrip("/").split("/")[-1]
    return re.sub(r"[^0-9A-Za-z._-]+", "-", base).strip("-") or "model"


@dataclass
class LensResolution:
    np_model_id: str
    hf_model_id: str | None


def resolve_neuronpedia_model_id(config: object, args: object) -> LensResolution:
    """Resolve the neuronpedia model id (and HF id when known) for lens lookup.

    Resolution order:
        1. Explicit ``--NEURONPEDIA_MODEL_ID`` argument (always wins).
        2. ``np_model_to_hf.json`` at the repo root: match the loaded model
           against the np->hf mapping.
        3. Otherwise raise (caller turns this into a non-fatal load failure).
    """
    explicit = getattr(args, "neuronpedia_model_id", None)
    mapping = _load_np_to_hf_mapping()

    model_id = getattr(config, "model_id", None)
    override_model_id = getattr(config, "override_model_id", None)
    custom_hf_model_id = getattr(config, "custom_hf_model_id", None)

    if explicit:
        hf_id = None
        if mapping is not None:
            hf_id = mapping.get(explicit)
        hf_id = hf_id or custom_hf_model_id
        return LensResolution(np_model_id=explicit, hf_model_id=hf_id)

    if mapping is None:
        raise ValueError(
            "Cannot resolve neuronpedia model id: np_model_to_hf.json not found at "
            "the repo root and --NEURONPEDIA_MODEL_ID was not provided."
        )

    # The server's model_id is normally already a neuronpedia model id.
    if model_id in mapping:
        return LensResolution(np_model_id=model_id, hf_model_id=mapping[model_id])

    # Otherwise reverse-map by the HF id we actually loaded.
    hf_candidates = [c for c in (custom_hf_model_id, override_model_id, model_id) if c is not None]
    for candidate in hf_candidates:
        for np_id, hf_id in mapping.items():
            if candidate in (hf_id, np_id):
                return LensResolution(np_model_id=np_id, hf_model_id=hf_id)

    raise ValueError(
        f"Cannot resolve neuronpedia model id for loaded model "
        f"(model_id={model_id!r}, override={override_model_id!r}, "
        f"custom_hf={custom_hf_model_id!r}). Pass --NEURONPEDIA_MODEL_ID."
    )


# Bytes of ``J_bar`` to keep on the compute device when no budget is supplied. The server
# measures a real one at startup (``resolve_jlens_gpu_budget_bytes``); this is the standalone
# default, sized to hold an ordinary lens whole -- 3.1 GiB at d_model=5120 x 63 layers in
# bf16 -- while still bounding the 9.9 GiB a d_model=8192 x 80-layer fit would want.
#
# A BYTE ceiling rather than a layer count, because a layer count cannot tell 8 x 5120^2
# from 8 x 8192^2, and the quantity that has to fit on the card is bytes. The count this
# replaced was 8, which on any lens with more than 8 fitted layers was not a partial cache
# but effectively no cache at all -- see the eviction note in ``_admit``.
DEFAULT_DEVICE_BUDGET_BYTES = 4 * 1024**3


def _normalize_device(device: torch.device) -> torch.device:
    """Resolve an index-less ``cuda`` to the concrete ``cuda:<n>`` a tensor would report.

    ``torch.device("cuda") != torch.device("cuda:0")``, but a tensor moved to the former
    reports the latter -- and ``--device cuda`` is how these pods are configured. Comparing
    the two forms directly makes every cache hit miss, so the lens is re-copied a layer at a
    time on every read-out batch: 27ms x 63 layers, which is the entire cost residency is
    meant to remove, silently still being paid.
    """
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


class LoadedJacobianLens:
    """A fitted Jacobian lens loaded from disk: per-layer ``J_bar`` + metadata.

    Standalone (does not import the ``jlens`` package).

    Jacobians are held at ``dtype`` (the served model's dtype) in host RAM, and
    :meth:`place_on_device` then uploads as many as ``device_budget_bytes`` allows and
    fixes ``transport_device`` as the place the transport runs. That device is a property
    of the LENS, not of the residual it carries: on the vLLM backend the
    residuals arrive from the worker as CPU tensors, and transporting them where they
    landed meant a 396 GFLOP host matmul per read-out batch at d_model=5120 x 63 layers
    (4.3s measured, against 4.8ms for the same sweep on an A100).

    On dtype: the upstream fitter writes fp16 unconditionally (``jlens/lens.py``'s
    ``save`` casts, whatever ``--dtype`` the fit ran under), so widening on load recovers
    no precision -- the values were already rounded -- while doubling a ``d_model^2``
    per-layer footprint. It is not held AS fp16 either: fp16 tops out at 65504, and the
    transport is a ``d_model``-long accumulation against late-layer residuals, so bf16's
    wider exponent is what makes the matmul safe. Hence "the model's dtype", not "the
    file's".
    """

    def __init__(
        self,
        jacobians: dict[int, torch.Tensor],
        *,
        source_layers: list[int],
        n_prompts: int,
        d_model: int,
        dtype: torch.dtype = torch.bfloat16,
        device_budget_bytes: int = DEFAULT_DEVICE_BUDGET_BYTES,
        residual: LensResidualSpec | None = None,
    ) -> None:
        self.dtype = dtype
        self.jacobians = {int(layer): J.to(dtype) for layer, J in jacobians.items()}
        self.source_layers = sorted(self.jacobians)
        self.n_prompts = n_prompts
        self.d_model = d_model
        # Which activation these matrices multiply, as the file declares it. None for a lens fitted
        # before the field existed, which the endpoint reads as the block output of a single-stream
        # trunk and refuses to read as anything on a multi-stream one -- see
        # `residual_spec.resolve`. Held rather than resolved here because resolving it needs the
        # served model's residual basis, which is not knowable at load time on the vLLM backend.
        self.residual = residual
        self.device_budget_bytes = max(0, int(device_budget_bytes))
        # None means "transport wherever the residual already is", which is what a CPU-only
        # pod wants and what every caller got before `place_on_device` existed.
        self.transport_device: torch.device | None = None
        # Set once the matrices are resident in the vLLM worker(s) instead of here. That
        # backend transports where the residuals were captured, so this process keeps the
        # host copy for metadata and the eager path but never uploads to its own device.
        self.worker_resident = False
        # Insertion-ordered, and deliberately NOT reordered on a hit -- see `_admit`.
        self._device_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._device_bytes = 0

    @classmethod
    def load(
        cls,
        path: str,
        dtype: torch.dtype = torch.bfloat16,
        device_budget_bytes: int = DEFAULT_DEVICE_BUDGET_BYTES,
    ) -> LoadedJacobianLens:
        # `weights_only=True` still admits the primitives and containers `provenance` is made of,
        # so the declaration below rides along without loosening the unpickler.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "J" not in checkpoint and "jacobians" in (checkpoint.get("parameters") or {}):
            # A J++ Lens file as safety-research/jpp_lens writes it: the same maps, keyed apart.
            config = checkpoint.get("config") or {}
            return cls(
                jacobians=checkpoint["parameters"]["jacobians"],
                source_layers=list(checkpoint.get("source_layers", [])),
                n_prompts=int(config.get("num_prompts_trained_on", 0)),
                d_model=int(config["d_model"]),
                dtype=dtype,
                device_budget_bytes=device_budget_bytes,
            )
        if "J" not in checkpoint:
            raise ValueError(f"{path} is not a Jacobian lens file (keys: {sorted(checkpoint)!r})")
        return cls(
            jacobians=checkpoint["J"],
            source_layers=list(checkpoint.get("source_layers", [])),
            n_prompts=int(checkpoint.get("n_prompts", 0)),
            d_model=int(checkpoint["d_model"]),
            dtype=dtype,
            device_budget_bytes=device_budget_bytes,
            residual=from_provenance(checkpoint.get("provenance")),
        )

    @property
    def resident_bytes(self) -> int:
        """Host bytes the Jacobians occupy. Worth logging: it is the largest single
        allocation the lens makes, and it scales with ``n_layers * d_model**2``."""
        return sum(J.numel() * J.element_size() for J in self.jacobians.values())

    @property
    def device_resident_bytes(self) -> int:
        """Bytes of device-side copies currently held, against ``device_budget_bytes``."""
        return self._device_bytes

    def fits_on_device(self) -> bool:
        """Whether every fitted layer can be resident at once.

        The case worth knowing about: below it, some layers are re-copied per read-out
        batch, so the transport goes from a matmul to a PCIe transfer.
        """
        return self.resident_bytes <= self.device_budget_bytes

    def place_on_device(self, device: torch.device, *, device_budget_bytes: int | None = None) -> None:
        """Upload what the budget allows and run every later transport on ``device``.

        Eager rather than lazy, because the alternative is a first-request stall (uploading
        63 layers costs ~300ms) and, worse, a lie to the memory accounting: the transient
        request budget is measured from free VRAM at the end of startup, so a lens that
        uploads itself later takes its gigabytes back out of memory already promised to
        concurrent requests.

        Layers are admitted in ascending order and admission stops at the first one that
        does not fit, rather than letting :meth:`_admit` evict to make room -- the result is
        the same resident prefix, without churning the cache to arrive at it.
        """
        if device_budget_bytes is not None:
            self.device_budget_bytes = max(0, int(device_budget_bytes))
        if self.device_budget_bytes <= 0 or device.type == "cpu":
            self.transport_device = None
            return
        device = _normalize_device(device)
        self.transport_device = device
        for layer in self.source_layers:
            host = self.jacobians[layer]
            if self._device_bytes + host.numel() * host.element_size() > self.device_budget_bytes:
                break
            self._admit(layer, host.to(device))

    def _admit(self, layer: int, tensor: torch.Tensor) -> None:
        """Hold ``tensor`` as ``layer``'s device copy, evicting to stay inside the budget.

        Eviction drops the MOST recently admitted entry, which is the opposite of the usual
        choice and is what the access pattern calls for. A read-out sweeps every fitted layer
        in the same ascending order on every batch, and under a cyclic scan LRU evicts
        precisely the entry wanted soonest: with a cache one layer short of the sweep it
        misses on all of them, so the cache does nothing at all. Dropping the newest instead
        keeps a stable prefix resident, so a lens too large to fit degrades in proportion to
        how much of it fits rather than falling off a cliff.
        """
        self._discard(layer)  # keeps the byte count right if this layer is already held
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > self.device_budget_bytes:
            # A single layer over budget: hand it back uncached rather than evicting
            # everything else to make room for something that still would not fit.
            return
        while self._device_bytes + nbytes > self.device_budget_bytes and self._device_cache:
            _, evicted = self._device_cache.popitem(last=True)
            self._device_bytes -= evicted.numel() * evicted.element_size()
        self._device_cache[layer] = tensor
        self._device_bytes += nbytes

    def _discard(self, layer: int) -> None:
        stale = self._device_cache.pop(layer, None)
        if stale is not None:
            self._device_bytes -= stale.numel() * stale.element_size()

    def placed_jacobians(self) -> dict[int, torch.Tensor]:
        """Each layer's ``J_bar`` where placement left it: the device copy if resident, else the host one.

        What a read-out in this process is handed. A host copy is moved per batch, one layer at a
        time, so a lens larger than its budget never has to fit on the device at once.
        """
        return {layer: self._device_cache.get(layer, host) for layer, host in self.jacobians.items()}

    def jacobian_on(self, layer: int, device: torch.device) -> torch.Tensor:
        device = _normalize_device(device)
        cached = self._device_cache.get(layer)
        if cached is not None:
            if cached.device == device:
                return cached
            # A caller wanting the layer somewhere else gets a copy, but the resident one
            # stays put. Dropping it here let one steer/swap request -- which asks for every
            # layer on the CPU, since vLLM's unembedding rows land there -- silently undo the
            # whole startup placement, so every read-out afterwards re-uploaded the lens.
            return self.jacobians[layer].to(device)
        moved = self.jacobians[layer].to(device)
        if moved is self.jacobians[layer]:
            # ``.to`` returned the storage tensor itself, so the transport is running where
            # the lens already lives (a CPU-only pod). There is no second copy to hold, and
            # counting one would ration memory nobody allocated.
            return moved
        self._admit(layer, moved)
        return moved


class JacobianLensStore:
    """Process-wide holder for the loaded lens and its load status."""

    _instance: LoadedJacobianLens | None = None
    _status: str = "not_loaded"  # one of: not_loaded, loaded, skipped, error
    _error: str | None = None
    _np_model_id: str | None = None

    @classmethod
    def set_loaded(cls, lens: LoadedJacobianLens, np_model_id: str) -> None:
        cls._instance = lens
        cls._status = "loaded"
        cls._error = None
        cls._np_model_id = np_model_id

    @classmethod
    def set_skipped(cls) -> None:
        cls._instance = None
        cls._status = "skipped"
        cls._error = None

    @classmethod
    def set_error(cls, error: str) -> None:
        cls._instance = None
        cls._status = "error"
        cls._error = error

    @classmethod
    def get(cls) -> LoadedJacobianLens | None:
        return cls._instance

    @classmethod
    def status(cls) -> str:
        return cls._status

    @classmethod
    def error(cls) -> str | None:
        return cls._error


class JppLensStore(JacobianLensStore):
    """Process-wide holder for the J++ Lens, apart from the Jacobian lens."""

    _instance: LoadedJacobianLens | None = None
    _status: str = "not_loaded"
    _error: str | None = None
    _np_model_id: str | None = None


@dataclass(frozen=True)
class LensKind:
    """One kind of fitted ``J_bar`` lens: where it is published and where it is held."""

    label: str
    store: type[JacobianLensStore]
    # The engine's J_bar set name (`LensSpec.jacobian_set`).
    engine_set: str
    # The HF path is `<np_model_id>/<folder>/<dataset>/<slug>_<stem>.pt`.
    folder: str
    stem: str
    # The `args` attributes are `<arg_prefix>_source`, `_dataset`, `_hf_repo` and `_hf_path`.
    arg_prefix: str
    default_repo: str


JACOBIAN_LENS_KIND = LensKind(
    "Jacobian lens", JacobianLensStore, "default", "jlens", "jacobian_lens", "jlens", "neuronpedia/jacobian-lens"
)
JPP_LENS_KIND = LensKind("J++ lens", JppLensStore, "jpp", "jpp", "jpp_lens", "jpp", "neuronpedia/jacobian-lens")
LENS_KINDS = (JACOBIAN_LENS_KIND, JPP_LENS_KIND)


def _find_local_lens_file(directory: str, stem: str = "jacobian_lens") -> str:
    matches = sorted(glob.glob(os.path.join(directory, f"*_{stem}.pt")))
    if not matches:
        raise FileNotFoundError(f"No *_{stem}.pt found in local lens source directory: {directory}")
    if len(matches) > 1:
        logger.warning("Multiple lens files in %s, using the first: %s", directory, matches[0])
    return matches[0]


def _list_hf_lens_path(repo_id: str, prefix: str) -> str | None:
    """List repo files under ``prefix`` and return the first lens ``.pt``, if any.

    Best-effort: returns ``None`` on any failure so the caller can surface a clear
    error after exhausting the deterministic candidates.
    """
    try:
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as exc:  # noqa: BLE001
        logger.warning("HF list_repo_files failed for %s: %s", repo_id, exc)
        return None
    matches = sorted(f for f in files if f.startswith(f"{prefix}/") and f.endswith(".pt"))
    return matches[0] if matches else None


def _download_lens_from_hf(
    repo_id: str,
    np_model_id: str,
    dataset: str,
    hf_model_id: str | None,
    explicit_path: str | None,
    *,
    folder: str = "jlens",
    stem: str = "jacobian_lens",
) -> str:
    """Download the lens from a HF model repo and return the local cache path.

    When ``explicit_path`` is given it is used verbatim. Otherwise we try the
    deterministic candidates under ``<np_model_id>/<folder>/<dataset>/``:
    ``<slug>_<stem>.pt`` first, then ``<slug>_<stem>_n1000.pt``,
    and finally fall back to the first ``.pt`` listed under that directory.
    """
    from huggingface_hub import hf_hub_download

    prefix = f"{np_model_id}/{folder}/{dataset}"

    candidate_paths: list[str] = []
    if explicit_path:
        candidate_paths.append(explicit_path)
    else:
        slug = _slug(hf_model_id) if hf_model_id is not None else _slug(np_model_id)
        candidate_paths.append(f"{prefix}/{slug}_{stem}.pt")
        candidate_paths.append(f"{prefix}/{slug}_{stem}_n1000.pt")

    last_error: Exception | None = None
    for filename in candidate_paths:
        try:
            logger.info("Downloading Jacobian lens from HF %s: %s", repo_id, filename)
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                cache_dir=_DOWNLOAD_CACHE_DIR,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("HF download failed for %s/%s: %s", repo_id, filename, exc)

    # Fall back to listing the directory to discover the exact filename.
    if not explicit_path:
        listed = _list_hf_lens_path(repo_id, prefix)
        if listed is not None and listed not in candidate_paths:
            try:
                logger.info(
                    "Downloading Jacobian lens from HF (from listing) %s: %s",
                    repo_id,
                    listed,
                )
                return hf_hub_download(
                    repo_id=repo_id,
                    filename=listed,
                    repo_type="model",
                    cache_dir=_DOWNLOAD_CACHE_DIR,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("HF download failed for %s/%s: %s", repo_id, listed, exc)

    raise FileNotFoundError(
        f"Could not download a Jacobian lens for '{np_model_id}' "
        f"(dataset='{dataset}') from HF repo '{repo_id}'. "
        f"Tried: {candidate_paths}. Last error: {last_error}"
    )


def _lens_dtype(config: object) -> torch.dtype:
    """The dtype to hold ``J_bar`` at: the served model's, falling back to bf16.

    ``--model_dtype auto`` (and anything unrecognized) lands on bf16 rather than the
    checkpoint's fp16 — see :class:`LoadedJacobianLens` on why the file's dtype is the
    wrong thing to inherit. fp32 is honored where a pod asks for it, since there the
    residuals are fp32 anyway and the transport would only have to widen again.
    """
    from neuronpedia_inference.shared import STR_TO_DTYPE

    return STR_TO_DTYPE.get(str(getattr(config, "model_dtype", "")), torch.bfloat16)


def _device_budget_bytes(config: object, args: object) -> int:
    """GPU bytes the lens may keep resident, measured against what is left of the card.

    The SAE cache is loaded by this point but may not be warmed, so the part of its
    residency budget it has not claimed yet is held back -- it is free memory that is
    already owed to something else.
    """
    from neuronpedia_inference.sae_cache import sae_cache
    from neuronpedia_inference.startup_memory import resolve_jlens_gpu_budget_bytes

    return resolve_jlens_gpu_budget_bytes(
        getattr(args, "jlens_gpu_budget_gib", None),
        device=getattr(config, "device", None) or getattr(args, "device", None),
        reserved_bytes=max(0, sae_cache.budget_bytes - sae_cache.resident_bytes),
    )


async def place_jacobian_lens_on_worker(
    config: object, args: object, model: object, kind: LensKind = JACOBIAN_LENS_KIND
) -> bool:
    """Upload the lens into the vLLM worker(s), where the residuals already are.

    The vLLM counterpart of :func:`place_jacobian_lens_on_device`, and the reason the
    read-out no longer ships residuals anywhere: with ``J_bar`` resident beside the model
    weights, ``lens_capture_readout`` transports and unembeds in the same process that
    captured the rows. Uploading here instead of to this process's device is not extra
    memory -- it is the same card, holding one copy instead of the other -- but note that
    under tensor parallelism every rank keeps a full copy, since ``J_bar`` is not sharded,
    and the budget below is measured on this process's device as a stand-in for all of them.

    All of the lens or none of it. Placement here can hold a resident PREFIX and re-copy the
    rest per batch, because it still has the host copy to fall back on; the worker has no
    such fallback, so a partial upload would read the missing layers out untransported --
    the wrong distribution, quietly. Not fitting therefore returns False and lets the caller
    fall back to placement here, which is slower but stays correct.

    A lens of any kind but the Jacobian lens has no such fallback on vLLM: a call carries only
    the default set's matrices, so a named set is read only where the worker holds it. Not
    fitting marks that lens unavailable.
    """
    from interp_engine import VLLMModel

    lens = kind.store.get()
    if lens is None or not lens.jacobians:
        return False
    # Only vLLM reads out in a separate process. Every other backend reads out here, where
    # `place_jacobian_lens_on_device` decides which layers sit on the device.
    if not isinstance(model, VLLMModel):
        return False

    def unavailable(reason: str) -> None:
        if kind is not JACOBIAN_LENS_KIND:
            kind.store.set_error(reason)

    budget = _device_budget_bytes(config, args)
    if lens.resident_bytes > budget:
        logger.warning(
            "%s (%.2f GiB) does not fit the vLLM worker budget (%.2f GiB), so it stays "
            "in this process, where a resident prefix plus host fallback keeps every layer "
            "transported. Raise JLENS_GPU_BUDGET_GIB if the card has room.",
            kind.label,
            lens.resident_bytes / 1024**3,
            budget / 1024**3,
        )
        unavailable(f"the {kind.label} does not fit the vLLM worker budget")
        return False

    try:
        if kind is JACOBIAN_LENS_KIND:
            nbytes = await model.set_lens_jacobians(lens.jacobians)
        else:
            nbytes = await model.set_lens_jacobians(lens.jacobians, name=kind.engine_set)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to upload the %s to the vLLM worker; falling back to this process", kind.label)
        unavailable(f"the {kind.label} failed to upload to the vLLM worker")
        return False
    lens.worker_resident = True
    lens.transport_device = None
    logger.info(
        "%s uploaded to the vLLM worker: %d layers, %.2f GiB per rank",
        kind.label,
        len(lens.jacobians),
        nbytes / 1024**3,
    )
    return True


async def install_named_lens(model: object, kind: LensKind) -> None:
    """Give the engine a named lens set by reference, where this process reads out.

    The Jacobian lens rides along with each call instead; a named set has to be installed.
    Run after :func:`place_jacobian_lens_on_device`, so each layer is where placement left it.
    """
    lens = kind.store.get()
    if kind is JACOBIAN_LENS_KIND or lens is None or lens.worker_resident:
        return
    try:
        await model.set_lens_jacobians(lens.placed_jacobians(), name=kind.engine_set)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to install the %s in the engine", kind.label)
        kind.store.set_error(str(exc))


def place_jacobian_lens_on_device(config: object, args: object, kind: LensKind = JACOBIAN_LENS_KIND) -> None:
    """Upload the loaded lens to the serving device. Call once, late in startup.

    Separate from :func:`load_jacobian_lens_at_startup` because of *when* it has to run.
    The budget is measured from free VRAM, and under vLLM the engine reserves its pool in
    a child process well after the lens file is read -- measuring at load time would report
    an empty card and promise the lens memory the engine is about to take. Running after
    the engine preload means one measurement is right for both backends, which is why this
    needs none of the ``is_vllm`` estimation :func:`resolve_sae_gpu_budget_bytes` does.

    Best-effort, like the load itself: on failure the lens stays in host RAM and the
    read-out still produces correct results, just slowly.
    """
    lens = kind.store.get()
    if lens is None or lens.worker_resident:
        return

    device_str = getattr(config, "device", None) or getattr(args, "device", None)
    if not device_str:
        return

    try:
        lens.place_on_device(torch.device(device_str), device_budget_bytes=_device_budget_bytes(config, args))
    except Exception:  # noqa: BLE001
        logger.exception("Failed to place the %s on %s; it stays in host RAM", kind.label, device_str)
        return

    if lens.transport_device is None:
        logger.info(
            "%s stays in host RAM (budget %.2f GiB): the transport will run on the CPU.",
            kind.label,
            lens.device_budget_bytes / 1024**3,
        )
        return

    logger.info(
        "%s placed on %s: %.2f GiB of %.2f GiB resident (budget %.2f GiB)",
        kind.label,
        lens.transport_device,
        lens.device_resident_bytes / 1024**3,
        lens.resident_bytes / 1024**3,
        lens.device_budget_bytes / 1024**3,
    )
    if not lens.fits_on_device():
        # Not fatal, but every layer past the resident prefix is re-copied across PCIe on
        # every read-out batch, so say so here rather than leaving it as unexplained latency.
        logger.warning(
            "%s is larger than its device budget (%.2f GiB > %.2f GiB): "
            "layers past the resident prefix are re-copied per read-out batch. Raise "
            "JLENS_GPU_BUDGET_GIB if the card has room.",
            kind.label,
            lens.resident_bytes / 1024**3,
            lens.device_budget_bytes / 1024**3,
        )


def _lens_kind_wanted(args: object, kind: LensKind) -> bool:
    """The Jacobian lens loads unless ``JLENS_SKIP``; the J++ lens only with ``JPP_LENS``."""
    if kind is JACOBIAN_LENS_KIND:
        return not getattr(args, "jlens_skip", False)
    return bool(getattr(args, "jpp_lens", False))


def load_jacobian_lens_at_startup(config: object, args: object, kind: LensKind = JACOBIAN_LENS_KIND) -> None:
    """Resolve + load one lens kind, updating its store (:class:`JacobianLensStore` by default).

    Never raises: failures are recorded as an error status so requests for that lens
    return a helpful message while the rest of the server runs normally.
    """
    store = kind.store
    if not _lens_kind_wanted(args, kind):
        logger.info("Not loading the %s at startup (not enabled).", kind.label)
        store.set_skipped()
        return

    try:
        resolution = resolve_neuronpedia_model_id(config, args)
        np_model_id = resolution.np_model_id
        store._np_model_id = np_model_id

        source = getattr(args, f"{kind.arg_prefix}_source", None)
        dataset = getattr(args, f"{kind.arg_prefix}_dataset", None) or "Salesforce-wikitext"

        if source:
            logger.info("Loading %s from local source: %s", kind.label, source)
            lens_path = _find_local_lens_file(source, kind.stem)
        else:
            repo_id = getattr(args, f"{kind.arg_prefix}_hf_repo", None) or kind.default_repo
            explicit_path = getattr(args, f"{kind.arg_prefix}_hf_path", None)
            lens_path = _download_lens_from_hf(
                repo_id,
                np_model_id,
                dataset,
                resolution.hf_model_id,
                explicit_path,
                folder=kind.folder,
                stem=kind.stem,
            )

        # Host RAM only. The device budget cannot be measured yet (see
        # `place_jacobian_lens_on_device`), and a zero budget keeps the lens off the card
        # rather than letting the default ration memory nothing has measured.
        lens = LoadedJacobianLens.load(lens_path, dtype=_lens_dtype(config), device_budget_bytes=0)
        store.set_loaded(lens, np_model_id)
        logger.info(
            "Loaded %s for %s: %d source layers (%s..%s), d_model=%d, "
            "n_prompts=%d, dtype=%s, reads %s (%.2f GiB resident)",
            kind.label,
            np_model_id,
            len(lens.source_layers),
            lens.source_layers[0] if lens.source_layers else "?",
            lens.source_layers[-1] if lens.source_layers else "?",
            lens.d_model,
            lens.n_prompts,
            lens.dtype,
            # Worth a word in the log rather than only in the file: on a multi-stream trunk this is
            # the difference between a read-out and a fluent wrong answer, and it is the one property
            # of the artifact that no later symptom reveals.
            lens.residual.describe() if lens.residual is not None else "an undeclared space",
            lens.resident_bytes / 1024**3,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load the %s at startup", kind.label)
        store.set_error(str(exc))
