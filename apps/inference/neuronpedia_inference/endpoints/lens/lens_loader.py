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
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch

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
    hf_candidates = [
        c for c in (custom_hf_model_id, override_model_id, model_id) if c is not None
    ]
    for candidate in hf_candidates:
        for np_id, hf_id in mapping.items():
            if candidate in (hf_id, np_id):
                return LensResolution(np_model_id=np_id, hf_model_id=hf_id)

    raise ValueError(
        f"Cannot resolve neuronpedia model id for loaded model "
        f"(model_id={model_id!r}, override={override_model_id!r}, "
        f"custom_hf={custom_hf_model_id!r}). Pass --NEURONPEDIA_MODEL_ID."
    )


class LoadedJacobianLens:
    """A fitted Jacobian lens loaded from disk: per-layer ``J_bar`` + metadata.

    Standalone (does not import the ``jlens`` package). Jacobians are kept on CPU
    as fp32 and moved to the compute device lazily on first use per layer.
    """

    def __init__(
        self,
        jacobians: dict[int, torch.Tensor],
        *,
        source_layers: list[int],
        n_prompts: int,
        d_model: int,
    ) -> None:
        self.jacobians = {int(layer): J.float() for layer, J in jacobians.items()}
        self.source_layers = sorted(self.jacobians)
        self.n_prompts = n_prompts
        self.d_model = d_model
        self._device_cache: dict[int, torch.Tensor] = {}

    @classmethod
    def load(cls, path: str) -> LoadedJacobianLens:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "J" not in checkpoint:
            raise ValueError(
                f"{path} is not a Jacobian lens file (keys: {sorted(checkpoint)!r})"
            )
        return cls(
            jacobians=checkpoint["J"],
            source_layers=list(checkpoint.get("source_layers", [])),
            n_prompts=int(checkpoint.get("n_prompts", 0)),
            d_model=int(checkpoint["d_model"]),
        )

    def jacobian_on(self, layer: int, device: torch.device) -> torch.Tensor:
        cached = self._device_cache.get(layer)
        if cached is not None and cached.device == device:
            return cached
        moved = self.jacobians[layer].to(device)
        self._device_cache[layer] = moved
        return moved

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """Map a residual at ``layer`` into the readout basis: ``residual @ J_bar.T``."""
        J_bar = self.jacobian_on(layer, residual.device).to(residual.dtype)
        return residual @ J_bar.T


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


def _find_local_lens_file(directory: str) -> str:
    matches = sorted(glob.glob(os.path.join(directory, "*_jacobian_lens.pt")))
    if not matches:
        raise FileNotFoundError(
            f"No *_jacobian_lens.pt found in local JLENS_SOURCE directory: {directory}"
        )
    if len(matches) > 1:
        logger.warning(
            "Multiple lens files in %s, using the first: %s", directory, matches[0]
        )
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
    matches = sorted(
        f for f in files if f.startswith(f"{prefix}/") and f.endswith(".pt")
    )
    return matches[0] if matches else None


def _download_lens_from_hf(
    repo_id: str,
    np_model_id: str,
    dataset: str,
    hf_model_id: str | None,
    explicit_path: str | None,
) -> str:
    """Download the lens from a HF model repo and return the local cache path.

    When ``explicit_path`` is given it is used verbatim. Otherwise we try the
    deterministic candidates under ``<np_model_id>/jlens/<dataset>/``:
    ``<slug>_jacobian_lens.pt`` first, then ``<slug>_jacobian_lens_n1000.pt``,
    and finally fall back to the first ``.pt`` listed under that directory.
    """
    from huggingface_hub import hf_hub_download

    prefix = f"{np_model_id}/jlens/{dataset}"

    candidate_paths: list[str] = []
    if explicit_path:
        candidate_paths.append(explicit_path)
    else:
        slug = _slug(hf_model_id) if hf_model_id is not None else _slug(np_model_id)
        candidate_paths.append(f"{prefix}/{slug}_jacobian_lens.pt")
        candidate_paths.append(f"{prefix}/{slug}_jacobian_lens_n1000.pt")

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


def load_jacobian_lens_at_startup(config: object, args: object) -> None:
    """Resolve + load the Jacobian lens, updating :class:`JacobianLensStore`.

    Never raises: failures are recorded as an error status so JACOBIAN_LENS
    requests return a helpful message while the rest of the server runs normally.
    """
    if getattr(args, "jlens_skip", False):
        logger.info("JLENS_SKIP set: not loading the Jacobian lens at startup.")
        JacobianLensStore.set_skipped()
        return

    try:
        resolution = resolve_neuronpedia_model_id(config, args)
        np_model_id = resolution.np_model_id
        JacobianLensStore._np_model_id = np_model_id

        source = getattr(args, "jlens_source", None)
        dataset = getattr(args, "jlens_dataset", "Salesforce-wikitext")

        if source:
            logger.info("Loading Jacobian lens from local source: %s", source)
            lens_path = _find_local_lens_file(source)
        else:
            repo_id = getattr(args, "jlens_hf_repo", "neuronpedia/jacobian-lens")
            explicit_path = getattr(args, "jlens_hf_path", None)
            lens_path = _download_lens_from_hf(
                repo_id,
                np_model_id,
                dataset,
                resolution.hf_model_id,
                explicit_path,
            )

        lens = LoadedJacobianLens.load(lens_path)
        JacobianLensStore.set_loaded(lens, np_model_id)
        logger.info(
            "Loaded Jacobian lens for %s: %d source layers (%s..%s), d_model=%d, "
            "n_prompts=%d",
            np_model_id,
            len(lens.source_layers),
            lens.source_layers[0] if lens.source_layers else "?",
            lens.source_layers[-1] if lens.source_layers else "?",
            lens.d_model,
            lens.n_prompts,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load Jacobian lens at startup")
        JacobianLensStore.set_error(str(exc))
