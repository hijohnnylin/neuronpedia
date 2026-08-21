"""PersonaData - Singleton for pre-loaded persona PCA data.

This module provides a singleton class that holds pre-loaded PCA data
and contrast vectors for persona analysis. Data is loaded once at server
startup (assistant-axis / persona feature).

Assets live at ``data/<hf_model_id>/`` and describe themselves in a ``persona.yaml``
manifest: which layers were fitted and in which files, what the PCs are called, and the
rendering conditions the fit assumed. Nothing about a specific asset is encoded here — no
id table, no filename convention to parse — so shipping a new one is a data change. A
checkpoint that reuses another's fit (a quantization of the same weights) is a symlink in
the data tree, which keeps the manifest with the fit it describes.
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# MeanScaler class - needed for unpickling the minimal PCA data files
# =============================================================================
def _to_numpy(x):
    """Convert tensor or array to numpy."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(x)}")


class MeanScaler:
    """Simple scaler that subtracts the mean."""

    def __init__(self, mean=None):
        self.mean = mean

    def _ensure_mean_numpy(self):
        if self.mean is None:
            return
        if isinstance(self.mean, torch.Tensor):
            self.mean = self.mean.detach().cpu().numpy()
        elif not isinstance(self.mean, np.ndarray):
            self.mean = _to_numpy(self.mean)

    def fit(self, X):
        X_np = _to_numpy(X)
        if self.mean is None:
            axes = tuple(range(X_np.ndim - 1))
            self.mean = X_np.mean(axis=axes, keepdims=False)
        else:
            self._ensure_mean_numpy()
        return self

    def transform(self, X):
        if self.mean is None:
            raise RuntimeError("MeanScaler not fitted")
        self._ensure_mean_numpy()
        X_np = _to_numpy(X)
        return X_np - self.mean

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class L2MeanScaler:
    """Scaler that subtracts mean and L2-normalizes."""

    def __init__(self, mean=None, eps: float = 1e-12):
        self.mean = mean
        self.eps = eps

    def _ensure_mean_numpy(self):
        if self.mean is None:
            return
        if isinstance(self.mean, torch.Tensor):
            self.mean = self.mean.detach().cpu().numpy()
        elif not isinstance(self.mean, np.ndarray):
            self.mean = _to_numpy(self.mean)

    def fit(self, X):
        X_np = _to_numpy(X)
        if self.mean is None:
            axes = tuple(range(X_np.ndim - 1))
            self.mean = X_np.mean(axis=axes, keepdims=False)
        else:
            self._ensure_mean_numpy()
        return self

    def transform(self, X):
        if self.mean is None:
            raise RuntimeError("L2MeanScaler not fitted")
        self._ensure_mean_numpy()
        X_np = _to_numpy(X)
        X_centered = X_np - self.mean
        norms = np.linalg.norm(X_centered, ord=2, axis=-1, keepdims=True)
        return X_centered / np.maximum(norms, self.eps)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# Register shims so torch.load can find scalers when loading pickled files
# The minimal files were saved with classes in __main__
main_module = sys.modules.get("__main__")
if main_module is not None:
    if not hasattr(main_module, "MeanScaler"):
        setattr(main_module, "MeanScaler", MeanScaler)  # noqa: B010
    if not hasattr(main_module, "L2MeanScaler"):
        setattr(main_module, "L2MeanScaler", L2MeanScaler)  # noqa: B010

MANIFEST_FILENAME = "persona.yaml"


@dataclass(frozen=True)
class PersonaPCA:
    """One fitted PCA: the layer it was fitted at and the file holding it."""

    layer: int
    file: str


@dataclass(frozen=True)
class PersonaFit:
    """A persona asset's ``persona.yaml``: how it was fitted and what it ships.

    A projection onto the fitted PCs only means anything if inference renders the
    conversation the way it was rendered during fitting, and only at the layer the fit
    was taken at — so all of this belongs to the **asset**, not to the model architecture
    or to the assistant-axis feature. It is read from the asset directory rather than
    kept in a table here so that adding an asset is a data change.
    """

    # The PCA was fitted with an empty system turn, so a caller-supplied system prompt
    # must be blanked or the activations land off-distribution. (On Llama 3 a non-empty
    # system turn also makes the template inject its knowledge-cutoff preamble.)
    blank_system_prompt: bool = False
    # Labels for the principal components, in component order. Length is how many PCs the
    # asset surfaces, which need not be how many were fitted.
    pc_titles: tuple[str, ...] = ()
    contrast_vectors: str = "contrast_vectors.pt"
    pca: tuple[PersonaPCA, ...] = ()

    @property
    def layers(self) -> list[int]:
        return [entry.layer for entry in self.pca]

    def file_for_layer(self, layer: int) -> str | None:
        for entry in self.pca:
            if entry.layer == layer:
                return entry.file
        return None


_DEFAULT_FIT = PersonaFit()


def _load_manifest(asset_dir: str) -> PersonaFit | None:
    """Read ``persona.yaml`` from an asset directory, or ``None`` if it has none."""
    path = os.path.join(asset_dir, MANIFEST_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping, got {type(raw).__name__}")

    pca_entries = raw.get("pca") or []
    if not isinstance(pca_entries, list):
        raise ValueError(f"{path}: 'pca' must be a list of layer/file entries")
    pca = tuple(PersonaPCA(layer=int(entry["layer"]), file=str(entry["file"])) for entry in pca_entries)

    return PersonaFit(
        blank_system_prompt=bool(raw.get("blank_system_prompt", False)),
        pc_titles=tuple(str(title) for title in raw.get("pc_titles") or ()),
        contrast_vectors=str(raw.get("contrast_vectors", "contrast_vectors.pt")),
        pca=tuple(sorted(pca, key=lambda entry: entry.layer)),
    )


class PersonaData:
    """
    Singleton class holding pre-loaded persona PCA data.

    Data is loaded once at server startup via initialize() and then
    accessed via get_instance().
    """

    _instance: "PersonaData | None" = None

    def __init__(self):
        """Initialize empty persona data container."""
        self._pca_data: dict[str, Any] = {}
        self._model_id: str | None = None
        self._initialized: bool = False
        self._layers: list[int] = []
        self._fit: PersonaFit = _DEFAULT_FIT

    @classmethod
    def get_instance(cls) -> "PersonaData":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if persona data has been initialized."""
        return cls._instance is not None and cls._instance._initialized

    @property
    def model_id(self) -> str | None:
        """Get the model ID for which data was loaded."""
        return self._model_id

    @property
    def layers(self) -> list[int]:
        """Layers with PCA data loaded, ascending. Empty when no asset is loaded."""
        return list(self._layers)

    @property
    def primary_layer(self) -> int | None:
        """The layer persona analysis runs on, or ``None`` when no asset is loaded."""
        return self._layers[0] if self._layers else None

    @property
    def fit(self) -> PersonaFit:
        """Fitting conditions for the loaded asset; defaults when nothing is loaded."""
        if not self._initialized:
            return _DEFAULT_FIT
        return self._fit

    @property
    def pc_titles(self) -> list[str]:
        """Labels for the PCs this asset surfaces, in component order."""
        return list(self.fit.pc_titles)

    def get_pca_data(self, layer: int) -> dict[str, Any] | None:
        """
        Get PCA data for a specific layer.

        Args:
            layer: Layer number

        Returns:
            Dict with 'pca' and 'scaler', or None if not loaded
        """
        cache_key = f"{self._model_id}:{layer}"
        return self._pca_data.get(cache_key)

    def initialize(self, model_id: str, layers: list[int] | None = None) -> None:
        """
        Load PCA data and contrast vectors for persona analysis.

        Args:
            model_id: HuggingFace model identifier of the served checkpoint. This is the
                asset directory name; a checkpoint that reuses another's fit (a
                quantization of the same weights, say) is a symlink in the data tree
                rather than an entry in a table here.
            layers: Layers to load data for. Defaults to the layers the asset's manifest
                declares.
        """
        # Reset first: every early return below is a failure to load, and must leave the
        # instance reporting itself unavailable rather than keeping a previous model's state.
        self._model_id = model_id
        self._layers = []
        self._initialized = False
        self._fit = _DEFAULT_FIT
        data_path = self._get_data_path()
        asset_dir = os.path.join(data_path, model_id)

        logger.info(f"Loading persona data for model {model_id} from {data_path}")

        try:
            manifest = _load_manifest(asset_dir)
        except (ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
            logger.warning(
                f"Could not read {os.path.join(asset_dir, MANIFEST_FILENAME)}: {exc}. Persona monitoring will not work"
            )
            return
        if manifest is None:
            logger.warning(f"No {MANIFEST_FILENAME} in {asset_dir}, persona monitoring will not work")
            return
        self._fit = manifest
        if os.path.islink(asset_dir):
            logger.info(
                f"Persona asset {model_id} is a symlink to {os.path.relpath(os.path.realpath(asset_dir), data_path)}"
            )

        # Load contrast vectors (needed for all layers)
        contrast_path = os.path.join(asset_dir, manifest.contrast_vectors)
        if not os.path.exists(contrast_path):
            logger.warning(f"Contrast vectors not found at {contrast_path}, persona monitoring will not work")
            return

        contrast_vectors = torch.load(contrast_path, weights_only=False)
        logger.info(f"Loaded contrast vectors with {len(contrast_vectors)} layers")

        if layers is None:
            layers = manifest.layers
            if not layers:
                logger.warning(
                    f"{MANIFEST_FILENAME} for {model_id} declares no PCA layers, persona monitoring will not work"
                )
                return
            logger.info(f"Persona asset {model_id} declares PCA layer(s) {layers}")

        # Load PCA data for each requested layer
        for layer in layers:
            pca_file = manifest.file_for_layer(layer)
            if pca_file is None:
                logger.warning(f"{MANIFEST_FILENAME} for {model_id} has no entry for layer {layer}, skipping")
                continue
            pca_path = os.path.join(asset_dir, pca_file)
            if not os.path.exists(pca_path):
                logger.warning(f"PCA data not found at {pca_path}, skipping layer {layer}")
                continue

            role_results = torch.load(pca_path, weights_only=False)

            # Get contrast vector for this layer
            if layer >= len(contrast_vectors):
                logger.warning(f"No contrast vector for layer {layer}, skipping")
                continue

            contrast_vector = contrast_vectors[layer]
            contrast_vector = F.normalize(contrast_vector, dim=0)

            # Replace PC1 with contrast vector (flipped)
            role_results["pca"].components_[0] = contrast_vector.float() * -1
            # only using the first role
            # role_results["pca"].components_[1] = role_results["pca"].components_[1] * -1
            # role_results["pca"].components_[2] = role_results["pca"].components_[2] * -1

            cache_key = f"{model_id}:{layer}"
            self._pca_data[cache_key] = role_results
            self._layers.append(layer)
            logger.info(f"Loaded PCA data for layer {layer}")

        # Only "initialized" once a layer actually loaded — otherwise there is nothing
        # to project onto and callers should treat persona monitoring as unavailable.
        if not self._layers:
            logger.warning(f"No persona PCA layers loaded for {model_id}, persona monitoring will not work")
            return

        # An asset with no pc_titles would load and then report every turn with an empty
        # set of PC values, which reads as "the axis is flat" rather than as a broken
        # manifest. Cheaper to say so here.
        if not self._fit.pc_titles:
            logger.warning(
                f"{MANIFEST_FILENAME} for {model_id} declares no pc_titles; "
                "assistant-axis responses will carry no PC values"
            )

        self._layers.sort()
        self._initialized = True
        logger.info(f"Persona data initialization complete for layer(s) {self._layers}")

    def _get_data_path(self) -> str:
        """Get the base path for persona data files."""
        # Data files are expected at inference_utils/persona/data/<model_id>/
        return os.path.join(os.path.dirname(__file__), "data")


def initialize_persona_data(model_id: str, layers: list[int] | None = None) -> None:
    """
    Initialize persona data at server startup.

    This function should be called during server initialization (loads for
    whichever backend is active; no-op when no persona data files are present).

    Args:
        model_id: HuggingFace model identifier
        layers: Layers to load. Defaults to whichever layers the asset ships.
    """
    persona_data = PersonaData.get_instance()
    persona_data.initialize(model_id, layers)
