"""PersonaData - Singleton for pre-loaded persona PCA data.

This module provides a singleton class that holds pre-loaded PCA data
and contrast vectors for persona analysis. Data is loaded once at server
startup when using chatspace mode.
"""

import logging
import os
import sys
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

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
        main_module.MeanScaler = MeanScaler
    if not hasattr(main_module, "L2MeanScaler"):
        main_module.L2MeanScaler = L2MeanScaler

# Default layer for persona analysis
DEFAULT_LAYER = 40

# PC titles for the three principal components
ROLE_PC_TITLES = [
    "- Role-playing ↔️ + Assistant-like",
    # "- Collective ↔️ + Individual",
    # "- Passionate ↔️ + Robotic",
]


class PersonaData:
    """
    Singleton class holding pre-loaded persona PCA data.
    
    Data is loaded once at server startup via initialize() and then
    accessed via get_instance().
    """
    
    _instance: Optional["PersonaData"] = None
    
    def __init__(self):
        """Initialize empty persona data container."""
        self._pca_data: dict[str, Any] = {}
        self._model_id: Optional[str] = None
        self._initialized: bool = False
    
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
    def model_id(self) -> Optional[str]:
        """Get the model ID for which data was loaded."""
        return self._model_id
    
    @property
    def pc_titles(self) -> list[str]:
        """Get the PC titles."""
        return ROLE_PC_TITLES
    
    def get_pca_data(self, layer: int) -> Optional[dict[str, Any]]:
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
            model_id: HuggingFace model identifier (used for data path)
            layers: List of layers to load data for (default: [DEFAULT_LAYER])
        """
        if layers is None:
            layers = [DEFAULT_LAYER]
        
        self._model_id = model_id
        data_path = self._get_data_path()
        
        logger.info(f"Loading persona data for model {model_id} from {data_path}")
        
        # Load contrast vectors (needed for all layers)
        contrast_path = os.path.join(data_path, model_id, "contrast_vectors.pt")
        if not os.path.exists(contrast_path):
            logger.warning(f"Contrast vectors not found at {contrast_path}, persona monitoring will not work")
            return
        
        contrast_vectors = torch.load(contrast_path, weights_only=False)
        logger.info(f"Loaded contrast vectors with {len(contrast_vectors)} layers")
        
        # Load PCA data for each requested layer
        for layer in layers:
            pca_path = os.path.join(data_path, model_id, "pca", f"roles_layer{layer}-min.pt")
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
            logger.info(f"Loaded PCA data for layer {layer}")
        
        self._initialized = True
        logger.info(f"Persona data initialization complete for {len(self._pca_data)} layer(s)")
    
    def _get_data_path(self) -> str:
        """Get the base path for persona data files."""
        # Data files are expected at endpoints/persona/data/<model_id>/
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def initialize_persona_data(model_id: str, layers: list[int] | None = None) -> None:
    """
    Initialize persona data at server startup.
    
    This function should be called during server initialization when
    chatspace mode is enabled.
    
    Args:
        model_id: HuggingFace model identifier
        layers: List of layers to load (default: [40])
    """
    persona_data = PersonaData.get_instance()
    persona_data.initialize(model_id, layers)

