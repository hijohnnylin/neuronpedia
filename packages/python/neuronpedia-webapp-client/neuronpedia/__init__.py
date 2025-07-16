"""
Neuronpedia Python Client Library

This library provides a clean interface to interact with Neuronpedia's API.

API Key Configuration:
There are three ways to configure your API key:

1. Context manager (recommended):
   ```python
   import neuronpedia

   with neuronpedia.api_key("your-api-key"):
       feature_request = neuronpedia.FeatureRequest()
   ```

2. Global context setting:
   ```python
   import neuronpedia

   neuronpedia.set_api_key("your-api-key")
   feature_request = neuronpedia.FeatureRequest()
   ```

3. Direct parameter (for explicit control):
   ```python
   feature_request = neuronpedia.FeatureRequest(api_key="your-api-key")
   ```

4. Environment variable (existing behavior):
   ```python
   # Set NEURONPEDIA_API_KEY environment variable
   feature_request = neuronpedia.FeatureRequest()
   ```
"""

import contextvars
from contextlib import contextmanager
from typing import Optional

# Context variable for API key
_api_key_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "neuronpedia_api_key", default=None
)


def set_api_key(api_key: str) -> None:
    """
    Set the API key in the current context.

    Args:
        api_key: Your Neuronpedia API key
    """
    _api_key_context.set(api_key)


def get_api_key() -> Optional[str]:
    """
    Get the API key from the current context.

    Returns:
        The API key if set in context, None otherwise
    """
    return _api_key_context.get()


@contextmanager
def api_key(api_key: str):
    """
    Context manager for temporarily setting an API key.

    Args:
        api_key: Your Neuronpedia API key

    Example:
        ```python
        with neuronpedia.api_key("your-key"):
            feature_request = FeatureRequest()
        ```
    """
    token = _api_key_context.set(api_key)
    try:
        yield
    finally:
        _api_key_context.reset(token)


from neuronpedia.requests.activation_request import ActivationRequest

# Import request classes for easy access
from neuronpedia.requests.feature_request import FeatureRequest
from neuronpedia.requests.graph_request import GraphRequest
from neuronpedia.requests.list_request import ListRequest
from neuronpedia.requests.model_request import ModelRequest
from neuronpedia.requests.sae_feature_request import SAEFeatureRequest
from neuronpedia.requests.source_set_request import SourceSetRequest
from neuronpedia.requests.steer_request import SteerChatRequest, SteerCompletionRequest
from neuronpedia.requests.vector_request import VectorRequest

__all__ = [
    "set_api_key",
    "get_api_key",
    "api_key",
    "FeatureRequest",
    "ModelRequest",
    "GraphRequest",
    "SourceSetRequest",
    "VectorRequest",
    "ListRequest",
    "SAEFeatureRequest",
    "SteerChatRequest",
    "SteerCompletionRequest",
    "ActivationRequest",
]
