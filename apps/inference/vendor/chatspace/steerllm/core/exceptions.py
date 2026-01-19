"""Custom exceptions for steerllm."""


class SteerLLMError(Exception):
    """Base exception for steerllm errors."""

    pass


class BackendError(SteerLLMError):
    """Error from a steering backend (vLLM, HuggingFace, etc.)."""

    pass


class CaptureError(SteerLLMError):
    """Error during activation capture or retrieval."""

    pass


class ValidationError(SteerLLMError):
    """Invalid steering configuration or input."""

    pass
