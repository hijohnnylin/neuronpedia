"""Single place that imports the optional vLLM bits.

vLLM is Linux/CUDA-only, so the app must import cleanly without it (macOS/CPU installs run the
EagerModel backend). ``VLLM_AVAILABLE`` gates the vLLM-only code paths and feeds the backend
auto-selector; ``SamplingParams`` is only ever *constructed* on those paths, since an
``VLLMModel`` being loaded implies vLLM imported fine.

Typing note: the import is written under ``TYPE_CHECKING`` so ``SamplingParams`` keeps its real
class type. With a plain ``try/except`` fallback to ``None``, every call site would have to
narrow away a ``None`` that cannot occur when that path runs.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import SamplingParams
else:
    try:
        from vllm import SamplingParams
    except ImportError:
        SamplingParams = None

VLLM_AVAILABLE: bool = SamplingParams is not None

__all__ = ["VLLM_AVAILABLE", "SamplingParams"]
