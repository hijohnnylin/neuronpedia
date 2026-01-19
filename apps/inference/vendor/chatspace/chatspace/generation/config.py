"""Configuration dataclasses for text generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationConfig:
    """Unified generation configuration for both HF and vLLM backends.

    This dataclass provides a common interface for generation parameters
    that can be converted to either HuggingFace `generate()` kwargs or
    vLLM `SamplingParams`.
    """

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    do_sample: bool = True
    seed: Optional[int] = None
    stop_strings: Optional[list[str]] = None

    def to_vllm_params(self):
        """Convert to vLLM SamplingParams.

        Returns
        -------
        SamplingParams
            vLLM sampling parameters object.
        """
        from vllm import SamplingParams

        kwargs = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_new_tokens,
        }
        if self.top_k > 0:
            kwargs["top_k"] = self.top_k
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.stop_strings:
            kwargs["stop"] = self.stop_strings

        return SamplingParams(**kwargs)
