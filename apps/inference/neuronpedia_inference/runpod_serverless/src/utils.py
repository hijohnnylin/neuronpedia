"""
Utility functions for the RunPod Serverless handler.
"""

import re
from enum import Enum
from typing import Any

import torch


class SteerType(Enum):
    """Steering types."""
    STEERED = "STEERED"
    DEFAULT = "DEFAULT"


class SteerMethod(Enum):
    """Steering methods."""
    SIMPLE_ADDITIVE = "SIMPLE_ADDITIVE"
    PROJECTION_CAP = "PROJECTION_CAP"


# Regex to match Llama 3's auto-injected knowledge cutoff preamble in system messages
_LLAMA3_SYSTEM_PREAMBLE_PATTERN = re.compile(
    r"^Cutting Knowledge Date:\s*[^\n]+\nToday Date:\s*[^\n]+\n*",
    re.MULTILINE
)


def format_sse_message(data: str | dict) -> str:
    """Format data as SSE message."""
    import json
    if isinstance(data, dict):
        data = json.dumps(data)
    return f"data: {data}\n\n"


def remove_sse_formatting(data: str) -> str:
    """Remove SSE formatting from data."""
    if data.startswith("data: "):
        data = data[6:]
    return data.rstrip("\n\n")


def _strip_llama3_system_preamble(content: str) -> str:
    """Strip Llama 3's auto-injected knowledge cutoff preamble from system message content."""
    content = _LLAMA3_SYSTEM_PREAMBLE_PATTERN.sub("", content)
    return content.strip()


def convert_to_chat_array(
    text: str,
    tokenizer: Any,
    custom_hf_model_id: str | None = None,
) -> list[dict[str, str]]:
    """
    Convert raw tokenized text back to chat array format.
    Specifically handles Llama 3.3 format.
    """
    if tokenizer is None:
        raise ValueError("Tokenizer cannot be None for chat array conversion")
    
    tokens = tokenizer.encode(text)
    conversation: list[dict[str, str]] = []
    
    # Llama 3.3 Instruct uses header tokens
    if hasattr(tokenizer, "name_or_path") and "llama-3" in tokenizer.name_or_path.lower():
        START_HEADER_ID = 128006
        END_HEADER_ID = 128007
        EOT_ID = 128009

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token == START_HEADER_ID:
                role_tokens = []
                i += 1
                while i < len(tokens) and tokens[i] != END_HEADER_ID:
                    role_tokens.append(tokens[i])
                    i += 1

                if i < len(tokens) and tokens[i] == END_HEADER_ID:
                    i += 1

                    content_tokens = []
                    while i < len(tokens) and tokens[i] != EOT_ID:
                        content_tokens.append(tokens[i])
                        i += 1

                    if role_tokens and content_tokens:
                        role = tokenizer.decode(role_tokens).strip()
                        content = tokenizer.decode(content_tokens).strip()

                        # Strip Llama 3's auto-injected knowledge cutoff preamble
                        if role == "system":
                            content = _strip_llama3_system_preamble(content)

                        if role and content:
                            conversation.append({
                                "role": role,
                                "content": content,
                            })

                    if i < len(tokens) and tokens[i] == EOT_ID:
                        i += 1
                    continue

            i += 1
    
    # Fallback for tokenizers without chat template
    elif not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        # Parse ChatML format: <|im_start|>{role}\n{content}<|im_end|>\n
        parts = text.split("<|im_start|>")
        
        for part in parts[1:]:
            if not part.strip():
                continue
            
            if "<|im_end|>" in part:
                content_part = part.split("<|im_end|>")[0]
                
                if "\n" in content_part:
                    role, content = content_part.split("\n", 1)
                    role = role.strip()
                    content = content.strip()
                    
                    if role and content:
                        conversation.append({
                            "role": role,
                            "content": content,
                        })
    
    return conversation


def apply_generic_chat_template(
    messages: list[dict[str, str]], add_generation_prompt: bool = True
) -> str:
    """
    Apply a generic ChatML template for models without native chat template support.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        add_generation_prompt: Whether to add the assistant generation prompt
        
    Returns:
        Formatted chat string ready for tokenization
    """
    formatted_text = ""
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        formatted_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    if add_generation_prompt:
        formatted_text += "<|im_start|>assistant\n"
    
    return formatted_text


class OrthogonalProjector:
    """Performs orthogonal projection steering for language model activations.

    This class implements low-rank orthogonal projection-based steering by projecting
    activations onto and orthogonal to a steering direction.

    Attributes:
        steering_vector: The direction to project onto/orthogonal to
        _P: Cached projection matrix
        _orthogonal_complement: Cached orthogonal complement matrix
    """

    def __init__(self, steering_vector: torch.Tensor):
        """Initializes projector with a steering vector.

        Args:
            steering_vector: Vector defining steering direction, shape (d,)
                           where d is activation dimension

        Raises:
            ValueError: If steering vector contains inf/nan values
        """
        self._P = None
        self._orthogonal_complement = None
        self.steering_vector = steering_vector.unsqueeze(1)

    def get_P(self) -> torch.Tensor:
        """Computes or returns cached projection matrix.

        Returns:
            Projection matrix P = vv^T/||v||^2, shape (d,d)

        Raises:
            ValueError: If projection computation fails or results in inf/nan
        """
        if self._P is None:
            # Compute the squared norm of the steering vector
            v_norm_squared = torch.sum(self.steering_vector * self.steering_vector)

            # Check for zero norm to avoid division by zero
            if v_norm_squared == 0:
                raise ValueError("Cannot create projection matrix from zero vector")

            # Compute the projection matrix: P = vv^T / ||v||^2
            self._P = (
                torch.matmul(self.steering_vector, self.steering_vector.T)
                / v_norm_squared
            )

            if not torch.isfinite(self._P).all():
                raise ValueError("Projection matrix contains inf or nan values")

        return self._P

    def get_orthogonal_complement(self) -> torch.Tensor:
        """Computes or returns cached orthogonal complement matrix.

        Returns:
            Matrix I-P where P is projection matrix, shape (d,d)

        Raises:
            ValueError: If computation fails
        """
        if self._orthogonal_complement is None:
            P = self.get_P()  # This may raise ValueError
            I = torch.eye(P.shape[0], dtype=P.dtype, device=P.device)  # noqa
            self._orthogonal_complement = I - P
            if not torch.isfinite(self._orthogonal_complement).all():
                raise ValueError(
                    "Orthogonal complement matrix contains inf or nan values"
                )
        return self._orthogonal_complement

    def project(
        self, activations: torch.Tensor, strength_multiplier: float = 1.0
    ) -> torch.Tensor:
        """Projects activations using orthogonal decomposition.

        Decomposes activations into components parallel and orthogonal to steering direction,
        then recombines with optional scaling of parallel component.

        Args:
            activations: Input activations to project, shape (d,)
            strength_multiplier: Scaling factor for parallel component

        Returns:
            Projected activations = (I-P)h + strength*Ph, shape (d,)
        """
        P = self.get_P()
        orthogonal_complement = self.get_orthogonal_complement()
        # use same dtype as activations
        orthogonal_complement = orthogonal_complement.to(activations.dtype)
        P = P.to(activations.dtype)
        return torch.matmul(
            activations, orthogonal_complement.T
        ) + strength_multiplier * torch.matmul(activations, P.T)

