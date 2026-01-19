"""Token bucketing and batching logic for efficient sequence padding."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import torch

from .config import SentenceTransformerConfig
from .utils import _next_power_of_two


@dataclass
class TokenBatch:
    """A batch of tokenized sequences ready for encoding."""

    rows: list[dict[str, Any]]
    features: dict[str, torch.Tensor]
    bucket_size: int


class _BucketBuffer:
    """Buffer for accumulating sequences of the same bucket size."""

    def __init__(self, bucket_size: int) -> None:
        self.bucket_size = bucket_size
        self.rows: list[dict[str, Any]] = []
        self.tokens: dict[str, list[torch.Tensor]] = defaultdict(list)

    def add(self, row: dict[str, Any], tokenized: dict[str, torch.Tensor]) -> None:
        """Add a row and its tokens to the buffer."""
        self.rows.append(row)
        for key, tensor in tokenized.items():
            self.tokens[key].append(tensor)

    def __len__(self) -> int:
        """Return number of rows in buffer."""
        return len(self.rows)

    def pop(self, count: int, pad_values: dict[str, int]) -> Optional[TokenBatch]:
        """Pop up to count rows and create a batch."""
        if count <= 0 or not self.rows:
            return None
        count = min(count, len(self.rows))
        rows = self.rows[:count]
        token_slices = {key: value[:count] for key, value in self.tokens.items()}
        features = _pad_and_stack_tokens(token_slices, self.bucket_size, pad_values)
        self.rows = self.rows[count:]
        for key in list(self.tokens.keys()):
            self.tokens[key] = self.tokens[key][count:]
        return TokenBatch(rows=rows, features=features, bucket_size=self.bucket_size)

    def flush(self, pad_values: dict[str, int]) -> Optional[TokenBatch]:
        """Flush all remaining rows as a batch."""
        return self.pop(len(self.rows), pad_values)


def _select_bucket_size(length: int, cfg: SentenceTransformerConfig) -> int:
    """Select appropriate bucket size for a sequence length."""
    bucket = max(cfg.bucket_min_tokens, _next_power_of_two(length))
    return min(bucket, cfg.bucket_max_tokens)


def _token_sequence_length(tokens: dict[str, torch.Tensor]) -> int:
    """Determine actual sequence length from tokenized output."""
    if "attention_mask" in tokens:
        return int(tokens["attention_mask"].sum().item())
    if "input_ids" in tokens:
        return int(tokens["input_ids"].shape[-1])
    for value in tokens.values():
        if isinstance(value, torch.Tensor):
            return int(value.shape[-1])
    return 0


def _pad_and_stack_tokens(
    token_slices: dict[str, list[torch.Tensor]], bucket_size: int, pad_values: dict[str, int]
) -> dict[str, torch.Tensor]:
    """Pad token tensors to bucket_size and stack into batch."""
    features: dict[str, torch.Tensor] = {}
    for key, tensors in token_slices.items():
        if not tensors:
            continue
        pad_value = pad_values.get(key, 0)
        padded: list[torch.Tensor] = []
        for tensor in tensors:
            if tensor.ndimension() == 0:
                tensor = tensor.unsqueeze(0)
            if tensor.ndimension() > 1:
                tensor = tensor.view(-1)
            current_len = tensor.shape[-1]
            if current_len > bucket_size:
                tensor = tensor[..., :bucket_size]
                current_len = bucket_size
            if current_len < bucket_size:
                pad_shape = (bucket_size - current_len,)
                pad_tensor = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
                tensor = torch.cat([tensor, pad_tensor], dim=-1)
            padded.append(tensor)
        if padded:
            stacked = torch.stack(padded, dim=0)
            features[key] = stacked
    return features


def _effective_batch_size(bucket_size: int, cfg: SentenceTransformerConfig) -> int:
    """Calculate effective batch size based on token budget or fixed size."""
    if cfg.tokens_per_batch is not None:
        sequences = cfg.tokens_per_batch // max(bucket_size, 1)
        return max(sequences, 1)
    return max(cfg.batch_size, 1)