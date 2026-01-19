"""Feature extraction utilities for steering vector evaluation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Sequence

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class _StopForward(RuntimeError):
    """Raised internally to terminate the forward pass after capturing a layer."""


@contextmanager
def _truncate_after_layer(model, layer_index: int, storage: dict[int, torch.Tensor]):
    """Context manager to capture a layer output and stop further computation."""
    handle = None

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        if 0 <= layer_index < len(layers):
            target_layer = layers[layer_index]

            def _hook(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                storage[layer_index] = hidden.detach()
                raise _StopForward

            handle = target_layer.register_forward_hook(_hook)
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


def extract_layer_hidden_states(
    records: Sequence[dict],
    model,
    tokenizer,
    *,
    target_layer: int,
    max_length: int = 4096,
    batch_size: int = 4,
    device: str | torch.device | None = None,
    truncate_after: bool = False,
    add_generation_prompt: bool = False,
    chat_template_kwargs: dict | None = None,
    use_tqdm: bool = False,
) -> np.ndarray:
    """Compute pooled hidden states from a specified transformer layer."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model.eval()

    iterator: Iterable[int] = range(0, len(records), batch_size)
    if use_tqdm and tqdm is not None:
        iterator = tqdm(iterator, desc="Extracting hidden states")

    pooled_rows: list[np.ndarray] = []
    chat_template_kwargs = chat_template_kwargs or {}

    for start in iterator:
        batch = records[start : start + batch_size]
        if not batch:
            continue

        chat_texts = [
            tokenizer.apply_chat_template(
                rec["messages"],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **chat_template_kwargs,
            )
            for rec in batch
        ]

        encoded = tokenizer(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        attention_mask = encoded.get("attention_mask")

        if truncate_after:
            captured: dict[int, torch.Tensor] = {}
            try:
                with _truncate_after_layer(model, target_layer, captured):
                    model(**encoded)
            except _StopForward:
                pass
            hidden = captured.get(target_layer)
            if hidden is None:
                raise RuntimeError(f"Failed to capture hidden states for layer {target_layer}")
        else:
            outputs = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states – enable truncate_after or check model config.")
            hidden = hidden_states[target_layer]

        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
            masked = hidden * mask
            summed = masked.sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts

        pooled_rows.append(pooled.cpu().numpy())

    if not pooled_rows:
        raise ValueError("No hidden states were extracted from the provided records.")

    return np.concatenate(pooled_rows, axis=0)
