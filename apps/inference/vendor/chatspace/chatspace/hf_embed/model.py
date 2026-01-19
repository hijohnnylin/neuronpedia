"""Model loading and inference management."""

from __future__ import annotations

import importlib.util
import logging
import time
from typing import Any, Iterable, Optional

import torch
from sentence_transformers import SentenceTransformer

from .config import SentenceTransformerConfig


def _is_flash_attn_available() -> bool:
    """Return True when the flash_attn module can be imported."""
    return importlib.util.find_spec("flash_attn") is not None


def _default_model_kwargs(cfg: SentenceTransformerConfig) -> dict[str, Any]:
    """Build model kwargs from config."""
    kwargs: dict[str, Any] = {}
    if cfg.attention_impl:
        if cfg.attention_impl == "flash_attention_2" and not _is_flash_attn_available():
            logging.warning(
                "flash_attention_2 requested but flash-attn is not installed; "
                "using the model default attention implementation instead."
            )
        else:
            kwargs["attn_implementation"] = cfg.attention_impl
    if cfg.device:
        kwargs["device_map"] = cfg.device
    if cfg.dtype:
        kwargs["dtype"] = cfg.dtype
    kwargs.update(cfg.model_kwargs)
    return kwargs


def _default_tokenizer_kwargs(cfg: SentenceTransformerConfig) -> dict[str, Any]:
    """Build tokenizer kwargs from config."""
    kwargs: dict[str, Any] = {
        "padding": True,
        "truncation": True,
    }
    if cfg.tokenizer_padding:
        kwargs["padding_side"] = cfg.tokenizer_padding
    kwargs.update(cfg.tokenizer_kwargs)
    return kwargs


def _load_model(cfg: SentenceTransformerConfig) -> SentenceTransformer:
    """Load SentenceTransformer model with config settings."""
    logging.info("Loading SentenceTransformer model: %s", cfg.model_name)
    model_kwargs = _default_model_kwargs(cfg)
    tokenizer_kwargs = _default_tokenizer_kwargs(cfg)
    try:
        model = SentenceTransformer(
            cfg.model_name,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            trust_remote_code=cfg.trust_remote_code,
        )
    except ValueError as exc:
        if cfg.attention_impl and "Flash Attention" in str(exc):
            logging.warning("Model %s does not support attention implementation '%s'; retrying with default.", cfg.model_name, cfg.attention_impl)
            model_kwargs.pop("attn_implementation", None)
            model = SentenceTransformer(
                cfg.model_name,
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs,
                trust_remote_code=cfg.trust_remote_code,
            )
        else:
            raise
    model = model.eval()
    if hasattr(model, "requires_grad_"):
        model.requires_grad_(False)
    return model


class _ModelRunner:
    """Manages model tokenization, compilation, and inference."""

    def __init__(self, model: SentenceTransformer, compile_enabled: bool, compile_mode: Optional[str]) -> None:
        self.model = model
        self.device = model.device
        self.compile_enabled = compile_enabled
        self.compile_mode = compile_mode or "default"
        self._compiled_forward: Optional[Any] = None
        self._compiled_cache: dict[int, Any] = {}

        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(f"Model {model} does not have a tokenizer attribute")
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError(f"Tokenizer {tokenizer} does not have a pad_token_id")
        self.pad_values: dict[str, int] = {
            "input_ids": pad_token_id,
            "attention_mask": 0,
            "token_type_ids": 0,
        }

        if self.compile_enabled and not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile requested but not available in this PyTorch build")

        if self.compile_enabled:
            def forward_fn(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
                return model.forward(features)

            self._compiled_forward = torch.compile(forward_fn, mode=self.compile_mode)

        self.model.eval()

    def tokenize(self, text: str, *, max_length: int) -> dict[str, torch.Tensor]:
        """Tokenize a single text string."""
        tokenized = self.model.tokenize([text], max_length=max_length, padding=False, truncation=True)
        return {key: value.squeeze(0) for key, value in tokenized.items()}

    def warmup(self, bucket_sizes: Iterable[int]) -> dict[int, float]:
        """Warmup compilation cache for all bucket sizes."""
        timings: dict[int, float] = {}
        if not self.compile_enabled or self._compiled_forward is None:
            return timings

        for size in bucket_sizes:
            if size in self._compiled_cache:
                continue

            features = self._dummy_features(size)
            start = time.perf_counter()
            with torch.inference_mode():
                self.forward(features)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings[size] = time.perf_counter() - start

        return timings

    def _dummy_features(self, seq_len: int) -> dict[str, torch.Tensor]:
        """Create dummy features for warmup."""
        features: dict[str, torch.Tensor] = {}
        for key, pad_value in self.pad_values.items():
            dtype = torch.long
            if key == "attention_mask":
                tensor = torch.ones((1, seq_len), dtype=dtype)
            else:
                tensor = torch.full((1, seq_len), pad_value, dtype=dtype)
            features[key] = tensor
        if "attention_mask" not in features:
            features["attention_mask"] = torch.ones((1, seq_len), dtype=torch.long)
        return features

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run model forward pass with optional compilation."""
        device_features = {key: value.to(self.device, non_blocking=True) for key, value in features.items()}
        if self.compile_enabled and self._compiled_forward is not None:
            bucket_size = next(iter(device_features.values())).shape[-1]
            compiled = self._compiled_cache.get(bucket_size)
            if compiled is None:
                compiled = self._compiled_forward
                self._compiled_cache[bucket_size] = compiled
            return compiled(device_features)
        return self.model.forward(device_features)
