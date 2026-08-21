"""Startup memory sizing for the NLA server (tuning knobs live here).

NLA is already concurrency-built (per-stage asyncio semaphores + env overrides:
NLA_MAX_CONCURRENT, NLA_MAX_NEW_TOKENS_LIMIT, ...). This module is the SINGLE place
to derive their DEFAULTS from the memory actually available at serve time, so an
operator doesn't have to hand-pick numbers per GPU. Any explicit env var still wins.

Kept in the NLA app (not the engine) because only the server knows how much memory
it spends on everything else (the source model, the reconstructor, SGLang/vLLM KV,
concept-injection buffers, ...). Edit the constants below to tune.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning knobs (edit here). Explicit env vars override the derived defaults.
# ---------------------------------------------------------------------------
TARGET_MAX_CONCURRENT = 4  # aim to serve at least this many parallel verbalizations
MAX_MAX_CONCURRENT = 24  # never auto-derive above this (NLA's historical default)
MIN_MAX_CONCURRENT = 1
TARGET_MAX_NEW_TOKENS = 4096
FLOOR_MAX_NEW_TOKENS = 512
# Rough per-concurrent-verbalization working-set budget (GiB) at TARGET tokens.
# Verbalizer KV + activations for a small verbalizer; tune per deployment.
GIB_PER_CONCURRENT_REQUEST = 2.0
MEMORY_SAFETY_FRACTION = 0.7


@dataclass
class ServingLimits:
    max_concurrent_requests: int
    max_new_tokens: int

    def __str__(self) -> str:
        return (
            f"ServingLimits(max_concurrent_requests={self.max_concurrent_requests}, "
            f"max_new_tokens={self.max_new_tokens})"
        )


def detect_free_memory_bytes(device: str) -> int:
    """Best-effort free memory (bytes): free VRAM on CUDA, else free system RAM."""
    dev = (device or "cpu").lower()
    try:
        if dev.startswith("cuda"):
            import torch

            idx = 0
            if ":" in dev:
                try:
                    idx = int(dev.split(":", 1)[1])
                except ValueError:
                    idx = 0
            free, _total = torch.cuda.mem_get_info(idx)
            return int(free)
        try:
            import psutil

            return int(psutil.virtual_memory().available)
        except Exception:  # noqa: BLE001
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not detect free memory for %r: %s", device, exc)
        return 0


def _env_int(name: str) -> int | None:
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def compute_serving_limits(device: str) -> ServingLimits:
    """Derive ``(max_concurrent_requests, max_new_tokens)`` defaults from free memory.

    Env overrides win: ``NLA_MAX_CONCURRENT``, ``NLA_MAX_NEW_TOKENS_LIMIT``.
    """
    env_concurrent = _env_int("NLA_MAX_CONCURRENT")
    env_tokens = _env_int("NLA_MAX_NEW_TOKENS_LIMIT")

    free = detect_free_memory_bytes(device)
    budget = free * MEMORY_SAFETY_FRACTION
    per_request = GIB_PER_CONCURRENT_REQUEST * (1024**3)

    if env_concurrent is not None:
        max_concurrent = env_concurrent
    elif budget <= 0:
        max_concurrent = TARGET_MAX_CONCURRENT
    else:
        derived = int(budget // per_request)
        max_concurrent = max(MIN_MAX_CONCURRENT, min(derived, MAX_MAX_CONCURRENT))
        max_concurrent = max(
            max_concurrent,
            TARGET_MAX_CONCURRENT if derived >= TARGET_MAX_CONCURRENT else max_concurrent,
        )

    max_new_tokens = env_tokens if env_tokens is not None else TARGET_MAX_NEW_TOKENS
    max_new_tokens = max(FLOOR_MAX_NEW_TOKENS if env_tokens is None else 1, max_new_tokens)

    limits = ServingLimits(int(max_concurrent), int(max_new_tokens))
    logger.info(
        "[nla.startup_memory] device=%s free=%.2fGiB -> %s",
        device,
        free / (1024**3),
        limits,
    )
    return limits
