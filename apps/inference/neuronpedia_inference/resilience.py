"""Server resilience: recover from irrecoverable CUDA states by restarting.

Some CUDA failures (device-side asserts, illegal memory accesses, launch failures,
and often post-OOM states) poison the CUDA context for the whole process -- every
subsequent kernel fails. There is no in-process recovery: the only fix is to
restart. This module detects those states and cleanly terminates the process so a
supervisor (systemd / docker/k8s restart policy / pod controller) relaunches it.

Critically, it kills child processes first: the engine-owned vLLM ``EngineCore``
runs in a child process that otherwise ORPHANS on a bare ``os._exit`` and keeps
holding GPU memory, so the restarted server can't allocate. Killing children
reclaims that memory before we exit.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os

logger = logging.getLogger(__name__)

# Substrings that indicate a poisoned / unrecoverable CUDA context (lowercased match).
_FATAL_CUDA_MARKERS = (
    "device-side assert",
    "illegal memory access",
    "unspecified launch failure",
    "misaligned address",
    "uncorrectable ecc",
    "cublas_status",
    "cuda error",
    "cuda assertion",
    "an illegal instruction",
    "out of memory",  # post-OOM the allocator/engine is frequently wedged; restart to be safe
)


def is_allocator_oom(exc: BaseException) -> bool:
    """True for a torch CUDA allocator OOM, which leaves the CUDA context INTACT.

    Distinct from the fatal states above: the allocator refuses the request before launching
    anything, so no kernel ever ran and nothing is poisoned. Once admission control bounds
    each request's working set (see :class:`~neuronpedia_inference.shared.VramBudget`), this
    means "too much in flight right now", which is a 503 the client can retry -- not a reason
    to drop every other in-flight request and reload the model.

    Deliberately narrow: matched by exception TYPE, not by an "out of memory" substring, so an
    OOM surfacing from the vLLM engine child (where the engine really may be dead) still
    counts as fatal below. And it is only optimistic, not a guarantee: `probe_cuda_or_die`
    runs after every request and restarts anyway if the context turns out to be wedged.
    """
    if type(exc).__name__ != "OutOfMemoryError":
        return False
    try:
        import torch

        return isinstance(exc, torch.cuda.OutOfMemoryError)
    except Exception:  # noqa: BLE001
        return False


def is_fatal_cuda_error(exc: BaseException) -> bool:
    """True if ``exc`` looks like an irrecoverable CUDA state (see markers above)."""
    if is_allocator_oom(exc):
        return False
    text = f"{type(exc).__name__}: {exc}".lower()
    # torch.cuda.OutOfMemoryError subclasses RuntimeError; catch by name too.
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return any(marker in text for marker in _FATAL_CUDA_MARKERS)


def reclaim_after_oom() -> None:
    """Return cached-but-unused blocks to the driver after an OOM.

    Worth the latency only on this path: the allocator is holding a fragmented pool that just
    failed to satisfy a request, and without this the next request inherits the fragmentation.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best effort; never mask the original error
        logger.warning("[RESILIENCE] empty_cache() failed after OOM", exc_info=True)


def terminate_for_restart(reason: str) -> None:
    """Kill child processes (reclaim GPU) then exit(1) so the supervisor restarts us."""
    logger.error("[RESILIENCE] Irrecoverable CUDA state; terminating for restart: %s", reason)
    try:
        import psutil

        me = psutil.Process()
        for child in me.children(recursive=True):
            with contextlib.suppress(Exception):
                child.kill()  # e.g. the vLLM EngineCore subprocess (frees its VRAM)
    except Exception:  # noqa: BLE001
        pass
    try:
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass
    os._exit(1)


def probe_cuda_or_die(device: str | None = None) -> None:
    """Cheap post-request health probe: if the CUDA context is poisoned, restart.

    Runs ``mem_get_info`` (a no-op query that nonetheless fails on a poisoned
    context) and, on a CUDA-error RuntimeError, escalates to ``terminate_for_restart``.
    Called after each request so a swallowed device-side error still triggers a
    restart instead of serving a wedged model.

    Every visible card is probed, not just ``cuda:0``. A sharded model runs work on all of
    them, and a device-side assert poisons the context of the card it happened on -- so
    probing one card would let a wedged shard keep taking requests.
    """
    if device is not None and not str(device).startswith("cuda"):
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        for index in range(torch.cuda.device_count()):
            torch.cuda.mem_get_info(torch.device(f"cuda:{index}"))
    except RuntimeError as exc:
        if is_fatal_cuda_error(exc):
            terminate_for_restart(f"post-request CUDA probe failed: {exc}")
    except Exception:  # noqa: BLE001 - never crash the probe itself
        pass
