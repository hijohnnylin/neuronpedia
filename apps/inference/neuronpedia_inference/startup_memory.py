"""Startup memory sizing: derive safe serving limits from the host + model.

This is the SINGLE place to tune how many concurrent requests we admit and how
many tokens a request may use, based on the memory that will actually be
available at serve time. It lives in the inference app (not the engine) because
only the server knows how much memory it spends on everything else (SAEs, the
vLLM KV cache reservation, activation/DFA capture buffers, etc.).

Contract:
- ``compute_serving_limits(...)`` returns ``ServingLimits(max_concurrent_requests,
  max_tokens)``. On vLLM we admit up to ``max_concurrent_requests`` in flight
  (vLLM batches them); off vLLM (EagerModel on CUDA/MPS/CPU) we always serve
  one request at a time, so ``max_concurrent_requests`` is pinned to 1.
- Everything is overridable at startup via env vars (see ``_env_*`` below) so ops
  can pin exact values without touching code.

The heuristic is intentionally conservative and approximate -- it is a safe
DEFAULT, not a guarantee. Tune the constants below.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning knobs (edit here). All are overridable by the matching env var.
# ---------------------------------------------------------------------------
# Aim to serve at least this many parallel requests. This is now a ceiling on the COUNT only:
# whether that many can actually run at once is decided per request by the byte budget
# (VramBudget + memory_cost.py), so a higher number lets many cheap requests (tokenize,
# single-source activations, lens) overlap without letting several expensive all-layers
# searches do the same. Before requests were costed individually, this number had to be low
# enough for the worst case of any four of them.
TARGET_MAX_CONCURRENT = 8
TARGET_MAX_TOKENS = 4096  # per-request sequence budget (prompt + generation)
MIN_MAX_CONCURRENT = 1
FLOOR_MAX_TOKENS = 512  # never shrink the token budget below this
# Fraction of *free* device memory we allow the concurrent KV/activation
# working set to use (leaves headroom for fragmentation + transient buffers).
KV_SAFETY_FRACTION = 0.7
# Per-token, per-request KV-cache bytes = 2 (K and V) * n_layers * n_kv_heads *
# head_dim * dtype_bytes. Activations add some more; pad with this multiplier.
KV_OVERHEAD_MULTIPLIER = 1.3
_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "float8": 1, "int8": 1}

# Fraction of the memory still free once EVERYTHING persistent is in place (model weights,
# SAE cache, Jacobian lens, and the vLLM engine's whole reservation) that we let concurrent
# per-request working sets occupy. The remainder absorbs allocator fragmentation and the
# error in the per-request cost estimates.
TRANSIENT_SAFETY_FRACTION = 0.8

# Below this there is no point rationing: report 0 (budget disabled) rather than admitting
# one request at a time against a budget too small for any real work.
MIN_TRANSIENT_BUDGET_BYTES = 256 * 1024**2

# --- SAE paging (see sae_cache.py) -----------------------------------------------------
# VRAM left for request working sets when deriving an "auto" SAE residency budget. The SAE
# cache gets a fraction of what remains AFTER this is set aside, so auto-sizing can never
# squeeze the request budget to nothing.
AUTO_SAE_TRANSIENT_RESERVE_BYTES = 4 * 1024**3
# ...and only this much of the remainder, leaving room for the estimate to be wrong.
AUTO_SAE_FRACTION = 0.8

# --- Jacobian lens residency (see lens_loader.py) ---------------------------------------
# VRAM left for request working sets when deriving an "auto" lens device budget. Same role as
# the SAE reserve above, but there is no fraction on top of it: a lens is a fixed size that
# does not grow with a request, so it is offered whatever is left and takes only what it
# needs. What it does not take is measured back as free by `measure_transient_budget`.
AUTO_JLENS_TRANSIENT_RESERVE_BYTES = 4 * 1024**3

# Host RAM held back from page-locking, for everything that is not an SAE: the vLLM engine
# process (which starts AFTER the SAEs are pinned, so it cannot be measured here), tokenizer
# and request buffers, and the CUDA runtime's own host allocations. Pinned pages can never be
# reclaimed by the kernel, so being wrong here degrades the whole box, not just this process.
HOST_PIN_RESERVE_BYTES = 12 * 1024**3
HOST_PIN_RESERVE_FRACTION_OF_TOTAL = 0.20
# Fraction of the remaining available RAM we are willing to page-lock. The rest of the SAEs
# fall back to pageable host memory, which stages in more slowly but costs nothing extra.
HOST_PIN_FRACTION = 0.6


@dataclass
class ModelMemoryInfo:
    """Minimal model dims the sizing heuristic needs (filled from the loaded backend)."""

    n_layers: int
    n_kv_heads: int
    head_dim: int
    dtype: str  # "bfloat16" | "float16" | "float32" | ...


@dataclass
class ServingLimits:
    max_concurrent_requests: int
    max_tokens: int

    def __str__(self) -> str:
        return f"ServingLimits(max_concurrent_requests={self.max_concurrent_requests}, max_tokens={self.max_tokens})"


def dtype_bytes(dtype: str) -> int:
    return _DTYPE_BYTES.get(str(dtype).lower().replace("torch.", ""), 2)


def detect_free_memory_bytes(device: str) -> int:
    """Best-effort free memory in bytes for the serving device.

    CUDA -> free VRAM on the device; MPS/CPU -> free system RAM. Returns 0 when it
    cannot be determined (callers then fall back to the target defaults).
    """
    dev = (device or "cpu").lower()
    try:
        if dev.startswith("cuda"):
            import torch

            # Pass the device explicitly: with num_gpus > 1 the SAE cache is loaded to
            # cuda:0 while the current device may be another shard, and the default would
            # then report the wrong card's free memory.
            free, _total = torch.cuda.mem_get_info(dev)
            return int(free)
        # MPS / CPU: use available system RAM.
        try:
            import psutil

            return int(psutil.virtual_memory().available)
        except Exception:  # noqa: BLE001 - psutil optional
            page_size = os.sysconf("SC_PAGE_SIZE")
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
            return int(page_size * avail_pages)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not detect free memory for device %r: %s", device, exc)
        return 0


def total_device_memory_bytes(device: str) -> int:
    """Total VRAM on the serving device, or 0 when it is not a CUDA device."""
    if not (device or "").lower().startswith("cuda"):
        return 0
    try:
        import torch

        _free, total = torch.cuda.mem_get_info(device)
        return int(total)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read total memory for device %r: %s", device, exc)
        return 0


def resolve_sae_gpu_budget_bytes(
    setting: str | None,
    *,
    device: str,
    is_vllm: bool,
    vllm_gpu_utilization: float,
) -> int:
    """How much VRAM the SAE residency cache may hold. 0 disables paging.

    ``setting`` comes from ``SAE_GPU_BUDGET_GIB``: a number of GiB, ``"auto"``, or unset.
    Unset keeps the historical behaviour (every SAE resident forever), which is why paging
    is opt-in rather than a silent change to every existing pod.

    ``"auto"`` derives the budget from what will be left of the card once the model is up.
    Under vLLM that cannot be measured at the point this is called -- the engine reserves
    ``gpu_memory_utilization`` of the WHOLE card later, in a child process -- so it is
    computed from the utilization figure instead. Under the eager backend the model is
    already loaded, so free memory is measured directly.
    """
    raw = (setting or "").strip().lower()
    if not raw or raw in {"0", "off", "false", "none"}:
        return 0
    if not (device or "").lower().startswith("cuda"):
        logger.info("[startup_memory] SAE paging ignored on non-CUDA device %r", device)
        return 0

    if raw != "auto":
        try:
            return max(0, int(float(raw) * 1024**3))
        except ValueError:
            logger.warning(
                "Ignoring non-numeric SAE_GPU_BUDGET_GIB=%r; SAE paging disabled",
                setting,
            )
            return 0

    if is_vllm:
        total = total_device_memory_bytes(device)
        available = int(total * max(0.0, 1.0 - float(vllm_gpu_utilization)))
    else:
        available = detect_free_memory_bytes(device)
    budget = int(max(0, available - AUTO_SAE_TRANSIENT_RESERVE_BYTES) * AUTO_SAE_FRACTION)
    logger.info(
        "[startup_memory] SAE_GPU_BUDGET_GIB=auto -> %.2f GiB "
        "(%.2f GiB expected free, less %.2f GiB reserved for requests, x%.2f)",
        budget / 1024**3,
        available / 1024**3,
        AUTO_SAE_TRANSIENT_RESERVE_BYTES / 1024**3,
        AUTO_SAE_FRACTION,
    )
    return budget


def resolve_jlens_gpu_budget_bytes(
    setting: str | None,
    *,
    device: str | None,
    reserved_bytes: int = 0,
) -> int:
    """GPU bytes the Jacobian lens's per-layer device cache may hold. 0 keeps it off the GPU.

    ``setting`` comes from ``JLENS_GPU_BUDGET_GIB``: a number of GiB, ``"auto"`` (the
    default), or an off switch. Unlike the SAE budget, ``auto`` is the default rather than
    opt-in, because what it sizes is not optional -- the read-out transports through every
    fitted layer whatever this returns, and the only question is whether it does so from
    resident copies or by re-reading the lens across PCIe on every batch.

    ``reserved_bytes`` is memory that is free right now but already spoken for: the SAE
    residency budget the cache has not warmed into yet, which would otherwise be offered to
    the lens and to the SAEs both.

    Measured rather than configured, and measured rather than estimated from
    ``gpu_memory_utilization`` the way :func:`resolve_sae_gpu_budget_bytes` has to be. Its
    caller runs after the vLLM engine has taken its pool as well as after the model weights
    and SAE cache, so free memory here is the real remainder on either backend -- see
    ``place_jacobian_lens_on_device``, which exists to make that ordering true. Whatever the
    lens does not take is measured back as free by :func:`measure_transient_budget`, which
    runs last.
    """
    raw = (setting or "auto").strip().lower()
    if raw in {"0", "off", "false", "none"}:
        return 0
    cuda_device = device or ""
    if not cuda_device.lower().startswith("cuda"):
        # Nothing to ration: the transport runs wherever the residuals already are, which off
        # CUDA is the lens's own storage rather than a second copy.
        return 0

    if raw != "auto":
        try:
            return max(0, int(float(raw) * 1024**3))
        except ValueError:
            logger.warning(
                "Ignoring non-numeric JLENS_GPU_BUDGET_GIB=%r; falling back to auto",
                setting,
            )

    free = detect_free_memory_bytes(cuda_device)
    budget = max(0, free - AUTO_JLENS_TRANSIENT_RESERVE_BYTES - max(0, reserved_bytes))
    logger.info(
        "[startup_memory] JLENS_GPU_BUDGET_GIB=auto -> %.2f GiB "
        "(%.2f GiB free, less %.2f GiB held for requests and %.2f GiB of unwarmed SAE residency)",
        budget / 1024**3,
        free / 1024**3,
        AUTO_JLENS_TRANSIENT_RESERVE_BYTES / 1024**3,
        max(0, reserved_bytes) / 1024**3,
    )
    return budget


def measure_pinnable_host_bytes(override_gib: float | None = None) -> int:
    """Host RAM we may page-lock for SAE master copies.

    Page-locked pages are unswappable and unreclaimable, so this is measured from what the
    host actually has free right now rather than configured, and a generous reserve is held
    back for the processes that allocate after this point (notably the vLLM engine).
    Returning less than the SAEs need is fine -- the overflow lives in pageable memory.
    """
    if override_gib is not None:
        return max(0, int(override_gib * 1024**3))
    try:
        import psutil

        vm = psutil.virtual_memory()
        available = int(vm.available)
        total = int(vm.total)
    except Exception as exc:  # noqa: BLE001 - psutil optional
        logger.warning("Could not measure host memory (%s); SAE pinning disabled", exc)
        return 0

    reserve = max(HOST_PIN_RESERVE_BYTES, int(total * HOST_PIN_RESERVE_FRACTION_OF_TOTAL))
    pinnable = int(max(0, available - reserve) * HOST_PIN_FRACTION)
    logger.info(
        "[startup_memory] pinnable host memory %.2f GiB (%.2f GiB available, %.2f GiB reserved, x%.2f)",
        pinnable / 1024**3,
        available / 1024**3,
        reserve / 1024**3,
        HOST_PIN_FRACTION,
    )
    return pinnable


def compute_activation_token_limit(
    *,
    budget_bytes: int,
    token_limit: int,
    d_sae: int,
    d_in: int,
    n_hooks: int,
    sae_dtype: str,
    model_dtype: str,
) -> int:
    """Largest prompt an all-layers activation search can hold in ``budget_bytes``.

    ``token_limit`` stays the completion/steer/tokenize cap (and the vLLM ``max_model_len``);
    this is a separate, never-higher bound for the activation endpoints, whose peak is
    ``O(d_sae * tokens)`` after the streaming top-K rewrite. Derived at the end of startup
    from the measured budget and the widest configured SAE, so a newly added wider SAE
    shrinks the cap by itself -- no pods.yaml edit.

    Returns ``token_limit`` unchanged when there is no budget to ration, or when the budget
    already fits ``token_limit`` tokens of the worst-case encode+capture.
    """
    if budget_bytes <= 0 or d_sae <= 0 or d_in <= 0:
        return max(1, int(token_limit))

    # Same dominant terms as memory_cost.activation_all_cost / sae_memory.transient_bytes:
    # one encode of the widest source (doubled for the activation temporary) plus one
    # capture tensor per distinct hook. Result-buffer cost is negligible next to these.
    sae_b = dtype_bytes(sae_dtype)
    model_b = dtype_bytes(model_dtype)
    per_token = int(KV_OVERHEAD_MULTIPLIER * (2 * d_sae * sae_b + max(1, n_hooks) * d_in * model_b))
    per_token = max(per_token, 1)
    # Fit at least one such request in the budget; concurrency for cheap requests is
    # handled by VramBudget, not by shrinking this further.
    derived = budget_bytes // per_token
    capped = min(int(token_limit), max(FLOOR_MAX_TOKENS, derived))
    logger.info(
        "[startup_memory] activation_token_limit=%d "
        "(token_limit=%d, budget=%.2fGiB, d_sae=%d, d_in=%d, n_hooks=%d, %dB/token)",
        capped,
        token_limit,
        budget_bytes / (1024**3),
        d_sae,
        d_in,
        n_hooks,
        per_token,
    )
    return capped


def measure_transient_budget(device: str) -> int:
    """Bytes available for concurrent per-request working sets. Call LAST, after warmup.

    Everything else in this module estimates; this one measures. It must run after the model,
    the SAE cache, the Jacobian lens and (crucially) the vLLM engine have all claimed their
    memory, because those are what determine how little is left:

    - The SAE cache lives in THIS process, outside vLLM's pool.
    - vLLM reserves ``VLLM_GPU_MEMORY_UTILIZATION`` of the whole card up front, in a child
      process. ``mem_get_info`` reports device-wide free memory, so it sees that reservation;
      ``torch.cuda.memory_allocated`` would not.

    What remains is the only memory a request's activation/DFA/lens buffers can come from --
    on an A40 running gemma-2-2b that is roughly 2.4 GiB, shared by every request in flight.

    Returns 0 when there is nothing to ration (non-CUDA device, or a budget too small to be
    useful), which callers treat as "budget disabled". Override with ``TRANSIENT_BUDGET_MIB``.
    """
    override_mib = _env_int("TRANSIENT_BUDGET_MIB")
    if override_mib is not None:
        budget = max(0, override_mib) * 1024**2
        logger.info(
            "[startup_memory] transient budget pinned by TRANSIENT_BUDGET_MIB: %.2f GiB",
            budget / (1024**3),
        )
        return budget

    if not (device or "").lower().startswith("cuda"):
        # One-at-a-time serving on eager CPU/MPS already bounds the working set.
        logger.info("[startup_memory] transient budget disabled on non-CUDA device %r", device)
        return 0

    free = detect_free_memory_bytes(device)
    budget = int(free * TRANSIENT_SAFETY_FRACTION)
    if budget < MIN_TRANSIENT_BUDGET_BYTES:
        logger.warning(
            "[startup_memory] only %.2f GiB free after warmup -- too little to ration, "
            "disabling the transient budget. Lower VLLM_GPU_MEMORY_UTILIZATION or the SAE "
            "cache if requests start OOMing.",
            free / (1024**3),
        )
        return 0

    logger.info(
        "[startup_memory] transient budget %.2f GiB (%.0f%% of %.2f GiB free after warmup)",
        budget / (1024**3),
        TRANSIENT_SAFETY_FRACTION * 100,
        free / (1024**3),
    )
    return budget


def _kv_bytes_per_token(info: ModelMemoryInfo) -> int:
    return int(2 * info.n_layers * info.n_kv_heads * info.head_dim * dtype_bytes(info.dtype))


def _env_int(name: str) -> int | None:
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return None
    try:
        return int(val)
    except ValueError:
        logger.warning("Ignoring non-integer env %s=%r", name, val)
        return None


def compute_serving_limits(
    *,
    device: str,
    is_vllm: bool,
    model_info: ModelMemoryInfo,
    kv_budget_bytes: int | None = None,
) -> ServingLimits:
    """Compute ``(max_concurrent_requests, max_tokens)`` for the loaded model.

    ``kv_budget_bytes`` optionally overrides the memory available for the
    concurrent KV/activation working set (e.g. pass the vLLM KV reservation);
    otherwise a fraction of detected free memory is used.

    Env overrides (take precedence over the heuristic): ``MAX_CONCURRENT_REQUESTS``,
    ``MAX_TOKENS``.
    """
    env_concurrent = _env_int("MAX_CONCURRENT_REQUESTS")
    env_tokens = _env_int("MAX_TOKENS")

    # Off vLLM we serve strictly one at a time (eager forward is not concurrency-safe).
    hard_max_concurrent = TARGET_MAX_CONCURRENT if is_vllm else 1

    if kv_budget_bytes is None:
        free = detect_free_memory_bytes(device)
        kv_budget_bytes = int(free * KV_SAFETY_FRACTION)

    per_token = max(_kv_bytes_per_token(model_info), 1)
    per_token = int(per_token * KV_OVERHEAD_MULTIPLIER)

    max_tokens = env_tokens if env_tokens is not None else TARGET_MAX_TOKENS
    max_concurrent = env_concurrent if env_concurrent is not None else hard_max_concurrent

    # If both are heuristic (no env pins) and the budget is known, shrink to fit:
    # first drop concurrency toward MIN, then the token budget toward the floor.
    if kv_budget_bytes > 0 and (env_concurrent is None or env_tokens is None):

        def fits(nc: int, mt: int) -> bool:
            return nc * mt * per_token <= kv_budget_bytes

        if env_concurrent is None:
            while max_concurrent > MIN_MAX_CONCURRENT and not fits(max_concurrent, max_tokens):
                max_concurrent -= 1
        if env_tokens is None:
            while max_tokens > FLOOR_MAX_TOKENS and not fits(max_concurrent, max_tokens):
                max_tokens = max(FLOOR_MAX_TOKENS, max_tokens // 2)

    max_concurrent = max(
        MIN_MAX_CONCURRENT,
        min(max_concurrent, hard_max_concurrent) if env_concurrent is None else max_concurrent,
    )
    max_tokens = max(FLOOR_MAX_TOKENS if env_tokens is None else 1, max_tokens)

    limits = ServingLimits(max_concurrent_requests=int(max_concurrent), max_tokens=int(max_tokens))
    logger.info(
        "[startup_memory] device=%s is_vllm=%s kv_budget=%.2fGiB kv_per_token=%dB -> %s",
        device,
        is_vllm,
        (kv_budget_bytes or 0) / (1024**3),
        per_token,
        limits,
    )
    return limits
