"""Page SAEs between host RAM and a byte-budgeted GPU residency cache.

Today every configured SAE is copied to the GPU at startup and stays there for the life of
the process. That is the fastest possible arrangement and also the most wasteful one: a
26-source gemmascope set is tens of GiB of VRAM sitting idle, and it is charged against the
same card the model weights, the vLLM KV pool and every request's working set come out of.
The cost is paid even though a single request touches one source at a time.

This module keeps the master copy of every SAE in host RAM (page-locked where the host can
afford it) and lets at most ``sae_gpu_budget_bytes`` of them be resident on the GPU at once,
evicting least-recently-used. Staging a 600 MiB SAE in from pinned memory takes tens of
milliseconds -- real, but small next to a forward pass, and it buys back VRAM that can go to
the KV pool or to request concurrency.

Two invariants make eviction safe without making ``get_sae()`` async:

1. A request RESERVES residency bytes before it runs, through :meth:`SAEGpuCache.reserve`,
   and total reservations never exceed the budget. So the bytes a request will need are
   already accounted for by the time it stages anything in.
2. A request holds AT MOST ONE source resident at a time -- ``acquire`` releases the
   previous hold -- and eviction skips whatever is currently held. Since held bytes are
   bounded by reserved bytes, and reserved bytes by the budget, there is always enough
   evictable residency to fit the next stage-in. Eviction can therefore never pull an SAE
   out from under a request that is mid-encode.

Endpoints that walk many sources (``/activation/all``) already read one source, finish with
it, then move on, which is exactly the access pattern invariant 2 describes. Anything that
needs two SAEs live simultaneously would have to reserve for both.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Log a stage-in that took longer than this, so a badly sized budget (thrashing) shows up in
# the logs as latency rather than as a mystery.
SLOW_STAGE_IN_SECONDS = 0.25


def _assign(module: Any, qualified_name: str, tensor: torch.Tensor) -> None:
    """Point ``module.<qualified_name>`` at ``tensor`` without replacing the Parameter.

    ``load_state_dict(assign=True)`` would swap in a NEW ``Parameter`` object, invalidating
    any reference an endpoint happens to be holding (``sae.W_dec``). Rebinding ``.data``
    keeps object identity and is what ``nn.Module.to()`` does internally.
    """
    parent_path, _, attr = qualified_name.rpartition(".")
    parent = module.get_submodule(parent_path) if parent_path else module
    current = getattr(parent, attr)
    if isinstance(current, torch.nn.Parameter):
        current.data = tensor
    else:
        setattr(parent, attr, tensor)


def _set_module_device(module: Any, device: str) -> None:
    """Update the placement attributes SAELens keeps alongside the parameters."""
    if hasattr(module, "device"):
        module.device = torch.device(device)
    cfg = getattr(module, "cfg", None)
    if cfg is not None and hasattr(cfg, "device"):
        cfg.device = device


def _collect_host_state(module: Any) -> dict[str, torch.Tensor]:
    """Every parameter and buffer of ``module``, keyed by qualified name."""
    state: dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters(recurse=True):
        state[name] = param.data
    for name, buf in module.named_buffers(recurse=True):
        if isinstance(buf, torch.Tensor):
            state[name] = buf
    return state


def state_nbytes(state: dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in state.values())


class HostPinner:
    """Page-locks SAE host copies up to a budget, falling back to ordinary memory.

    Pinned memory is what makes a stage-in fast (a DMA straight from the page-locked pages
    instead of a bounce through a staging buffer), but it is memory the kernel can never
    reclaim, so over-pinning starves everything else on the box -- including vLLM's own host
    allocations. The budget is measured at startup rather than configured
    (``startup_memory.measure_pinnable_host_bytes``), and running out of it is not an error:
    the remaining SAEs simply live in pageable memory and stage in more slowly.
    """

    def __init__(self) -> None:
        self._budget = 0
        self._used = 0
        self._refused = 0

    def configure(self, budget_bytes: int) -> None:
        self._budget = max(0, int(budget_bytes))
        self._used = 0
        self._refused = 0

    @property
    def used_bytes(self) -> int:
        return self._used

    @property
    def budget_bytes(self) -> int:
        return self._budget

    @property
    def refused_count(self) -> int:
        return self._refused

    def pin(self, state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
        """Return a page-locked copy of ``state``, or ``state`` itself if it does not fit."""
        nbytes = state_nbytes(state)
        if self._budget <= 0 or self._used + nbytes > self._budget:
            self._refused += 1
            return state, False
        try:
            pinned = {name: tensor.pin_memory() for name, tensor in state.items()}
        except RuntimeError as exc:  # cudaHostAlloc can fail well before the budget does
            logger.warning(
                "[SAE-CACHE] could not pin %.2f GiB of host memory (%s); continuing with pageable memory",
                nbytes / 1024**3,
                exc,
            )
            self._refused += 1
            return state, False
        self._used += nbytes
        return pinned, True


class PagedSAE:
    """One SAE whose weights live on the host and are copied to the GPU on demand."""

    def __init__(
        self,
        sae_id: str,
        module: Any,
        host_state: dict[str, torch.Tensor],
        pinned: bool,
    ) -> None:
        self.sae_id = sae_id
        self.module = module
        self.host_state = host_state
        self.pinned = pinned
        self.nbytes = state_nbytes(host_state)
        self.on_gpu = False
        self.stage_ins = 0

    def stage_in(self, device: str) -> None:
        if self.on_gpu:
            return
        started = time.monotonic()
        # non_blocking is only actually asynchronous from page-locked source memory, and is
        # safe either way: the copy is enqueued on the same stream as the kernels that read
        # it, so it is ordered before them.
        self.module.to(device, non_blocking=self.pinned)
        self.on_gpu = True
        self.stage_ins += 1
        elapsed = time.monotonic() - started
        if elapsed > SLOW_STAGE_IN_SECONDS:
            logger.info(
                "[SAE-CACHE] staged in %s (%.2f GiB, pinned=%s) in %.0f ms",
                self.sae_id,
                self.nbytes / 1024**3,
                self.pinned,
                elapsed * 1000,
            )

    def stage_out(self) -> None:
        """Rebind the module to its host tensors, dropping the GPU copies.

        Deliberately not ``module.to("cpu")``: that would copy the weights back down and
        leave the module pointing at fresh pageable tensors, throwing away the page-locked
        master and making the next stage-in slower. Rebinding is free and keeps the master.
        """
        if not self.on_gpu:
            return
        for name, tensor in self.host_state.items():
            _assign(self.module, name, tensor)
        # SAELens tracks placement outside the parameters (`SAE.to` sets both), and code
        # reads `sae.device` to place its own tensors. Rebinding skips that, so mirror it.
        _set_module_device(self.module, "cpu")
        self.on_gpu = False


class _RequestResidency:
    """One in-flight request's claim on the GPU cache.

    ``held`` is the single source the request is currently using; eviction treats it as
    untouchable. ``reserved`` is what the request was admitted for, which bounds ``held``.
    """

    __slots__ = ("reserved", "held")

    def __init__(self, reserved: int) -> None:
        self.reserved = reserved
        self.held: str | None = None


_current_residency: ContextVar[_RequestResidency | None] = ContextVar("np_sae_residency", default=None)


class SAEGpuCache:
    """LRU cache of GPU-resident SAEs, bounded in bytes rather than in count.

    Disabled (``enabled == False``) unless a positive budget is configured, in which case
    every call here is a no-op and :class:`~neuronpedia_inference.sae_manager.SAEManager`
    keeps its historical behaviour of holding every SAE on the GPU forever.
    """

    def __init__(self) -> None:
        self._budget_bytes = 0
        self._device = "cpu"
        self._records: dict[str, PagedSAE] = {}
        self._resident: OrderedDict[str, PagedSAE] = OrderedDict()
        self._resident_bytes = 0
        self._lock = threading.RLock()
        self._reserved_bytes = 0
        self._active: set[_RequestResidency] = set()
        self._condition = asyncio.Condition()
        self.pinner = HostPinner()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # -- configuration -----------------------------------------------------------------

    def configure(self, *, budget_bytes: int, device: str, pinned_host_bytes: int) -> None:
        self._budget_bytes = max(0, int(budget_bytes))
        self._device = device
        self.pinner.configure(pinned_host_bytes)
        if self.enabled:
            logger.info(
                "[SAE-CACHE] paging enabled: %.2f GiB GPU residency on %s, %.2f GiB pinnable host memory",
                self._budget_bytes / 1024**3,
                device,
                pinned_host_bytes / 1024**3,
            )
        else:
            logger.info("[SAE-CACHE] paging disabled (all SAEs stay GPU-resident)")

    def reset(self) -> None:
        """Drop all state. For tests and for a re-init inside one process."""
        with self._lock:
            for record in list(self._resident.values()):
                record.stage_out()
            self._records.clear()
            self._resident.clear()
            self._resident_bytes = 0
            self._reserved_bytes = 0
            self._active.clear()
            self._budget_bytes = 0
            self.hits = self.misses = self.evictions = 0

    @property
    def enabled(self) -> bool:
        return self._budget_bytes > 0

    @property
    def budget_bytes(self) -> int:
        return self._budget_bytes

    @property
    def resident_bytes(self) -> int:
        return self._resident_bytes

    @property
    def host_bytes(self) -> int:
        return sum(record.nbytes for record in self._records.values())

    # -- registration ------------------------------------------------------------------

    def register(self, sae_id: str, module: Any) -> int:
        """Take ownership of a CPU-resident SAE. Returns its size in bytes.

        The module must already be on the CPU and fully transformed (folded, reshaped) --
        after this it is only ever moved between host and device, never modified.
        """
        state = _collect_host_state(module)
        host_state, pinned = self.pinner.pin(state)
        if pinned:
            for name, tensor in host_state.items():
                _assign(module, name, tensor)
        record = PagedSAE(sae_id, module, host_state, pinned)
        with self._lock:
            self._records[sae_id] = record
            # A budget smaller than a single SAE would make that source permanently
            # unservable, so it wins over the configured number. Better to overshoot the
            # VRAM target than to 503 every request for one set.
            if record.nbytes > self._budget_bytes:
                logger.warning(
                    "[SAE-CACHE] raising GPU residency budget from %.2f to %.2f GiB: "
                    "%s alone does not fit in the configured budget",
                    self._budget_bytes / 1024**3,
                    record.nbytes / 1024**3,
                    sae_id,
                )
                self._budget_bytes = record.nbytes
        return record.nbytes

    def unregister(self, sae_id: str) -> None:
        with self._lock:
            record = self._records.pop(sae_id, None)
            if record is None:
                return
            if record.sae_id in self._resident:
                self._evict_locked(record.sae_id)

    def is_registered(self, sae_id: str) -> bool:
        return sae_id in self._records

    def nbytes_for(self, sae_id: str) -> int:
        record = self._records.get(sae_id)
        return record.nbytes if record else 0

    def largest_nbytes(self) -> int:
        return max((r.nbytes for r in self._records.values()), default=0)

    # -- residency ---------------------------------------------------------------------

    def acquire(self, sae_id: str) -> Any:
        """Return the SAE, staging it onto the GPU first if it is not already there.

        Releases the calling request's previous hold, so a loop over sources keeps exactly
        one of them pinned against eviction.
        """
        record = self._records.get(sae_id)
        if record is None:
            return None
        residency = _current_residency.get()
        with self._lock:
            if residency is not None:
                residency.held = sae_id
            if record.on_gpu:
                self._resident.move_to_end(sae_id)
                self.hits += 1
                return record.module
            self.misses += 1
            self._make_room_locked(record.nbytes, keep=sae_id)
            record.stage_in(self._device)
            self._resident[sae_id] = record
            self._resident_bytes += record.nbytes
            return record.module

    def peek(self, sae_id: str) -> Any:
        """The module without touching residency. Only safe for CPU-side metadata reads."""
        record = self._records.get(sae_id)
        return record.module if record else None

    def _make_room_locked(self, needed: int, *, keep: str) -> None:
        held = {r.held for r in self._active if r.held is not None}
        held.add(keep)
        for sae_id in list(self._resident):
            if self._resident_bytes + needed <= self._budget_bytes:
                return
            if sae_id in held:
                continue
            self._evict_locked(sae_id)
        if self._resident_bytes + needed > self._budget_bytes:
            # Only reachable if a caller staged in without reserving, or holds two sources
            # at once. Staging anyway is still the least-bad option: refusing here would
            # fail a request that probably fits, and the allocator will tell us if it
            # does not.
            logger.warning(
                "[SAE-CACHE] over budget: %.2f GiB resident + %.2f GiB needed > %.2f GiB "
                "(every eviction candidate is in use)",
                self._resident_bytes / 1024**3,
                needed / 1024**3,
                self._budget_bytes / 1024**3,
            )

    def _evict_locked(self, sae_id: str) -> None:
        record = self._resident.pop(sae_id, None)
        if record is None:
            return
        record.stage_out()
        self._resident_bytes -= record.nbytes
        self.evictions += 1
        logger.debug(
            "[SAE-CACHE] evicted %s (%.2f GiB), %.2f GiB resident",
            sae_id,
            record.nbytes / 1024**3,
            self._resident_bytes / 1024**3,
        )

    def warm(self, sae_ids: list[str]) -> None:
        """Fill the GPU cache up to the budget at startup, in ``sae_ids`` order.

        Not just a latency optimisation: it also makes the post-startup free-VRAM
        measurement honest, because the request budget is measured from what is left once
        the SAE cache has claimed its share.
        """
        if not self.enabled:
            return
        with self._lock:
            for sae_id in sae_ids:
                record = self._records.get(sae_id)
                if record is None or record.on_gpu:
                    continue
                if self._resident_bytes + record.nbytes > self._budget_bytes:
                    continue
                record.stage_in(self._device)
                self._resident[sae_id] = record
                self._resident_bytes += record.nbytes
            logger.info(
                "[SAE-CACHE] warmed %d/%d SAEs onto %s (%.2f of %.2f GiB)",
                len(self._resident),
                len(self._records),
                self._device,
                self._resident_bytes / 1024**3,
                self._budget_bytes / 1024**3,
            )

    # -- admission ---------------------------------------------------------------------

    @asynccontextmanager
    async def reserve(self, nbytes: int, *, timeout: float = 0.0):
        """Reserve room for the largest single SAE this request will stage in.

        Held for the whole request, alongside the transient VRAM reservation, so the number
        of requests admitted is bounded by SAE residency as well as by working set.
        """
        if not self.enabled or nbytes <= 0:
            yield None
            return
        nbytes = min(int(nbytes), self._budget_bytes)

        async with self._condition:
            has_room = self._condition.wait_for(lambda: self._reserved_bytes + nbytes <= self._budget_bytes)
            if timeout and timeout > 0:
                await asyncio.wait_for(has_room, timeout=timeout)
            else:
                await has_room
            self._reserved_bytes += nbytes

        residency = _RequestResidency(nbytes)
        with self._lock:
            self._active.add(residency)
        token = _current_residency.set(residency)
        try:
            yield residency
        finally:
            _current_residency.reset(token)
            with self._lock:
                self._active.discard(residency)
            async with self._condition:
                self._reserved_bytes = max(0, self._reserved_bytes - nbytes)
                self._condition.notify_all()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "budget_bytes": self._budget_bytes,
                "resident_bytes": self._resident_bytes,
                "resident_count": len(self._resident),
                "registered_count": len(self._records),
                "host_bytes": self.host_bytes,
                "pinned_bytes": self.pinner.used_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }


# Global cache (configured at startup from SAEManager).
sae_cache = SAEGpuCache()
