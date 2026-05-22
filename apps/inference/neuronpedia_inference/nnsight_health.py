"""Health tracking for the nnsight runtime.

Background: nnsight v0.5.x has a bug where its C-level `unmount` (in
`nnsight._c.py_mount`) doesn't propagate `PyDict_DelItemString` failures.
When the process-wide `Globals.stack` counter and the underlying mount
registry get out of sync, `Globals.exit()` calls `unmount("save")` against
a registry that no longer has "save", and the C function returns
`Py_RETURN_NONE` with a `KeyError` already set, which Python surfaces as
`SystemError: <built-in function unmount> returned a result with an
exception set`. Once this happens, the C-level mount registry stays
corrupt for the rest of the process lifetime and every subsequent
nnsight tracer request fails.

We mitigate the bug by disabling `nnsight.CONFIG.APP.PYMOUNT` at startup
(see `server.py`), so the buggy code path is never exercised. This
module additionally exposes a fast health probe so the inference server
can be marked unhealthy and replaced if we ever do observe the
corruption (e.g. from a code path we missed, a leftover request from
before the fix, or a regression).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import AsyncIterator
from typing import Any, TypeVar

import nnsight
from nnsight.intervention.tracing.globals import Globals

T = TypeVar("T")

logger = logging.getLogger(__name__)

_state_lock = threading.Lock()
_corruption_detected: bool = False
_corruption_reason: str | None = None
_corruption_observed_at: float | None = None


def mark_nnsight_corrupted(reason: str) -> None:
    """Mark the nnsight runtime as corrupted.

    Idempotent: only the first reason is recorded. Safe to call from any
    thread / async context.
    """
    global _corruption_detected, _corruption_reason, _corruption_observed_at
    with _state_lock:
        if _corruption_detected:
            return
        _corruption_detected = True
        _corruption_reason = reason
        _corruption_observed_at = time.time()
    logger.error("[NNSIGHT HEALTH] Runtime marked corrupted: %s", reason)


def is_nnsight_corruption_exception(exc: BaseException) -> bool:
    """Return True if `exc` (or its cause/context) is the nnsight unmount
    corruption signature.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, SystemError):
            msg = str(cur)
            if "unmount" in msg or "mount" in msg:
                return True
        if isinstance(cur, KeyError):
            key = cur.args[0] if cur.args else None
            if key in ("save", "stop"):
                return True
        cur = cur.__cause__ or cur.__context__
    return False


def detect_and_mark(exc: BaseException) -> bool:
    """If `exc` matches the nnsight corruption signature, mark the runtime
    corrupted and return True. Otherwise return False.
    """
    if is_nnsight_corruption_exception(exc):
        mark_nnsight_corrupted(
            f"{type(exc).__name__}: {exc} "
            f"(cause={type(exc.__cause__).__name__ if exc.__cause__ else None})"
        )
        return True
    return False


async def watch_async_iter(source: AsyncIterator[T]) -> AsyncIterator[T]:
    """Pass-through async iterator that detects nnsight corruption errors
    and marks the runtime as corrupted before re-raising.

    Wrap any async generator that may raise the nnsight unmount
    `SystemError` so the health endpoint can report the broken state
    without needing to attempt another (slow) tracer request.
    """
    try:
        async for item in source:
            yield item
    except BaseException as exc:
        detect_and_mark(exc)
        raise


def get_nnsight_health() -> tuple[bool, dict[str, Any]]:
    """Fast probe of nnsight runtime health.

    Does NOT run a tracer / model — it only inspects:
      * the persistent corruption flag set by `mark_nnsight_corrupted`
      * `nnsight.CONFIG.APP.PYMOUNT` (we expect this to be False)
      * `Globals.stack` (should be 0 when no request is in flight; a
        non-zero "at rest" value indicates leaked state from a prior
        request)
    """
    with _state_lock:
        corruption_detected = _corruption_detected
        corruption_reason = _corruption_reason
        corruption_observed_at = _corruption_observed_at

    pymount = bool(nnsight.CONFIG.APP.PYMOUNT)
    stack = int(Globals.stack)

    details: dict[str, Any] = {
        "pymount_enabled": pymount,
        "globals_stack": stack,
        "corruption_detected": corruption_detected,
    }
    if corruption_detected:
        details["corruption_reason"] = corruption_reason
        details["corruption_observed_at"] = corruption_observed_at

    healthy = True
    reasons: list[str] = []
    if corruption_detected:
        healthy = False
        reasons.append("nnsight unmount corruption observed")
    if pymount:
        healthy = False
        reasons.append("nnsight PYMOUNT is enabled (expected disabled)")
    if stack < 0:
        healthy = False
        reasons.append(f"nnsight Globals.stack is negative ({stack})")

    if not healthy:
        details["unhealthy_reasons"] = reasons

    return healthy, details
