# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Logging setup helper.

The library logs through :mod:`logging` (logger names under ``jlens.*``) and
does not configure handlers itself — that's the application's job. This module
provides a one-liner for scripts and notebooks that want a readable default::

    import jlens
    jlens.configure_logging()

which prints ``[<elapsed> +<delta>] message`` to stderr — the elapsed time
since :func:`configure_logging` was called (human-readable: ``4s``, ``12m34s``,
``2h15m``, ``1d04h``) and the delta since the previous log line, so a slow step
stands out at a glance::

    [   4s +  0.01s] fit: n_layers=64 d_model=5120, fitting 63 source layers …
    [  12m34s +  4.21s] prompt 100/200  seq_len=128 …
    [  12m38s +  4.30s] prompt 101/200  seq_len=128 …
    [  13m43s + 64.40s] eval done            ← ah, eval is slow
"""

from __future__ import annotations

import logging
import time


def _human_duration(seconds: float) -> str:
    """``4`` → ``4s``, ``754`` → ``12m34s``, ``8100`` → ``2h15m``, ``100800`` → ``1d04h``."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h{m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d{h:02d}h"


class _DeltaFormatter(logging.Formatter):
    """Adds ``%(elapsed)s`` (human-readable since construction) and
    ``%(delta)s`` (``+X.XXs`` since the previous record) to log records."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._t0 = self._last = time.monotonic()

    def format(self, record: logging.LogRecord) -> str:
        now = time.monotonic()
        record.elapsed = f"{_human_duration(now - self._t0):>7}"
        record.delta = f"+{now - self._last:6.2f}s"
        self._last = now
        return super().format(record)


def configure_logging(level: int | str = logging.INFO) -> None:
    """Attach a stderr handler with the ``[elapsed +delta] message`` format
    to the ``jlens`` logger.

    Idempotent — repeated calls don't stack handlers. Intended for scripts and
    notebooks; library callers that already configure :mod:`logging` should not
    call this.

    Args:
        level: Log level for the ``jlens`` logger.
    """
    package_logger = logging.getLogger("jlens")
    package_logger.setLevel(level)
    if any(isinstance(h, logging.StreamHandler) for h in package_logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(_DeltaFormatter("[%(elapsed)s %(delta)s] %(message)s"))
    package_logger.addHandler(handler)
    package_logger.propagate = False
