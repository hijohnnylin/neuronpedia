"""CPU-only contract for incremental capture drains (``worker_drain_request``).

Streaming a lens read-out means reading a request's captured rows WHILE it generates,
rather than once at the end. The drain has to hand back exactly the rows appended since
the previous call, in forward order, and leave the registration alone -- if it dropped
rows or deregistered, positions would silently vanish from the response; if it returned
rows twice, positions would be paired with the wrong token.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from interp_engine import Address
from interp_engine.vllm_capture import (
    _get_demux,
    decode_tensor_payload,
    worker_collect_request,
    worker_drain_request,
)

RID = "np-capgen-test"
POINT = Address("resid_post", 3)
# The wire key is the canonical address string, derived here rather than spelled out so this
# fake cannot drift from the grammar the worker actually stores rows under.
KEY = str(POINT)


def _worker() -> SimpleNamespace:
    """A worker holding an empty model: the drain reads it to scale what vLLM scales outside the
    hooked module, and ``resid_post`` is not one of those points, so nothing is looked up on it."""
    return SimpleNamespace(model_runner=SimpleNamespace(model=torch.nn.Module()))


def _forward(worker: object, rid: str, rows: list[int], key: str = KEY) -> None:
    """Append what a capture hook would append for one forward: ``[n_rows, width]``."""
    demux = _get_demux(worker)
    demux.registered.add(rid)
    demux.cap_points.setdefault(rid, {POINT})
    block = torch.tensor([[float(r), 0.5] for r in rows])
    demux.captures.setdefault(rid, {}).setdefault(key, []).append(block)


def _drained_rows(worker: object, rid: str = RID, key: str = KEY) -> list[float]:
    out = worker_drain_request(worker, rid)
    if key not in out:
        return []
    return [row[0].item() for row in decode_tensor_payload(out[key])]


def test_drain_returns_only_new_rows_in_forward_order():
    worker = _worker()
    _forward(worker, RID, [0, 1, 2])  # prefill
    assert _drained_rows(worker) == [0, 1, 2]

    _forward(worker, RID, [3])  # decode step
    _forward(worker, RID, [4])  # another, before we drained
    assert _drained_rows(worker) == [3, 4]

    # Nothing new: an empty payload, not a repeat of what was already handed out.
    assert worker_drain_request(worker, RID) == {}


def test_drain_keeps_the_request_registered():
    """Hooks and registration must survive a drain, or capture stops mid-generation."""
    worker = _worker()
    _forward(worker, RID, [0])
    worker_drain_request(worker, RID)
    demux = _get_demux(worker)
    assert RID in demux.registered
    assert RID in demux.cap_points
    _forward(worker, RID, [1])
    assert _drained_rows(worker) == [1]


def test_collect_after_drains_returns_only_the_tail():
    """The final collect closes the request out with whatever arrived after the last drain."""
    worker = _worker()
    _forward(worker, RID, [0, 1])
    worker_drain_request(worker, RID)
    _forward(worker, RID, [2])

    out = worker_collect_request(worker, RID)
    assert [row[0].item() for row in decode_tensor_payload(out[KEY])] == [2]
    demux = _get_demux(worker)
    assert RID not in demux.captures
    assert RID not in demux.cap_points


def test_drain_of_unknown_request_is_empty():
    assert worker_drain_request(_worker(), "never-registered") == {}


def test_drain_is_per_request():
    """Concurrent requests share the hooks; a drain must not touch another's rows."""
    worker = _worker()
    other = "np-capgen-other"
    _forward(worker, RID, [0, 1])
    _forward(worker, other, [7])
    assert _drained_rows(worker, RID) == [0, 1]
    assert _drained_rows(worker, other) == [7]
