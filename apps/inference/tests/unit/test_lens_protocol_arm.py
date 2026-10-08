"""CPU-only contract for what the lens server still owns on top of ``generate_with_lens``.

- An intervention is the engine's ``steer()`` block, opened around the read: the lens's three ops
  as the engine's own spec, BOS positions masked, generated positions written only when asked.
- ``_unembed_vectors_by_id`` asks the protocol for its rows, once per distinct id.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import torch
from interp_engine import InterpModel, eager_residual_basis
from interp_engine.steer import active_steering
from interp_engine.steer_specs import AblateSpec, NormScaledAddSpec, SwapSpec

from neuronpedia_inference.endpoints.lens.prompt import (
    _build_lens_steering_spec,
    _lens_intervention,
    _unembed_vectors_by_id,
)
from neuronpedia_inference.endpoints.lens.residual_spec import BLOCK_OUTPUT

D_MODEL = 4
PROMPT = [11, 12, 13]


class _ProtocolOnlyBackend:
    """The protocol's unembed surface and nothing of eager's or vLLM's."""

    residual_basis = eager_residual_basis(architecture="GPT2LMHeadModel")

    def __init__(self) -> None:
        self.unembed_calls: list[list[int]] = []

    async def unembed_rows(self, token_ids):
        ids = [int(t) for t in token_ids]
        self.unembed_calls.append(ids)
        return torch.stack([torch.full((D_MODEL,), float(t)) for t in ids])


def _seen(model: Any, **kwargs: Any):
    """What ``steer()`` has recorded for ``model`` inside the lens's intervention block."""
    kwargs = {
        "steer_strength": 1.0,
        "steer_ablate": False,
        "swap_deltas": None,
        "steer_generated": False,
        "bos_token_id": PROMPT[0],
        **kwargs,
    }
    with _lens_intervention(cast(Any, model), PROMPT, residual=BLOCK_OUTPUT, **kwargs):
        return active_steering(model)


def test_an_intervention_rides_a_steer_block_with_the_lens_scope() -> None:
    """BOS at position 0 masked, generated positions left alone."""
    model = _ProtocolOnlyBackend()
    seen = _seen(model, steer_deltas={0: torch.ones(D_MODEL), 2: torch.ones(D_MODEL)}, steer_strength=1.5)
    assert seen is not None
    assert seen.position_mask == [0] and seen.generated is False
    (spec,) = seen.specs
    assert sorted(spec.layers) == [0, 2]
    (op,) = spec.layers[2].operations
    assert isinstance(op, NormScaledAddSpec) and op.strength == 1.5
    assert active_steering(model) is None, "the block closes with the request"


def test_steer_generated_and_no_bos_widen_the_scope() -> None:
    seen = _seen(
        _ProtocolOnlyBackend(),
        steer_deltas={0: torch.ones(D_MODEL)},
        steer_ablate=True,
        steer_generated=True,
        bos_token_id=None,
    )
    assert seen is not None
    assert seen.position_mask is None and seen.generated is True
    assert isinstance(seen.specs[0].layers[0].operations[0], AblateSpec)


def test_a_read_out_with_no_intervention_opens_no_block() -> None:
    assert _seen(_ProtocolOnlyBackend(), steer_deltas={0: torch.ones(D_MODEL)}, steer_strength=0.0) is None


def test_the_spec_builder_keeps_the_lens_precedence_and_skips_a_zero_direction() -> None:
    src, tgt, zero = torch.ones(D_MODEL), torch.arange(D_MODEL, dtype=torch.float32), torch.zeros(D_MODEL)
    swap = _build_lens_steering_spec({0: src, 1: src}, 2.0, True, {0: tgt, 1: zero})
    assert swap is not None and list(swap.layers) == [0], "swap wins over steer/ablate; a zero target is skipped"
    assert isinstance(swap.layers[0].operations[0], SwapSpec)
    assert swap.point == "resid_post" and swap.stream is None
    assert _build_lens_steering_spec({0: zero}, 1.0, False, None) is None
    assert _build_lens_steering_spec({0: src}, 0.0, False, None) is None, "strength 0 without ablate is no steer"


def test_unembedding_rows_come_from_the_protocol_once_per_distinct_id() -> None:
    model = _ProtocolOnlyBackend()
    rows = asyncio.run(_unembed_vectors_by_id(cast(InterpModel, model), [5, 7, 5]))
    assert model.unembed_calls == [[5, 7]]
    assert set(rows) == {5, 7}
    assert rows[7].dtype == torch.float32 and float(rows[7][0]) == 7.0
