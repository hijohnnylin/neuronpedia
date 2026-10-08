"""The vector activation paths read through the engine's ``project``.

On vLLM the worker projects its own rows, so a request's ``[seq, d_model]`` capture no longer
crosses to the server. What is worth pinning here is the routing: a plain hook becomes one
``DirectionSet`` at the hook's point, and a hook this server derives after the capture
(``hook_normalized``) is captured, divided, then projected, since the worker never sees the
division.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from interp_engine import Address, InterpModel, pre_gain_normalized
from interp_engine.api import DirectionSet

from neuronpedia_inference import engine_adapter
from neuronpedia_inference.endpoints.activation.single import vector_activation_values
from neuronpedia_inference.endpoints.activation.single_batch import process_vector_activations_batch

D = 8
EPS = 1e-6


class _Model:
    """A hooked model with no forward: ``capture`` and ``project`` answer from fixed rows."""

    hf_model_id = "fake/project-vector"
    hooks_available = True
    config = SimpleNamespace(rms_norm_eps=EPS)

    def __init__(self) -> None:
        self.projected: list[tuple[list[int], list[DirectionSet]]] = []
        self.captured: list[list[Address]] = []

    def rows(self, n: int) -> torch.Tensor:
        return torch.arange(n * D, dtype=torch.float32).reshape(n, D) / 10 - 2

    def refuses(self, address: Any) -> None:
        return None

    async def project(self, ids: list[int], sets: list[DirectionSet]) -> list[torch.Tensor]:
        self.projected.append((ids, sets))
        return [self.rows(len(ids)) @ s.vectors.T for s in sets]

    async def capture(self, ids: list[int], points: list[Address]) -> dict[Address, torch.Tensor]:
        self.captured.append(points)
        return {p: self.rows(len(ids)) for p in points}


@pytest.fixture(autouse=True)
def _forget_memoized_eps():
    engine_adapter._RMS_NORM_EPS.clear()
    yield
    engine_adapter._RMS_NORM_EPS.clear()


def _vector(seed: int = 0) -> torch.Tensor:
    return torch.randn(D, generator=torch.Generator().manual_seed(seed))


def test_a_plain_hook_is_one_direction_set_at_its_point() -> None:
    model, v = _Model(), _vector()
    got = asyncio.run(
        engine_adapter.project_vector_async(
            cast(InterpModel, model), torch.tensor([5, 6, 7]), "blocks.2.hook_resid_post", v
        )
    )
    ((ids, sets),) = model.projected
    assert ids == [5, 6, 7] and not model.captured
    assert sets[0].point == Address("resid_post", 2) and tuple(sets[0].vectors.shape) == (1, D)
    torch.testing.assert_close(got, model.rows(3) @ v)


def test_a_normalized_hook_is_captured_divided_then_projected() -> None:
    model, v = _Model(), _vector()
    got = asyncio.run(
        engine_adapter.project_vector_async(
            cast(InterpModel, model), torch.tensor([1, 2]), "blocks.3.ln2.hook_normalized", v
        )
    )
    assert not model.projected and model.captured == [[Address("resid_mid", 3)]]
    torch.testing.assert_close(got, pre_gain_normalized(model.rows(2), EPS) @ v)


def test_the_batch_path_projects_each_prompt_in_order() -> None:
    model, v = _Model(), _vector()
    prompts = [torch.tensor([1, 2, 3, 4]), torch.tensor([9, 8])]
    got = asyncio.run(
        process_vector_activations_batch(v.tolist(), prompts, "blocks.0.hook_resid_post", cast(InterpModel, model))
    )
    assert [ids for ids, _ in model.projected] == [[1, 2, 3, 4], [9, 8]]
    assert [len(r.values) for r in got] == [4, 2]
    expect = (model.rows(2) @ v).tolist()
    assert got[1].values == pytest.approx(expect)
    assert got[1].max_value_index == expect.index(max(expect))


def test_the_values_peak_where_the_projection_does() -> None:
    result = vector_activation_values(torch.tensor([0.5, 2.0, -1.0]))
    assert result.values == [0.5, 2.0, -1.0] and result.max_value == 2.0 and result.max_value_index == 1
