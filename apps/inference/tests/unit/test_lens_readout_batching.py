"""CPU-only contract for the lens read-out's residual staging and vLLM streaming.

Properties that each cost a large multiple of the request's runtime when they regress:

- ``_stack_chunk_residuals`` transports a whole chunk per layer instead of per position.
  ``J_bar`` is ``d_model**2`` floats (21 MB at gemma-2-2b's 2304), so a per-position
  matvec re-streams it for every position and the staging becomes bandwidth-bound: 8.5s
  vs 0.23s for a 550-token gemma-2-2b prompt. It also stages every layer on the lens's
  ``transport_device``, including the unfitted ones. The row LAYOUT must not change with
  either -- the read-out unfolds these rows as position-major groups of ``n_layers``.
- ``_iter_readout_vllm`` is the serving path: the worker transports and unembeds where it
  captured, so what streams back is top-k rather than residuals. What it still has to get
  right is placing each position on the global axis and pairing it with a token id, since
  generation runs ahead of the read-outs and ``skip_before`` moves the start.
- ``_iter_residuals_vllm`` and ``_to_wire_dtype`` are the FALLBACK, for a lens that could
  not be made worker-resident and so can only be applied in this process. Same streaming
  contract, with the residuals crossing ``collective_rpc`` in both directions -- which is
  why the rows are narrowed to the served model's dtype first, but only where the
  configured dtype names something concrete, since ``model_dtype`` is ``"auto"`` by
  default and a wrong guess would be a real precision loss.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
import torch
from interp_engine import vllm_residual_basis
from interp_engine.vllm_backend import VLLMModel

from neuronpedia_inference.config import Config
from neuronpedia_inference.endpoints.lens.lens_loader import LoadedJacobianLens
from neuronpedia_inference.endpoints.lens.prompt import (
    LensType,
    _iter_readout_vllm,
    _iter_residuals_vllm,
    _stack_chunk_residuals,
    _to_wire_dtype,
)

D_MODEL = 4
LAYERS = [0, 1, 2]


class _FakeLens:
    """Stand-in for ``LoadedJacobianLens``: a distinct J per fitted layer."""

    def __init__(self, fitted: list[int], transport_device: torch.device | None = None):
        self.jacobians = {
            layer: torch.eye(D_MODEL) * (layer + 2) + 0.1 * torch.arange(D_MODEL**2).reshape(D_MODEL, D_MODEL)
            for layer in fitted
        }
        self.source_layers = sorted(self.jacobians)
        self.transport_device = transport_device
        self.transported_on: list[torch.device] = []

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        self.transported_on.append(residual.device)
        return residual @ self.jacobians[layer].to(residual.device).T


def _reference_stack(lens_type, layers, residuals_list, lens) -> torch.Tensor:
    """The per-position formulation this batches: one matvec per (position, layer)."""
    use_jacobian = lens_type == LensType.JACOBIAN_LENS and lens is not None
    rows = []
    for residuals in residuals_list:
        for layer in layers:
            residual = residuals[layer]
            if use_jacobian:
                assert lens is not None
                if layer in lens.jacobians:
                    residual = lens.transport(residual.float(), layer)
            rows.append(residual.float())
    return torch.stack(rows, dim=0)


def _residuals(n_positions: int, dtype=torch.float32) -> list[dict[int, torch.Tensor]]:
    torch.manual_seed(0)
    return [{layer: torch.randn(D_MODEL, dtype=dtype) for layer in LAYERS} for _ in range(n_positions)]


@pytest.mark.parametrize("n_positions", [1, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_batched_transport_matches_per_position(n_positions: int, dtype):
    """Same rows, same order. Layer 2 is unfitted, so it must pass through untransported."""
    lens = cast(LoadedJacobianLens, _FakeLens(fitted=[0, 1]))
    residuals_list = _residuals(n_positions, dtype)
    out = _stack_chunk_residuals(LensType.JACOBIAN_LENS, LAYERS, residuals_list, lens)
    ref = _reference_stack(LensType.JACOBIAN_LENS, LAYERS, residuals_list, lens)
    assert out.shape == (n_positions * len(LAYERS), D_MODEL)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref)


def test_row_layout_is_position_major():
    """Row ``pos * n_layers + i`` is position ``pos``, layer ``layers[i]`` -- the
    layout the read-out reshapes into ``[n_positions, n_layers, vocab]``."""
    residuals_list = _residuals(3)
    out = _stack_chunk_residuals(LensType.LOGIT_LENS, LAYERS, residuals_list, None)
    for pos in range(3):
        for i, layer in enumerate(LAYERS):
            torch.testing.assert_close(out[pos * len(LAYERS) + i], residuals_list[pos][layer])


def test_logit_lens_never_transports():
    """A LOGIT_LENS read-out decodes the raw residual even when a lens is loaded."""
    lens = cast(LoadedJacobianLens, _FakeLens(fitted=[0, 1, 2]))
    residuals_list = _residuals(2)
    out = _stack_chunk_residuals(LensType.LOGIT_LENS, LAYERS, residuals_list, lens)
    torch.testing.assert_close(out, _reference_stack(LensType.LOGIT_LENS, LAYERS, residuals_list, None))


def test_cpu_residuals_are_staged_on_the_lens_transport_device():
    """The vLLM shape: residuals arrive on the CPU, and the transport must not follow them.

    Uses "meta" as a stand-in device so the placement is observable without a GPU. What is
    pinned is that every block -- including unfitted layer 2, which is only along for the
    stack -- is moved before the transport, so no block is left behind on the host.
    """
    device = torch.device("meta")
    lens = cast(LoadedJacobianLens, _FakeLens(fitted=[0, 1], transport_device=device))
    out = _stack_chunk_residuals(LensType.JACOBIAN_LENS, LAYERS, _residuals(4), lens)

    assert out.device == device
    assert [d.type for d in cast(Any, lens).transported_on] == ["meta", "meta"]


@pytest.mark.parametrize(
    ("model_dtype", "expected"),
    [
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
        ("float32", torch.float32),
        # Unrecognised names keep the staged float32: "auto" only resolves inside vLLM, and
        # narrowing to a dtype the model does not use would lose precision for nothing.
        ("auto", torch.float32),
        ("something-new", torch.float32),
    ],
)
def test_staged_rows_cross_the_rpc_at_the_served_model_dtype(monkeypatch, model_dtype, expected):
    config = Config.get_instance()
    monkeypatch.setattr(config, "model_dtype", model_dtype, raising=False)
    staged = torch.randn(6, D_MODEL, dtype=torch.float32)
    assert _to_wire_dtype(staged).dtype == expected


def test_narrowing_to_the_wire_dtype_preserves_the_values_it_can_represent(monkeypatch):
    """The cast is the one the worker would do on arrival, so it must not reorder anything."""
    monkeypatch.setattr(Config.get_instance(), "model_dtype", "bfloat16", raising=False)
    staged = torch.randn(4, D_MODEL, dtype=torch.float32)
    narrowed = _to_wire_dtype(staged)
    assert torch.equal(narrowed.float(), staged.to(torch.bfloat16).float())


def test_staging_stays_put_when_the_lens_has_no_transport_device():
    """A CPU-only pod: nothing to move to, and moving would only cost a copy."""
    lens = cast(LoadedJacobianLens, _FakeLens(fitted=[0, 1], transport_device=None))
    out = _stack_chunk_residuals(LensType.JACOBIAN_LENS, LAYERS, _residuals(4), lens)

    assert out.device.type == "cpu"
    assert [d.type for d in cast(Any, lens).transported_on] == ["cpu", "cpu"]


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


class _FakeStreamingBackend:
    """Minimal ``VLLMModel`` surface: scripted capture drains per step.

    ``steps`` is a list of ``(n_new_rows, token_ids_so_far)``, mimicking an engine that
    yields cumulative token ids while the capture hooks accumulate rows independently
    (so the two can be out of step in either direction).
    """

    # A conventional trunk, which is what these tests are about: the iterators read it off the model
    # to resolve the lens's capture point and its stream reduction, and a single-stream verdict means
    # `resid_post` with nothing to reduce, exactly as before that was a choice.
    residual_basis = vllm_residual_basis(architecture="GPT2LMHeadModel")

    def __init__(self, prompt_len: int, steps: list[tuple[int, list[int]]]):
        self.prompt_len = prompt_len
        self.steps = steps
        self.next_row = 0
        self.stream_calls = 0
        self.capture_calls = 0
        self.calls: list[dict[str, Any]] = []

    def _rows(self, n: int, layer: int) -> torch.Tensor:
        # Row r at layer L is identifiable as (r, L) so pairing errors are visible.
        base = torch.arange(self.next_row, self.next_row + n, dtype=torch.float32)
        return torch.stack([base, torch.full((n,), float(layer))], dim=1)

    async def capture(self, prompt_token_ids, points):
        self.capture_calls += 1
        return {p: self._rows(len(prompt_token_ids), p.layer) for p in points}

    async def capture_generation_stream(self, prompt_token_ids, points, **kwargs):
        self.stream_calls += 1
        self.calls.append(kwargs)
        for n_rows, token_ids in self.steps:
            caps = {p: self._rows(n_rows, p.layer) for p in points} if n_rows else {}
            self.next_row += n_rows
            yield caps, token_ids


def _collect_batches(model: Any, prompt_token_ids, **kwargs) -> list[list[tuple]]:
    async def _run():
        return [
            batch
            async for batch in _iter_residuals_vllm(
                cast(VLLMModel, model),
                prompt_token_ids,
                LAYERS,
                temperature=0.0,
                **kwargs,
            )
        ]

    return asyncio.run(_run())


def _collect(model, prompt_token_ids, **kwargs) -> list[tuple]:
    return [step for batch in _collect_batches(model, prompt_token_ids, **kwargs) for step in batch]


def _tokens(steps: list[tuple]) -> list[tuple[int, bool]]:
    return [(tid, gen) for tid, gen, _ in steps]


def test_prompt_only_request_uses_plain_capture():
    """No generation and no intervention: a single prefill capture, no streaming."""
    model = _FakeStreamingBackend(prompt_len=3, steps=[])
    steps = _collect(model, [10, 11, 12], num_completion_tokens=0)
    assert model.capture_calls == 1
    assert model.stream_calls == 0
    assert _tokens(steps) == [(10, False), (11, False), (12, False)]
    for pos, (_, _, residuals) in enumerate(steps):
        assert residuals[1].tolist() == [float(pos), 1.0]


def test_generated_positions_stream_as_they_land():
    """The prefill's positions must be emitted on the FIRST drain, not after generation.

    The regression this guards: consuming the whole completion before yielding anything,
    which left the client with no read-outs until generation finished.
    """
    model = _FakeStreamingBackend(
        prompt_len=2,
        # prefill (2 rows, 1 token sampled), then one row per decode step.
        steps=[(2, [90]), (1, [90, 91]), (1, [90, 91, 92])],
    )

    async def _run() -> list[tuple[int, bool]]:
        seen: list[tuple[int, bool]] = []
        async for batch in _iter_residuals_vllm(
            cast(VLLMModel, model),
            [10, 11],
            LAYERS,
            num_completion_tokens=3,
            temperature=0.0,
        ):
            # Every prompt position is available while generation is still running.
            if len(seen) < 2:
                assert model.next_row <= 2
            seen.extend((tid, gen) for tid, gen, _ in batch)
        return seen

    # positions 0,1 are the prompt; 2,3 are the first two generated tokens. The last
    # sampled token (92) is never forwarded, so it has no residual and no read-out.
    assert asyncio.run(_run()) == [(10, False), (11, False), (90, True), (91, True)]


def test_available_positions_are_handed_over_as_one_batch():
    """Positions available together must arrive together.

    Each batch is staged with one Jacobian transport per layer, so splitting the
    prefill's positions across batches would re-read every ``J_bar`` per split --
    the cost this whole path is built to avoid. Nothing may be held back either: the
    batch boundary is the drain, not a fixed size.
    """
    model = _FakeStreamingBackend(
        prompt_len=4,
        # prefill (4 rows), then a drain that caught two decode steps at once.
        steps=[(4, [90]), (2, [90, 91, 92])],
    )
    batches = _collect_batches(model, [10, 11, 12, 13], num_completion_tokens=3)
    assert [len(b) for b in batches] == [4, 2]


def test_prompt_only_prefill_is_one_batch():
    model = _FakeStreamingBackend(prompt_len=3, steps=[])
    batches = _collect_batches(model, [10, 11, 12], num_completion_tokens=0)
    assert [len(b) for b in batches] == [3]


def test_position_waits_for_both_row_and_token_id():
    """A drained row whose token id has not arrived yet must not be emitted early."""
    model = _FakeStreamingBackend(
        prompt_len=2,
        # The engine ran two decode steps before we drained, but reported one token id.
        steps=[(3, [90]), (0, [90, 91])],
    )
    steps = _collect(model, [10, 11], num_completion_tokens=4)
    assert _tokens(steps) == [(10, False), (11, False), (90, True)]
    # The generated position pairs row 2 with token 90.
    assert steps[2][2][0].tolist() == [2.0, 0.0]


def test_completion_cap_is_respected():
    """More generated rows than requested (engine overrun) must not be emitted."""
    model = _FakeStreamingBackend(prompt_len=1, steps=[(1, [90]), (1, [90, 91]), (1, [90, 91, 92])])
    steps = _collect(model, [10], num_completion_tokens=1)
    assert _tokens(steps) == [(10, False), (90, True)]


def test_intervention_only_prefill_drops_the_throwaway_token():
    """num_completion_tokens == 0 with steering still generates one token to run the
    intervened forward; that position is not a result."""
    model = _FakeStreamingBackend(prompt_len=2, steps=[(2, [90]), (1, [90, 91])])
    steps = _collect(
        model,
        [10, 11],
        num_completion_tokens=0,
        steer_deltas={0: torch.ones(D_MODEL)},
        steer_strength=0.5,
    )
    assert model.stream_calls == 1
    assert _tokens(steps) == [(10, False), (11, False)]


# --------------------------------------------------------------------------- #
# Streaming, worker-side read-out (the serving path)
# --------------------------------------------------------------------------- #

TYPES = [LensType.JACOBIAN_LENS, LensType.LOGIT_LENS]
LAYERS_BY_TYPE = {LensType.JACOBIAN_LENS: [0, 1, 2], LensType.LOGIT_LENS: [1, 2]}
TOP_N = 4


class _FakeReadoutBackend:
    """Minimal ``VLLMModel`` surface for :func:`_iter_readout_vllm`.

    ``steps`` is ``(first_position, n_positions, token_ids_so_far)`` per yield, mimicking a
    worker that reads out what it captured while the sampler runs ahead of it. Each row is
    filled with its own ``(position, layer_index)`` so a misplaced position is visible in the
    values rather than only in the shapes.
    """

    residual_basis = vllm_residual_basis(architecture="GPT2LMHeadModel")

    def __init__(self, steps: list[tuple[int, int, list[int]]]):
        self.steps = steps
        self.calls: list[dict[str, Any]] = []

    async def lens_capture_readout_stream(self, prompt_token_ids, points, specs, **kwargs):
        self.calls.append({"points": points, "specs": specs, **kwargs})
        for first, n, token_ids in self.steps:
            idx, probs = [], []
            for spec in specs:
                n_layers = len(spec["layers"])
                # A step with no positions still yields, carrying zero-row tensors: the real
                # backend does that to report ids the engine sampled while the read-out RPCs
                # were in flight (see `lens_capture_readout_stream`).
                rows = torch.tensor(
                    [[first + p, layer] for p in range(n) for layer in range(n_layers)],
                    dtype=torch.int64,
                ).reshape(n * n_layers, 2)
                idx.append(rows.repeat(1, TOP_N // 2))
                probs.append(rows.float().repeat(1, TOP_N // 2))
            yield first, idx, probs, token_ids


def _collect_readout(model, prompt_token_ids, **kwargs) -> list[list[tuple]]:
    async def _run():
        return [
            batch
            async for batch in _iter_readout_vllm(
                cast(VLLMModel, model),
                prompt_token_ids,
                TYPES,
                LAYERS_BY_TYPE,
                temperature=0.0,
                top_n=TOP_N,
                softcap=None,
                word_mask=None,
                chunk_positions=8,
                skip_before=0,
                **kwargs,
            )
        ]

    return asyncio.run(_run())


def test_the_worker_is_asked_for_one_spec_per_type_over_the_union_of_layers():
    """Only JACOBIAN_LENS asks for the transport, and the capture covers both types."""
    model = _FakeReadoutBackend(steps=[(0, 2, [90])])
    _collect_readout(model, [10, 11], num_completion_tokens=0)
    call = model.calls[0]
    assert call["specs"] == [
        {"layers": [0, 1, 2], "jacobian": True},
        {"layers": [1, 2], "jacobian": False},
    ]
    assert [p.layer for p in call["points"]] == [0, 1, 2]


def test_each_position_gets_its_own_layer_rows_per_type():
    """The flat ``[n_positions * n_layers, k]`` result is unfolded position-major."""
    model = _FakeReadoutBackend(steps=[(0, 3, [90])])
    steps = [s for batch in _collect_readout(model, [10, 11, 12], num_completion_tokens=0) for s in batch]

    assert _tokens(steps) == [(10, False), (11, False), (12, False)]
    for position, (_, _, per_type) in enumerate(steps):
        for spec_index, lens_type in enumerate(TYPES):
            top_idx, _ = per_type[spec_index]
            n_layers = len(LAYERS_BY_TYPE[lens_type])
            assert top_idx.shape == (n_layers, TOP_N)
            assert top_idx[:, 0].tolist() == [position] * n_layers
            assert top_idx[:, 1].tolist() == list(range(n_layers))


def test_generated_positions_stream_as_they_are_read_out():
    model = _FakeReadoutBackend(steps=[(0, 2, [90]), (2, 1, [90, 91]), (3, 1, [90, 91, 92])])
    batches = _collect_readout(model, [10, 11], num_completion_tokens=2)
    assert [[(tid, gen) for tid, gen, _ in batch] for batch in batches] == [
        [(10, False), (11, False)],
        [(90, True)],
        [(91, True)],
    ]


def test_a_position_waits_for_its_token_id():
    """The read-out can outrun the sampler; a position without an id must not be emitted."""
    model = _FakeReadoutBackend(steps=[(0, 3, [90]), (3, 0, [90, 91])])
    steps = [s for batch in _collect_readout(model, [10, 11], num_completion_tokens=4) for s in batch]
    assert _tokens(steps) == [(10, False), (11, False), (90, True)]


def test_ids_arriving_alone_release_positions_already_read_out():
    """The mirror of the test above, and the one that was broken.

    The engine outruns the read-out RPCs, so one step comes back with the rows for several
    sampled tokens and the steps after it carry ids and nothing else. Those ids are what
    release the positions already in hand: while they were withheld, a request for 3 tokens
    returned 1, and the shortfall moved with how far the engine happened to run ahead.
    """
    model = _FakeReadoutBackend(steps=[(0, 4, [90]), (4, 0, [90, 91]), (4, 0, [90, 91, 92])])
    steps = [s for batch in _collect_readout(model, [10, 11], num_completion_tokens=3) for s in batch]
    assert _tokens(steps) == [(10, False), (11, False), (90, True), (91, True)]


def test_the_completion_cap_still_drops_an_engine_overrun():
    model = _FakeReadoutBackend(steps=[(0, 1, [90]), (1, 1, [90, 91]), (2, 1, [90, 91, 92])])
    steps = [s for batch in _collect_readout(model, [10], num_completion_tokens=1) for s in batch]
    assert _tokens(steps) == [(10, False), (90, True)]


def test_an_intervention_only_prefill_drops_the_throwaway_token():
    model = _FakeReadoutBackend(steps=[(0, 2, [90]), (2, 1, [90, 91])])
    steps = [
        s
        for batch in _collect_readout(
            model,
            [10, 11],
            num_completion_tokens=0,
            steer_deltas={0: torch.ones(D_MODEL)},
            steer_strength=0.5,
        )
        for s in batch
    ]
    assert _tokens(steps) == [(10, False), (11, False)]
    assert model.calls[0]["lens_intervention"] is not None


def test_a_reused_prefix_is_skipped_in_the_worker_and_numbered_from_there():
    """`skip_before` reaches the worker, and the positions that come back start past it."""

    async def _run():
        return [
            batch
            async for batch in _iter_readout_vllm(
                cast(VLLMModel, model),
                [10, 11, 12, 13],
                TYPES,
                LAYERS_BY_TYPE,
                num_completion_tokens=0,
                temperature=0.0,
                top_n=TOP_N,
                softcap=None,
                word_mask=None,
                chunk_positions=8,
                skip_before=2,
            )
        ]

    model = _FakeReadoutBackend(steps=[(2, 2, [90])])
    steps = [s for batch in asyncio.run(_run()) for s in batch]
    assert model.calls[0]["skip_before"] == 2
    assert _tokens(steps) == [(12, False), (13, False)]


# --------------------------------------------------------------------------- #
# How many tokens the engine is asked for -- both vLLM paths, one rule
# --------------------------------------------------------------------------- #

_RESIDUAL_PATH = (
    lambda model, **kw: _collect(model, [10, 11], **kw),
    lambda: _FakeStreamingBackend(prompt_len=2, steps=[(2, [90])]),
)
_READOUT_PATH = (
    lambda model, **kw: _collect_readout(model, [10, 11], **kw),
    lambda: _FakeReadoutBackend(steps=[(0, 2, [90])]),
)


@pytest.mark.parametrize(("collect", "backend"), [_RESIDUAL_PATH, _READOUT_PATH], ids=["residuals", "readout"])
def test_one_more_token_is_sampled_than_the_request_asks_for(collect, backend):
    """The engine has to run a forward *on* the last requested token, not merely sample it.

    A position's read-out comes from the pass that has that token as its input, and nothing
    runs after the final sample -- so asking for exactly n left the nth token with no row, and
    a request for 3 displayed 2. The eager path has always run that forward, sampling and then
    forwarding once per step, so this is the vLLM path matching it rather than a new policy.
    The surplus token is past the emit cap and is dropped by it, like an engine overrun.
    """
    model = backend()
    collect(model, num_completion_tokens=3)
    assert model.calls[0]["max_tokens"] == 4


@pytest.mark.parametrize(("collect", "backend"), [_RESIDUAL_PATH, _READOUT_PATH], ids=["residuals", "readout"])
def test_an_intervention_only_request_still_asks_for_exactly_one(collect, backend):
    """``num_completion_tokens == 0`` wants one throwaway token to run the intervened forward
    and no result position, so it must not grow to two along with everything else."""
    model = backend()
    collect(model, num_completion_tokens=0, steer_deltas={0: torch.ones(D_MODEL)}, steer_strength=0.5)
    assert model.calls[0]["max_tokens"] == 1
