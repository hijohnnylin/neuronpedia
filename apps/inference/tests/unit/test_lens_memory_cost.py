"""What a lens request reserves, which decides how many can run at once.

The staged residual rows moved from host RAM to the device when the Jacobian transport did,
so the reservation stopped being a fixed number. Under-reserving here does not throttle, it
OOMs: the admission controller lets in as many requests as the claims say fit.

The capture term is the one that has actually cost a pod. It was missing because the
read-out is chunked, which made a lens request look like it had a fixed cost -- but the
harvest that feeds the read-out is held for the whole sequence, so it grows with the
conversation. DeepSeek-V4-Flash OOMed inside `collective_rpc` on the second turn of a chat
with the reservation reporting a few hundred MiB.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from neuronpedia_inference import memory_cost
from neuronpedia_inference.memory_cost import FLAT_LENS_BYTES, lens_cost

# Qwen3.6-27B, the pod this was sized against: 64 read-out layers at d_model 5120, staged a
# full `_TRANSPORT_BATCH_SIZE` at a time.
QWEN = {"staged_positions": 128, "d_model": 5120}
ONE_BLOCK_BYTES = 128 * 64 * 5120 * 4

# DeepSeek-V4-Flash on one B200: 43 capture points at d_model 4096, on a hyper-connection
# trunk carrying 4 residual streams. The pod that OOMed.
DSV4 = {"n_capture_points": 43, "d_model": 4096, "n_streams": 4}
GIB = 1024**3


@pytest.fixture(autouse=True)
def _bf16_model():
    """Pin the model dtype the capture term is sized in, so the numbers below are stable."""
    config = SimpleNamespace(token_limit=4096, sae_dtype="bfloat16", model_dtype="bfloat16")
    with patch.object(memory_cost.Config, "get_instance", return_value=config):
        yield


class TestLensCost:
    def test_never_reserves_less_than_the_old_flat_figure(self):
        # The rows are not the whole working set -- the eager backend also holds a
        # vocab-sized decode chunk -- so the flat number stays as the floor.
        assert lens_cost(staged_positions=1, layer_counts=[1], d_model=64) == FLAT_LENS_BYTES
        assert lens_cost(staged_positions=0, layer_counts=[], d_model=5120) == FLAT_LENS_BYTES

    def test_both_lens_types_cost_more_than_one(self):
        one = lens_cost(layer_counts=[64], **QWEN)
        both = lens_cost(layer_counts=[64, 64], **QWEN)
        assert both > one

    def test_a_both_types_request_exceeds_the_old_flat_figure(self):
        # The regression: at max_concurrent=8 the flat 512 MiB admitted 4 GiB of claims for
        # a working set nearer 5 GiB.
        assert lens_cost(layer_counts=[64, 64], **QWEN) > FLAT_LENS_BYTES

    def test_covers_every_resident_block_plus_the_one_being_stacked(self):
        both = lens_cost(layer_counts=[64, 64], **QWEN)
        assert both >= 3 * ONE_BLOCK_BYTES

    def test_scales_with_positions_layers_and_width(self):
        base = lens_cost(layer_counts=[64], **QWEN)
        assert lens_cost(staged_positions=256, layer_counts=[64], d_model=5120) > base
        assert lens_cost(staged_positions=128, layer_counts=[128], d_model=5120) > base
        assert lens_cost(staged_positions=128, layer_counts=[64], d_model=10240) > base

    def test_a_short_reused_tail_stays_at_the_floor(self):
        # A follow-up turn that reuses its cached prefix stages a handful of positions, and
        # must not be charged for the whole conversation.
        assert lens_cost(staged_positions=3, layer_counts=[64], d_model=5120) == FLAT_LENS_BYTES

    def test_a_caller_that_omits_the_capture_geometry_gets_the_old_answer(self):
        # The capture arguments default to "unknown", which must not become "free x 0 = crash".
        assert lens_cost(layer_counts=[64], **QWEN) == lens_cost(
            layer_counts=[64], capture_positions=0, n_capture_points=0, **QWEN
        )


class TestLensCaptureTerm:
    """The harvest held for the whole sequence, which the staged-rows term never saw."""

    def _dsv4(self, capture_positions: int) -> int:
        return lens_cost(
            staged_positions=8,
            layer_counts=[43, 43],
            capture_positions=capture_positions,
            **DSV4,
        )

    def test_it_grows_with_the_conversation(self):
        # The whole point: turn two of a chat costs more than turn one, which is the
        # difference the flat reservation could not see.
        assert self._dsv4(2048) > self._dsv4(1024)

    def test_it_is_linear_in_the_sequence_length(self):
        # Doubling the conversation doubles the harvest, so the dominant term has to double
        # too. The staged rows and the overhead factor keep this from being exact.
        short, long = self._dsv4(2048), self._dsv4(4096)
        assert 1.9 < long / short < 2.1

    def test_the_v4_pod_at_4096_exceeds_what_that_card_had_free(self):
        # ~7 GiB was free outside the vLLM pool after the lens was placed. The request the
        # reservation waved through wanted more than that, and OOMed in `collective_rpc`.
        assert self._dsv4(4096) > 7 * GIB

    def test_the_v4_pod_at_2048_fits_that_same_headroom(self):
        # Halving the cap is what made the pod usable again, so the estimate has to agree --
        # otherwise the fix for the OOM is a blanket rejection instead.
        assert self._dsv4(2048) < 5 * GIB

    def test_a_hyper_connection_trunk_costs_several_times_a_conventional_one(self):
        # A capture site is `n_streams x d_model` wide, and the reduced copy is held beside
        # it. Same layer count and width, so the ratio is the trunk and nothing else.
        wide = lens_cost(staged_positions=8, layer_counts=[43], capture_positions=2048, **DSV4)
        flat = lens_cost(
            staged_positions=8,
            layer_counts=[43],
            capture_positions=2048,
            n_capture_points=43,
            d_model=4096,
            n_streams=1,
        )
        assert 4.5 < wide / flat < 5.5

    def test_a_conventional_trunk_is_charged_no_reduced_copy(self):
        # `reduce_streams(..., "none")` hands back the captured tensor, so there is no second
        # allocation to charge for -- only the harvest itself.
        one_stream = memory_cost._lens_capture_bytes(
            capture_positions=1024, n_capture_points=32, d_model=4096, n_streams=1
        )
        assert one_stream == 32 * 1024 * 4096 * 2

    def test_a_small_model_stays_on_the_floor(self):
        # gpt2-small at its 550-token limit: the capture is tens of MiB, so this must not
        # start throttling the pods that were never the problem.
        assert (
            lens_cost(staged_positions=8, layer_counts=[12], capture_positions=550, n_capture_points=12, d_model=768)
            == FLAT_LENS_BYTES
        )
