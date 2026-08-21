"""Per-request cost estimates: the properties that make admission control safe.

The exact byte counts are not the contract -- they are estimates, and tuning them should not
break tests. What must hold is the shape of the function, because the admission controller is
only as good as these orderings:

- An empty `selected_sources` means the WHOLE source set, so it must not be costed as the
  cheapest possible request when it is the most expensive one the UI issues.
- A wider SAE must cost more than a narrow one, with no configuration change. That is what
  makes a newly added source self-throttling.
- Longer prompts, more sources, more results, bigger batches all cost more.
- A source with DFA enabled costs more than the same source without it, and superlinearly in
  the token count, because DFA recomputes an attention pattern.
- Nothing may under-estimate by being fooled into a zero.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from neuronpedia_inference import memory_cost
from neuronpedia_inference.memory_cost import (
    activation_all_batch_cost,
    activation_all_cost,
    activation_single_cost,
    activation_source_cost,
    activation_topk_by_token_cost,
    estimate_tokens,
    sae_residency_bytes,
    similarity_matrix_cost,
    steer_cost,
)

SOURCE_SET = "res-test"
NARROW_SOURCES = [f"{layer}-res-test" for layer in range(4)]
WIDE_SOURCE = "0-res-wide"

# Attention sources, which are the ones that enable DFA. Same width as the narrow residual
# sources so that a comparison between the two isolates the DFA term.
ATT_SET = "att-test"
ATT_SOURCES = [f"{layer}-att-test" for layer in range(4)]

D_SAE_NARROW = 16_384
D_SAE_WIDE = 1_048_576
D_IN = 2304

# Attention geometry, stubbed so the DFA assertions do not depend on the module's fallbacks.
N_HEADS, N_KV_HEADS, HEAD_DIM = 8, 4, 256

# Captured before the autouse fixture stubs it, for the one test that wants the real thing.
_real_attention_dims = memory_cost._attention_dims


@pytest.fixture(autouse=True)
def stub_singletons():
    """Stub the SAE manager and config the estimators read, with no weights involved."""
    dims: dict[str, tuple[int, int]] = dict.fromkeys([*NARROW_SOURCES, *ATT_SOURCES], (D_SAE_NARROW, D_IN))
    dims[WIDE_SOURCE] = (D_SAE_WIDE, D_IN)
    dfa_enabled = set(ATT_SOURCES)

    manager = MagicMock()
    manager.sae_set_to_saes = {
        SOURCE_SET: list(NARROW_SOURCES),
        ATT_SET: list(ATT_SOURCES),
    }
    manager.get_d_sae.side_effect = lambda s: dims.get(s, (None, None))[0]
    manager.get_d_in.side_effect = lambda s: dims.get(s, (None, None))[1]
    manager.get_sae_hook.side_effect = lambda s: f"blocks.{s.split('-')[0]}.hook_{s.split('-')[1]}"
    # Spelled out rather than left to the mock's default truthy return, which would charge
    # every source in this file for DFA.
    manager.is_dfa_enabled.side_effect = lambda s: s in dfa_enabled
    # Weight sizes as recorded at load time (see sae_cache), scaled off d_sae so the wide
    # source is the expensive one to page in as well as to encode.
    manager.get_sae_nbytes.side_effect = lambda s: dims.get(s, (0, 0))[0] * D_IN * 4

    config = SimpleNamespace(token_limit=1024, sae_dtype="float32", model_dtype="bfloat16")

    with (
        patch.object(memory_cost.SAEManager, "get_instance", return_value=manager),
        patch.object(memory_cost.Config, "get_instance", return_value=config),
        patch.object(
            memory_cost,
            "_attention_dims",
            return_value=(N_HEADS, N_KV_HEADS, HEAD_DIM),
        ),
    ):
        yield


def _all_request(**overrides):
    base = {
        "prompt": "a prompt of some length",
        "source_set": SOURCE_SET,
        "selected_sources": list(NARROW_SOURCES),
        "num_results": 50,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _att_request(**overrides):
    """The same request against the attention set, where every source enables DFA."""
    base = {"source_set": ATT_SET, "selected_sources": list(ATT_SOURCES)}
    base.update(overrides)
    return _all_request(**base)


def test_an_empty_source_list_is_costed_as_the_whole_set():
    """The all-layers default. Costing it as "no sources" would admit the single most
    expensive request the UI issues as though it were free."""
    explicit = activation_all_cost(_all_request())
    implicit = activation_all_cost(_all_request(selected_sources=[]))

    assert implicit == explicit
    assert implicit > 0


def test_a_wider_sae_costs_more_with_no_configuration_change():
    """Why a newly added SAE throttles itself: the estimate reads its recorded width."""
    narrow = activation_all_cost(_all_request(selected_sources=[NARROW_SOURCES[0]]))
    wide = activation_all_cost(_all_request(selected_sources=[WIDE_SOURCE]))

    assert wide > narrow
    # The encode term dominates and scales linearly in d_sae, so the gap should be within
    # spitting distance of the width ratio rather than a rounding difference.
    assert wide > narrow * (D_SAE_WIDE / D_SAE_NARROW) / 2


def test_the_widest_source_sets_the_cost_not_the_sum():
    """The estimate has to match the streaming reduction it is estimating: one source is
    encoded at a time, so adding narrow sources to a wide one must not multiply the cost."""
    wide_alone = activation_all_cost(_all_request(selected_sources=[WIDE_SOURCE]))
    wide_plus_narrow = activation_all_cost(_all_request(selected_sources=[WIDE_SOURCE, *NARROW_SOURCES]))

    # More sources still cost more (each adds a capture tensor), but nowhere near 5x.
    assert wide_alone < wide_plus_narrow < 2 * wide_alone


def test_more_sources_more_results_and_longer_prompts_all_cost_more():
    base = _all_request(selected_sources=NARROW_SOURCES[:1], num_results=10, prompt="hi")

    assert activation_all_cost(
        _all_request(selected_sources=NARROW_SOURCES, num_results=10, prompt="hi")
    ) > activation_all_cost(base)
    assert activation_all_cost(
        _all_request(selected_sources=NARROW_SOURCES[:1], num_results=500, prompt="hi")
    ) > activation_all_cost(base)
    assert activation_all_cost(
        _all_request(selected_sources=NARROW_SOURCES[:1], num_results=10, prompt="hi" * 200)
    ) > activation_all_cost(base)


def test_a_bigger_batch_costs_more():
    one = SimpleNamespace(
        prompts=["a prompt"],
        source_set=SOURCE_SET,
        selected_sources=list(NARROW_SOURCES),
        num_results=50,
    )
    four = SimpleNamespace(
        prompts=["a prompt"] * 4,
        source_set=SOURCE_SET,
        selected_sources=list(NARROW_SOURCES),
        num_results=50,
    )

    assert activation_all_batch_cost(four) > activation_all_batch_cost(one)


def test_dfa_costs_more_than_the_same_search_without_it():
    """DFA recomputes an attention pattern, so it cannot be free.

    The two sources here are the same width, so the encode and capture terms match and the
    difference is the pattern alone -- which these estimates omitted entirely, admitting an
    attention search as though it cost what a residual one does.
    """
    residual = activation_all_cost(_all_request(selected_sources=[NARROW_SOURCES[0]]))
    attention = activation_all_cost(_att_request(selected_sources=[ATT_SOURCES[0]]))

    assert attention > residual


def test_dfa_grows_superlinearly_in_the_token_count():
    """The pattern is [n_heads, dest, src], so DFA puts a quadratic term in the estimate.

    Superlinearly, not quadratically: as with the similarity matrix, the linear encode term
    still dominates at these lengths and widths. The point is that 4x the tokens now costs
    more than 4x, where before it cost exactly 4x.
    """
    short = activation_all_cost(_att_request(prompt="x" * 100))
    long = activation_all_cost(_att_request(prompt="x" * 400))

    assert long > 4 * short


def test_a_residual_source_pays_nothing_for_dfa():
    """Costing DFA off `is_dfa_enabled` is what keeps the residual sets -- the common case --
    exactly as cheap to admit as they were."""
    assert memory_cost._dfa_bytes(NARROW_SOURCES, 512, layers_held=50) == 0
    assert memory_cost._dfa_bytes([], 512, layers_held=50) == 0


def test_the_dfa_memo_is_bounded_by_the_layers_that_can_be_in_it():
    """/activation/all memoizes a capture per layer and never evicts, so more results mean
    more patterns alive at once -- but only up to the number of DFA sources selected, after
    which further results reuse a pattern already held."""
    held_by_results = memory_cost._dfa_bytes(ATT_SOURCES, 256, layers_held=2)

    assert held_by_results > memory_cost._dfa_bytes(ATT_SOURCES, 256, layers_held=1)
    assert memory_cost._dfa_bytes(ATT_SOURCES, 256, layers_held=50) == memory_cost._dfa_bytes(
        ATT_SOURCES, 256, layers_held=len(ATT_SOURCES)
    )


def test_attention_dims_fall_back_when_no_model_is_loaded():
    """These estimators run before admission, and the DFA callers can reach them before a
    model exists. Raising there would fail the admission check rather than the request being
    sized, so the unloaded case has to resolve to the (deliberately large) fallbacks."""
    from neuronpedia_inference.shared import Model

    with (
        patch.object(memory_cost, "_attention_dims", _real_attention_dims),
        # `create` because `_instance` is only an annotation until a model is set, which is
        # itself one of the ways this can be reached unloaded.
        patch.object(Model, "_instance", None, create=True),
    ):
        n_heads, n_kv_heads, head_dim = memory_cost._attention_dims()

    assert (n_heads, n_kv_heads, head_dim) == (
        memory_cost._FALLBACK_N_HEADS,
        memory_cost._FALLBACK_N_HEADS,
        memory_cost._FALLBACK_HEAD_DIM,
    )


def test_similarity_matrix_cost_grows_faster_than_the_token_count():
    """The similarity matrix is [n_tokens, n_tokens], so cost has to grow SUPERlinearly.

    Only superlinearly, not quadratically: at any realistic SAE width the linear encode term
    still dominates the matrix at these lengths. The point is that the quadratic term is
    accounted for at all -- it is why this endpoint needed a token limit.
    """
    short = similarity_matrix_cost(SimpleNamespace(sourceId=NARROW_SOURCES[0], text="x" * 100))
    long = similarity_matrix_cost(SimpleNamespace(sourceId=NARROW_SOURCES[0], text="x" * 400))

    assert long > 4 * short


def test_estimate_tokens_never_underestimates_and_respects_the_limit():
    """A token is at least one character, and the endpoint's own limit caps it."""
    assert estimate_tokens("hello") >= 1
    assert estimate_tokens("x" * 50) >= 50
    # token_limit is 1024 in the stub.
    assert estimate_tokens("x" * 100_000) == 1024
    assert estimate_tokens("") == 1
    assert estimate_tokens(None) == 1


def test_an_unknown_source_falls_back_to_something_expensive():
    """An unrecorded source (a `neurons` source) must not be costed as free."""
    known = activation_single_cost(SimpleNamespace(source=NARROW_SOURCES[0], prompt="a prompt"))
    unknown = activation_single_cost(SimpleNamespace(source="9-not-recorded", prompt="a prompt"))

    assert unknown > known


@pytest.mark.parametrize(
    ("estimator", "request_obj"),
    [
        (activation_all_cost, _all_request()),
        (
            activation_single_cost,
            SimpleNamespace(source=NARROW_SOURCES[0], prompt="hi"),
        ),
        (
            activation_topk_by_token_cost,
            SimpleNamespace(source=NARROW_SOURCES[0], prompt="hi", top_k=20),
        ),
        (
            activation_source_cost,
            SimpleNamespace(source=NARROW_SOURCES[0], prompts=["hi"]),
        ),
        (
            similarity_matrix_cost,
            SimpleNamespace(sourceId=NARROW_SOURCES[0], text="hi"),
        ),
        (steer_cost, SimpleNamespace()),
    ],
    ids=["all", "single", "topk", "source", "similarity", "steer"],
)
def test_every_estimator_returns_a_positive_int(estimator, request_obj):
    """A zero would silently disable rationing for that endpoint."""
    cost = estimator(request_obj)
    assert isinstance(cost, int)
    assert cost > 0


def test_a_missing_top_k_does_not_zero_the_estimate():
    """`top_k=None` means the endpoint's default, not "no work"."""
    cost = activation_topk_by_token_cost(SimpleNamespace(source=NARROW_SOURCES[0], prompt="hi", top_k=None))
    assert cost > 0


def test_sae_residency_is_the_largest_single_source_not_the_sum():
    """Endpoints hold one SAE at a time, so admission reserves for one, not for all of them."""
    mixed = _all_request(selected_sources=[*NARROW_SOURCES, WIDE_SOURCE])
    reserved = sae_residency_bytes(mixed)

    assert reserved == D_SAE_WIDE * D_IN * 4
    assert reserved < sum(D_SAE_NARROW * D_IN * 4 for _ in NARROW_SOURCES) + reserved


def test_sae_residency_expands_the_all_layers_default():
    """An empty selection is the whole set here too, or the widest request reserves nothing."""
    assert sae_residency_bytes(_all_request(selected_sources=[])) > 0


@pytest.mark.parametrize(
    "request_obj",
    [
        SimpleNamespace(source=WIDE_SOURCE),
        SimpleNamespace(sourceId=WIDE_SOURCE),
        SimpleNamespace(features=[SimpleNamespace(source=WIDE_SOURCE, index=3)]),
    ],
    ids=["source", "sourceId", "steer-features"],
)
def test_sae_residency_finds_the_source_in_every_request_shape(request_obj):
    """Missing a shape would admit a steer or util request with no residency reserved."""
    assert sae_residency_bytes(request_obj) == D_SAE_WIDE * D_IN * 4
