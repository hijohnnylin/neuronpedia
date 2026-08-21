"""The engine kwargs this server chooses for vLLM, beyond what ``load_model`` derives.

The one that needs explaining is ``hf_overrides={"is_mm_prefix_lm": False}``, which decides the
attention backend for every layer of a multimodal-architecture model.

vLLM sets ``is_mm_prefix_lm`` -- "image tokens attend bidirectionally" -- from ``model_type``
against a hardcoded list holding ``"gemma3"``. Every attention layer then asks for a backend that
can express that mask, and ``validate_configuration`` rejects FlashAttention with "partial
multimodal token full attention not supported", leaving Triton unified attention. That applies to a
pure-text request, because the flag is a property of the checkpoint rather than the batch: on this
box ``get_attn_backend(head_size=256, ..., use_mm_prefix=True)`` returns ``TRITON_ATTN`` and the
same call with ``False`` returns ``FLASH_ATTN``. It also splits Gemma 3 by size, since
``gemma-3-12b-it`` is ``model_type "gemma3"`` while ``gemma-3-1b-it`` is ``"gemma3_text"``.

The override is sound here only because no endpoint on this server accepts an image, which is what
the last test is for: it is a tripwire on that premise, not a test of the override.
"""

from __future__ import annotations

import pytest

from neuronpedia_inference.server import (
    _DEFAULT_CUDAGRAPH_CAPTURE_SIZES,
    _engine_context_len,
    _parse_static_points,
    _vllm_backend_kwargs,
    _vllm_engine_backend,
)


def test_the_model_is_declared_text_only_so_flash_attention_stays_eligible():
    kwargs = _vllm_backend_kwargs(2048, backend="vllm")
    assert kwargs["extra_vllm_kwargs"]["hf_overrides"] == {"is_mm_prefix_lm": False}


def test_the_token_limit_is_the_engine_context_length():
    assert _vllm_backend_kwargs(4096, backend="vllm")["max_model_len"] == 4096


def test_the_engine_context_covers_the_lens_cap_when_it_is_the_higher_one():
    """A pod at --token_limit 256 with the default 1024 lens cap must still be built for 1024.

    Built for 256, a 267-token lens conversation passed the endpoint's own check (which bounds
    against lens_token_limit) and then failed inside vLLM as "maximum context length is 256".
    """
    assert _engine_context_len(256, 1024) == 1024


def test_the_engine_context_covers_the_prompt_cap_when_it_is_the_higher_one():
    assert _engine_context_len(8192, 1024) == 8192


def test_the_two_flags_pick_one_of_the_three_vllm_backends():
    """The whole mode decision, so a new flag combination cannot quietly land on the wrong one."""
    assert _vllm_engine_backend(generation_only=False, static_points=None) == "vllm"
    assert _vllm_engine_backend(generation_only=False, static_points="auto") == "vllm-static"
    assert _vllm_engine_backend(generation_only=True, static_points=None) == "vllm-generate"


def test_generation_only_wins_over_a_declared_set():
    """Unreachable through startup, which refuses the pair, but the mapping stays total."""
    assert _vllm_engine_backend(generation_only=True, static_points="auto") == "vllm-generate"


def test_enforce_eager_is_left_to_the_engine_on_every_backend():
    """The engine owns it per backend: True for hooked vLLM, False for the two graph modes, and it
    refuses a caller that passes True to a graph mode. Saying nothing here is what lets it."""
    for backend in ("vllm", "vllm-static", "vllm-generate"):
        points = "auto" if backend == "vllm-static" else None
        assert "enforce_eager" not in _vllm_backend_kwargs(2048, backend=backend, static_points=points)


def test_a_declared_set_reaches_the_engine_kwargs():
    from interp_engine.address import Address

    points = [Address("resid_post", 7)]
    kwargs = _vllm_backend_kwargs(2048, backend="vllm-static", static_points=points)
    assert kwargs["static_points"] is points


def test_generate_pods_declare_nothing_because_the_backend_name_says_it():
    kwargs = _vllm_backend_kwargs(2048, backend="vllm-generate")
    assert "static_points" not in kwargs


def test_hooked_vllm_does_not_set_cuda_graph_capture_sizes():
    extra = _vllm_backend_kwargs(2048, backend="vllm")["extra_vllm_kwargs"]
    assert "compilation_config" not in extra


def test_graph_pods_pass_power_of_two_capture_sizes_to_vllm():
    expected = list(_DEFAULT_CUDAGRAPH_CAPTURE_SIZES)
    static = _vllm_backend_kwargs(2048, backend="vllm-static", static_points="auto")
    gen = _vllm_backend_kwargs(2048, backend="vllm-generate")
    assert static["extra_vllm_kwargs"]["compilation_config"]["cudagraph_capture_sizes"] == expected
    assert gen["extra_vllm_kwargs"]["compilation_config"]["cudagraph_capture_sizes"] == expected


def test_capture_sizes_env_overrides_the_default(monkeypatch):
    monkeypatch.setenv("VLLM_CUDAGRAPH_CAPTURE_SIZES", "1,8,32")
    kwargs = _vllm_backend_kwargs(2048, backend="vllm-static", static_points="auto")
    assert kwargs["extra_vllm_kwargs"]["compilation_config"]["cudagraph_capture_sizes"] == [1, 8, 32]


def test_the_prefill_chunk_is_left_to_the_engine_unless_pinned():
    extra = _vllm_backend_kwargs(2048, backend="vllm-static", static_points="auto")["extra_vllm_kwargs"]
    assert "max_num_batched_tokens" not in extra


def test_a_pinned_prefill_chunk_reaches_every_kind_of_vllm_pod(monkeypatch):
    """All three backends: it sizes the engine, not the endpoint set."""
    monkeypatch.setenv("MAX_NUM_BATCHED_TOKENS", "1024")
    for kwargs in (
        _vllm_backend_kwargs(2048, backend="vllm"),
        _vllm_backend_kwargs(2048, backend="vllm-static", static_points="auto"),
        _vllm_backend_kwargs(2048, backend="vllm-generate"),
    ):
        assert kwargs["extra_vllm_kwargs"]["max_num_batched_tokens"] == 1024


def test_a_non_integer_prefill_chunk_is_refused_rather_than_ignored(monkeypatch):
    monkeypatch.setenv("MAX_NUM_BATCHED_TOKENS", "lots")
    with pytest.raises(ValueError, match="MAX_NUM_BATCHED_TOKENS"):
        _vllm_backend_kwargs(2048, backend="vllm")


def test_parse_static_points_accepts_auto_sae_and_json():
    assert _parse_static_points(None) is None
    assert _parse_static_points("") is None
    assert _parse_static_points("auto") == "auto"
    assert _parse_static_points("sae") == "sae"
    assert _parse_static_points('[["resid_post", 7]]') == [["resid_post", 7]]


def test_an_empty_declared_set_points_at_generation_only_instead_of_loading_a_useless_pod():
    """It used to mean "graphs, no taps". That mode is GENERATION_ONLY now, and leaving both
    spellings alive is how a pod ends up in a mode nobody can name from its config."""
    with pytest.raises(ValueError, match="GENERATION_ONLY"):
        _parse_static_points("[]")


def test_a_malformed_static_points_is_refused_rather_than_ignored():
    with pytest.raises(ValueError, match="STATIC_POINTS"):
        _parse_static_points("resid_post")


def test_no_endpoint_accepts_an_image():
    """The premise the override rests on, checked rather than asserted in a comment.

    If this fails, an endpoint has grown an image input and the override above is now a
    correctness bug: those tokens would attend causally instead of bidirectionally.
    """
    import inspect
    from pathlib import Path

    import neuronpedia_inference.endpoints as endpoints

    root = Path(inspect.getfile(endpoints)).parent
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        # The request models are what a client can actually send; a field named for an image is
        # the cheapest signal that one has appeared.
        if any(
            marker in path.read_text()
            for marker in ("image_url", "image_data", "images:", "image:", "multi_modal_data")
        )
    ]
    assert offenders == [], f"image input reached an endpoint: {offenders}"
