"""How a request's sampling knobs become the settings a generation runs with.

Every generating endpoint resolves the same way: the request's value, else the generating
checkpoint's ``generation_config.json``, else neutral. The response says what was used.
"""

from interp_engine import RecommendedSampling

from server import (
    CompletionRequest,
    DescribeRequest,
    SamplingReport,
    _resolve_sampling,
    _sampling_kwargs,
    _sampling_report,
)

QWEN_THINKING = RecommendedSampling(temperature=0.6, top_k=20, top_p=0.95, source="test")


def test_unset_knobs_take_the_checkpoint_recommendation():
    settings = _resolve_sampling(QWEN_THINKING, DescribeRequest(activations=[[0.0]]))
    assert (settings.temperature, settings.top_k, settings.top_p) == (0.6, 20, 0.95)
    assert settings.presence_penalty == 0.0


def test_set_knobs_win_over_the_recommendation():
    req = DescribeRequest(activations=[[0.0]], temperature=0.0, top_k=0, presence_penalty=1.5)
    settings = _resolve_sampling(QWEN_THINKING, req)
    assert settings.temperature == 0.0
    assert settings.top_k is None, "0 spells 'keep all', reported as null"
    assert settings.top_p == 0.95, "an untouched knob still inherits"
    assert settings.presence_penalty == 1.5


def test_no_recommendation_is_neutral():
    settings = _resolve_sampling(RecommendedSampling(), CompletionRequest(text="hi"))
    assert settings.temperature == 1.0
    assert settings.top_k is None and settings.top_p is None
    assert settings.presence_penalty == 0.0


def test_report_carries_every_decided_knob():
    settings = _resolve_sampling(QWEN_THINKING, CompletionRequest(text="hi", presence_penalty=0.5))
    report = _sampling_report(settings)
    assert report == SamplingReport(temperature=0.6, top_k=20, top_p=0.95, presence_penalty=0.5)
    assert set(report.model_dump()) == {"temperature", "top_k", "top_p", "presence_penalty"}


def test_completion_generate_keywords():
    greedy = _sampling_kwargs(
        _resolve_sampling(RecommendedSampling(), CompletionRequest(text="hi", temperature=0.0)), 3
    )
    assert greedy["do_sample"] is False
    assert "temperature" not in greedy

    sampled = _sampling_kwargs(_resolve_sampling(QWEN_THINKING, CompletionRequest(text="hi", presence_penalty=1.0)), 3)
    assert sampled["do_sample"] is True
    assert (sampled["temperature"], sampled["top_k"], sampled["top_p"]) == (0.6, 20, 0.95)
    assert len(sampled["logits_processor"]) == 1, "the presence penalty rides along as a logits processor"
