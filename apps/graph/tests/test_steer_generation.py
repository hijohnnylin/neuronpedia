"""`/steer` generates through `transformers` while its wire contract stays TransformerLens-shaped.

The endpoint used to call `HookedTransformer.generate` directly, so on the default `interp_engine`
it raised `AttributeError: 'InterpEngineReplacementModel' object has no attribute 'generate'`. The
attribute was the loud half. The quiet half is everything `steer_generation` translates: keywords
`transformers` rejects, sampling defaults it reads out of the checkpoint, and a continuation it
hands back as text when the endpoint needs the token ids.

Everything here runs on a randomly initialized two-layer GPT-2 built from a local config, so there
is no download, no GPU and no tokenizer to fetch: what is under test is the keywords and the ids,
not what a model would say.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from neuronpedia_graph.steer_generation import (
    _generation_kwargs,
    _SequenceCollector,
    _TransformerLensSampling,
    generate_default,
    generate_steered,
)

PROMPT_IDS = torch.tensor([0, 7, 11, 19])
NEW_TOKENS = 6


@pytest.fixture(scope="module")
def hf_model() -> GPT2LMHeadModel:
    config = GPT2Config(vocab_size=64, n_positions=64, n_embd=32, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(config).eval()
    # An untrained model emits token ids uniformly, so it would hit EOS partway through and every
    # length below would depend on the seed. No EOS means every run reaches `max_new_tokens`.
    model.generation_config.eos_token_id = None
    return model


class FakeInterpEngineModel:
    """The surface `steer_generation` uses from `InterpEngineReplacementModel`.

    `feature_intervention_generate` mirrors the real one where it matters here: keywords reach
    `transformers`' `generate` untouched, and the continuation comes back decoded rather than as
    ids. That is what makes this a test of the translation -- a keyword `generate` will not accept
    raises here exactly as it does in production.
    """

    backend = "interp_engine"

    def __init__(self, model: GPT2LMHeadModel) -> None:
        self.hf_model = model
        self.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=None)
        self.device = torch.device("cpu")
        self.sequences: torch.Tensor | None = None

    def ensure_tokenized(self, prompt: str) -> torch.Tensor:
        assert isinstance(prompt, str)
        return PROMPT_IDS

    def feature_intervention_generate(
        self,
        inputs: str,
        interventions: Any,
        freeze_attention: bool = True,
        **kwargs: Any,
    ) -> tuple[str, torch.Tensor, None]:
        input_ids = self.ensure_tokenized(inputs).unsqueeze(0)
        output = self.hf_model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            pad_token_id=0,
            use_cache=True,
            return_dict_in_generate=True,
            output_logits=True,
            **kwargs,
        )
        self.sequences = output.sequences
        return "the continuation, as text", torch.cat(output.logits, dim=0), None


def test_temperature_zero_asks_for_greedy_decoding():
    """The steer modal sends 0 by default, and `transformers` raises on it where TL means argmax."""
    kwargs = _generation_kwargs(NEW_TOKENS, temperature=0.0, freq_penalty=0.0)
    assert kwargs["do_sample"] is False
    assert "logits_processor" not in kwargs
    # `generate` logs a line for every flag it was handed and will not use, and none of the sampling
    # knobs apply while greedy.
    assert "top_k" not in kwargs and "temperature" not in kwargs


def test_sampling_ignores_the_checkpoints_own_generation_config():
    """Qwen3 ships temperature 0.6 / top_p 0.95 / top_k 20; TransformerLens applies none of them."""
    kwargs = _generation_kwargs(NEW_TOKENS, temperature=0.7, freq_penalty=0.0)
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 1.0
    assert kwargs["top_k"] == 0
    assert kwargs["top_p"] == 1.0
    assert kwargs["repetition_penalty"] == 1.0


def test_frequency_penalty_is_applied_after_the_temperature():
    """`sample_logits` divides first, then subtracts. Reversed, the penalty is scaled by 1/temperature."""
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    tokens = torch.tensor([[1, 1, 3]])

    penalized = _TransformerLensSampling(temperature=2.0, freq_penalty=0.5)(tokens, scores.clone())

    counts = torch.tensor([[0.0, 2.0, 0.0, 1.0]])
    assert torch.allclose(penalized, scores / 2.0 - 0.5 * counts)


def test_non_positive_frequency_penalty_does_nothing():
    """The slider goes to -2, and `sample_logits` applies the penalty only when it is positive."""
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    tokens = torch.tensor([[1, 1, 3]])

    penalized = _TransformerLensSampling(temperature=1.0, freq_penalty=-0.5)(tokens, scores.clone())

    assert torch.allclose(penalized, scores)


def test_default_generation_returns_the_prompt_and_the_continuation(hf_model: GPT2LMHeadModel):
    tokens = generate_default(
        FakeInterpEngineModel(hf_model),
        "a prompt",
        max_new_tokens=NEW_TOKENS,
        temperature=0.0,
        freq_penalty=0.0,
    )

    assert tokens.tolist()[: len(PROMPT_IDS)] == PROMPT_IDS.tolist()
    assert len(tokens) == len(PROMPT_IDS) + NEW_TOKENS


@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_steered_generation_returns_ids_rather_than_the_text(hf_model: GPT2LMHeadModel, temperature: float):
    """The ids have to be the ones `generate` chose, which is what the streamer is there for.

    Re-tokenizing the returned string was the previous approach, and it shifted every logit row by
    one on families whose tokenizer prepends BOS (commit bff40b1e).
    """
    model = FakeInterpEngineModel(hf_model)

    torch.manual_seed(0)
    tokens, logits = generate_steered(
        model,
        "a prompt",
        [],
        max_new_tokens=NEW_TOKENS,
        temperature=temperature,
        freq_penalty=0.0,
        freeze_attention=True,
    )

    assert model.sequences is not None
    assert tokens.tolist() == model.sequences[0].tolist()
    assert len(tokens) == len(PROMPT_IDS) + NEW_TOKENS
    # One row of logits per generated token: the response pairs them up position by position.
    assert logits.shape[0] == NEW_TOKENS


@pytest.mark.parametrize("engine", ["nnsight", "transformerlens"])
def test_an_engine_that_cannot_steer_says_which_one_it_was(engine: str):
    """Refusing beats translating: no graph pod runs either, and both would need their own path."""
    with pytest.raises(NotImplementedError, match=engine):
        generate_default(
            SimpleNamespace(backend=engine),
            "a prompt",
            max_new_tokens=NEW_TOKENS,
            temperature=0.0,
            freq_penalty=0.0,
        )


def test_a_model_that_is_not_a_replacement_model_is_named_by_type():
    """The lm-saes-crm engine loads one, and it has no `backend` to report."""
    with pytest.raises(NotImplementedError, match="SimpleNamespace"):
        generate_default(
            SimpleNamespace(),
            "a prompt",
            max_new_tokens=NEW_TOKENS,
            temperature=0.0,
            freq_penalty=0.0,
        )


def test_the_streamer_collects_the_prompt_and_then_each_token():
    """`generate` hands over the prompt in one call and one token per step after it."""
    collector = _SequenceCollector()
    collector.put(torch.tensor([[4, 5, 6]]))
    collector.put(torch.tensor([7]))
    collector.put(torch.tensor([8]))
    collector.end()

    assert collector.sequence().tolist() == [4, 5, 6, 7, 8]
