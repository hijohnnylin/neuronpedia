"""Tests for turn-end token detection.

`/completion` tags the generated turn-end token as `section: "footer"` so clients
can drop it from message content without matching literal `<|im_end|>`-style
strings. The id set comes from the model's own config, and that resolution is
what these tests pin down. Model-free: stubs stand in for the tokenizer and the
model's `generation_config`.
"""

from __future__ import annotations

from chat_template_spans import terminal_token_ids


class StubTokenizer:
    def __init__(self, eos_token_id=None):
        self.eos_token_id = eos_token_id


class StubGenerationConfig:
    def __init__(self, eos_token_id=None):
        self.eos_token_id = eos_token_id


class StubModel:
    def __init__(self, eos_token_id=None):
        self.generation_config = StubGenerationConfig(eos_token_id)


def test_scalar_eos_from_tokenizer():
    assert terminal_token_ids(StubTokenizer(151645)) == {151645}


def test_generation_config_list_is_unioned_in():
    # Llama/Qwen ship a list here, covering both <|eot_id|> and <|eom_id|>.
    tok = StubTokenizer(128009)
    model = StubModel([128009, 128008])
    assert terminal_token_ids(tok, model) == {128008, 128009}


def test_generation_config_scalar_is_included():
    assert terminal_token_ids(StubTokenizer(1), StubModel(2)) == {1, 2}


def test_no_eos_anywhere_yields_empty_set():
    assert terminal_token_ids(StubTokenizer(None), StubModel(None)) == set()


def test_model_omitted_falls_back_to_tokenizer_only():
    assert terminal_token_ids(StubTokenizer(7)) == {7}


def test_none_entries_inside_a_list_are_skipped():
    assert terminal_token_ids(StubTokenizer(None), StubModel([3, None, 4])) == {3, 4}


def test_tokenizer_missing_eos_attribute_is_tolerated():
    class Bare:
        pass

    assert terminal_token_ids(Bare()) == set()
