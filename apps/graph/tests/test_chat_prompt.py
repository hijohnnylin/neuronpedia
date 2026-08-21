"""Tests for round-tripping a rendered chat prompt back into structured turns.

Model-free: stub tokenizers reimplement the template families we serve (ChatML,
Qwen 3, Gemma, Llama 3) closely enough that the sentinel probing sees the same
delimiters a real tokenizer would emit. Downloading real tokenizers to check this
would make it a network-bound test, not a unit test.

Where a stub is more permissive than the template it stands in for, the probing
learns delimiters that no real prompt contains, so the strictness below is part
of what is under test: gemma's alternation rule, Qwen 3's positional `<think>`
block, and transformers' truncating implementation of `continue_final_message`.
"""

from __future__ import annotations

import pytest

from neuronpedia_graph.chat_prompt import (
    bos_token_positions,
    learn_turn_delimiters,
    parse_chat_prompt,
    render_prompt_from_messages,
    strip_leading_bos,
    unsteerable_token_positions,
)


def continued(rendered: str, messages) -> str:
    """`continue_final_message` as transformers implements it: cut the render at
    the end of the final message's content.

    Modelled here rather than by having each stub skip the final footer, because
    the difference is the point — an empty final content matches at the very end
    of the string, so nothing is cut and the turn stays closed.
    """
    final = messages[-1]["content"]
    return rendered[: rendered.rindex(final) + len(final)]


class QwenTokenizer:
    """ChatML, and auto-inserts a default system block when none is given."""

    bos_token = None

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=False,
    ):
        out = ""
        if messages[0]["role"] != "system":
            out += "<|im_start|>system\nYou are Qwen.<|im_end|>\n"
        for m in messages:
            out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
        return continued(out, messages) if continue_final_message else out


class Qwen3Tokenizer:
    """ChatML, but decorates the assistant turn it expects to be continued.

    Qwen3 wraps a trailing assistant turn in an empty `<think>` block and strips
    that block back out of earlier ones, so an assistant header learned from a
    turn in final position is not a literal that matches every occurrence.
    """

    bos_token = None

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=False,
    ):
        last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
        out = ""
        for i, m in enumerate(messages):
            content = m["content"]
            if m["role"] == "assistant":
                reasoning = ""
                if "</think>" in content:
                    reasoning = content.split("</think>")[0].split("<think>")[-1]
                    content = content.split("</think>")[-1].lstrip("\n")
                if i > last_user:
                    reasoning = reasoning.strip("\n")
                    content = f"<think>\n{reasoning}\n</think>\n\n{content}"
            out += f"<|im_start|>{m['role']}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
        return continued(out, messages) if continue_final_message else out


class GemmaTokenizer:
    """Gemma: BOS baked in, no system role, and `assistant` renamed to `model`.

    Enforces alternation starting at `user`, as the real template does. That
    strictness is load-bearing here: it is what makes a lone `assistant` and a
    repeated `user` unrenderable, which is the case the delimiter probing has to
    survive.
    """

    bos_token = "<bos>"

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=False,
    ):
        if any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")
        for i, m in enumerate(messages):
            if (m["role"] == "user") != (i % 2 == 0):
                raise ValueError("Conversation roles must alternate user/assistant")
        out = "<bos>"
        for m in messages:
            role = "model" if m["role"] == "assistant" else m["role"]
            out += f"<start_of_turn>{role}\n{m['content']}<end_of_turn>\n"
        if add_generation_prompt:
            out += "<start_of_turn>model\n"
        return continued(out, messages) if continue_final_message else out


class Llama3Tokenizer:
    """Llama 3 header/eot form, with a date-stamped preamble on the system turn."""

    bos_token = "<|begin_of_text|>"

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=False,
    ):
        out = "<|begin_of_text|>"
        for m in messages:
            out += f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
        if add_generation_prompt:
            out += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return continued(out, messages) if continue_final_message else out


class NoTemplateTokenizer:
    """A base model: no chat template at all."""

    bos_token = "<bos>"

    def apply_chat_template(self, *args, **kwargs):
        raise ValueError("Cannot use chat template: no template is set")


ALL = [QwenTokenizer(), Qwen3Tokenizer(), GemmaTokenizer(), Llama3Tokenizer()]


def roundtrip(tokenizer, messages):
    """Render, then parse back, the way /generate-graph and Remix pair up."""
    prompt = render_prompt_from_messages(tokenizer, messages)
    parsed = parse_chat_prompt(strip_leading_bos(tokenizer, prompt), learn_turn_delimiters(tokenizer))
    # `None` means "this is plain text, not a chat prompt". Every caller here rendered a chat
    # prompt on the line above, so a None is the roundtrip failing, not a case to handle.
    assert parsed is not None, "a rendered chat prompt did not parse back as chat"
    return prompt, parsed


@pytest.mark.parametrize("tokenizer", ALL, ids=lambda t: type(t).__name__)
def test_single_user_turn_roundtrips(tokenizer):
    _, parsed = roundtrip(tokenizer, [{"role": "user", "content": "hello there"}])
    # Qwen's auto-inserted system block is legitimately part of the prompt, so it
    # comes back as an editable turn; the user turn is always the last one.
    assert parsed[-1] == {"role": "user", "content": "hello there"}


@pytest.mark.parametrize("tokenizer", ALL, ids=lambda t: type(t).__name__)
def test_multi_turn_roundtrips(tokenizer):
    messages = [
        {"role": "user", "content": "what is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "and 3+3?"},
    ]
    _, parsed = roundtrip(tokenizer, messages)
    assert parsed[-3:] == messages


@pytest.mark.parametrize(
    "tokenizer",
    [QwenTokenizer(), Qwen3Tokenizer(), Llama3Tokenizer()],
    ids=lambda t: type(t).__name__,
)
def test_explicit_system_turn_roundtrips(tokenizer):
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "hi"},
    ]
    _, parsed = roundtrip(tokenizer, messages)
    assert parsed == messages


@pytest.mark.parametrize("tokenizer", ALL, ids=lambda t: type(t).__name__)
def test_content_containing_a_newline_survives(tokenizer):
    messages = [{"role": "user", "content": "line one\nline two\n"}]
    _, parsed = roundtrip(tokenizer, messages)
    assert parsed[-1]["content"] == "line one\nline two\n"


@pytest.mark.parametrize("tokenizer", ALL, ids=lambda t: type(t).__name__)
def test_gemma_model_role_comes_back_as_assistant(tokenizer):
    # The canonical role is returned, not the template's literal label, so the
    # webapp's editor never sees "model".
    _, parsed = roundtrip(
        tokenizer,
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
    )
    assert parsed[-1]["role"] == "assistant"


@pytest.mark.parametrize("tokenizer", ALL, ids=lambda t: type(t).__name__)
def test_plain_text_prompt_is_not_treated_as_chat(tokenizer):
    parsed = parse_chat_prompt("The capital of France is", learn_turn_delimiters(tokenizer))
    assert parsed is None


def test_gemma_headers_carry_no_bos():
    # Gemma emits BOS only at the very start of a render, so a header learned
    # from a lone user turn would otherwise hold one. Prompts are parsed with
    # their BOS stripped, so such a header matches nothing and the whole prompt
    # comes back as plain text.
    delimiters = learn_turn_delimiters(GemmaTokenizer())
    assert delimiters["user"][0] == "<start_of_turn>user\n"
    assert delimiters["assistant"][0] == "<start_of_turn>model\n"


def test_stored_prompt_with_an_open_final_turn_parses():
    # What a graph actually stores: turns rendered for generation, so the last
    # one is an empty model turn left open for the model to continue.
    prompt = "<start_of_turn>user\nWhat is the capital of Texas?<end_of_turn>\n<start_of_turn>model\n"
    parsed = parse_chat_prompt(prompt, learn_turn_delimiters(GemmaTokenizer()))
    assert parsed == [
        {"role": "user", "content": "What is the capital of Texas?"},
        {"role": "assistant", "content": ""},
    ]


def test_qwen3_assistant_header_excludes_the_think_block():
    # A header carrying `<think>\n\n</think>\n\n` only matches an assistant turn
    # whose reasoning happens to be empty; every other one is missed and its text
    # is swallowed into the turn before it, tokens and all.
    delimiters = learn_turn_delimiters(Qwen3Tokenizer())
    assert delimiters["assistant"][0] == "<|im_start|>assistant\n"


def test_qwen3_thinking_text_stays_in_the_message_content():
    # The editor models the think block as part of the assistant's content, so
    # the parse has to leave it there rather than absorb it into the delimiter.
    messages = [
        {"role": "user", "content": "What is the capital of Texas?"},
        {
            "role": "assistant",
            "content": "<think>\nRecall US states\n</think>\n\nAustin",
        },
    ]
    _, parsed = roundtrip(Qwen3Tokenizer(), messages)
    assert parsed == messages


@pytest.mark.parametrize("tokenizer", ALL, ids=lambda t: type(t).__name__)
def test_empty_final_assistant_turn_stays_open(tokenizer):
    # The reason to keep a trailing model turn at all is to see what the model
    # says. Close it and the graph instead shows the model guessing what the
    # *user* would say next, which is a different question entirely.
    rendered = render_prompt_from_messages(
        tokenizer,
        [
            {"role": "user", "content": "The capital of Texas is"},
            {"role": "assistant", "content": ""},
        ],
    )
    generation_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "The capital of Texas is"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert rendered == generation_prompt


@pytest.mark.parametrize("tokenizer", ALL, ids=lambda t: type(t).__name__)
def test_nonempty_final_turn_is_continued_not_reopened(tokenizer):
    # The other half: a final turn WITH content is continued from, so the render
    # must not close it or append a second header after it.
    rendered = render_prompt_from_messages(
        tokenizer,
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "The capital of Texas is"},
        ],
    )
    assert rendered.endswith("The capital of Texas is")


def test_model_without_a_chat_template_yields_no_delimiters():
    tokenizer = NoTemplateTokenizer()
    assert learn_turn_delimiters(tokenizer) == {}
    assert parse_chat_prompt("anything", {}) is None


def test_leading_bos_is_stripped_once_and_only_when_present():
    tokenizer = GemmaTokenizer()
    assert strip_leading_bos(tokenizer, "<bos>hi") == "hi"
    assert strip_leading_bos(tokenizer, "hi") == "hi"
    assert strip_leading_bos(tokenizer, "<bos><bos>hi") == "<bos>hi"


def test_tokenizer_without_bos_leaves_the_prompt_alone():
    assert strip_leading_bos(QwenTokenizer(), "<|im_start|>user\nhi") == ("<|im_start|>user\nhi")


class Encoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class IdTokenizer:
    """Tokenizes by whitespace, mapping `<bos>` to the BOS id."""

    bos_token = "<bos>"
    bos_token_id = 2

    def __call__(self, prompt):
        return Encoding([2 if word == "<bos>" else 100 for word in prompt.split()])


def test_bos_position_is_found_at_the_start():
    assert bos_token_positions(IdTokenizer(), "<bos> hello world") == {0}


def test_every_bos_occurrence_is_reported():
    # A stored graph prompt can carry a baked-in <bos> that a re-tokenization
    # duplicates, so a mid-prompt occurrence has to be caught too.
    assert bos_token_positions(IdTokenizer(), "<bos> hi <bos> there") == {0, 2}


def test_prompt_with_no_bos_has_no_bos_positions():
    assert bos_token_positions(IdTokenizer(), "hello world") == set()


def test_tokenizer_without_a_bos_id_reports_nothing():
    class NoBos:
        bos_token_id = None

    assert bos_token_positions(NoBos(), "anything") == set()


def test_unsteerable_positions_are_the_bos_positions():
    # `/steer` drops features here and `/parse-chat-prompt` reports the same set
    # so the UI can hide those sliders. Pinning them equal is what keeps the
    # controls a client hides identical to what the server refuses.
    tokenizer = IdTokenizer()
    for prompt in ("<bos> hello world", "<bos> hi <bos> there", "hello world"):
        assert unsteerable_token_positions(tokenizer, prompt) == bos_token_positions(tokenizer, prompt)


def test_turn_markers_stay_steerable():
    # Only BOS is refused. A turn marker carries content the user may well want
    # to steer, and `steer_special_tokens` is the flag that governs those.
    class TurnTokenizer:
        bos_token = "<bos>"
        bos_token_id = 2

        def __call__(self, prompt):
            ids = {"<bos>": 2, "<start_of_turn>": 105, "<end_of_turn>": 106}
            return Encoding([ids.get(word, 100) for word in prompt.split()])

    assert unsteerable_token_positions(TurnTokenizer(), "<bos> <start_of_turn> hi <end_of_turn>") == {0}
