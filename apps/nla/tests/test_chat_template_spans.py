"""Tests for the canonical NLA tokenization.

Model-free: a stub tokenizer records the ``add_special_tokens`` flag it was called with, which
is the whole decision under test. Real-tokenizer behaviour (gemma renders 17 tokens with the
duplicate and 16 without, llama 43 and 42) is a GPU/download-bound check, not a unit test.
"""

from __future__ import annotations

import re

from chat_template_spans import compute_spans, encode_with_special

BOS = "<bos>"


class StubTokenizer:
    """Callable like a HF tokenizer, recording how it was invoked."""

    def __init__(self, bos_token: str | None = BOS):
        self.bos_token = bos_token
        self.calls: list[bool] = []

    def __call__(self, text: str, add_special_tokens: bool = True):
        self.calls.append(add_special_tokens)
        # Stand-in ids; only the flag matters here.
        return {"input_ids": [1, 2, 3]}


def test_rendered_chat_starting_with_bos_does_not_add_another():
    tok = StubTokenizer()
    encode_with_special(tok, f"{BOS}<start_of_turn>user\nhi<end_of_turn>\n")
    assert tok.calls == [False]


def test_raw_text_still_gets_its_bos():
    tok = StubTokenizer()
    encode_with_special(tok, "What is 2+2?")
    assert tok.calls == [True]


def test_tokenizer_without_bos_always_adds_specials():
    """Qwen ships `bos_token=None`; there is nothing to duplicate, so don't change its path."""
    tok = StubTokenizer(bos_token=None)
    encode_with_special(tok, "<|im_start|>user\nhi<|im_end|>\n")
    assert tok.calls == [True]


def test_bos_not_at_the_start_does_not_count():
    """Only a leading BOS is the template's own prefix; one mid-string is ordinary content."""
    tok = StubTokenizer()
    encode_with_special(tok, f"tell me about {BOS} tokens")
    assert tok.calls == [True]


def test_returns_the_tokenizers_input_ids():
    tok = StubTokenizer()
    assert encode_with_special(tok, "hi") == [1, 2, 3]


def test_tokenizer_missing_bos_attribute_is_tolerated():
    """Partial tokenizer stand-ins shouldn't crash the encode path."""

    class NoBosAttr:
        def __call__(self, text: str, add_special_tokens: bool = True):
            return {"input_ids": [7]}

    assert encode_with_special(NoBosAttr(), "hi") == [7]


# --- header/content split for a held-open final message ----------------------------------------

# One token per special marker, per whitespace run, and per word — enough structure for the span
# diffing, which only ever compares token ids.
_TOKEN_RE = re.compile(r"<\|[^|]*\|>|</?think>|\s+|[^\s<]+|.")


class TrimmingTemplateTokenizer:
    """A Qwen3.5/3.6-shaped template: content is ``|trim``-ed and the assistant header ends in
    spacing (``</think>\\n\\n``) that the header/content split has to recognise as its own.

    ``apply_chat_template`` mirrors how transformers (5.x) actually implements
    ``continue_final_message``: it appends a sentinel to the final message's content, renders, and
    cuts the string at the sentinel — falling back to ``rstrip()`` when the template trimmed the
    sentinel's trailing space, which it always does here. That fallback is the whole bug: with the
    empty content used to isolate the wrapper there is nothing left to anchor on, so the
    template's own ``\\n\\n`` disappears from the wrapper and reads as message content instead.
    """

    TAG = "CONTINUE_FINAL_MESSAGE_TAG "

    def __init__(self):
        self.bos_token = None
        self._vocab: dict[str, int] = {}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,  # noqa: ARG002 - callers here always render to a string
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
    ) -> str:
        msgs = [dict(m) for m in messages]
        if continue_final_message:
            msgs[-1]["content"] = msgs[-1]["content"] + self.TAG
        out = ""
        for m in msgs:
            content = m["content"].strip()
            opener = f"<|im_start|>{m['role']}\n"
            if m["role"] == "assistant":
                opener += "<think>\n\n</think>\n\n"
            out += f"{opener}{content}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        if continue_final_message:
            loc = out.rindex(self.TAG.strip())
            out = out[:loc] if out[loc : loc + len(self.TAG)] == self.TAG else out[:loc].rstrip()
        return out

    def encode(self, text: str) -> list[int]:
        ids = []
        for match in _TOKEN_RE.findall(text):
            ids.append(self._vocab.setdefault(match, len(self._vocab)))
        return ids

    def decode(self, ids: list[int], clean_up_tokenization_spaces: bool = False) -> str:  # noqa: ARG002
        reverse = {v: k for k, v in self._vocab.items()}
        return "".join(reverse[i] for i in ids)


def test_held_open_final_message_content_is_only_the_prefill():
    tok = TrimmingTemplateTokenizer()
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "The answer is"}]
    _, spans = compute_spans(
        tok,
        messages,
        encode=tok.encode,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    assistant = [s for s in spans if s.message_index == 1]
    assert "".join(s.token_str for s in assistant if s.section == "content") == "The answer is"
    # An open turn has no footer, and the header runs unbroken up to the content.
    sections = [s.section for s in assistant]
    header_len = sections.index("content")
    assert sections == ["header"] * header_len + ["content"] * (len(sections) - header_len)


def test_closed_message_still_gets_its_footer():
    """The fix must not cost a closed turn its ``<|im_end|>`` footer."""
    tok = TrimmingTemplateTokenizer()
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "4"}]
    _, spans = compute_spans(tok, messages, encode=tok.encode, add_generation_prompt=True)
    assistant = [s for s in spans if s.message_index == 1]
    assert "".join(s.token_str for s in assistant if s.section == "content") == "4"
    assert "".join(s.token_str for s in assistant if s.section == "footer") == "<|im_end|>\n"
