"""Server-side chat templating + per-token span metadata for NLA.

The webapp frontend used to hand-build model-family chat-template strings
(`<|im_start|>`, `<start_of_turn>`, `<|eot_id|>`, ...) and re-derive message
structure from those literal tokens. This module moves that to the server: it
applies the model's REAL chat template and computes per-token span metadata
(role / channel / section / message index) so the frontend can render the
conversation without any per-family knowledge.

Family-agnostic and driven purely by the tokenizer's own chat template — it is a
trimmed adaptation of ``interp_engine.Tokenize.message_spans``, whose ``TokenSpan`` is
field-identical to the one below. The engine **is** a live dependency of this app now (the
comment here used to say it could not be imported yet, which stopped being true), so this is
a redundant copy rather than a necessary one, and folding it back is worth doing.

It is not a rename, which is why it has not happened yet: the alignment contract below is
against :func:`encode_with_special`, while the engine renders through its own ``Tokenize``
and BOS convention (``default_prepend_bos``). Those must be shown to agree first, because a
one-token disagreement shifts every position in every span and the spans still look
well-formed. ``tests/test_chat_template_spans.py`` is what a migration has to keep green.

Alignment contract: the NLA endpoints tokenize the rendered string with
:func:`encode_with_special`, and so does this module (render with
``tokenize=False``, then encode) instead of ``apply_chat_template(tokenize=True)``,
so spans line up 1:1 with the positions the endpoints produce.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TokenSpan:
    """Per-token span metadata for one position in a chat-formatted sequence."""

    position: int
    token_id: int
    token_str: str
    message_index: int | None
    role: str | None
    channel: str | None
    section: str  # "header" | "content" | "footer" | "scaffold"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def encode_with_special(tokenizer: Any, text: str) -> list[int]:
    """The canonical NLA tokenization: add the special prefix only if it isn't already there.

    Raw text carries no BOS, so the tokenizer has to supply it. A chat-template render already
    opens with one on families whose template emits it (gemma's ``<bos>``, llama's
    ``<|begin_of_text|>``), and asking the tokenizer to add another produces an off-distribution
    ``<bos><bos>`` prefix that also shifts every later position by one. Skipping the automatic
    prefix in that case still yields the BOS the template wrote, because it is a registered added
    token and encodes to the same id. Families whose ``bos_token`` is None (Qwen) are unaffected
    either way.
    """
    bos = getattr(tokenizer, "bos_token", None)
    already_prefixed = bool(bos) and text.startswith(bos)
    return tokenizer(text, add_special_tokens=not already_prefixed)["input_ids"]


def terminal_token_ids(tokenizer: Any, model: Any = None) -> set[int]:
    """Token ids that end a generated turn, taken from the model's own config.

    ``generate()`` pushes the terminating token to the streamer before stopping
    criteria are evaluated, so it always reaches the caller. Deriving the id set
    from config (rather than a hand-written table of ``<|im_end|>``-style
    strings) is what lets a client drop the turn-end marker structurally.
    ``generation_config.eos_token_id`` is a list on Qwen/Llama, covering e.g.
    both ``<|eot_id|>`` and ``<|eom_id|>``.
    """
    candidates = [getattr(tokenizer, "eos_token_id", None)]
    gen_config = getattr(model, "generation_config", None)
    if gen_config is not None:
        candidates.append(getattr(gen_config, "eos_token_id", None))
    ids: set[int] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, int):
            ids.add(candidate)
        elif isinstance(candidate, (list, tuple, set)):
            ids.update(int(c) for c in candidate if c is not None)
    return ids


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _common_suffix_len(a: list[int], b: list[int]) -> int:
    n = 0
    la, lb = len(a), len(b)
    while n < la and n < lb and a[la - 1 - n] == b[lb - 1 - n]:
        n += 1
    return n


def has_chat_template(tokenizer: Any) -> bool:
    return getattr(tokenizer, "chat_template", None) is not None


def render_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
    **template_kwargs: Any,
) -> str:
    """Render ``messages`` to a string via the model's real chat template."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        **template_kwargs,
    )


def compute_spans(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    encode: Callable[[str], list[int]],
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
    **template_kwargs: Any,
) -> tuple[str, list[TokenSpan]]:
    """Render ``messages`` and compute per-token spans over ``encode(rendered)``.

    ``encode`` must be the exact tokenization the caller will use downstream
    (i.e. :func:`encode_with_special` bound to the same tokenizer) so the returned
    span positions line up 1:1 with the caller's token ids.

    Boundaries come from the longest common token-prefix of ``full_ids`` with the
    encoding of each growing message prefix; within a block, header/content/footer
    are separated by re-rendering the same message with EMPTY content and diffing.
    """

    def render(msgs: list[dict[str, str]], *, agp: bool, cfm: bool) -> str:
        return render_chat(
            tokenizer,
            msgs,
            add_generation_prompt=agp,
            continue_final_message=cfm,
            **template_kwargs,
        )

    rendered = render(messages, agp=add_generation_prompt, cfm=continue_final_message)
    full_ids = encode(rendered)
    n = len(messages)
    total = len(full_ids)

    prefix_end: list[int] = []
    for j in range(0, n + 1):
        if j == 0:
            prefix_end.append(0)
            continue
        ids_j = full_ids if j == n and continue_final_message else encode(render(messages[:j], agp=False, cfm=False))
        b = min(_common_prefix_len(ids_j, full_ids), total)
        if prefix_end:
            b = max(b, prefix_end[-1])
        prefix_end.append(b)

    roles: list[str | None] = [None] * total
    channels: list[str | None] = [None] * total
    msg_idx: list[int | None] = [None] * total
    sections: list[str] = ["scaffold"] * total

    for k in range(n):
        start, end = prefix_end[k], prefix_end[k + 1]
        if end <= start:
            continue
        block_len = end - start
        hdr, ftr = _header_footer_split(
            tokenizer,
            messages,
            k,
            start,
            end,
            full_ids,
            encode=encode,
            continue_final_message=continue_final_message and k == n - 1,
            **template_kwargs,
        )
        role = messages[k].get("role")
        channel = messages[k].get("channel")
        for pos in range(start, end):
            msg_idx[pos] = k
            roles[pos] = role
            channels[pos] = channel
            rel = pos - start
            if rel < hdr:
                sections[pos] = "header"
            elif rel >= block_len - ftr:
                sections[pos] = "footer"
            else:
                sections[pos] = "content"

    # Trailing generation prompt: the assistant turn opener the model continues.
    for pos in range(prefix_end[n], total):
        roles[pos] = "assistant"
        sections[pos] = "header"

    spans = [
        TokenSpan(
            position=pos,
            token_id=int(tid),
            token_str=tokenizer.decode([tid], clean_up_tokenization_spaces=False),
            message_index=msg_idx[pos],
            role=roles[pos],
            channel=channels[pos],
            section=sections[pos],
        )
        for pos, tid in enumerate(full_ids)
    ]
    return rendered, spans


def _header_footer_split(
    tokenizer: Any,
    messages: list[dict[str, str]],
    k: int,
    start: int,
    end: int,
    full_ids: list[int],
    *,
    encode: Callable[[str], list[int]],
    continue_final_message: bool,
    **template_kwargs: Any,
) -> tuple[int, int]:
    """Return ``(header_len, footer_len)`` for message ``k``'s block ``full_ids[start:end]``.

    The empty-content wrapper is rendered CLOSED even for a final message held open by
    ``continue_final_message``: transformers implements that flag by cutting the rendered string
    at a sentinel appended to the content, and templates that trim content take its fallback
    ``rstrip()`` path, which with empty content also strips the spacing the template puts between
    the header and the content. An open turn has no footer to find anyway. See the long-form
    version of this in ``interp_engine.Tokenize._header_footer_split``, which this mirrors.
    """
    block = full_ids[start:end]
    block_len = end - start
    empty_messages = [dict(m) for m in messages[: k + 1]]
    empty_messages[k] = {**empty_messages[k], "content": ""}
    try:
        ids_empty = encode(
            render_chat(
                tokenizer,
                empty_messages,
                add_generation_prompt=False,
                continue_final_message=False,
                **template_kwargs,
            )
        )
    except Exception:  # noqa: BLE001 - some templates reject empty content
        return 0, 0
    struct = ids_empty[start:]
    if not struct:
        return 0, 0
    hp = min(_common_prefix_len(struct, block), block_len)
    if continue_final_message:
        return hp, 0
    fs = _common_suffix_len(struct, block)
    fs = min(fs, block_len - hp, max(0, len(struct) - hp))
    return hp, fs
