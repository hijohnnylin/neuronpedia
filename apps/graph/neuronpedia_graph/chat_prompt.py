"""Chat-prompt rendering and parsing against a model's own chat template.

Graphs store their prompt as a single rendered string. The webapp's "Remix" flow
needs to get structured turns back out of that string so the user can edit them,
which used to mean the frontend carried a hand-written table of each family's
special tokens (`<|im_start|>`, `<start_of_turn>`, `<|start_header_id|>`, ...).

Everything here derives the delimiters from the tokenizer's real chat template by
probing it with a sentinel, so a new model family needs no code change and the
frontend needs no special-case table. Kept free of server/model globals so it can
be exercised without loading a model.
"""

from __future__ import annotations

from typing import Any

# A sentinel no chat template will alter or escape, used to locate where a
# message's content lands inside the rendered output.
SENTINEL = "\u0000NPPARSE\u0000"

# Roles worth probing. A template that rejects one (e.g. gemma has no system
# role) simply won't contribute delimiters for it.
PROBE_ROLES = ("system", "user", "assistant")


def render_prompt_from_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Render structured chat turns to a prompt string via the real chat template.

    The final turn is kept OPEN so the model continues from it rather than
    predicting the next speaker's turn — that continuation is the whole point of
    the graph.

    An empty final assistant turn means the same thing, but
    ``continue_final_message`` cannot express it: transformers implements that
    flag by cutting the render at the end of the final message's content, and an
    empty content matches at the very end of the string, so nothing is cut and
    the turn closes. A generation prompt says it in a way every template
    supports.
    """
    if len(messages) > 1 and messages[-1]["role"] == "assistant" and not messages[-1].get("content", "").strip():
        return tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=True,
            add_generation_prompt=False,
        )
    except Exception:  # noqa: BLE001 - some templates reject continue_final_message
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _probe(tokenizer: Any, roles: tuple[str, ...]) -> list[str] | None:
    """Render ``roles`` with every content replaced by the sentinel, then split.

    Returns the ``len(roles) + 1`` literal chunks surrounding the contents, or
    ``None`` if the template rejects this role sequence or does not reproduce
    each content exactly once.
    """
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": role, "content": SENTINEL} for role in roles],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:  # noqa: BLE001 - template may reject this role or ordering
        return None
    chunks = rendered.split(SENTINEL)
    return chunks if len(chunks) == len(roles) + 1 else None


def _header_after_user(chunks: list[str], user_footer: str) -> str | None:
    """The second turn's header, given chunks from a probe starting with `user`."""
    between = chunks[1]
    if not between.startswith(user_footer):
        return None
    return between[len(user_footer) :]


def learn_turn_delimiters(tokenizer: Any) -> dict[str, tuple[str, str]]:
    """Map each supported role to the (header, footer) literals its turns get."""
    delimiters: dict[str, tuple[str, str]] = {}
    lone_user = _probe(tokenizer, ("user",))
    if lone_user is None:
        return delimiters
    user_prefix, user_footer = lone_user

    for role in PROBE_ROLES:
        # A turn's header has to be one literal to be searchable, but templates
        # decorate turns by position: Qwen3 injects an empty `<think>` block into
        # a trailing assistant turn and none into earlier ones. Sandwiching the
        # probe between user turns gets the undecorated header, which leaves any
        # such decoration inside the message content, where the editor wants it.
        sandwich = _probe(tokenizer, ("user", role, "user"))
        pair = _probe(tokenizer, ("user", role))
        header = None
        if sandwich is not None:
            header = _header_after_user(sandwich, user_footer)
        if header is None and pair is not None:
            header = _header_after_user(pair, user_footer)
        if header is None and role == "user":
            # Templates that refuse two user turns in a row (gemma insists the
            # roles alternate) leave only the lone render to learn from, and its
            # prefix opens the conversation, so it carries the BOS. Prompts reach
            # the parser with their BOS already stripped, so a header holding one
            # would never match a single turn.
            header = strip_leading_bos(tokenizer, user_prefix)
        if header is None:
            continue
        # The footer is what closes the turn, which no template varies by
        # position, so the shortest render that produced this role will do.
        footer = pair[-1] if pair is not None else user_footer
        delimiters[role] = (header, footer)
    return delimiters


def strip_leading_bos(tokenizer: Any, prompt: str) -> str:
    """Drop a leading BOS token from a stored prompt string.

    Consumers re-tokenize and add their own BOS, so leaving a baked-in one would
    doubly prepend it and shift every token position by one.
    """
    bos = getattr(tokenizer, "bos_token", None)
    if bos and prompt.startswith(bos):
        return prompt[len(bos) :]
    return prompt


def bos_token_positions(tokenizer: Any, prompt: str) -> set[int]:
    """Token positions in ``prompt`` that hold a BOS token."""
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None:
        return set()
    input_ids = tokenizer(prompt).input_ids
    return {i for i, token_id in enumerate(input_ids) if token_id == bos_id}


def unsteerable_token_positions(tokenizer: Any, prompt: str) -> set[int]:
    """Positions ``/steer`` refuses to steer, given a prompt it has normalized.

    BOS only, today. It carries no content of its own and perturbing it
    destabilizes the whole generation, whereas turn markers and other special
    tokens are legitimately steerable.

    This is the one definition of that rule: ``/steer`` enforces it and
    ``/parse-chat-prompt`` reports it. A steer UI greys out what this returns
    rather than matching token strings, so the controls it hides and the
    features the server drops cannot drift apart. Both callers must pass the
    same normalized prompt — ``strip_leading_bos`` applied, chat turns already
    rendered — since these are indices into that tokenization.
    """
    return bos_token_positions(tokenizer, prompt)


def parse_chat_prompt(prompt: str, delimiters: dict[str, tuple[str, str]]) -> list[dict[str, str]] | None:
    """Recover structured chat turns from an already-rendered prompt string.

    The inverse of :func:`render_prompt_from_messages`. Returns ``None`` when the
    model has no chat template or the prompt carries no recognizable turn header,
    i.e. it is plain (non-chat) text and should be handed back to the caller as-is.

    Roles come back canonical, not as the literal label in the template, so
    gemma's ``model`` turns are reported as ``assistant``.
    """
    if not delimiters:
        return None

    matches: list[tuple[int, str, str]] = []  # (index, role, header)
    for role, (header, _footer) in delimiters.items():
        start = 0
        while True:
            idx = prompt.find(header, start)
            if idx == -1:
                break
            matches.append((idx, role, header))
            start = idx + len(header)
    if not matches:
        return None

    # Longest header wins at a shared offset, so a role whose header is a prefix
    # of another's can't claim the match.
    matches.sort(key=lambda m: (m[0], -len(m[2])))
    kept: list[tuple[int, str, str]] = []
    consumed_until = -1
    for idx, role, header in matches:
        if idx < consumed_until:
            continue
        kept.append((idx, role, header))
        consumed_until = idx + len(header)

    messages: list[dict[str, str]] = []
    for i, (idx, role, header) in enumerate(kept):
        content_start = idx + len(header)
        content_end = kept[i + 1][0] if i + 1 < len(kept) else len(prompt)
        content = prompt[content_start:content_end]
        footer = delimiters[role][1]
        if footer and content.endswith(footer):
            content = content[: -len(footer)]
        messages.append({"role": role, "content": content})
    return messages
