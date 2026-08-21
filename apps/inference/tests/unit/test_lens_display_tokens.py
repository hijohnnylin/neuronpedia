"""Multi-token characters: what the chips show, and what text they add up to.

A byte-level tokenizer splits one emoji across several tokens, and each fragment decodes on
its own to a lone replacement char, so the glyph would never appear. `_decode_display_tokens`
repairs that by decoding the run together and giving the whole glyph to EVERY position in it —
so a 5-token emoji renders as five chips of the same emoji, which is deliberate.

That repetition is lossy for anything rebuilding text: jlens stores the assistant turn to
re-send as history, and joining the display strings emitted one emoji per contributing token
(the reported bug: an emoji doubling in the previous turn on the next message). The run's
2nd..nth positions therefore carry `continuation`, and the string alone cannot substitute for
it — two adjacent split emoji are indistinguishable from one repeated across its fragments,
which the last case here pins.

Model-free: the stub decodes UTF-8 bytes with `errors="replace"`, which is what makes the real
tokenizers produce the fragments in the first place.
"""

from __future__ import annotations

from neuronpedia_inference.endpoints.lens.prompt import _decode_display_tokens

THUMBS_UP = "👍🏽"  # 8 UTF-8 bytes: emoji + skin-tone modifier
GRIN = "😀"  # 4 UTF-8 bytes


class ByteTokenizer:
    """Ids map to byte strings; decoding joins them and replaces incomplete sequences."""

    def __init__(self, pieces: list[bytes]):
        self._pieces = pieces

    def decode(self, ids: list[int], clean_up_tokenization_spaces: bool = False) -> str:  # noqa: ARG002
        return b"".join(self._pieces[i] for i in ids).decode("utf-8", errors="replace")


def _fragment(text: str, per_token: int) -> list[bytes]:
    """Split ``text`` into ``per_token``-byte pieces, as a byte-level vocab would."""
    raw = text.encode("utf-8")
    return [raw[i : i + per_token] for i in range(0, len(raw), per_token)]


def _display(pieces: list[bytes]) -> list[tuple[str, bool]]:
    tok = ByteTokenizer(pieces)
    return [(d.token, d.continuation) for d in _decode_display_tokens(tok, list(range(len(pieces))), {})]


def _text(pieces: list[bytes]) -> str:
    """Rebuild the message text the way a consumer must: skip the continuations."""
    return "".join(token for token, continuation in _display(pieces) if not continuation)


def test_whole_tokens_are_never_continuations():
    pieces = [b"Hi", b" there"]
    assert _display(pieces) == [("Hi", False), (" there", False)]


def test_split_emoji_shows_at_every_position_but_counts_once():
    pieces = [b"Hi "] + _fragment(GRIN, 1)
    display = _display(pieces)
    assert display == [("Hi ", False), (GRIN, False), (GRIN, True), (GRIN, True), (GRIN, True)]
    assert _text(pieces) == f"Hi {GRIN}"


def test_a_grapheme_of_two_codepoints_repairs_as_two_runs():
    """Repair is per codepoint, not per grapheme: `👍🏽` is a thumbs-up plus a skin-tone
    modifier, so it renders as two pairs of chips rather than four identical ones. The
    rebuilt text is still the whole grapheme."""
    base, modifier = THUMBS_UP[0], THUMBS_UP[1]
    pieces = _fragment(THUMBS_UP, 2) + [b"!"]
    assert _display(pieces) == [
        (base, False),
        (base, True),
        (modifier, False),
        (modifier, True),
        ("!", False),
    ]
    assert _text(pieces) == f"{THUMBS_UP}!"


def test_two_adjacent_split_emoji_stay_two():
    """The case the display string cannot express on its own: identical adjacent runs."""
    pieces = _fragment(GRIN, 2) + _fragment(GRIN, 2)
    assert _display(pieces) == [(GRIN, False), (GRIN, True), (GRIN, False), (GRIN, True)]
    assert _text(pieces) == GRIN + GRIN


def test_emoji_that_are_each_one_token_are_left_alone():
    pieces = [GRIN.encode(), GRIN.encode()]
    assert _display(pieces) == [(GRIN, False), (GRIN, False)]
    assert _text(pieces) == GRIN + GRIN


def test_an_unrepairable_fragment_keeps_its_replacement_char_alone():
    """A truncated sequence at the end of the stream has nothing to complete it, and a lone
    replacement char stands for itself rather than continuing a neighbour."""
    pieces = [b"Hi", GRIN.encode()[:2]]
    display = _display(pieces)
    assert display[0] == ("Hi", False)
    assert display[1][0].startswith("\ufffd") and display[1][1] is False
