"""Locks the key set of every streamed lens NDJSON frame.

The lens frames are the one wire surface FastAPI cannot police for us: they are emitted
one JSON object per line rather than returned as a response body, so they never reach
`openapi.json` and nothing regenerates a type from them.

They are also the one wire surface the webapp does not rename. The frames go verbatim into
`/api/lens/prompt`'s response and into the stored share blobs, so these field names are the
public contract, and `PublicFrameSchema` keeps them un-aliased for exactly that reason.
Renaming a field here changes what existing viewers and existing stored shares read, so it
fails this test first.
"""

from neuronpedia_inference.endpoints.lens.prompt import (
    LensDoneMessage,
    LensErrorMessage,
    LensMetaMessage,
    LensPromptToken,
    LensPromptTokensMessage,
    LensTokenMessage,
    LensTypeSlice,
)
from neuronpedia_inference.schemas import BaseSchema, PublicFrameSchema

EXPECTED_KEYS: dict[type, set[str]] = {
    LensTypeSlice: {"type", "top_tokens", "top_probs"},
    LensMetaMessage: {
        "kind",
        "model",
        "types",
        "layers_by_type",
        "top_n",
        "prompt_len",
        "num_completion_tokens",
        "temperature",
        "prepend_bos",
        "reuse_len",
    },
    LensPromptToken: {
        "position",
        "token",
        "id",
        "is_generated",
        "is_char_continuation",
        "message_index",
        "role",
        "channel",
        "section",
    },
    LensPromptTokensMessage: {"kind", "tokens"},
    LensTokenMessage: {
        "kind",
        "position",
        "token",
        "id",
        "is_generated",
        "is_char_continuation",
        "results",
        "message_index",
        "role",
        "channel",
        "section",
    },
    LensDoneMessage: {"kind", "seq_len", "prompt_len", "vocab_size", "completion"},
    LensErrorMessage: {"kind", "error"},
}


def test_every_frame_model_is_pinned():
    """A new frame model must be added here rather than shipping unchecked.

    Discovery is by `PublicFrameSchema`, so a frame declared on plain `BaseSchema` would be
    camelCased *and* invisible here. The second assertion closes that: nothing in this module
    may be an un-aliased-by-omission `BaseSchema`.
    """
    from neuronpedia_inference.endpoints.lens import prompt

    declared = {
        obj
        for obj in vars(prompt).values()
        if isinstance(obj, type) and obj.__module__ == prompt.__name__ and issubclass(obj, BaseSchema)
    }
    assert declared == set(EXPECTED_KEYS), "lens frame models changed; update EXPECTED_KEYS"
    assert all(issubclass(model, PublicFrameSchema) for model in declared), (
        "a lens frame is on BaseSchema, so it will be camelCased and break existing readers"
    )


def test_frame_keys_are_exact_and_not_camel_cased():
    for model, expected in EXPECTED_KEYS.items():
        actual = set(model.model_json_schema()["properties"])
        assert actual == expected, f"{model.__name__} wire keys drifted"


def test_serialized_frames_use_the_pinned_keys():
    """Schema and serializer can disagree; only a real dump proves the alias override took."""
    done = LensDoneMessage(seq_len=3, prompt_len=2, vocab_size=50257, completion="hi")
    assert set(done.model_dump()) == EXPECTED_KEYS[LensDoneMessage]

    token = LensTokenMessage(position=0, token="a", id=1, is_generated=False, results=[])
    assert set(token.model_dump()) == EXPECTED_KEYS[LensTokenMessage]

    assert LensErrorMessage(error="boom").model_dump() == {"kind": "error", "error": "boom"}


def test_the_alias_generator_is_actually_off():
    """`PublicFrameSchema` overrides one inherited config key; nothing else may reintroduce it."""
    assert PublicFrameSchema.model_config.get("alias_generator") is None
    assert BaseSchema.model_config.get("alias_generator") is not None, "BaseSchema must still alias"
    assert LensDoneMessage.model_fields["seq_len"].alias is None
