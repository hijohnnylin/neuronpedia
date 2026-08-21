"""Pins the SSE frames, which no spec can reach.

``/completion``, ``/describe`` and ``/explain`` emit a stream of frames rather than a response
body, so they contribute nothing to ``openapi.json`` and the webapp has no generated type for
them. Four files parse them by hand -- ``nla-provider.tsx``, ``nla-input-chat.tsx``,
``explanations-pane.tsx`` and ``app/api/nla/explain/route.ts`` -- reading fields like
``layer_index``, ``prompt_length``, ``token_id`` and ``message_index`` directly.

Renaming one of those fields would compile everywhere and break in the browser, and for
``/explain`` it would also change what gets written into ``NlaExplainCache.resultJson``, which
backs permanent public share URLs. This test is the only thing standing between a rename and
that outcome, so it pins key sets rather than merely checking the models parse.

Deliberately self-contained: it reads no webapp files. A python test that greps TypeScript
couples two apps that should only agree on the wire, and breaks whenever the frontend is
refactored.
"""

import inspect

import pytest

import server
from server import (
    CompletionDoneFrame,
    CompletionPromptFrame,
    CompletionTokenFrame,
    DescribeProgressFrame,
    DescriptionResult,
    ExplainMetaFrame,
    ExplainProgressFrame,
    ExplainResult,
    NlaFrameSchema,
)

# The exact keys each frame puts on the wire. Update only alongside the consumers listed in the
# module docstring -- and, for ExplainResult, only alongside a migration for rows already in
# NlaExplainCache.resultJson.
EXPECTED_KEYS = {
    CompletionPromptFrame: {"type", "prompt_length", "tokens"},
    CompletionTokenFrame: {"type", "token"},
    CompletionDoneFrame: {"type", "text"},
    DescribeProgressFrame: {"index", "text", "done"},
    ExplainMetaFrame: {"layer_index", "total", "prompt_length"},
    ExplainProgressFrame: {"position", "text", "done"},
}


@pytest.mark.parametrize("model", EXPECTED_KEYS, ids=lambda m: m.__name__)
def test_frame_keys_are_pinned(model):
    assert set(model.model_fields) == EXPECTED_KEYS[model]


def test_every_frame_model_is_declared_here():
    """A new frame type must be added to EXPECTED_KEYS, not just defined.

    Discovery is by base class, so declaring one on NlaSchema instead of NlaFrameSchema would
    hide it from this test -- which is exactly the mistake worth catching.
    """
    declared = {
        obj
        for _, obj in inspect.getmembers(server, inspect.isclass)
        if issubclass(obj, NlaFrameSchema) and obj is not NlaFrameSchema
    }
    assert declared == set(EXPECTED_KEYS), (
        f"undeclared frame models: {sorted(m.__name__ for m in declared - set(EXPECTED_KEYS))}, "
        f"stale entries: {sorted(m.__name__ for m in set(EXPECTED_KEYS) - declared)}"
    )


def test_progress_frames_carry_no_description():
    """Both stream loops tell a partial frame from a final one by looking for ``description``.

    Adding that key to a progress frame would make the server's own completion counter finish
    early and truncate the stream, and would make the browser render a partial generation as a
    finished explanation.
    """
    assert "description" not in DescribeProgressFrame.model_fields
    assert "description" not in ExplainProgressFrame.model_fields
    assert "description" in DescriptionResult.model_fields
    assert "description" in ExplainResult.model_fields


def test_the_persisted_explain_keys_are_pinned():
    """``ExplainResult`` is a response model, but it is also written to the database verbatim.

    ``app/api/nla/explain/route.ts`` stores these records into ``NlaExplainCache.resultJson``
    without mapping any names, and serves them back to permanent ``/nla/[shareId]`` URLs. A
    rename here strands every existing row.
    """
    assert set(ExplainResult.model_fields) == {
        "token",
        "token_id",
        "position",
        "l2_norm",
        "description",
        "mse",
        "cosine_similarity",
        "generated",
        "fragment_index",
        "fragment_count",
        "role",
        "section",
        "channel",
        "message_index",
    }
