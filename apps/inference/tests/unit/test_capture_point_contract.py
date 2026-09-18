"""Pins `NPCapturePoint` to the engine's point table.

A read spec names where in a layer an activation is captured, and the value goes straight into an
`interp_engine.Address`. So the wire enum is not a vocabulary of its own: every member must be an
engine point, spelled as the engine spells it, that is layer-scoped and `d_model` wide -- the only
points a direction of the model's hidden size can be projected onto.

The webapp cannot import the engine. It gets this enum through `openapi.json` and the generated
`inference.d.ts`, so this test is the one place the two vocabularies are held together.
"""

from interp_engine.points import POINTS, Scope, Width

from neuronpedia_inference.schemas import NPCapturePoint


def _engine_readable_points() -> set[str]:
    return {p.name for p in POINTS if p.scope is Scope.LAYER and p.width is Width.D_MODEL}


def test_every_capture_point_is_an_engine_point():
    unknown = {member.value for member in NPCapturePoint} - _engine_readable_points()
    assert not unknown, f"NPCapturePoint names points the engine does not declare as layer-scoped d_model: {unknown}"


def test_capture_point_values_match_member_names():
    # `resid_post` on the wire, `RESID_POST` in Python: the value is the engine's spelling and the
    # member is its constant-case, so a member cannot quietly alias one point under another name.
    for member in NPCapturePoint:
        assert member.name == member.value.upper()
