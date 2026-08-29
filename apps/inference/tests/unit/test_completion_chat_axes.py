"""Requesting several readout axes at once: agreement, capture points, and shaping.

Three properties matter, and the shipped assets make each of them load-bearing rather than
speculative -- the 8B model ships six traits spread over five layers:

- **Rendering conditions have to agree.** They are applied to the prompt before generation, so
  they change the text itself. Two axes fitted under different conditions cannot both be
  measured in one generation, and serving one axis's numbers off the other's prompt would be a
  wrong answer rather than a missing one.
- **Layers are captured together.** One forward per condition, however many axes ask for it.
- **Readouts are keyed by id, not title.** A title is a display string; these values get
  persisted.
"""

from __future__ import annotations

import dataclasses

import torch

from neuronpedia_inference.endpoints.steer.completion_chat import (
    _agreed_render_conditions,
    _axis_capture_points,
    build_axis_readouts,
)
from neuronpedia_inference.inference_utils.persona.axis_data import AxisAsset, RenderConditions
from neuronpedia_inference.schemas import NPSteerChatMessage, NPSteerType

D_MODEL = 4


def _axis(
    axis_id: str,
    layer: int,
    *,
    render: RenderConditions | None = None,
    center: float = 0.0,
    scale_pos: float = 1.0,
    scale_neg: float = 1.0,
    caveat: str | None = None,
    quantiles: tuple[list[float], list[float]] | None = None,
) -> AxisAsset:
    """One axis in memory. ``quantiles`` is ``(pos, neg)``, and absent by default -- an asset
    fitted before the tables reports no percentile, which is the case to keep working."""
    unit = torch.zeros(D_MODEL)
    unit[0] = 1.0
    levels: torch.Tensor | None = None
    table_pos: torch.Tensor | None = None
    table_neg: torch.Tensor | None = None
    if quantiles is not None:
        pos, neg = quantiles
        levels = torch.linspace(0.0, 1.0, len(pos))
        table_pos = torch.tensor(pos)
        table_neg = torch.tensor(neg)
    return AxisAsset(
        id=axis_id,
        author=axis_id.split("_", 1)[0],
        title=f"title for {axis_id}",
        layer=layer,
        normalize="none",
        center=center,
        scale_pos=scale_pos,
        scale_neg=scale_neg,
        render=render or RenderConditions(),
        direction=unit,
        scaler_mean=torch.zeros(D_MODEL),
        pca_mean=torch.zeros(D_MODEL),
        caveat=caveat,
        quantile_levels=levels,
        quantiles_pos=table_pos,
        quantiles_neg=table_neg,
    )


def _acts(*first_components: float) -> torch.Tensor:
    acts = torch.zeros(len(first_components), D_MODEL)
    for row, value in enumerate(first_components):
        acts[row][0] = value
    return acts


class TestRenderAgreement:
    def test_no_axes_changes_nothing_about_the_prompt(self):
        render, conflict = _agreed_render_conditions([])
        assert conflict is None
        assert render == RenderConditions()

    def test_axes_sharing_conditions_agree(self):
        pinned = RenderConditions(template_kwargs={"date_string": "26 Jul 2024"})
        render, conflict = _agreed_render_conditions([_axis("t_a", 13, render=pinned), _axis("t_b", 19, render=pinned)])
        assert conflict is None
        assert render.template_kwargs == {"date_string": "26 Jul 2024"}

    def test_disagreeing_axes_are_a_conflict_naming_both_sides(self):
        # The 70B assistant axis blanks the system prompt and the 8B traits do not, so a
        # caller combining assets across fits hits this rather than getting wrong numbers.
        blanked = _axis("lu_assistant-axis", 40, render=RenderConditions(blank_system_prompt=True))
        kept = _axis("mit_empathy", 19, render=RenderConditions())
        _render, conflict = _agreed_render_conditions([blanked, kept])
        assert conflict is not None
        assert "lu_assistant-axis" in conflict
        assert "mit_empathy" in conflict
        assert "blank_system_prompt" in conflict

    def test_a_differing_template_kwarg_is_a_conflict(self):
        # Same flag, different date: the same template renders a different system block, so
        # the two fits saw different distributions.
        july = _axis("t_a", 19, render=RenderConditions(template_kwargs={"date_string": "26 Jul 2024"}))
        january = _axis("t_b", 19, render=RenderConditions(template_kwargs={"date_string": "01 Jan 2025"}))
        _render, conflict = _agreed_render_conditions([july, january])
        assert conflict is not None


class TestCapturePoints:
    def test_nothing_requested_captures_nothing(self):
        assert _axis_capture_points([]) == {}

    def test_layers_are_deduplicated(self):
        # Two axes at layer 19 must not cost two captures; the 8B asset has exactly that pair.
        points = _axis_capture_points([_axis("mit_empathy", 19), _axis("mit_erudite", 19), _axis("mit_toxic", 13)])
        assert sorted(points) == [13, 19]

    def test_each_point_reads_the_residual_stream_at_its_layer(self):
        points = _axis_capture_points([_axis("t_a", 29)])
        assert str(points[29]) == "resid_post.29"


class TestBuildReadouts:
    @staticmethod
    def _conversation() -> list[NPSteerChatMessage]:
        return [
            NPSteerChatMessage(role="system", content="be helpful"),
            NPSteerChatMessage(role="user", content="hi"),
            NPSteerChatMessage(role="assistant", content="hello there"),
        ]

    def test_one_readout_per_axis_keyed_by_id(self):
        axes = [_axis("mit_empathy", 19), _axis("mit_toxic", 13)]
        readouts = build_axis_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            axes,
            {19: _acts(0.0, 0.0, 0.5), 13: _acts(0.0, 0.0, -0.5)},
            None,
        )
        assert [readout.id for readout in readouts] == ["mit_empathy", "mit_toxic"]
        assert [readout.title for readout in readouts] == ["title for mit_empathy", "title for mit_toxic"]
        assert [readout.layer for readout in readouts] == [19, 13]

    def test_only_assistant_turns_are_reported(self):
        # Located by role, not position: the system message would otherwise shift every value
        # by one turn.
        readouts = build_axis_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            [_axis("t_a", 19)],
            {19: _acts(7.0, 8.0, 0.25)},
            None,
        )
        turns = readouts[0].turns
        assert turns is not None
        assert len(turns) == 1
        assert turns[0].value == 0.25
        assert turns[0].snippet == "hello there"

    def test_post_cap_values_are_reported_when_captured(self):
        readouts = build_axis_readouts(
            self._conversation(),
            NPSteerType.STEERED,
            [_axis("t_a", 19)],
            {19: _acts(0.0, 0.0, 0.25)},
            {19: _acts(0.0, 0.0, 0.75)},
        )
        turns = readouts[0].turns
        assert turns is not None
        assert turns[0].value == 0.25
        assert turns[0].value_post_cap == 0.75

    def test_post_cap_is_absent_without_a_steered_capture(self):
        readouts = build_axis_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            [_axis("t_a", 19)],
            {19: _acts(0.0, 0.0, 0.25)},
            None,
        )
        turns = readouts[0].turns
        assert turns is not None
        assert turns[0].value_post_cap is None

    def test_each_axis_reads_its_own_layer(self):
        # The single-layer format could not express this, and getting it wrong would report
        # one trait's numbers under another's name.
        axes = [_axis("t_shallow", 13), _axis("t_deep", 29)]
        readouts = build_axis_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            axes,
            {13: _acts(0.0, 0.0, -1.0), 29: _acts(0.0, 0.0, 1.0)},
            None,
        )
        by_id = {readout.id: readout for readout in readouts}
        assert by_id["t_shallow"].turns[0].value == -1.0  # type: ignore[index]
        assert by_id["t_deep"].turns[0].value == 1.0  # type: ignore[index]

    def test_calibration_is_applied_per_axis(self):
        axes = [_axis("t_raw", 19), _axis("t_calibrated", 19, center=0.5, scale_pos=0.25)]
        readouts = build_axis_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            axes,
            {19: _acts(0.0, 0.0, 1.0)},
            None,
        )
        by_id = {readout.id: readout for readout in readouts}
        assert by_id["t_raw"].turns[0].value == 1.0  # type: ignore[index]
        assert by_id["t_calibrated"].turns[0].value == 2.0  # type: ignore[index]

    def test_an_axis_whose_layer_failed_to_capture_is_dropped(self):
        # Reported empty it would draw as a flat line at zero, which reads as a measurement.
        readouts = build_axis_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            [_axis("t_present", 19), _axis("t_absent", 29)],
            {19: _acts(0.0, 0.0, 0.25)},
            None,
        )
        assert [readout.id for readout in readouts] == ["t_present"]

    def test_a_caveat_travels_with_the_readout(self):
        readouts = build_axis_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            [_axis("mit_sycophantic", 15, caveat="Turn bias 0.40")],
            {15: _acts(0.0, 0.0, 0.25)},
            None,
        )
        assert readouts[0].caveat == "Turn bias 0.40"

    def test_what_the_poles_mean_travels_with_the_readout(self):
        # The names are what a chart labels its two ends with, and the descriptions are what it
        # can show when someone asks what "sycophantic" was taken to mean. Neither is recoverable
        # from the values, so a readout that dropped them could only be drawn against a hardcoded
        # table of the axes someone happened to know about.
        axis = _axis("mit_toxic", 13)
        axis = dataclasses.replace(
            axis,
            pole_positive="toxic",
            pole_negative="respectful",
            pole_positive_description="insulting or demeaning",
            pole_negative_description="considerate",
            source_revision="a" * 40,
        )
        readout = build_axis_readouts(
            self._conversation(), NPSteerType.DEFAULT, [axis], {13: _acts(0.0, 0.0, 1.0)}, None
        )[0]
        assert (readout.pole_positive, readout.pole_negative) == ("toxic", "respectful")
        assert readout.pole_positive_description == "insulting or demeaning"
        assert readout.pole_negative_description == "considerate"
        # Which commit an axis fetched for this request was read at: what makes the numbers
        # reproducible once the branch it came from has moved on.
        assert readout.source_revision == "a" * 40

    def test_an_axis_that_names_no_poles_reports_none(self):
        readout = build_axis_readouts(
            self._conversation(), NPSteerType.DEFAULT, [_axis("t_a", 19)], {19: _acts(0.0, 0.0, 1.0)}, None
        )[0]
        assert readout.pole_positive is None
        assert readout.source_revision is None

    def test_a_percentile_is_reported_beside_the_value(self):
        # Both readings of one turn: the ratio says how many spreads from centre, the percentile
        # says how much of the corpus it is past. A display shows the second and stores the first.
        readouts = build_axis_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            [_axis("t_a", 19, scale_pos=0.5, quantiles=([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]))],
            {19: _acts(0.0, 0.0, 1.0)},
            None,
        )
        turns = readouts[0].turns
        assert turns is not None
        assert turns[0].value == 2.0
        assert turns[0].percentile == 0.5

    def test_a_percentile_past_the_corpus_stops_at_full_scale_while_the_value_does_not(self):
        # The point of carrying both, in one turn: the ratio is free to say 4.0 and the
        # percentile refuses to say more than 1.0.
        readouts = build_axis_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            [_axis("t_a", 19, scale_pos=0.5, quantiles=([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]))],
            {19: _acts(0.0, 0.0, 2.0)},
            None,
        )
        turns = readouts[0].turns
        assert turns is not None
        assert turns[0].value == 4.0
        assert turns[0].percentile == 1.0

    def test_the_steered_measurement_gets_its_own_percentile(self):
        readouts = build_axis_readouts(
            self._conversation(),
            NPSteerType.STEERED,
            [_axis("t_a", 19, quantiles=([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]))],
            {19: _acts(0.0, 0.0, 1.0)},
            {19: _acts(0.0, 0.0, 2.0)},
        )
        turns = readouts[0].turns
        assert turns is not None
        assert (turns[0].percentile, turns[0].percentile_post_cap) == (0.5, 1.0)

    def test_an_axis_without_tables_reports_no_percentile(self):
        # `lu_assistant-axis` is this case. Absent rather than 0, which would draw as centre.
        readouts = build_axis_readouts(
            self._conversation(),
            NPSteerType.STEERED,
            [_axis("t_a", 19)],
            {19: _acts(0.0, 0.0, 0.25)},
            {19: _acts(0.0, 0.0, 0.75)},
        )
        turns = readouts[0].turns
        assert turns is not None
        assert turns[0].value == 0.25
        assert turns[0].percentile is None
        assert turns[0].percentile_post_cap is None

    def test_a_conversation_with_no_assistant_turn_yields_no_turns(self):
        readouts = build_axis_readouts(
            [NPSteerChatMessage(role="user", content="hi")],
            NPSteerType.DEFAULT,
            [_axis("t_a", 19)],
            {19: _acts(0.0)},
            None,
        )
        assert readouts[0].turns == []
