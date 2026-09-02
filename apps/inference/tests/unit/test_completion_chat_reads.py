"""Requesting several readout reads at once: agreement, capture points, and shaping.

Three properties matter, and the shipped assets make each of them load-bearing rather than
speculative -- the 8B model ships six traits spread over five layers:

- **Rendering conditions have to agree.** They are applied to the prompt before generation, so
  they change the text itself. Two reads fitted under different conditions cannot both be
  measured in one generation, and serving one vector's numbers off the other's prompt would be a
  wrong answer rather than a missing one.
- **Captures are shared.** One forward per condition, however many reads ask for it. What is shared
  is the point in the model; two reads that reduce it differently still cost one forward.
- **Readouts are keyed by id, not title.** A title is a display string; these values get
  persisted.

Read specs are the fourth, and the one a bare layer number could not express: a vector reads the
site, pooling and messages its own fit used, so two reads at one layer may disagree about all three.
"""

from __future__ import annotations

import dataclasses

import torch

from neuronpedia_inference.endpoints.steer.completion_chat import (
    _agreed_render_conditions,
    _declared_points,
    _read_capture_points,
    build_readouts,
)
from neuronpedia_inference.inference_utils.vectors.vector_data import (
    CaptureKey,
    Pooling,
    ReadSpec,
    RenderConditions,
    VectorAsset,
)
from neuronpedia_inference.schemas import NPSteerChatMessage, NPSteerType

D_MODEL = 4


def _read(
    vector_id: str,
    layer: int,
    *,
    render: RenderConditions | None = None,
    read: ReadSpec | None = None,
    center: float = 0.0,
    scale_pos: float = 1.0,
    scale_neg: float = 1.0,
    caveat: str | None = None,
    quantiles: tuple[list[float], list[float]] | None = None,
) -> VectorAsset:
    """One vector in memory. ``quantiles`` is ``(pos, neg)``, and absent by default -- an asset
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
    return VectorAsset(
        id=vector_id,
        author=vector_id.split("_", 1)[0],
        title=f"title for {vector_id}",
        layer=layer,
        normalize="none",
        center=center,
        scale_pos=scale_pos,
        scale_neg=scale_neg,
        render=render or RenderConditions(),
        read=read or ReadSpec(),
        direction=unit,
        scaler_mean=torch.zeros(D_MODEL),
        pca_mean=torch.zeros(D_MODEL),
        caveat=caveat,
        quantile_levels=levels,
        quantiles_pos=table_pos,
        quantiles_neg=table_neg,
    )


def _cap(layer: int, pool: Pooling = "mean") -> CaptureKey:
    """The key a vector with default-ish read conditions captures under."""
    return CaptureKey(site="resid_post", layer=layer, pool=pool)


def _acts(*first_components: float) -> torch.Tensor:
    acts = torch.zeros(len(first_components), D_MODEL)
    for row, value in enumerate(first_components):
        acts[row][0] = value
    return acts


class TestRenderAgreement:
    def test_no_reads_changes_nothing_about_the_prompt(self):
        render, conflict = _agreed_render_conditions([])
        assert conflict is None
        assert render == RenderConditions()

    def test_reads_sharing_conditions_agree(self):
        pinned = RenderConditions(template_kwargs={"date_string": "26 Jul 2024"})
        render, conflict = _agreed_render_conditions([_read("t_a", 13, render=pinned), _read("t_b", 19, render=pinned)])
        assert conflict is None
        assert render.template_kwargs == {"date_string": "26 Jul 2024"}

    def test_disagreeing_reads_are_a_conflict_naming_both_sides(self):
        # The 70B assistant vector blanks the system prompt and the 8B traits do not, so a
        # caller combining assets across fits hits this rather than getting wrong numbers.
        blanked = _read("lu_assistant-axis", 40, render=RenderConditions(blank_system_prompt=True))
        kept = _read("mit_empathy", 19, render=RenderConditions())
        _render, conflict = _agreed_render_conditions([blanked, kept])
        assert conflict is not None
        assert "lu_assistant-axis" in conflict
        assert "mit_empathy" in conflict
        assert "blank_system_prompt" in conflict

    def test_a_differing_template_kwarg_is_a_conflict(self):
        # Same flag, different date: the same template renders a different system block, so
        # the two fits saw different distributions.
        july = _read("t_a", 19, render=RenderConditions(template_kwargs={"date_string": "26 Jul 2024"}))
        january = _read("t_b", 19, render=RenderConditions(template_kwargs={"date_string": "01 Jan 2025"}))
        _render, conflict = _agreed_render_conditions([july, january])
        assert conflict is not None

    def test_reads_pooling_differently_are_not_a_conflict(self):
        # The line the two specs draw: render conditions change the text every vector is measured
        # off, so they must agree, while a pooling changes only that vector's own reduction.
        mean = _read("t_mean", 19)
        last = _read("t_last", 19, read=ReadSpec(pool="last"))
        _render, conflict = _agreed_render_conditions([mean, last])
        assert conflict is None


class TestCapturePoints:
    def test_nothing_requested_captures_nothing(self):
        assert _read_capture_points([]) == {}

    def test_layers_are_deduplicated(self):
        # Two reads at layer 19 must not cost two captures; the 8B asset has exactly that pair.
        points = _read_capture_points([_read("mit_empathy", 19), _read("mit_erudite", 19), _read("mit_toxic", 13)])
        assert sorted(key.layer for key in points) == [13, 19]

    def test_each_point_reads_the_residual_stream_at_its_layer(self):
        points = _read_capture_points([_read("t_a", 29)])
        assert str(points[_cap(29)]) == "resid_post.29"

    def test_two_poolings_at_one_layer_are_two_captures(self):
        # A layer used to be the whole key, so the second of these read the first's mean under
        # its own name -- a plausible number from the wrong reduction.
        points = _read_capture_points([_read("t_mean", 19), _read("t_last", 19, read=ReadSpec(pool="last"))])
        assert sorted(key.pool for key in points) == ["last", "mean"]

    def test_two_poolings_at_one_layer_declare_one_address(self):
        # And the point of keying by the reduction rather than capturing twice: the generation is
        # asked for that layer once, and the one tensor is pooled two ways.
        points = _read_capture_points([_read("t_mean", 19), _read("t_last", 19, read=ReadSpec(pool="last"))])
        assert [str(point) for point in _declared_points(points)] == ["resid_post.19"]


class TestBuildReadouts:
    @staticmethod
    def _conversation() -> list[NPSteerChatMessage]:
        return [
            NPSteerChatMessage(role="system", content="be helpful"),
            NPSteerChatMessage(role="user", content="hi"),
            NPSteerChatMessage(role="assistant", content="hello there"),
        ]

    def test_one_readout_per_vector_keyed_by_id(self):
        reads = [_read("mit_empathy", 19), _read("mit_toxic", 13)]
        readouts = build_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            reads,
            {_cap(19): _acts(0.0, 0.0, 0.5), _cap(13): _acts(0.0, 0.0, -0.5)},
            None,
        )
        assert [readout.id for readout in readouts] == ["mit_empathy", "mit_toxic"]
        assert [readout.title for readout in readouts] == ["title for mit_empathy", "title for mit_toxic"]
        assert [readout.layer for readout in readouts] == [19, 13]

    def test_only_assistant_turns_are_reported(self):
        # Located by role, not position: the system message would otherwise shift every value
        # by one turn.
        readouts = build_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            [_read("t_a", 19)],
            {_cap(19): _acts(7.0, 8.0, 0.25)},
            None,
        )
        turns = readouts[0].turns
        assert turns is not None
        assert len(turns) == 1
        assert turns[0].value == 0.25
        assert turns[0].snippet == "hello there"

    def test_a_vector_reading_all_turns_reports_every_message(self):
        # What a probe fitted on the whole conversation needs, and what the assistant-turn
        # selection above deliberately drops.
        readouts = build_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            [_read("t_a", 19, read=ReadSpec(tokens="all_turns"))],
            {_cap(19): _acts(1.0, 2.0, 3.0)},
            None,
        )
        turns = readouts[0].turns
        assert turns is not None
        assert [turn.value for turn in turns] == [1.0, 2.0, 3.0]
        assert [turn.snippet for turn in turns] == ["be helpful", "hi", "hello there"]

    def test_selection_is_per_vector(self):
        # Two reads in one readout, disagreeing about what a turn is. Nothing about the request
        # is shared here, which is why the selection is read off the vector rather than the request.
        readouts = build_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            [_read("t_assistant", 19), _read("t_all", 19, read=ReadSpec(tokens="all_turns"))],
            {_cap(19): _acts(1.0, 2.0, 3.0)},
            None,
        )
        by_id = {readout.id: readout for readout in readouts}
        assert [turn.value for turn in by_id["t_assistant"].turns or []] == [3.0]
        assert [turn.value for turn in by_id["t_all"].turns or []] == [1.0, 2.0, 3.0]

    def test_each_vector_reads_the_pooling_it_was_fitted_with(self):
        # The two captures of one layer, kept apart all the way to the readout.
        reads = [_read("t_mean", 19), _read("t_last", 19, read=ReadSpec(pool="last"))]
        readouts = build_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            reads,
            {_cap(19): _acts(0.0, 0.0, 0.25), _cap(19, "last"): _acts(0.0, 0.0, 0.75)},
            None,
        )
        by_id = {readout.id: readout for readout in readouts}
        assert by_id["t_mean"].turns[0].value == 0.25  # type: ignore[index]
        assert by_id["t_last"].turns[0].value == 0.75  # type: ignore[index]

    def test_post_cap_values_are_reported_when_captured(self):
        readouts = build_readouts(
            self._conversation(),
            NPSteerType.STEERED,
            [_read("t_a", 19)],
            {_cap(19): _acts(0.0, 0.0, 0.25)},
            {_cap(19): _acts(0.0, 0.0, 0.75)},
        )
        turns = readouts[0].turns
        assert turns is not None
        assert turns[0].value == 0.25
        assert turns[0].value_post_cap == 0.75

    def test_post_cap_is_absent_without_a_steered_capture(self):
        readouts = build_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            [_read("t_a", 19)],
            {_cap(19): _acts(0.0, 0.0, 0.25)},
            None,
        )
        turns = readouts[0].turns
        assert turns is not None
        assert turns[0].value_post_cap is None

    def test_each_vector_reads_its_own_layer(self):
        # The single-layer format could not express this, and getting it wrong would report
        # one trait's numbers under another's name.
        reads = [_read("t_shallow", 13), _read("t_deep", 29)]
        readouts = build_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            reads,
            {_cap(13): _acts(0.0, 0.0, -1.0), _cap(29): _acts(0.0, 0.0, 1.0)},
            None,
        )
        by_id = {readout.id: readout for readout in readouts}
        assert by_id["t_shallow"].turns[0].value == -1.0  # type: ignore[index]
        assert by_id["t_deep"].turns[0].value == 1.0  # type: ignore[index]

    def test_calibration_is_applied_per_vector(self):
        reads = [_read("t_raw", 19), _read("t_calibrated", 19, center=0.5, scale_pos=0.25)]
        readouts = build_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            reads,
            {_cap(19): _acts(0.0, 0.0, 1.0)},
            None,
        )
        by_id = {readout.id: readout for readout in readouts}
        assert by_id["t_raw"].turns[0].value == 1.0  # type: ignore[index]
        assert by_id["t_calibrated"].turns[0].value == 2.0  # type: ignore[index]

    def test_a_vector_whose_layer_failed_to_capture_is_dropped(self):
        # Reported empty it would draw as a flat line at zero, which reads as a measurement.
        readouts = build_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            [_read("t_present", 19), _read("t_absent", 29)],
            {_cap(19): _acts(0.0, 0.0, 0.25)},
            None,
        )
        assert [readout.id for readout in readouts] == ["t_present"]

    def test_a_vector_whose_pooling_failed_to_capture_is_dropped(self):
        # The same rule one level finer. Falling back to the mean that is sitting right there
        # would report a number under a spec that did not produce it.
        readouts = build_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            [_read("t_mean", 19), _read("t_last", 19, read=ReadSpec(pool="last"))],
            {_cap(19): _acts(0.0, 0.0, 0.25)},
            None,
        )
        assert [readout.id for readout in readouts] == ["t_mean"]

    def test_a_caveat_travels_with_the_readout(self):
        readouts = build_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            [_read("mit_sycophantic", 15, caveat="Turn bias 0.40")],
            {_cap(15): _acts(0.0, 0.0, 0.25)},
            None,
        )
        assert readouts[0].caveat == "Turn bias 0.40"

    def test_what_the_poles_mean_travels_with_the_readout(self):
        # The names are what a chart labels its two ends with, and the descriptions are what it
        # can show when someone asks what "sycophantic" was taken to mean. Neither is recoverable
        # from the values, so a readout that dropped them could only be drawn against a hardcoded
        # table of the reads someone happened to know about.
        vector = _read("mit_toxic", 13)
        vector = dataclasses.replace(
            vector,
            pole_positive="toxic",
            pole_negative="respectful",
            pole_positive_description="insulting or demeaning",
            pole_negative_description="considerate",
            source_revision="a" * 40,
        )
        readout = build_readouts(
            self._conversation(), NPSteerType.DEFAULT, [vector], {_cap(13): _acts(0.0, 0.0, 1.0)}, None
        )[0]
        assert (readout.pole_positive, readout.pole_negative) == ("toxic", "respectful")
        assert readout.pole_positive_description == "insulting or demeaning"
        assert readout.pole_negative_description == "considerate"
        # Which commit a vector fetched for this request was read at: what makes the numbers
        # reproducible once the branch it came from has moved on.
        assert readout.source_revision == "a" * 40

    def test_a_vector_that_names_no_poles_reports_none(self):
        readout = build_readouts(
            self._conversation(), NPSteerType.DEFAULT, [_read("t_a", 19)], {_cap(19): _acts(0.0, 0.0, 1.0)}, None
        )[0]
        assert readout.pole_positive is None
        assert readout.source_revision is None

    def test_a_percentile_is_reported_beside_the_value(self):
        # Both readings of one turn: the ratio says how many spreads from centre, the percentile
        # says how much of the corpus it is past. A display shows the second and stores the first.
        readouts = build_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            [_read("t_a", 19, scale_pos=0.5, quantiles=([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]))],
            {_cap(19): _acts(0.0, 0.0, 1.0)},
            None,
        )
        turns = readouts[0].turns
        assert turns is not None
        assert turns[0].value == 2.0
        assert turns[0].percentile == 0.5

    def test_a_percentile_past_the_corpus_stops_at_full_scale_while_the_value_does_not(self):
        # The point of carrying both, in one turn: the ratio is free to say 4.0 and the
        # percentile refuses to say more than 1.0.
        readouts = build_readouts(
            self._conversation(),
            NPSteerType.DEFAULT,
            [_read("t_a", 19, scale_pos=0.5, quantiles=([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]))],
            {_cap(19): _acts(0.0, 0.0, 2.0)},
            None,
        )
        turns = readouts[0].turns
        assert turns is not None
        assert turns[0].value == 4.0
        assert turns[0].percentile == 1.0

    def test_the_steered_measurement_gets_its_own_percentile(self):
        readouts = build_readouts(
            self._conversation(),
            NPSteerType.STEERED,
            [_read("t_a", 19, quantiles=([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]))],
            {_cap(19): _acts(0.0, 0.0, 1.0)},
            {_cap(19): _acts(0.0, 0.0, 2.0)},
        )
        turns = readouts[0].turns
        assert turns is not None
        assert (turns[0].percentile, turns[0].percentile_post_cap) == (0.5, 1.0)

    def test_a_vector_without_tables_reports_no_percentile(self):
        # `lu_assistant-axis` is this case. Absent rather than 0, which would draw as centre.
        readouts = build_readouts(
            self._conversation(),
            NPSteerType.STEERED,
            [_read("t_a", 19)],
            {_cap(19): _acts(0.0, 0.0, 0.25)},
            {_cap(19): _acts(0.0, 0.0, 0.75)},
        )
        turns = readouts[0].turns
        assert turns is not None
        assert turns[0].value == 0.25
        assert turns[0].percentile is None
        assert turns[0].percentile_post_cap is None

    def test_a_conversation_with_no_assistant_turn_yields_no_turns(self):
        readouts = build_readouts(
            [NPSteerChatMessage(role="user", content="hi")],
            NPSteerType.DEFAULT,
            [_read("t_a", 19)],
            {_cap(19): _acts(0.0)},
            None,
        )
        assert readouts[0].turns == []
