"""Readout axis assets: everything specific to a fit comes out of the asset directory.

The layer, the label, the scaling and the rendering conditions are all properties of how a
given direction was fitted, so they live in that axis's ``axis.yaml`` and never in the code.
These tests pin that, the projection arithmetic, and what a malformed manifest does.

The server ships no assets of its own -- an axis arrives with the request -- so a directory here
is only the clearest way to state a manifest, not a thing the server would ever discover.
"""

import numpy as np
import pytest
import torch
import yaml
from safetensors.torch import save_file

from neuronpedia_inference.inference_utils.persona.axis_data import (
    MANIFEST_FILENAME,
    TENSORS_FILENAME,
    AxisAsset,
    RenderConditions,
    calibrate,
    load_axis,
    project_axis,
    project_axis_with_percentile,
)

# The fitted titles carry a variation selector. Spelled with an escape because a combining
# mark in a .py file breaks CodeQL's extractor for the whole file (see AGENTS.md).
ASSISTANT_AXIS_TITLE = "- Role-playing \u2194\ufe0f + Assistant-like"
D_MODEL = 4


def write_axis(
    root,
    model_id: str,
    axis_id: str,
    *,
    direction: list[float] | None = None,
    scaler_mean: list[float] | None = None,
    pca_mean: list[float] | None = None,
    tensors: bool = True,
    quantiles_pos: list[float] | None = None,
    quantiles_neg: list[float] | None = None,
    quantile_levels: list[float] | None = None,
    **manifest_overrides,
) -> None:
    """Lay out one axis directory the way ``initialize`` expects to find it.

    The author defaults to the id's own prefix, so a fixture follows the ``<author>_<name>``
    convention by construction; pass ``author=`` to write an asset that breaks it.

    The quantile tables are written only when asked for, since an asset without them is the
    case every consumer still has to handle. Passing one of the three writes just that one,
    which is how the incomplete-set rejection is exercised.
    """
    assert "_" in axis_id or "author" in manifest_overrides, (
        f"{axis_id!r} has no author prefix, so this fixture could not load; "
        "name it <author>_<name> or pass author= explicitly"
    )
    axis_dir = root / model_id / axis_id
    axis_dir.mkdir(parents=True, exist_ok=True)

    if tensors:
        payload = {
            "direction": torch.tensor(direction if direction is not None else [1.0] + [0.0] * (D_MODEL - 1)),
            "scaler_mean": torch.tensor(scaler_mean if scaler_mean is not None else [0.0] * D_MODEL),
            "pca_mean": torch.tensor(pca_mean if pca_mean is not None else [0.0] * D_MODEL),
        }
        given = {
            "quantile_levels": quantile_levels,
            "quantiles_pos": quantiles_pos,
            "quantiles_neg": quantiles_neg,
        }
        if any(table is not None for table in given.values()):
            # Levels default to an even grid across however many entries the tables have, so a
            # test states the table it cares about and nothing else.
            width = len(quantiles_pos or quantiles_neg or quantile_levels or [])
            if given["quantile_levels"] is None and width:
                given["quantile_levels"] = list(np.linspace(0.0, 1.0, width))
            payload.update(
                {name: torch.tensor(table, dtype=torch.float32) for name, table in given.items() if table is not None}
            )
        save_file(payload, str(axis_dir / TENSORS_FILENAME))

    manifest = {
        "author": axis_id.split("_", 1)[0],
        "title": f"title for {axis_id}",
        "layer": 7,
        **manifest_overrides,
    }
    (axis_dir / MANIFEST_FILENAME).write_text(yaml.safe_dump(manifest, allow_unicode=True), encoding="utf-8")


@pytest.fixture
def asset_root(tmp_path):
    return tmp_path


def loaded(root, model: str, axis_id: str) -> AxisAsset:
    """Read back an axis written by :func:`write_axis`.

    A directory is still the clearest way for a test to state a manifest, even though the server
    no longer discovers any: what reaches it now is one artifact at a time, which is exactly what
    :func:`load_axis` takes.
    """
    return load_axis(axis_id, str(root / model / axis_id))


class TestManifestValidation:
    """A malformed asset raises rather than loading something half-read.

    The server ships no assets, so there is no set of axes for a bad one to be isolated from:
    an axis arrives with the request, and refusing it is what tells the caller their fit is
    wrong. :mod:`.axis_request` turns each of these into a 400 naming the axis.
    """

    def test_a_malformed_manifest_is_rejected(self, asset_root):
        write_axis(asset_root, "m", "t_broken")
        (asset_root / "m" / "t_broken" / MANIFEST_FILENAME).write_text("title: [not, a, string\n", encoding="utf-8")
        with pytest.raises(Exception):  # noqa: B017 - yaml's own error type is not the contract
            loaded(asset_root, "m", "t_broken")

    def test_missing_tensors_are_rejected(self, asset_root):
        write_axis(asset_root, "m", "t_no-tensors", tensors=False)
        with pytest.raises(Exception):  # noqa: B017 - a missing file, raised by safetensors
            loaded(asset_root, "m", "t_no-tensors")

    def test_an_empty_title_and_no_poles_is_rejected(self, asset_root):
        # An axis with no label at all reaches the UI as an unnamed line on a chart. A manifest
        # that names its poles instead is fine, and TestPoles covers that.
        write_axis(asset_root, "m", "t_a", title="")
        with pytest.raises(ValueError, match="title"):
            loaded(asset_root, "m", "t_a")

    @pytest.mark.parametrize("scales", [{"scale": 0.0}, {"scale_pos": 0.0}, {"scale_neg": 0.0}])
    def test_a_zero_divisor_is_rejected(self, asset_root, scales):
        # Dividing by it would report inf for every turn on that side of the centre. Rejected
        # whichever key carries it, including the shared one, which reaches both poles.
        write_axis(asset_root, "m", "t_a", **scales)
        with pytest.raises(ValueError):
            loaded(asset_root, "m", "t_a")

    def test_an_unknown_normalize_mode_is_rejected(self, asset_root):
        write_axis(asset_root, "m", "t_a", normalize="whitened")
        with pytest.raises(ValueError):
            loaded(asset_root, "m", "t_a")

    def test_mismatched_tensor_shapes_are_rejected(self, asset_root):
        write_axis(asset_root, "m", "t_a", scaler_mean=[0.0, 0.0])
        with pytest.raises(ValueError):
            loaded(asset_root, "m", "t_a")


class TestAuthor:
    """Who fitted an axis is recorded with it, so a reading can be attributed."""

    def test_the_author_is_read_from_the_manifest(self, asset_root):
        write_axis(asset_root, "m", "mit_empathy")
        assert loaded(asset_root, "m", "mit_empathy").author == "mit"

    def test_two_authors_may_fit_the_same_trait(self, asset_root):
        # The reason the author is in the id by convention: these are different measurements of
        # the same word, and a caller has to be able to ask for one of them.
        write_axis(asset_root, "m", "mit_empathy", layer=19)
        write_axis(asset_root, "m", "lu_empathy", layer=40)
        assert {loaded(asset_root, "m", axis_id).author for axis_id in ("mit_empathy", "lu_empathy")} == {"mit", "lu"}

    def test_the_id_need_not_match_the_author(self, asset_root):
        # The id is the caller's to choose and names the reading in the response; the author
        # names who fitted the direction. A published artifact agrees on both by convention,
        # but a request may report one under any id it likes.
        write_axis(asset_root, "m", "mit_empathy", author="lu")
        assert loaded(asset_root, "m", "mit_empathy").author == "lu"

    def test_a_missing_author_is_rejected(self, asset_root):
        write_axis(asset_root, "m", "mit_empathy", author="")
        with pytest.raises(ValueError, match="author"):
            loaded(asset_root, "m", "mit_empathy")


class TestPoles:
    """What each end of an axis means, which is two fields and used to be a parsed string.

    The manifests in this tree still spell their poles inside a display title, so both readings
    have to work: an artifact that declares them, and one that only ever had the title.
    """

    def test_declared_poles_are_read_from_the_manifest(self, asset_root):
        write_axis(
            asset_root,
            "m",
            "mit_toxic",
            pole_positive="toxic",
            pole_negative="respectful",
            pole_positive_description="insulting or demeaning",
        )
        axis = load_axis("mit_toxic", str(asset_root / "m" / "mit_toxic"))
        assert (axis.pole_positive, axis.pole_negative) == ("toxic", "respectful")
        assert axis.pole_positive_description == "insulting or demeaning"
        assert axis.pole_negative_description is None

    def test_poles_are_recovered_from_a_title_that_predates_the_fields(self, asset_root):
        write_axis(asset_root, "m", "mit_toxic", title="- respectful \u2194\ufe0f + toxic")
        axis = load_axis("mit_toxic", str(asset_root / "m" / "mit_toxic"))
        assert (axis.pole_positive, axis.pole_negative) == ("toxic", "respectful")

    def test_a_declared_pole_wins_over_the_title(self, asset_root):
        write_axis(
            asset_root,
            "m",
            "mit_toxic",
            title="- respectful \u2194\ufe0f + toxic",
            pole_positive="hostile",
            pole_negative="civil",
        )
        axis = load_axis("mit_toxic", str(asset_root / "m" / "mit_toxic"))
        assert (axis.pole_positive, axis.pole_negative) == ("hostile", "civil")

    def test_a_manifest_that_declares_poles_needs_no_title(self, asset_root):
        # What a published artifact looks like: the poles in their own fields, and no title to
        # parse them back out of. The synthesized one is what the old manifests spelled by hand,
        # so a caller that displays a title sees no difference across the changeover.
        write_axis(asset_root, "m", "mit_toxic", title=None, pole_positive="toxic", pole_negative="respectful")
        axis = load_axis("mit_toxic", str(asset_root / "m" / "mit_toxic"))
        assert axis.title == "- respectful \u2194\ufe0f + toxic"
        assert (axis.pole_positive, axis.pole_negative) == ("toxic", "respectful")

    def test_a_manifest_with_neither_a_title_nor_both_poles_is_rejected(self, asset_root):
        # One pole is not a label for an axis, and nothing may invent the other end's name.
        write_axis(asset_root, "m", "mit_toxic", title=None, pole_positive="toxic")
        with pytest.raises(ValueError, match="'title', or both 'pole_positive' and 'pole_negative'"):
            load_axis("mit_toxic", str(asset_root / "m" / "mit_toxic"))

    def test_a_title_that_names_no_poles_leaves_them_absent(self, asset_root):
        # Absent rather than guessed. An axis is a direction first, and inventing "Tone" as a pole
        # would put a made-up word on a chart.
        write_axis(asset_root, "m", "mit_toxic", title="Tone")
        axis = load_axis("mit_toxic", str(asset_root / "m" / "mit_toxic"))
        assert (axis.pole_positive, axis.pole_negative) == (None, None)


class TestRenderConditions:
    def test_defaults_keep_the_system_prompt(self, asset_root):
        write_axis(asset_root, "m", "t_a")
        axis = loaded(asset_root, "m", "t_a")
        assert axis.render == RenderConditions()
        assert axis.render.blank_system_prompt is False

    def test_template_kwargs_are_read(self, asset_root):
        write_axis(
            asset_root,
            "m",
            "t_a",
            render={"blank_system_prompt": False, "template_kwargs": {"date_string": "26 Jul 2024"}},
        )
        axis = loaded(asset_root, "m", "t_a")
        assert axis.render.template_kwargs == {"date_string": "26 Jul 2024"}

    def test_axes_agree_only_when_every_field_matches(self):
        pinned = RenderConditions(template_kwargs={"date_string": "26 Jul 2024"})
        assert pinned.key() == RenderConditions(template_kwargs={"date_string": "26 Jul 2024"}).key()
        # A differing date is a different distribution, not a cosmetic difference.
        assert pinned.key() != RenderConditions(template_kwargs={"date_string": "01 Jan 2025"}).key()
        assert pinned.key() != RenderConditions(blank_system_prompt=True).key()


class TestProjection:
    def test_uncentered_dot_product(self, asset_root):
        write_axis(asset_root, "m", "t_a", direction=[1.0, 0.0, 0.0, 0.0])
        axis = load_axis("t_a", str(asset_root / "m" / "t_a"))
        acts = torch.tensor([[2.0, 9.0, 9.0, 9.0], [-3.0, 0.0, 0.0, 0.0]])
        np.testing.assert_allclose(project_axis(acts, axis), [2.0, -3.0], atol=1e-6)

    def test_means_are_subtracted_before_the_dot_product(self, asset_root):
        write_axis(
            asset_root,
            "m",
            "t_a",
            direction=[1.0, 0.0, 0.0, 0.0],
            scaler_mean=[1.0, 0.0, 0.0, 0.0],
            pca_mean=[0.5, 0.0, 0.0, 0.0],
        )
        axis = load_axis("t_a", str(asset_root / "m" / "t_a"))
        acts = torch.tensor([[4.0, 0.0, 0.0, 0.0]])
        np.testing.assert_allclose(project_axis(acts, axis), [2.5], atol=1e-6)

    def test_l2_normalize_makes_it_a_cosine(self, asset_root):
        write_axis(asset_root, "m", "t_a", direction=[1.0, 0.0, 0.0, 0.0], normalize="l2")
        axis = load_axis("t_a", str(asset_root / "m" / "t_a"))
        # Scaling an activation cannot change a cosine, which is the property "uncentered
        # cosine" assets rely on.
        acts = torch.tensor([[3.0, 4.0, 0.0, 0.0], [30.0, 40.0, 0.0, 0.0]])
        np.testing.assert_allclose(project_axis(acts, axis), [0.6, 0.6], atol=1e-6)

    def test_center_and_scale_are_applied(self, asset_root):
        write_axis(asset_root, "m", "t_a", direction=[1.0, 0.0, 0.0, 0.0], center=1.0, scale=0.5)
        axis = load_axis("t_a", str(asset_root / "m" / "t_a"))
        acts = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
        np.testing.assert_allclose(project_axis(acts, axis), [2.0], atol=1e-6)

    def test_values_are_not_clipped(self, asset_root):
        # Calibration puts most turns in [-1, 1] and roughly 1% outside it. Pinning that
        # spill to the boundary is the bug the recalibration exists to undo, so a value
        # past the limit has to come through as itself.
        write_axis(asset_root, "m", "t_a", direction=[1.0, 0.0, 0.0, 0.0], center=0.0, scale=0.4)
        axis = load_axis("t_a", str(asset_root / "m" / "t_a"))
        acts = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        np.testing.assert_allclose(project_axis(acts, axis), [2.5], atol=1e-6)

    def test_accepts_a_list_of_per_turn_tensors(self, asset_root):
        write_axis(asset_root, "m", "t_a", direction=[1.0, 0.0, 0.0, 0.0])
        axis = load_axis("t_a", str(asset_root / "m" / "t_a"))
        turns = [torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.tensor([2.0, 0.0, 0.0, 0.0])]
        np.testing.assert_allclose(project_axis(turns, axis), [1.0, 2.0], atol=1e-6)


class TestPerPoleScale:
    """Each pole is divided by its own spread.

    A fitted axis is rarely symmetric about its centre, and one divisor has to be the larger of
    the two spreads, so it squeezes the tighter pole towards zero. ``mit_erudite`` shipped that
    way: its negative tail is 5.8x its positive spread, so "sophisticated" could not read past
    +0.21 and looked flat whatever the prompt.
    """

    def test_each_pole_uses_its_own_divisor(self, asset_root):
        write_axis(
            asset_root,
            "m",
            "t_a",
            direction=[1.0, 0.0, 0.0, 0.0],
            center=0.0,
            scale_pos=0.1,
            scale_neg=0.5,
        )
        axis = load_axis("t_a", str(asset_root / "m" / "t_a"))
        acts = torch.tensor([[0.1, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.0]])
        np.testing.assert_allclose(project_axis(acts, axis), [1.0, -1.0], atol=1e-6)

    def test_the_tight_pole_reaches_full_scale(self, asset_root):
        # The regression itself, in the numbers that produced it: erudite's constants, and a raw
        # projection one positive spread above centre. Under the negative divisor -- which is
        # what max(pos, neg) picks -- this reads +0.17 instead of +1.0.
        write_axis(
            asset_root,
            "m",
            "t_a",
            direction=[1.0, 0.0, 0.0, 0.0],
            center=0.1780307,
            scale_pos=0.10189773,
            scale_neg=0.58756445,
        )
        axis = load_axis("t_a", str(asset_root / "m" / "t_a"))
        acts = torch.tensor([[0.1780307 + 0.10189773, 0.0, 0.0, 0.0]])
        np.testing.assert_allclose(project_axis(acts, axis), [1.0], atol=1e-6)

    def test_the_two_branches_agree_at_the_centre(self, asset_root):
        # The map is only usable if it is continuous where it changes branch: a jump at the
        # centre would put a gap in the middle of every axis, which is where readings cluster.
        write_axis(asset_root, "m", "t_a", center=0.25, scale_pos=0.1, scale_neg=10.0)
        axis = load_axis("t_a", str(asset_root / "m" / "t_a"))
        raw = torch.tensor([0.25 - 1e-9, 0.25, 0.25 + 1e-9])
        np.testing.assert_allclose(calibrate(raw, axis).numpy(), [0.0, 0.0, 0.0], atol=1e-8)

    def test_rank_order_survives_the_piecewise_map(self, asset_root):
        # What the trade is: the map is piecewise, so it is not linear in the raw projection,
        # but it is monotone, and a drift panel is read as an ordering.
        write_axis(asset_root, "m", "t_a", center=0.3, scale_pos=0.05, scale_neg=0.9)
        axis = load_axis("t_a", str(asset_root / "m" / "t_a"))
        raw = torch.linspace(-2.0, 2.0, 101)
        values = calibrate(raw, axis).numpy()
        assert np.all(np.diff(values) > 0)

    def test_a_lone_scale_still_means_both_poles(self, asset_root):
        # The spelling every asset used before the poles were separated, and what
        # `lu_assistant-axis` still ships. It has to keep reporting the numbers it always did.
        write_axis(asset_root, "m", "t_a", direction=[1.0, 0.0, 0.0, 0.0], scale=0.5)
        axis = load_axis("t_a", str(asset_root / "m" / "t_a"))
        assert (axis.scale_pos, axis.scale_neg) == (0.5, 0.5)
        acts = torch.tensor([[0.5, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.0]])
        np.testing.assert_allclose(project_axis(acts, axis), [1.0, -1.0], atol=1e-6)

    def test_an_absent_scale_is_uncalibrated_on_both_poles(self, asset_root):
        write_axis(asset_root, "m", "t_a")
        axis = load_axis("t_a", str(asset_root / "m" / "t_a"))
        assert (axis.scale_pos, axis.scale_neg) == (1.0, 1.0)

    def test_one_pole_may_be_given_on_its_own(self, asset_root):
        # A partial manifest takes the shared value for the pole it does not name, rather than
        # silently falling back to 1.0 and reporting that pole in raw projection units.
        write_axis(asset_root, "m", "t_a", scale=0.4, scale_pos=0.1)
        axis = load_axis("t_a", str(asset_root / "m" / "t_a"))
        assert (axis.scale_pos, axis.scale_neg) == (0.1, 0.4)


class TestPercentile:
    """The bounded reading, from the quantile tables.

    ``value`` is a ratio to the fit's p99 landmark, so it passes 1 for about 2% of turns by
    construction. That is a true measurement and a confusing gauge, so an asset may also carry
    the tables these tests exercise: the same projection expressed as the share of the
    calibration corpus it is past, which cannot leave [-1, 1] no matter the input. Boundedness
    is the property, so most of what follows is about the ends.
    """

    # A deliberately lopsided pair: the positive pole spans ten times what the negative does,
    # so a test that accidentally ranked against one table would read visibly wrong on the
    # other. Three entries put the levels at 0, 0.5 and 1.
    POS = [0.0, 1.0, 2.0]
    NEG = [0.0, 0.1, 0.2]

    def _axis(self, asset_root, **overrides):
        write_axis(
            asset_root,
            "m",
            "t_a",
            direction=[1.0, 0.0, 0.0, 0.0],
            quantiles_pos=overrides.pop("quantiles_pos", self.POS),
            quantiles_neg=overrides.pop("quantiles_neg", self.NEG),
            **overrides,
        )
        return load_axis("t_a", str(asset_root / "m" / "t_a"))

    def _percentiles(self, axis, raws: list[float]) -> np.ndarray:
        acts = torch.tensor([[raw, 0.0, 0.0, 0.0] for raw in raws])
        _, percentiles = project_axis_with_percentile(acts, axis)
        assert percentiles is not None
        return percentiles

    def test_a_reading_at_the_top_of_the_corpus_is_full_scale(self, asset_root):
        axis = self._axis(asset_root)
        np.testing.assert_allclose(self._percentiles(axis, [2.0, -0.2]), [1.0, -1.0], atol=1e-6)

    def test_a_reading_past_the_corpus_cannot_exceed_full_scale(self, asset_root):
        # The whole reason this scale exists: no input, however far off distribution, produces
        # the "102%" that the ratio legitimately reports.
        axis = self._axis(asset_root)
        assert np.abs(self._percentiles(axis, [3.0, 50.0, 1e6, -1e6])).max() == 1.0

    def test_each_pole_is_ranked_within_its_own_half(self, asset_root):
        # Both poles reach full scale despite spanning different distances, which is the same
        # reason the divisors are per pole: a shared table would flatten the tighter side.
        axis = self._axis(asset_root)
        np.testing.assert_allclose(self._percentiles(axis, [1.0, -0.1]), [0.5, -0.5], atol=1e-6)

    def test_the_centre_reads_zero_from_either_side(self, asset_root):
        # A step at the centre would put a gap where readings cluster, so the two branches have
        # to meet. Approached from both sides rather than evaluated once at zero.
        axis = self._axis(asset_root, center=0.25)
        values = self._percentiles(axis, [0.25, 0.25 - 1e-9, 0.25 + 1e-9])
        np.testing.assert_allclose(values, [0.0, 0.0, 0.0], atol=1e-6)

    def test_the_table_is_interpolated_between_its_entries(self, asset_root):
        # Without this the scale would be a staircase with 100 steps, and two readings a
        # percent apart would show as identical.
        axis = self._axis(asset_root)
        np.testing.assert_allclose(self._percentiles(axis, [0.5, 1.5]), [0.25, 0.75], atol=1e-6)

    def test_order_is_preserved_across_the_whole_range(self, asset_root):
        # The property that makes it honest to display: it is a relabelling of the measurement,
        # so it can never reorder two turns. Only non-decreasing, since the ends are flat.
        axis = self._axis(asset_root, center=0.1)
        values = self._percentiles(axis, list(np.linspace(-5.0, 5.0, 201)))
        assert np.all(np.diff(values) >= 0)

    def test_the_measurement_is_unchanged_by_having_tables(self, asset_root):
        # The percentile is added beside `value`, not in place of it: the same activations must
        # still project to the same number, or every stored reading would shift.
        acts = torch.tensor([[1.3, 0.0, 0.0, 0.0]])
        with_tables = self._axis(asset_root, center=0.2, scale_pos=0.5, scale_neg=0.4)
        write_axis(asset_root, "m", "t_b", direction=[1.0, 0.0, 0.0, 0.0], center=0.2, scale_pos=0.5, scale_neg=0.4)
        without = load_axis("t_b", str(asset_root / "m" / "t_b"))
        np.testing.assert_allclose(project_axis(acts, with_tables), project_axis(acts, without), atol=1e-9)

    def test_an_axis_without_tables_reports_no_percentile(self, asset_root):
        # `lu_assistant-axis` is this case, so it has to be a clean absence rather than a zero:
        # 0 would read as "dead centre" on a display.
        write_axis(asset_root, "m", "t_a", direction=[1.0, 0.0, 0.0, 0.0])
        axis = load_axis("t_a", str(asset_root / "m" / "t_a"))
        assert axis.has_percentile is False
        values, percentiles = project_axis_with_percentile(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), axis)
        assert percentiles is None
        np.testing.assert_allclose(values, [1.0], atol=1e-6)

    def test_an_incomplete_set_of_tables_is_rejected(self, asset_root):
        # Two of the three cannot be interpreted without the third, and guessing the missing
        # one would put readings at the wrong place on the scale rather than fail.
        write_axis(asset_root, "m", "t_a", quantiles_pos=self.POS)
        with pytest.raises(ValueError, match="incomplete"):
            load_axis("t_a", str(asset_root / "m" / "t_a"))

    def test_an_unsorted_table_is_rejected(self, asset_root):
        # The one failure a bounded scale must not have: a non-monotone table reorders readings
        # instead of merely misplacing them, and nothing downstream could detect it.
        write_axis(asset_root, "m", "t_a", quantiles_pos=[0.0, 2.0, 1.0], quantiles_neg=self.NEG)
        with pytest.raises(ValueError, match="nondecreasing"):
            load_axis("t_a", str(asset_root / "m" / "t_a"))

    def test_a_flat_table_does_not_divide_by_zero(self, asset_root):
        # A degenerate pole -- every training turn at the same distance -- is a bad axis, not a
        # crash, and it still has to answer with something inside the bound.
        axis = self._axis(asset_root, quantiles_pos=[0.0, 0.0, 0.0])
        values = self._percentiles(axis, [0.0, 1.0, 100.0])
        assert np.all(np.isfinite(values))
        assert np.abs(values).max() <= 1.0
