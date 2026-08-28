"""Readout axes supplied with a request: what is accepted, what is refused, and why.

An on-disk asset was built by a converter that checked it. One arriving in a request was not, so
every check that protects a projection has to happen at the door -- and the two that only the
serving model can answer, width and depth, can happen nowhere else. A direction of the wrong
length is a 500 from a matmul otherwise, and a non-finite entry is worse than an error: it returns
a fully populated readout in which every number is NaN.

The other half of these tests is that an accepted payload becomes the *same* thing an asset
directory becomes. Everything downstream reads an ``AxisAsset``, so if the two paths agree there,
a request-supplied axis is measured by exactly the code the shipped ones are.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest
import torch
import yaml
from pydantic import ValidationError
from safetensors.torch import save_file

from neuronpedia_inference.inference_utils.persona.axis_data import (
    MANIFEST_FILENAME,
    TENSORS_FILENAME,
    project_axis,
    project_axis_with_percentile,
)
from neuronpedia_inference.inference_utils.persona.axis_request import (
    DEFAULT_AUTHOR,
    MAX_CUSTOM_AXES,
    AxisRequestError,
    _artifact_at,
    asset_from_payload,
    resolve_request_axes,
)
from neuronpedia_inference.schemas import NPAxis

D_MODEL = 4
N_LAYERS = 8
UNIT = [1.0, 0.0, 0.0, 0.0]


def payload(**overrides) -> NPAxis:
    """The smallest axis a caller can send, with room to break one thing at a time."""
    return NPAxis.model_validate({"id": "np_trait", "direction": UNIT, "layer": 3, **overrides})


def build(**overrides):
    return asset_from_payload(payload(**overrides), hidden_size=D_MODEL, n_layers=N_LAYERS)


def acts(*first_components: float) -> torch.Tensor:
    rows = torch.zeros(len(first_components), D_MODEL)
    for row, value in enumerate(first_components):
        rows[row][0] = value
    return rows


def resolve(payloads: list[NPAxis]):
    return asyncio.run(resolve_request_axes(payloads, hidden_size=D_MODEL, n_layers=N_LAYERS))


class TestPayloadShape:
    """Which combinations of fields are an axis at all. Refused by the model, before any model."""

    def test_a_direction_and_a_layer_are_enough(self):
        assert payload().layer == 3

    def test_a_source_alone_is_enough(self):
        axis = NPAxis.model_validate({"id": "np_trait", "source": {"hfRepoId": "org/axes", "hfFolder": "trait"}})
        assert axis.source is not None
        assert axis.direction is None

    def test_a_direction_without_a_layer_is_refused(self):
        with pytest.raises(ValidationError, match="layer"):
            NPAxis.model_validate({"id": "np_trait", "direction": UNIT})

    def test_neither_a_source_nor_a_direction_is_refused(self):
        with pytest.raises(ValidationError, match="direction"):
            NPAxis.model_validate({"id": "np_trait"})

    def test_a_source_carrying_the_axis_cannot_be_overridden(self):
        # Half a published artifact with a hand-edited direction is not the artifact anyone can go
        # and look at, and the response could not say which of the two produced the numbers.
        with pytest.raises(ValidationError, match="direction"):
            NPAxis.model_validate(
                {
                    "id": "np_trait",
                    "source": {"hfRepoId": "org/axes", "hfFolder": "trait"},
                    "direction": UNIT,
                    "layer": 3,
                }
            )

    def test_a_source_cannot_be_relabelled_either(self):
        with pytest.raises(ValidationError, match="pole_positive"):
            NPAxis.model_validate(
                {
                    "id": "np_trait",
                    "source": {"hfRepoId": "org/axes", "hfFolder": "trait"},
                    "polePositive": "toxic",
                }
            )

    def test_the_wire_names_are_camel_case(self):
        axis = payload(preNormMean=[0.5] * D_MODEL, polePositive="toxic", scalePos=2.0)
        assert axis.pre_norm_mean == [0.5] * D_MODEL
        assert axis.pole_positive == "toxic"
        assert axis.scale_pos == 2.0


class TestDefaults:
    """What a two-field axis means. Everything optional has to have an answer, and the answer has
    to be the identity: a caller who sent a direction and a layer gets the dot product, not a
    transformed version of it they did not ask for."""

    def test_the_reading_is_the_dot_product(self):
        assert project_axis(acts(2.0, -3.0), build()).tolist() == [2.0, -3.0]

    def test_no_mean_is_subtracted(self):
        axis = build()
        assert axis.scaler_mean.tolist() == [0.0] * D_MODEL
        assert axis.pca_mean.tolist() == [0.0] * D_MODEL

    def test_nothing_is_normalized(self):
        assert build().normalize == "none"

    def test_there_is_no_percentile_without_tables(self):
        values, percentiles = project_axis_with_percentile(acts(2.0), build())
        assert values.tolist() == [2.0]
        assert percentiles is None

    def test_an_unclaimed_axis_is_not_attributed_to_whoever_the_id_names(self):
        # The `<author>_<name>` convention holds for what this server ships; a caller's id is
        # theirs to choose, so reading an author out of it would attribute a fit to a stranger.
        assert build(id="mit_trait").author == DEFAULT_AUTHOR

    def test_a_claimed_axis_is_attributed_as_sent(self):
        assert build(author="Some Lab").author == "Some Lab"

    def test_the_title_is_not_a_parsed_string(self):
        # Poles are their own fields now. A title survives only for displays that read one, and
        # nothing should be able to recover the poles from it.
        assert build(polePositive="toxic", poleNegative="respectful").title == "np_trait"
        assert build(displayName="Tone").title == "Tone"


class TestCarried:
    """Everything a caller sends and expects back, or expects to take effect."""

    def test_poles_and_their_descriptions(self):
        axis = build(
            polePositive="toxic",
            poleNegative="respectful",
            polePositiveDescription="insulting or demeaning",
            poleNegativeDescription="considerate",
        )
        assert (axis.pole_positive, axis.pole_negative) == ("toxic", "respectful")
        assert axis.pole_positive_description == "insulting or demeaning"
        assert axis.pole_negative_description == "considerate"

    def test_a_caveat(self):
        assert build(caveat="fitted on 200 turns").caveat == "fitted on 200 turns"

    def test_render_conditions(self):
        axis = build(render={"blankSystemPrompt": True, "templateKwargs": {"date_string": "26 Jul 2024"}})
        assert axis.render.blank_system_prompt is True
        assert axis.render.template_kwargs == {"date_string": "26 Jul 2024"}

    def test_the_two_means_are_subtracted_in_their_own_places(self):
        # Without normalizing they are interchangeable, so the test that tells them apart has to
        # normalize between them: pre-norm shifts what gets scaled, post-norm shifts the result.
        pre = build(normalize="l2", preNormMean=[1.0, 0.0, 0.0, 0.0])
        post = build(normalize="l2", postNormMean=[1.0, 0.0, 0.0, 0.0])
        assert project_axis(acts(3.0), pre).tolist() == [1.0]
        assert project_axis(acts(3.0), post).tolist() == [0.0]

    def test_l2_normalizing_makes_the_reading_a_direction(self):
        assert project_axis(acts(3.0, 60.0), build(normalize="l2")).tolist() == [1.0, 1.0]

    def test_the_two_scales_are_per_pole(self):
        axis = build(center=1.0, scalePos=2.0, scaleNeg=4.0)
        assert project_axis(acts(5.0, -3.0), axis).tolist() == [2.0, -1.0]

    def test_quantile_tables_produce_a_percentile(self):
        axis = build(quantilesPos=[0.0, 1.0, 2.0], quantilesNeg=[0.0, 1.0, 2.0])
        values, percentiles = project_axis_with_percentile(acts(1.0, -2.0, 9.0), axis)
        assert values.tolist() == [1.0, -2.0, 9.0]
        assert percentiles is not None
        # Half way up the table, the end of it, and past everything the fit saw -- which holds at
        # 1.0 rather than reporting a share above the whole.
        assert percentiles.tolist() == [0.5, -1.0, 1.0]

    def test_the_levels_can_be_uneven(self):
        axis = build(quantilesPos=[0.0, 1.0], quantilesNeg=[0.0, 1.0], quantileLevels=[0.0, 0.5])
        _, percentiles = project_axis_with_percentile(acts(1.0), axis)
        assert percentiles is not None
        assert percentiles.tolist() == [0.5]


class TestRefusals:
    """A payload this model cannot measure. Each of these is a 400 naming the axis and the field,
    because every one of them is something the caller can fix -- and every one of them, unchecked,
    produces a response that looks like an answer."""

    def test_a_direction_of_the_wrong_width(self):
        with pytest.raises(AxisRequestError, match="3 entries.*4-dimensional"):
            build(direction=[1.0, 0.0, 0.0])

    def test_a_mean_of_the_wrong_width(self):
        with pytest.raises(AxisRequestError, match="preNormMean"):
            build(preNormMean=[0.0, 0.0])

    def test_a_direction_of_all_zeros(self):
        # Every turn would read exactly `center`: a model with no variation on this trait, which
        # is not what "you forgot to fill in the direction" looks like.
        with pytest.raises(AxisRequestError, match="all zeros"):
            build(direction=[0.0] * D_MODEL)

    def test_a_non_finite_direction(self):
        with pytest.raises(AxisRequestError, match="non-finite"):
            build(direction=[float("nan"), 0.0, 0.0, 0.0])

    def test_a_non_finite_quantile_table(self):
        with pytest.raises(AxisRequestError, match="non-finite"):
            build(quantilesPos=[0.0, float("inf")], quantilesNeg=[0.0, 1.0])

    def test_a_layer_this_model_does_not_have(self):
        with pytest.raises(AxisRequestError, match="out of range"):
            build(layer=N_LAYERS)

    def test_a_negative_layer(self):
        with pytest.raises(AxisRequestError, match="out of range"):
            build(layer=-1)

    @pytest.mark.parametrize("field", ["scalePos", "scaleNeg"])
    def test_a_scale_of_zero(self, field: str):
        with pytest.raises(AxisRequestError, match="may not be zero"):
            build(**{field: 0.0})

    def test_one_pole_of_a_quantile_table(self):
        with pytest.raises(AxisRequestError, match="quantilesNeg is missing"):
            build(quantilesPos=[0.0, 1.0])

    def test_quantile_tables_of_different_lengths(self):
        with pytest.raises(AxisRequestError, match="2 entries and quantilesNeg has 3"):
            build(quantilesPos=[0.0, 1.0], quantilesNeg=[0.0, 1.0, 2.0])

    def test_levels_that_do_not_match_the_tables(self):
        with pytest.raises(AxisRequestError, match="quantileLevels has 3"):
            build(quantilesPos=[0.0, 1.0], quantilesNeg=[0.0, 1.0], quantileLevels=[0.0, 0.5, 1.0])

    def test_levels_without_any_table(self):
        with pytest.raises(AxisRequestError, match="without any quantile table"):
            build(quantileLevels=[0.0, 1.0])

    def test_a_single_point_table(self):
        with pytest.raises(AxisRequestError, match="at least two entries"):
            build(quantilesPos=[1.0], quantilesNeg=[1.0])

    def test_an_unsorted_table(self):
        # Interpolating it is non-monotone, which reorders readings rather than merely misplacing
        # them: the one failure a bounded scale must not have.
        with pytest.raises(AxisRequestError, match="nondecreasing"):
            build(quantilesPos=[1.0, 0.0], quantilesNeg=[0.0, 1.0])


class TestResolvingSeveral:
    def test_order_is_the_order_they_were_sent(self):
        assets = resolve([payload(id="a_one"), payload(id="b_two", layer=5)])
        assert [(a.id, a.layer) for a in assets] == [("a_one", 3), ("b_two", 5)]

    def test_two_axes_cannot_share_an_id(self):
        with pytest.raises(AxisRequestError, match="Duplicate"):
            resolve([payload(id="a_one"), payload(id="a_one", layer=5)])

    def test_more_axes_than_a_request_may_carry(self):
        with pytest.raises(AxisRequestError, match=f"At most {MAX_CUSTOM_AXES}"):
            resolve([payload(id=f"np_{index}") for index in range(MAX_CUSTOM_AXES + 1)])

    def test_the_cap_itself_is_allowed(self):
        assert len(resolve([payload(id=f"np_{index}") for index in range(MAX_CUSTOM_AXES)])) == MAX_CUSTOM_AXES


def write_artifact(root, *, layer: int = 3, width: int = D_MODEL, **manifest_overrides) -> None:
    """One published artifact on a fake Hub: the same two files an asset directory holds."""
    root.mkdir(parents=True, exist_ok=True)
    direction = [1.0] + [0.0] * (width - 1)
    save_file(
        {
            "direction": torch.tensor(direction),
            "scaler_mean": torch.zeros(width),
            "pca_mean": torch.zeros(width),
        },
        str(root / TENSORS_FILENAME),
    )
    manifest = {
        "author": "mit",
        "title": "- respectful \u2194\ufe0f + toxic",
        "layer": layer,
        **manifest_overrides,
    }
    (root / MANIFEST_FILENAME).write_text(yaml.safe_dump(manifest, allow_unicode=True), encoding="utf-8")


class FakeHub:
    """The Hub as this module uses it: resolve a revision, then fetch two files at that commit."""

    def __init__(self, root):
        self.root = root
        self.downloads: list[str] = []
        self.sha = "0" * 40
        self.repo_error: Exception | None = None

    def repo_info(self, repo_id: str, revision: str | None = None, repo_type: str | None = None):
        if self.repo_error is not None:
            raise self.repo_error
        return SimpleNamespace(sha=self.sha)

    def download(self, *, repo_id: str, filename: str, revision: str, repo_type: str) -> str:
        from huggingface_hub.errors import EntryNotFoundError

        self.downloads.append(f"{repo_id}@{revision}/{filename}")
        path = self.root / revision / filename
        if not path.exists():
            raise EntryNotFoundError(f"{filename} not found in {repo_id}")
        return str(path)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A fake Hub with one artifact at ``trait/``, and the artifact cache cleared around it."""
    fake = FakeHub(tmp_path)
    write_artifact(tmp_path / fake.sha / "trait")
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: fake)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake.download)
    _artifact_at.cache_clear()
    yield fake
    _artifact_at.cache_clear()


def from_source(folder: str = "trait", **source_overrides) -> list:
    return resolve(
        [
            NPAxis.model_validate(
                {
                    "id": "np_mine",
                    "source": {"hfRepoId": "org/axes", "hfFolder": folder, **source_overrides},
                }
            )
        ]
    )


class TestPublishedArtifact:
    def test_the_artifact_becomes_an_asset(self, hub):
        (axis,) = from_source()
        assert axis.layer == 3
        assert axis.author == "mit"
        assert (axis.pole_positive, axis.pole_negative) == ("toxic", "respectful")
        assert project_axis(acts(2.0), axis).tolist() == [2.0]

    def test_it_is_reported_under_the_id_the_caller_chose(self, hub):
        # The id keys the readout, so it is the caller's. Who fitted the axis comes from the
        # artifact, which is why an unprefixed id is not a misattribution here.
        (axis,) = from_source()
        assert axis.id == "np_mine"

    def test_the_commit_it_was_read_at_comes_back(self, hub):
        (axis,) = from_source()
        assert axis.source_revision == hub.sha

    def test_the_revision_is_resolved_every_time_but_read_once(self, hub):
        from_source()
        from_source()
        # Two requests, two artifacts parsed only if the cache missed. A branch name is a moving
        # target, so what gets cached is the commit it resolved to, not the name.
        assert len(hub.downloads) == 2  # axis.yaml + axis.safetensors, for the first call only

    def test_a_new_commit_is_read_again(self, hub):
        from_source()
        hub.sha = "1" * 40
        write_artifact(hub.root / hub.sha / "trait", layer=5)
        (axis,) = from_source()
        assert axis.layer == 5
        assert axis.source_revision == "1" * 40

    def test_a_folder_that_is_not_an_axis(self, hub):
        with pytest.raises(AxisRequestError, match=f"has no {MANIFEST_FILENAME}") as caught:
            from_source("not-an-axis")
        assert caught.value.status_code == 400

    def test_a_repo_that_cannot_be_read_is_not_the_caller_s_fault(self, hub):
        # A Hub that will not answer is ours to report as such: a 400 would tell the caller to fix
        # a request that was fine.
        hub.repo_error = OSError("connection reset")
        with pytest.raises(AxisRequestError, match="connection reset") as caught:
            from_source()
        assert caught.value.status_code == 502

    def test_a_download_that_fails_for_any_other_reason(self, hub, monkeypatch):
        def explode(**_kwargs):
            raise OSError("connection reset")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", explode)
        with pytest.raises(AxisRequestError) as caught:
            from_source()
        assert caught.value.status_code == 502

    def test_an_artifact_fitted_for_a_different_model(self, hub):
        write_artifact(hub.root / hub.sha / "wide", width=D_MODEL + 1)
        with pytest.raises(AxisRequestError, match="5-dimensional.*4-dimensional"):
            from_source("wide")

    def test_an_artifact_fitted_at_a_layer_this_model_does_not_have(self, hub):
        write_artifact(hub.root / hub.sha / "deep", layer=N_LAYERS + 1)
        with pytest.raises(AxisRequestError, match="does not have"):
            from_source("deep")

    def test_a_malformed_artifact_names_what_was_wrong_with_it(self, hub):
        write_artifact(hub.root / hub.sha / "broken", scale_pos=0.0)
        with pytest.raises(AxisRequestError, match="not a valid axis"):
            from_source("broken")

    def test_a_folder_cannot_walk_out_of_the_repo(self, hub):
        with pytest.raises(AxisRequestError, match=r"\.\."):
            from_source("../../etc")

    def test_a_folder_at_the_repo_root(self, hub):
        write_artifact(hub.root / hub.sha, layer=6)
        (axis,) = from_source("")
        assert axis.layer == 6
        assert os.path.basename(hub.downloads[0]) == MANIFEST_FILENAME
