"""Tests for Config and the SAELens directory helpers it is built on.

The SAE config Config exposes is the SAELens directory narrowed down twice: once to the
requested sets, then again by the include/exclude patterns. Most of what is worth testing
is that second step, so these tests stub out the generated config and vary only the
patterns applied to it.
"""

from typing import Any
from unittest.mock import patch

import pytest

from neuronpedia_inference.config import (
    Config,
    config_to_json,
    get_saelens_neuronpedia_directory_df,
)

RES_JB_SAES = [f"{i}-res-jb" for i in range(13)]


def sae_set(name: str, saes: list[str]) -> dict[str, Any]:
    """One entry shaped the way _generate_sae_config returns them, pre-filtering."""
    return {
        "model": "gpt2-small",
        "local": False,
        "set": name,
        "type": "saelens-1",
        "saes": saes,
    }


def build_config(generated: list[dict[str, Any]], **overrides: Any) -> Config:
    """Build a Config whose generated SAE config is fixed rather than looked up.

    The real _generate_sae_config reads the SAELens directory over the network. Stubbing
    it keeps these tests offline and deterministic, and leaves the filtering (which is
    what they actually exercise) running for real.
    """
    settings: dict[str, Any] = {
        "model_id": "gpt2-small",
        "sae_sets": ["res-jb"],
        "model_dtype": "float16",
        "sae_dtype": "float32",
        "secret": "test_secret",
        "token_limit": 100,
        "device": "cpu",
    }
    settings.update(overrides)
    with patch.object(Config, "_generate_sae_config", return_value=generated):
        return Config(**settings)


@pytest.fixture
def config() -> Config:
    return build_config(
        [sae_set("res-jb", RES_JB_SAES)],
        valid_completion_types=["DEFAULT", "STEERED"],
    )


class TestConfigInitialization:
    def test_settings_are_kept_as_given(self, config: Config):
        assert config.model_id == "gpt2-small"
        assert config.model_dtype == "float16"
        assert config.sae_dtype == "float32"
        assert config.secret == "test_secret"
        assert config.token_limit == 100
        assert config.device == "cpu"
        assert config.valid_completion_types == ["DEFAULT", "STEERED"]

    def test_port_falls_back_to_its_default(self, config: Config):
        assert config.port == 5000

    def test_sae_config_passes_through_untouched_without_patterns(self, config: Config):
        assert config.sae_config == [sae_set("res-jb", RES_JB_SAES)]


class TestSaeFiltering:
    @pytest.mark.parametrize(
        ("include", "exclude", "expected"),
        [
            pytest.param(None, None, RES_JB_SAES, id="no-patterns-keeps-everything"),
            pytest.param(
                [r"^[0-5]-res-jb"],
                None,
                [f"{i}-res-jb" for i in range(6)],
                id="include-keeps-only-what-matches",
            ),
            pytest.param(
                None,
                [r"^[0-5]-res-jb"],
                [f"{i}-res-jb" for i in range(6, 13)],
                id="exclude-drops-what-matches",
            ),
            pytest.param(
                [r"^[0-9]-res-jb"],
                [r"^[5-9]-res-jb"],
                [f"{i}-res-jb" for i in range(5)],
                id="exclude-overrides-include",
            ),
        ],
    )
    def test_patterns_select_saes(self, include: list[str] | None, exclude: list[str] | None, expected: list[str]):
        config = build_config(
            [sae_set("res-jb", RES_JB_SAES)],
            include_sae=include,
            exclude_sae=exclude,
        )
        assert [entry["saes"] for entry in config.sae_config] == [expected]

    def test_single_digit_patterns_do_not_catch_double_digit_ids(self):
        # `^[0-9]-` cannot match "10-res-jb": the character after the digit is "0",
        # not the hyphen. Worth pinning, since layers 10-12 silently going missing is
        # the kind of thing an over-eager pattern would cause.
        config = build_config([sae_set("res-jb", RES_JB_SAES)], include_sae=[r"^[0-9]-res-jb"])
        assert config.sae_config[0]["saes"] == [f"{i}-res-jb" for i in range(10)]

    def test_patterns_apply_across_every_set(self):
        config = build_config(
            [
                sae_set("res-jb", RES_JB_SAES),
                sae_set("res_fs768-jb", ["8-res_fs768-jb"]),
            ],
            sae_sets=["res-jb", "res_fs768-jb"],
            # "^8-res" spans both sets; the exclude then removes one id from the first.
            include_sae=[r"^[0-3]-res-jb", r"^8-res"],
            exclude_sae=[r"^2-res-jb"],
        )
        assert [entry["saes"] for entry in config.sae_config] == [
            ["0-res-jb", "1-res-jb", "3-res-jb", "8-res-jb"],
            ["8-res_fs768-jb"],
        ]

    def test_sets_keep_their_order_and_metadata(self):
        generated = [
            sae_set("res-jb", RES_JB_SAES),
            sae_set("res_fs768-jb", ["8-res_fs768-jb"]),
            sae_set("res_fs1536-jb", ["8-res_fs1536-jb"]),
        ]
        config = build_config(generated, sae_sets=["res-jb", "res_fs768-jb", "res_fs1536-jb"])

        assert config.sae_config == generated
        assert [entry["set"] for entry in config.sae_config] == [
            "res-jb",
            "res_fs768-jb",
            "res_fs1536-jb",
        ]


class TestSaeLensDirectory:
    def test_directory_has_the_columns_config_reads(self):
        directory_df = get_saelens_neuronpedia_directory_df()

        assert directory_df is not None
        assert directory_df.shape[0] > 0
        assert {
            "release",
            "model",
            "neuronpedia_id",
            "sae_lens_id",
            "neuronpedia_set",
        } <= set(directory_df.columns)

    def test_config_to_json_returns_each_requested_set(self):
        selected_sets = ["res-jb", "res_fs768-jb", "res_fs1536-jb"]

        json_output = config_to_json(
            get_saelens_neuronpedia_directory_df(),
            selected_sets_neuronpedia=selected_sets,
            selected_model="gpt2-small",
        )

        assert isinstance(json_output, list)
        assert {entry["set"] for entry in json_output} >= set(selected_sets)
