from unittest.mock import patch

import pytest

from neuronpedia_inference.config import (
    Config,
    config_to_json,
    get_saelens_neuronpedia_directory_df,
)


@pytest.fixture
def mock_config():
    with patch.object(Config, "_generate_sae_config") as mock_gen:
        mock_gen.return_value = [
            {
                "model": "gpt2-small",
                "local": False,
                "set": "res-jb",
                "type": "saelens-1",
                "saes": [f"{i}-res-jb" for i in range(13)],
            }
        ]
        config = Config(
            model_id="gpt2-small",
            sae_sets=["res-jb"],
            model_dtype="float16",
            sae_dtype="float32",
            secret="test_secret",
            token_limit=100,
            device="cpu",
            valid_completion_types=[
                "DEFAULT",
                "STEERED",
            ],
        )
        return config  # noqa: RET504


def test_config_initialization(mock_config: Config):
    assert mock_config.model_id == "gpt2-small"
    assert mock_config.model_dtype == "float16"
    assert mock_config.sae_dtype == "float32"
    assert mock_config.secret == "test_secret"
    assert mock_config.port == 5000
    assert mock_config.token_limit == 100
    assert mock_config.valid_completion_types == ["DEFAULT", "STEERED"]
    assert mock_config.device == "cpu"

    # Check SAE config
    assert mock_config.sae_config is not None
    assert mock_config.sae_config[0] == {
        "model": "gpt2-small",
        "local": False,
        "set": "res-jb",
        "type": "saelens-1",
        "saes": [f"{i}-res-jb" for i in range(13)],
    }


def test_get_saelens_neuronpedia_directory_df():
    directory_df = get_saelens_neuronpedia_directory_df()

    assert directory_df is not None
    assert directory_df.shape[0] > 0
    for col_name in [
        "release",
        "model",
        "neuronpedia_id",
        "sae_lens_id",
        "neuronpedia_set",
    ]:
        assert col_name in directory_df.columns


def test_config_to_json():
    directory_df = get_saelens_neuronpedia_directory_df()

    # GPT2 Small SERVER 1
    selected_model = "gpt2-small"
    selected_sets = [
        "res-jb",
        "res_fs768-jb",
        "res_fs1536-jb",
    ]

    json_output = config_to_json(
        directory_df,
        selected_sets_neuronpedia=selected_sets,
        selected_model=selected_model,
    )

    # Verify basic structure
    assert isinstance(json_output, list)

    # Just check that we have entries for each set
    sets_in_output = {entry["set"] for entry in json_output}
    assert all(s in sets_in_output for s in selected_sets)


def test_config_no_filtering(mock_config: Config):
    assert mock_config.model_id == "gpt2-small"
    assert mock_config.model_dtype == "float16"
    assert mock_config.sae_dtype == "float32"
    assert mock_config.secret == "test_secret"
    assert mock_config.port == 5000
    assert mock_config.token_limit == 100
    assert mock_config.valid_completion_types == ["DEFAULT", "STEERED"]
    assert mock_config.device == "cpu"

    # Check SAE config
    assert mock_config.sae_config is not None
    assert mock_config.sae_config[0] == {
        "model": "gpt2-small",
        "local": False,
        "set": "res-jb",
        "type": "saelens-1",
        "saes": [f"{i}-res-jb" for i in range(13)],
    }


def test_config_include_filtering():
    with patch.object(Config, "_generate_sae_config") as mock_gen:
        # Return raw config before filtering
        mock_gen.return_value = [
            {
                "model": "gpt2-small",
                "local": False,
                "set": "res-jb",
                "type": "saelens-1",
                "saes": [f"{i}-res-jb" for i in range(13)],
            }
        ]

        config = Config(
            model_id="gpt2-small",
            sae_sets=["res-jb"],
            include_sae=[r"^[0-5]-res-jb"],
            model_dtype="float16",
            sae_dtype="float32",
            secret="test_secret",
            token_limit=100,
            device="cpu",
        )

        assert len(config.sae_config) == 1
        assert len(config.sae_config[0]["saes"]) == 6
        assert all(
            sae.startswith(("0-", "1-", "2-", "3-", "4-", "5-"))
            for sae in config.sae_config[0]["saes"]
        )


def test_config_exclude_filtering():
    with patch.object(Config, "_generate_sae_config") as mock_gen:
        # Return raw config before filtering
        mock_gen.return_value = [
            {
                "model": "gpt2-small",
                "local": False,
                "set": "res-jb",
                "type": "saelens-1",
                "saes": [f"{i}-res-jb" for i in range(13)],
            }
        ]

        config = Config(
            model_id="gpt2-small",
            sae_sets=["res-jb"],
            exclude_sae=[r"^[0-5]-res-jb"],
            model_dtype="float16",
            sae_dtype="float32",
            secret="test_secret",
            token_limit=100,
            device="cpu",
        )

        assert len(config.sae_config) == 1
        assert len(config.sae_config[0]["saes"]) == 7
        assert all(
            not sae.startswith(("0-", "1-", "2-", "3-", "4-", "5-"))
            for sae in config.sae_config[0]["saes"]
        )


def test_config_include_exclude_filtering():
    with patch.object(Config, "_generate_sae_config") as mock_gen:
        # Return raw config before filtering
        mock_gen.return_value = [
            {
                "model": "gpt2-small",
                "local": False,
                "set": "res-jb",
                "type": "saelens-1",
                "saes": [f"{i}-res-jb" for i in range(13)],
            }
        ]

        config = Config(
            model_id="gpt2-small",
            sae_sets=["res-jb"],
            include_sae=[r"^[0-9]-res-jb"],
            exclude_sae=[r"^[5-9]-res-jb"],
            model_dtype="float16",
            sae_dtype="float32",
            secret="test_secret",
            token_limit=100,
            device="cpu",
        )

        assert len(config.sae_config) == 1
        assert len(config.sae_config[0]["saes"]) == 5
        assert all(
            sae.startswith(("0-", "1-", "2-", "3-", "4-"))
            for sae in config.sae_config[0]["saes"]
        )


def test_config_filtering_with_multiple_patterns():
    with patch.object(Config, "_generate_sae_config") as mock_gen:
        # Return raw config before filtering
        mock_gen.return_value = [
            {
                "model": "gpt2-small",
                "local": False,
                "set": "res-jb",
                "type": "saelens-1",
                "saes": [f"{i}-res-jb" for i in range(13)],
            },
            {
                "model": "gpt2-small",
                "local": False,
                "set": "res_fs768-jb",
                "type": "saelens-1",
                "saes": ["8-res_fs768-jb"],
            },
        ]

        config = Config(
            model_id="gpt2-small",
            sae_sets=["res-jb", "res_fs768-jb"],
            include_sae=[r"^[0-3]-res-jb", r"^8-res"],
            exclude_sae=[r"^2-res-jb"],
            model_dtype="float16",
            sae_dtype="float32",
            secret="test_secret",
            token_limit=100,
            device="cpu",
        )

        assert len(config.sae_config) == 2
        assert len(config.sae_config[0]["saes"]) == 4
        assert set(config.sae_config[0]["saes"]) == {
            "0-res-jb",
            "1-res-jb",
            "3-res-jb",
            "8-res-jb",
        }
        assert len(config.sae_config[1]["saes"]) == 1
        assert config.sae_config[1]["saes"][0] == "8-res_fs768-jb"


def test_config_multiple_sae_sets():
    with patch.object(Config, "_generate_sae_config") as mock_gen:
        # Return raw config before filtering
        mock_gen.return_value = [
            {
                "model": "gpt2-small",
                "local": False,
                "set": "res-jb",
                "type": "saelens-1",
                "saes": [f"{i}-res-jb" for i in range(13)],
            },
            {
                "model": "gpt2-small",
                "local": False,
                "set": "res_fs768-jb",
                "type": "saelens-1",
                "saes": ["8-res_fs768-jb"],
            },
            {
                "model": "gpt2-small",
                "local": False,
                "set": "res_fs1536-jb",
                "type": "saelens-1",
                "saes": ["8-res_fs1536-jb"],
            },
        ]

        config = Config(
            model_id="gpt2-small",
            sae_sets=["res-jb", "res_fs768-jb", "res_fs1536-jb"],
            model_dtype="float16",
            sae_dtype="float32",
            secret="test_secret",
            token_limit=100,
            device="cpu",
        )

        assert len(config.sae_config) == 3

        # Check the first set (res-jb)
        assert config.sae_config[0]["set"] == "res-jb"
        assert len(config.sae_config[0]["saes"]) == 13
        assert all(sae.endswith("-res-jb") for sae in config.sae_config[0]["saes"])

        # Check the second set (res_fs768-jb)
        assert config.sae_config[1]["set"] == "res_fs768-jb"
        assert len(config.sae_config[1]["saes"]) == 1
        assert config.sae_config[1]["saes"][0] == "8-res_fs768-jb"

        # Check the third set (res_fs1536-jb)
        assert config.sae_config[2]["set"] == "res_fs1536-jb"
        assert len(config.sae_config[2]["saes"]) == 1
        assert config.sae_config[2]["saes"][0] == "8-res_fs1536-jb"

        # Check common properties for all sets
        for sae_config in config.sae_config:
            assert sae_config["model"] == "gpt2-small"
            assert sae_config["local"] is False
            assert sae_config["type"] == "saelens-1"
