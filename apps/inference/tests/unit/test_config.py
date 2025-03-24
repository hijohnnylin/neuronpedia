import os
import sys

import pytest

from neuronpedia_inference.config import (
    Config,
    config_to_json,
    get_saelens_neuronpedia_directory_df,
)


@pytest.fixture
def mock_config():
    return Config(
        model_id="gpt2-small",
        sae_sets=["res-jb"],
        model_dtype="float16",
        sae_dtype="float32",
        secret="test_secret",
        token_limit=100,
        device="cpu",
    )


def test_config_initialization(mock_config: Config):
    assert mock_config.MODEL_ID == "gpt2-small"
    assert mock_config.MODEL_DTYPE == "float16"
    assert mock_config.SAE_DTYPE == "float32"
    assert mock_config.SECRET == "test_secret"
    assert mock_config.PORT == 5000
    assert mock_config.TOKEN_LIMIT == 100
    assert mock_config.SAES_PATH is None
    assert mock_config.VALID_COMPLETION_TYPES == ["DEFAULT", "STEERED"]
    assert mock_config.DEVICE == "cpu"

    # Check SAE config
    assert mock_config.SAE_CONFIG is not None
    assert mock_config.SAE_CONFIG[0] == {
        "model": "gpt2-small",
        "local": False,
        "set": "res-jb",
        "type": "saelens-1",
        "saes": [f"{i}-res-jb" for i in range(13)],
    }


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


OUTPUT_FOLDER = "server_files"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


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
    # server_name = "gpt2-small-server-1"
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

    assert isinstance(json_output, list)
    assert len(json_output) == len(selected_sets)

    expected_json = [
        {
            "model": "gpt2-small",
            "set": "res-jb",
            "type": "saelens-1",
            "local": False,
            "saes": [
                "0-res-jb",
                "1-res-jb",
                "2-res-jb",
                "3-res-jb",
                "4-res-jb",
                "5-res-jb",
                "6-res-jb",
                "7-res-jb",
                "8-res-jb",
                "9-res-jb",
                "10-res-jb",
                "11-res-jb",
                "12-res-jb",
            ],
        },
        {
            "model": "gpt2-small",
            "set": "res_fs768-jb",
            "type": "saelens-1",
            "local": False,
            "saes": ["8-res_fs768-jb"],
        },
        {
            "model": "gpt2-small",
            "set": "res_fs1536-jb",
            "type": "saelens-1",
            "local": False,
            "saes": ["8-res_fs1536-jb"],
        },
    ]

    assert json_output == expected_json


def test_config_no_filtering(mock_config: Config):
    assert mock_config.MODEL_ID == "gpt2-small"
    assert mock_config.MODEL_DTYPE == "float16"
    assert mock_config.SAE_DTYPE == "float32"
    assert mock_config.SECRET == "test_secret"
    assert mock_config.PORT == 5000
    assert mock_config.TOKEN_LIMIT == 100
    assert mock_config.SAES_PATH is None
    assert mock_config.VALID_COMPLETION_TYPES == ["DEFAULT", "STEERED"]
    assert mock_config.DEVICE == "cpu"

    # Check SAE config
    assert mock_config.SAE_CONFIG is not None
    assert mock_config.SAE_CONFIG[0] == {
        "model": "gpt2-small",
        "local": False,
        "set": "res-jb",
        "type": "saelens-1",
        "saes": [f"{i}-res-jb" for i in range(13)],
    }


def test_config_include_filtering():
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
    assert len(config.SAE_CONFIG) == 1
    assert len(config.SAE_CONFIG[0]["saes"]) == 6, config.SAE_CONFIG[0]["saes"]
    assert all(
        sae.startswith(("0-", "1-", "2-", "3-", "4-", "5-"))
        for sae in config.SAE_CONFIG[0]["saes"]
    )


def test_config_exclude_filtering():
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
    assert len(config.SAE_CONFIG) == 1
    assert len(config.SAE_CONFIG[0]["saes"]) == 7
    assert all(
        not sae.startswith(("0-", "1-", "2-", "3-", "4-", "5-"))
        for sae in config.SAE_CONFIG[0]["saes"]
    )


def test_config_include_exclude_filtering():
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
    assert len(config.SAE_CONFIG) == 1
    assert len(config.SAE_CONFIG[0]["saes"]) == 5
    assert all(
        sae.startswith(("0-", "1-", "2-", "3-", "4-"))
        for sae in config.SAE_CONFIG[0]["saes"]
    )


def test_config_filtering_with_multiple_patterns():
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
    assert len(config.SAE_CONFIG) == 2
    assert len(config.SAE_CONFIG[0]["saes"]) == 4
    assert set(config.SAE_CONFIG[0]["saes"]) == {
        "0-res-jb",
        "1-res-jb",
        "3-res-jb",
        "8-res-jb",
    }
    assert len(config.SAE_CONFIG[1]["saes"]) == 1
    assert config.SAE_CONFIG[1]["saes"][0] == "8-res_fs768-jb"


def test_config_multiple_sae_sets():
    config = Config(
        model_id="gpt2-small",
        sae_sets=["res-jb", "res_fs768-jb", "res_fs1536-jb"],
        model_dtype="float16",
        sae_dtype="float32",
        secret="test_secret",
        token_limit=100,
        device="cpu",
    )

    assert len(config.SAE_CONFIG) == 3

    # Check the first set (res-jb)
    assert config.SAE_CONFIG[0]["set"] == "res-jb"
    assert len(config.SAE_CONFIG[0]["saes"]) == 13
    assert all(sae.endswith("-res-jb") for sae in config.SAE_CONFIG[0]["saes"])

    # Check the second set (res_fs768-jb)
    assert config.SAE_CONFIG[1]["set"] == "res_fs768-jb"
    assert len(config.SAE_CONFIG[1]["saes"]) == 1
    assert config.SAE_CONFIG[1]["saes"][0] == "8-res_fs768-jb"

    # Check the third set (res_fs1536-jb)
    assert config.SAE_CONFIG[2]["set"] == "res_fs1536-jb"
    assert len(config.SAE_CONFIG[2]["saes"]) == 1
    assert config.SAE_CONFIG[2]["saes"][0] == "8-res_fs1536-jb"

    # Check common properties for all sets
    for sae_config in config.SAE_CONFIG:
        assert sae_config["model"] == "gpt2-small"
        assert sae_config["local"] is False
        assert sae_config["type"] == "saelens-1"
