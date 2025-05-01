from unittest.mock import MagicMock, patch

import pytest
from sae_lens.sae import SAE

from neuronpedia_inference.config import Config
from neuronpedia_inference.sae_manager import SAEManager


class MockSAE:
    def __init__(self):
        self.cfg = MagicMock()
        self.cfg.hook_name = "blocks.5.hook_resid_pre"
        self.cfg.neuronpedia_id = "mock-neuronpedia-id"

    def to(self, device, dtype):  # type: ignore
        return self

    def fold_W_dec_norm(self):
        pass

    def eval(self):
        pass


def mock_from_pretrained(*args, **kwargs):  # type: ignore # noqa: ARG001
    return MockSAE(), {}, {}


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
        include_sae=[r"^5-res-jb"],
        max_loaded_saes=3,
    )


@pytest.fixture
def mock_config_2():
    return Config(
        model_id="gemma-2-2b",
        sae_sets=["gemmascope-mlp-16k"],  # This seems to be loading two SAEs
        model_dtype="float16",
        sae_dtype="float32",
        secret="test_secret",
        port=5000,
        token_limit=2048,
        override_model_id="gemma-2-2b-it",
        include_sae=[r"^1-gemmascope-mlp-16k$"],
    )


@pytest.fixture
def mock_config_multiple_sae_sets():
    return Config(
        model_id="gpt2-small",
        sae_sets=["res-jb", "att-kk"],
        model_dtype="float16",
        sae_dtype="float32",
        secret="test_secret",
        token_limit=200,
        device="cpu",
        exclude_sae=[r"^(?!5-res-jb|7-att-kk).*"],
    )


def test_sae_manager_initialize(mock_config: Config) -> None:
    with (
        patch.object(SAE, "from_pretrained", side_effect=mock_from_pretrained),
        patch(
            "neuronpedia_inference.sae_manager.Config.get_instance",
            return_value=mock_config,
        ),
    ):
        sae_manager = SAEManager(
            num_layers=12,
            device="cpu",
        )
        sae_manager.load_saes()  # Call load_saes explicitly
    assert sae_manager is not None
    assert len(sae_manager.sae_data) == 13  # 12 layers + 1 SAE


def test_sae_manager_initialize_different_model(mock_config_2: Config) -> None:
    with (
        patch.object(SAE, "from_pretrained", side_effect=mock_from_pretrained),
        patch(
            "neuronpedia_inference.sae_manager.Config.get_instance",
            return_value=mock_config_2,
        ),
    ):
        sae_manager = SAEManager(
            num_layers=12,
            device="cpu",
        )
        sae_manager.load_saes()  # Call load_saes explicitly

    assert sae_manager is not None
    assert len(sae_manager.sae_data) == 13  # 12 layers + 1 SAE
    assert "1-gemmascope-mlp-16k" in sae_manager.sae_data
    assert isinstance(sae_manager.sae_data["1-gemmascope-mlp-16k"]["sae"], MockSAE)

    # check valid models
    expected_accepted_model_ids = {"gemma-2-2b"}
    assert sae_manager.config.get_valid_model_ids() == expected_accepted_model_ids


@pytest.fixture
def mock_sae_manager(mock_config: Config) -> SAEManager:
    with (
        patch("neuronpedia_inference.sae_manager.SaeLensSAE") as mock_sae_lens,
        patch(
            "neuronpedia_inference.sae_manager.Config.get_instance",
            return_value=mock_config,
        ),
    ):
        mock_sae_lens.load.return_value = (MagicMock(), "mock_hook")
        sae_manager = SAEManager(
            num_layers=12,
            device="cpu",
        )
        sae_manager.load_saes()  # Call load_saes explicitly
        return sae_manager


def test_lru_loading_up_to_limit(mock_sae_manager: SAEManager) -> None:
    for i in range(3):
        mock_sae_manager.get_sae(f"{i}-res-jb")

    assert len(mock_sae_manager.loaded_saes) == 3
    assert list(mock_sae_manager.loaded_saes.keys()) == [
        "0-res-jb",
        "1-res-jb",
        "2-res-jb",
    ]


def test_lru_unloading_least_recently_used(mock_sae_manager: SAEManager) -> None:
    for i in range(4):
        mock_sae_manager.get_sae(f"{i}-res-jb")

    assert len(mock_sae_manager.loaded_saes) == 3
    assert list(mock_sae_manager.loaded_saes.keys()) == [
        "1-res-jb",
        "2-res-jb",
        "3-res-jb",
    ]


def test_lru_accessing_loaded_sae(mock_sae_manager: SAEManager) -> None:
    for i in range(3):
        mock_sae_manager.get_sae(f"{i}-res-jb")

    mock_sae_manager.get_sae("1-res-jb")

    assert list(mock_sae_manager.loaded_saes.keys()) == [
        "0-res-jb",
        "2-res-jb",
        "1-res-jb",
    ]


def test_lru_order_reflects_usage(mock_sae_manager: SAEManager) -> None:
    for i in range(3):
        mock_sae_manager.get_sae(f"{i}-res-jb")

    mock_sae_manager.get_sae("0-res-jb")
    mock_sae_manager.get_sae("2-res-jb")

    assert list(mock_sae_manager.loaded_saes.keys()) == [
        "1-res-jb",
        "0-res-jb",
        "2-res-jb",
    ]


def test_lru_unload_and_reload(mock_sae_manager: SAEManager) -> None:
    for i in range(3):
        mock_sae_manager.get_sae(f"{i}-res-jb")

    mock_sae_manager.unload_sae("1-res-jb")
    assert "1-res-jb" not in mock_sae_manager.loaded_saes

    mock_sae_manager.get_sae("1-res-jb")
    assert list(mock_sae_manager.loaded_saes.keys()) == [
        "0-res-jb",
        "2-res-jb",
        "1-res-jb",
    ]


def test_lru_stress_test(mock_sae_manager: SAEManager) -> None:
    # Simulate a more complex usage pattern
    access_pattern = [0, 1, 2, 3, 1, 4, 0, 2, 1, 3]
    for i in access_pattern:
        mock_sae_manager.get_sae(f"{i}-res-jb")

    assert len(mock_sae_manager.loaded_saes) == 3
    assert list(mock_sae_manager.loaded_saes.keys()) == [
        "2-res-jb",
        "1-res-jb",
        "3-res-jb",
    ]
