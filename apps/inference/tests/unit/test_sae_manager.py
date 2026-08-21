"""Tests for SAEManager: which SAEs it knows about, and which it keeps resident.

Unlike test_config.py these build a real Config, so the SAE config really is resolved
against the SAELens directory; only the SAE weights themselves are stubbed. That keeps
the ids under test honest while avoiding any actual downloads.

The residency tests drive get_sae/unload_sae and assert on the order of loaded_saes,
whose last entry is the most recently used and whose first is the next to be evicted.
"""

from collections import OrderedDict
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sae_lens.saes.sae import SAE

from neuronpedia_inference.config import Config
from neuronpedia_inference.sae_manager import SAEManager

MAX_LOADED_SAES = 3


class StubSAE:
    """Stands in for a loaded SAE, with just the surface the manager touches."""

    def __init__(self):
        self.cfg = MagicMock()
        self.cfg.hook_name = "blocks.5.hook_resid_pre"
        self.cfg.neuronpedia_id = "mock-neuronpedia-id"

    def to(self, *args: Any, **kwargs: Any):  # type: ignore # noqa: ARG002
        # Mirror torch's flexible .to(): production SaeLensSAE.load calls .to(device)
        # with a single positional arg.
        return self

    def fold_W_dec_norm(self):
        pass

    def eval(self):
        pass


def build_config(**overrides: Any) -> Config:
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
    return Config(**settings)


def make_stub_sae(*args: Any, **kwargs: Any) -> StubSAE:  # noqa: ARG001
    return StubSAE()


def build_manager_over_real_loader(config: Config) -> SAEManager:
    """A manager built through the real SaeLensSAE.load, with only weights stubbed.

    Patching at SAE.from_pretrained rather than at SaeLensSAE leaves the adapter's own
    load path running -- dtype handling, hook naming, the fold and eval calls -- so
    these tests still cover it.
    """
    with (
        patch.object(SAE, "from_pretrained", new=make_stub_sae),
        patch(
            "neuronpedia_inference.sae_manager.Config.get_instance",
            return_value=config,
        ),
    ):
        manager = SAEManager(num_layers=12, device="cpu")
        manager.load_saes()
        return manager


def build_manager(config: Config) -> SAEManager:
    """A manager over `config` with SAE loading skipped entirely.

    Residency is about which ids are held, not what they hold, so the loader is stubbed
    wholesale here to keep those tests fast.
    """
    with (
        patch("neuronpedia_inference.sae_manager.SaeLensSAE") as mock_sae_lens,
        patch(
            "neuronpedia_inference.sae_manager.Config.get_instance",
            return_value=config,
        ),
    ):
        mock_sae_lens.load.return_value = (MagicMock(), "mock_hook")
        manager = SAEManager(num_layers=12, device="cpu")
        manager.load_saes()
        return manager


@pytest.fixture
def manager() -> SAEManager:
    return build_manager(build_config(include_sae=[r"^5-res-jb"], max_loaded_saes=MAX_LOADED_SAES))


def resident(manager: SAEManager) -> list[str]:
    """Resident SAE ids, least recently used first."""
    return list(manager.loaded_saes.keys())


class TestLoadSaes:
    def test_registers_one_entry_per_layer_plus_the_selected_sae(self):
        manager = build_manager_over_real_loader(
            build_config(include_sae=[r"^5-res-jb"], max_loaded_saes=MAX_LOADED_SAES)
        )

        assert len(manager.sae_data) == 13  # 12 layers + 1 selected SAE

    def test_resolves_aliases_for_an_overridden_model(self):
        config = build_config(
            model_id="gemma-2-2b",
            sae_sets=["gemmascope-mlp-16k"],
            token_limit=2048,
            port=5000,
            override_model_id="gemma-2-2b-it",
            include_sae=[r"^1-gemmascope-mlp-16k$"],
        )
        manager = build_manager_over_real_loader(config)

        assert "1-gemmascope-mlp-16k" in manager.sae_data
        assert isinstance(manager.sae_data["1-gemmascope-mlp-16k"]["sae"], StubSAE)
        # The SAELens id, the override, and the HF spelling of each (np_model_to_hf.json)
        # all have to be accepted, or requests naming the model the other way get a 400.
        assert manager.config.get_valid_model_ids() == {
            "gemma-2-2b",
            "google/gemma-2-2b",
            "gemma-2-2b-it",
            "google/gemma-2-2b-it",
        }


class TestResidency:
    @pytest.mark.parametrize(
        ("accesses", "expected"),
        [
            pytest.param([0, 1, 2], [0, 1, 2], id="fills-up-to-the-limit"),
            pytest.param([0, 1, 2, 3], [1, 2, 3], id="evicts-the-least-recent"),
            pytest.param([0, 1, 2, 1], [0, 2, 1], id="rereading-promotes-to-newest"),
            pytest.param([0, 1, 2, 0, 2], [1, 0, 2], id="order-tracks-usage"),
            pytest.param(
                [0, 1, 2, 3, 1, 4, 0, 2, 1, 3],
                [2, 1, 3],
                id="survives-a-longer-access-pattern",
            ),
        ],
    )
    def test_eviction_order(self, manager: SAEManager, accesses: list[int], expected: list[int]):
        for i in accesses:
            manager.get_sae(f"{i}-res-jb")

        assert resident(manager) == [f"{i}-res-jb" for i in expected]
        assert len(manager.loaded_saes) <= MAX_LOADED_SAES

    def test_never_exceeds_the_limit_midway(self, manager: SAEManager):
        for i in range(6):
            manager.get_sae(f"{i}-res-jb")
            assert len(manager.loaded_saes) <= MAX_LOADED_SAES

    def test_unloaded_sae_comes_back_as_the_newest(self, manager: SAEManager):
        for i in range(3):
            manager.get_sae(f"{i}-res-jb")

        manager.unload_sae("1-res-jb")
        assert "1-res-jb" not in manager.loaded_saes

        manager.get_sae("1-res-jb")
        assert resident(manager) == ["0-res-jb", "2-res-jb", "1-res-jb"]

    def test_residency_is_ordered(self, manager: SAEManager):
        # The eviction assertions above are only meaningful because this is ordered.
        manager.get_sae("0-res-jb")
        assert isinstance(manager.loaded_saes, OrderedDict)
