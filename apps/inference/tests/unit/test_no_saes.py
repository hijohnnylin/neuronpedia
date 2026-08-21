"""A server configured with no SAE sets must start and stay useful.

Pods are brought up for things that have nothing to do with SAEs -- raw residual capture,
the lens endpoints, steering by vector -- and paying for the SAELens directory (a network
read) plus a set of weights they never touch is pure cost. These pin the "no SAEs" path:
nothing is loaded, nothing is looked up, and the derived limits behave as if the SAE terms
were simply absent.
"""

from unittest.mock import patch

from neuronpedia_inference.config import Config
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.startup_memory import compute_activation_token_limit


def build_config(**overrides: object) -> Config:
    settings: dict[str, object] = {
        "model_id": "gpt2-small",
        "sae_sets": [],
        "model_dtype": "float16",
        "device": "cpu",
        "token_limit": 200,
    }
    settings.update(overrides)
    return Config(**settings)  # type: ignore[arg-type]


def test_empty_sae_sets_never_reads_the_saelens_directory():
    with patch("neuronpedia_inference.config.get_saelens_neuronpedia_directory_df") as directory:
        config = build_config()

    directory.assert_not_called()
    assert config.sae_config == []


def test_the_configured_model_is_still_recognised_without_saes():
    # get_valid_model_ids draws partly on the SAE config; with none, the configured ids
    # have to carry it alone, or nothing would be recognised at all.
    config = build_config(model_id="openai-community/gpt2")
    assert "openai-community/gpt2" in config.get_valid_model_ids()


def test_an_unknown_model_is_logged_rather_than_rejected():
    config = build_config()
    # No return value and no raise: the only contract is that it does not reject.
    assert config.check_requested_model("something/else") is None
    assert config.check_requested_model(None) is None


def test_load_saes_is_a_no_op_but_still_registers_neuron_layers():
    config = build_config()
    with patch.object(Config, "get_instance", return_value=config):
        manager = SAEManager(num_layers=12, device="cpu")
        manager.config = config
        manager.load_saes()

    assert manager.loaded_saes == {}
    assert manager.get_valid_sae_sets() == []
    # setup_neuron_layers still runs, so the neurons source set is described even though
    # the engine backends do not serve it.
    assert len(manager.sae_set_to_saes[SAEManager.NEURONS_SOURCESET]) == 12
    assert manager.widest_activation_dims() == (0, 0, 0)


def test_activation_token_limit_is_untouched_when_there_are_no_saes():
    # The cap exists to bound an SAE encode. With no SAE there is nothing to bound, so the
    # activation endpoints keep the full token limit instead of being throttled by a
    # zero-width formula.
    assert (
        compute_activation_token_limit(
            budget_bytes=8 * 1024**3,
            token_limit=2048,
            d_sae=0,
            d_in=0,
            n_hooks=0,
            sae_dtype="bfloat16",
            model_dtype="bfloat16",
        )
        == 2048
    )
