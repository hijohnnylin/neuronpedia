import torch

from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.shared import Model
from tests.conftest import SAE_SELECTED_SOURCES, TEST_PROMPT


def test_initialize(initialize_models: None):  # noqa: ARG001
    """
    Test that the model and SAE are properly initialized when using the /initialize endpoint.
    """
    # Check that the model is loaded
    model = Model.get_instance()
    assert model is not None

    # Check that the SAE is loaded
    sae_manager = SAEManager.get_instance()
    assert sae_manager is not None
    expected_sae_source = SAE_SELECTED_SOURCES[0]
    assert expected_sae_source in sae_manager.sae_data, (
        f"Expected SAE source '{expected_sae_source}' not found in loaded SAEs. "
        f"Loaded SAEs: {list(sae_manager.sae_data.keys())}"
    )
    sae = sae_manager.sae_data[expected_sae_source]["sae"]
    assert sae is not None

    # Test a simple forward pass
    tokens = model.to_tokens(TEST_PROMPT)
    with torch.no_grad():
        logits = model(tokens)
    assert logits is not None
    assert logits.shape[0] == 1  # batch size of 1
    assert logits.shape[1] == len(tokens[0])  # sequence length
    assert logits.shape[2] == model.cfg.d_vocab  # vocabulary size
