import torch
from interp_engine import EagerModel

from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.shared import Model
from tests.conftest import TEST_PROMPT


def test_initialize(initialize_models: None):  # noqa: ARG001
    """
    Test that the model and SAE are properly initialized when using the /initialize endpoint.
    """
    # Check that the model is loaded (the session fixture forces the EagerModel backend).
    model = Model.get_instance()
    assert isinstance(model, EagerModel)

    # Check that the SAE is loaded
    sae_manager = SAEManager.get_instance()
    assert sae_manager is not None
    assert "7-res-jb" in sae_manager.sae_data
    sae = sae_manager.sae_data["7-res-jb"]["sae"]
    assert sae is not None

    # Test a simple forward pass through the raw HF model the engine wraps.
    tokens = model.to_tokens(TEST_PROMPT)
    with torch.no_grad():
        logits = model.hf_model(tokens).logits
    assert logits is not None
    assert logits.shape[0] == 1  # batch size of 1
    assert logits.shape[1] == len(tokens[0])  # sequence length
    assert logits.shape[2] == model.vocab_size  # vocabulary size
