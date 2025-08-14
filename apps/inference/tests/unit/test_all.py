from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from neuronpedia_inference_client.models.activation_all_post200_response import (
    ActivationAllPost200Response,
)
from neuronpedia_inference_client.models.activation_all_post_request import (
    ActivationAllPostRequest,
)
from sae_lens import SAE
from transformer_lens import ActivationCache, HookedTransformer

from neuronpedia_inference.endpoints.activation.all import ActivationProcessor


@pytest.fixture
def mock_model() -> Mock:
    """Mock transformer model for testing."""
    model = Mock(spec=HookedTransformer)
    model.cfg = Mock()
    model.cfg.n_layers = 4
    model.cfg.default_prepend_bos = True
    model.to_tokens.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    model.to_str_tokens.return_value = ["<BOS>", "hello", "world", "test", "<EOS>"]

    # Mock cache data
    cache_data = {}
    cache_data["blocks.0.hook_mlp_out"] = torch.randn(1, 5, 64)
    cache_data["blocks.1.hook_mlp_out"] = torch.randn(1, 5, 64)
    # Add cache data for all layers that might be accessed
    for layer in range(4):  # n_layers = 4
        cache_data["v", layer] = torch.randn(1, 5, 4, 16)
        cache_data["pattern", layer] = torch.randn(1, 4, 5, 5)

    cache = Mock(spec=ActivationCache)
    # cache.__getitem__ = lambda self, key: cache_data[key]
    cache.__getitem__ = lambda _, key: cache_data[key]

    model.run_with_cache.return_value = (None, cache)
    return model


@pytest.fixture
def mock_sae() -> Mock:
    """Mock SAE for testing."""
    sae = Mock(spec=SAE)
    sae.cfg = Mock()
    sae.cfg.prepend_bos = True
    sae.encode.return_value = torch.randn(1, 5, 128)
    return sae


@pytest.fixture
def mock_sae_manager() -> Mock:
    """Mock SAE manager for testing."""
    manager = Mock()
    # Create a more flexible mock SAE that allows attribute access
    mock_sae = Mock()
    mock_sae.cfg = Mock()
    mock_sae.cfg.prepend_bos = True
    mock_sae.encode.return_value = torch.randn(1, 5, 128)

    manager.get_sae.return_value = mock_sae
    manager.get_sae_hook.return_value = "blocks.0.hook_mlp_out"
    manager.get_sae_type.return_value = "features"
    manager.is_dfa_enabled.return_value = False
    manager.sae_set_to_saes = {"test_set": ["0-test_set", "1-test_set"]}
    return manager


@pytest.fixture
def mock_config() -> Mock:
    """Mock config for testing."""
    config = Mock()
    config.device = "cpu"
    config.token_limit = 1000
    return config


@pytest.fixture
def sample_request() -> ActivationAllPostRequest:
    """Sample activation request for testing."""
    return ActivationAllPostRequest(
        model="test_model",
        prompt="hello world test",
        source_set="test_set",
        selected_sources=["0-test_set", "1-test_set"],
        num_results=10,
        sort_by_token_indexes=[],
        ignore_bos=False,
        feature_filter=None,
    )


@pytest.fixture
def processor() -> ActivationProcessor:
    """ActivationProcessor instance for testing."""
    return ActivationProcessor()


@patch("neuronpedia_inference.endpoints.activation.all.get_activations_by_index")
@patch("neuronpedia_inference.endpoints.activation.all.Model")
@patch("neuronpedia_inference.endpoints.activation.all.SAEManager")
@patch("neuronpedia_inference.endpoints.activation.all.Config")
def test_process_activations_basic(
    mock_config_class: MagicMock,
    mock_sae_manager_class: MagicMock,
    mock_model_class: MagicMock,
    mock_get_activations: MagicMock,
    processor: ActivationProcessor,
    sample_request: ActivationAllPostRequest,
    mock_model: Mock,
    mock_sae_manager: Mock,
    mock_config: Mock,
):
    """Test basic process_activations functionality."""
    # Setup mocks
    mock_model_class.get_instance.return_value = mock_model
    mock_sae_manager_class.get_instance.return_value = mock_sae_manager
    mock_config_class.get_instance.return_value = mock_config
    mock_get_activations.return_value = torch.randn(128, 5)
    # Execute
    result = processor.process_activations(sample_request)
    # Verify result structure
    assert isinstance(result, ActivationAllPost200Response)
    assert hasattr(result, "activations")
    assert hasattr(result, "tokens")
    assert hasattr(result, "counts")
    assert isinstance(result.tokens, list)
    assert isinstance(result.counts, list)
    # Verify model interactions
    mock_model.to_tokens.assert_called_once()
    mock_model.to_str_tokens.assert_called_once()
    mock_model.run_with_cache.assert_called_once()


@patch("neuronpedia_inference.endpoints.activation.all.get_activations_by_index")
@patch("neuronpedia_inference.endpoints.activation.all.Model")
@patch("neuronpedia_inference.endpoints.activation.all.SAEManager")
@patch("neuronpedia_inference.endpoints.activation.all.Config")
def test_process_activations_with_sort_by_token_indexes(
    mock_config_class: MagicMock,
    mock_sae_manager_class: MagicMock,
    mock_model_class: MagicMock,
    mock_get_activations: MagicMock,
    processor: ActivationProcessor,
    sample_request: ActivationAllPostRequest,
    mock_model: Mock,
    mock_sae_manager: Mock,
    mock_config: Mock,
):
    """Test process_activations with sort_by_token_indexes."""
    # Setup mocks
    mock_model_class.get_instance.return_value = mock_model
    mock_sae_manager_class.get_instance.return_value = mock_sae_manager
    mock_config_class.get_instance.return_value = mock_config
    mock_get_activations.return_value = torch.randn(128, 5)
    # Modify request to include sort_by_token_indexes
    sample_request.sort_by_token_indexes = [1, 2, 3]
    # Execute
    result = processor.process_activations(sample_request)
    # Verify result
    assert isinstance(result, ActivationAllPost200Response)
    assert len(result.tokens) == 5  # Should match mock token length


@patch("neuronpedia_inference.endpoints.activation.all.get_activations_by_index")
@patch("neuronpedia_inference.endpoints.activation.all.Model")
@patch("neuronpedia_inference.endpoints.activation.all.SAEManager")
@patch("neuronpedia_inference.endpoints.activation.all.Config")
def test_process_activations_with_feature_filter(
    mock_config_class: MagicMock,
    mock_sae_manager_class: MagicMock,
    mock_model_class: MagicMock,
    mock_get_activations: MagicMock,
    processor: ActivationProcessor,
    sample_request: ActivationAllPostRequest,
    mock_model: Mock,
    mock_sae_manager: Mock,
    mock_config: Mock,
):
    """Test process_activations with feature filter."""
    # Setup mocks
    mock_model_class.get_instance.return_value = mock_model
    mock_sae_manager_class.get_instance.return_value = mock_sae_manager
    mock_config_class.get_instance.return_value = mock_config
    mock_get_activations.return_value = torch.randn(128, 5)
    # Modify request for single layer with feature filter
    sample_request.selected_sources = ["0-test_set"]
    sample_request.feature_filter = [0, 1, 5, 10]
    # Execute
    result = processor.process_activations(sample_request)
    # Verify result
    assert isinstance(result, ActivationAllPost200Response)


@patch("neuronpedia_inference.endpoints.activation.all.Model")
@patch("neuronpedia_inference.endpoints.activation.all.SAEManager")
@patch("neuronpedia_inference.endpoints.activation.all.Config")
def test_process_activations_with_neurons_sae_type(
    mock_config_class: MagicMock,
    mock_sae_manager_class: MagicMock,
    mock_model_class: MagicMock,
    processor: ActivationProcessor,
    sample_request: ActivationAllPostRequest,
    mock_model: Mock,
    mock_sae_manager: Mock,
    mock_config: Mock,
):
    """Test process_activations with neurons SAE type."""
    # Setup mocks
    mock_model_class.get_instance.return_value = mock_model
    mock_sae_manager_class.get_instance.return_value = mock_sae_manager
    mock_config_class.get_instance.return_value = mock_config
    # Configure SAE manager to return neurons type
    mock_sae_manager.get_sae_type.return_value = "neurons"
    # Execute
    result = processor.process_activations(sample_request)
    # Verify result
    assert isinstance(result, ActivationAllPost200Response)


@patch("neuronpedia_inference.endpoints.activation.all.get_activations_by_index")
@patch("neuronpedia_inference.endpoints.activation.all.Model")
@patch("neuronpedia_inference.endpoints.activation.all.SAEManager")
@patch("neuronpedia_inference.endpoints.activation.all.Config")
def test_process_activations_with_dfa_enabled(
    mock_config_class: MagicMock,
    mock_sae_manager_class: MagicMock,
    mock_model_class: MagicMock,
    mock_get_activations: MagicMock,
    processor: ActivationProcessor,
    sample_request: ActivationAllPostRequest,
    mock_model: Mock,
    mock_sae_manager: Mock,
    mock_config: Mock,
):
    """Test process_activations with DFA calculation enabled."""
    # Setup mocks
    mock_model_class.get_instance.return_value = mock_model
    mock_sae_manager_class.get_instance.return_value = mock_sae_manager
    mock_config_class.get_instance.return_value = mock_config
    mock_get_activations.return_value = torch.randn(128, 5)
    # Enable DFA
    mock_sae_manager.is_dfa_enabled.return_value = True
    # Mock calculate_per_source_dfa function
    with patch(
        "neuronpedia_inference.endpoints.activation.all.calculate_per_source_dfa"
    ) as mock_dfa:
        # Return a 2D tensor where first dimension can be indexed
        mock_dfa.return_value = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        # Execute
        result = processor.process_activations(sample_request)
        # Verify result
        assert isinstance(result, ActivationAllPost200Response)
        # Check that DFA was called
        mock_dfa.assert_called()


@patch("neuronpedia_inference.endpoints.activation.all.Model")
@patch("neuronpedia_inference.endpoints.activation.all.SAEManager")
@patch("neuronpedia_inference.endpoints.activation.all.Config")
def test_process_activations_invalid_token_index(
    mock_config_class: MagicMock,
    mock_sae_manager_class: MagicMock,
    mock_model_class: MagicMock,
    processor: ActivationProcessor,
    sample_request: ActivationAllPostRequest,
    mock_model: Mock,
    mock_sae_manager: Mock,
    mock_config: Mock,
):
    """Test process_activations raises error for invalid sort_by_token_indexes."""
    # Setup mocks
    mock_model_class.get_instance.return_value = mock_model
    mock_sae_manager_class.get_instance.return_value = mock_sae_manager
    mock_config_class.get_instance.return_value = mock_config
    # Set invalid token index (beyond token length)
    sample_request.sort_by_token_indexes = [10]  # tokens only go 0-4
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        processor.process_activations(sample_request)
    assert exc_info.value.status_code == 400
    assert "Sort by token index" in exc_info.value.detail
    assert "is out of range" in exc_info.value.detail


@patch("neuronpedia_inference.endpoints.activation.all.get_activations_by_index")
@patch("neuronpedia_inference.endpoints.activation.all.Model")
@patch("neuronpedia_inference.endpoints.activation.all.SAEManager")
@patch("neuronpedia_inference.endpoints.activation.all.Config")
def test_process_activations_ignore_bos(
    mock_config_class: MagicMock,
    mock_sae_manager_class: MagicMock,
    mock_model_class: MagicMock,
    mock_get_activations: MagicMock,
    processor: ActivationProcessor,
    sample_request: ActivationAllPostRequest,
    mock_model: Mock,
    mock_sae_manager: Mock,
    mock_config: Mock,
):
    """Test process_activations with ignore_bos=True."""
    # Setup mocks
    mock_model_class.get_instance.return_value = mock_model
    mock_sae_manager_class.get_instance.return_value = mock_sae_manager
    mock_config_class.get_instance.return_value = mock_config
    mock_get_activations.return_value = torch.randn(128, 5)
    sample_request.ignore_bos = True
    # Execute
    result = processor.process_activations(sample_request)
    # Verify result
    assert isinstance(result, ActivationAllPost200Response)
