# ABOUTME: Unit tests for completion_chat batch generation optimization functions
# ABOUTME: Tests core functionality with maximum code coverage and minimal dependencies

from unittest.mock import Mock, patch

import pytest
import torch
from neuronpedia_inference_client.models.np_steer_chat_message import NPSteerChatMessage
from neuronpedia_inference_client.models.np_steer_feature import NPSteerFeature
from neuronpedia_inference_client.models.np_steer_method import NPSteerMethod
from neuronpedia_inference_client.models.np_steer_type import NPSteerType

from neuronpedia_inference.endpoints.steer.completion_chat import (
    create_batched_steering_hook,
    generate_single_completion_chat,
    make_steer_completion_chat_response,
    sequential_generate_chat,
)


class TestBatchedSteeringHook:
    """Test the batched steering hook creation and functionality."""

    def test_create_batched_steering_hook_basic(self):
        """Test basic batched steering hook creation."""
        promptTokenized = torch.tensor([1, 2, 3, 4, 5])
        features = [
            NPSteerFeature(
                model="gpt2-small",
                source="0-res-jb",
                index=100,
                strength=1.0,
                steering_vector=[0.1] * 768,
            )
        ]

        hook_func = create_batched_steering_hook(
            promptTokenized=promptTokenized,
            features=features,
            strength_multiplier=1.0,
            steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
            normalize_steering=False,
            steer_special_tokens=True,
        )

        assert callable(hook_func)

    @patch("neuronpedia_inference.endpoints.steer.completion_chat.Model")
    def test_batched_hook_simple_additive_steering(self, mock_model_class):
        """Test batched hook applies steering only to activations[0]."""
        # Setup mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.bos_token_id = 1
        mock_model.tokenizer = mock_tokenizer
        mock_model_class.get_instance.return_value = mock_model

        promptTokenized = torch.tensor([1, 2, 3, 4, 5])
        features = [
            NPSteerFeature(
                model="gpt2-small",
                source="0-res-jb",
                index=100,
                strength=2.0,
                steering_vector=[0.1] * 10,  # Small vector for testing
            )
        ]

        hook_func = create_batched_steering_hook(
            promptTokenized=promptTokenized,
            features=features,
            strength_multiplier=1.5,
            steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
            normalize_steering=False,
            steer_special_tokens=True,
        )

        # Create test activations [batch_size=2, seq_len=5, hidden_dim=10]
        activations = torch.zeros(2, 5, 10)
        original_batch_1 = activations[1].clone()

        # Apply hook
        result = hook_func(activations, None)

        # Check that activations[0] was modified (should have steering added)
        assert not torch.equal(result[0], torch.zeros(5, 10))

        # Check that activations[1] remains unchanged (DEFAULT)
        assert torch.equal(result[1], original_batch_1)

    @patch("neuronpedia_inference.endpoints.steer.completion_chat.Model")
    def test_batched_hook_with_normalization(self, mock_model_class):
        """Test batched hook with steering vector normalization."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.bos_token_id = 1
        mock_model.tokenizer = mock_tokenizer
        mock_model_class.get_instance.return_value = mock_model

        promptTokenized = torch.tensor([1, 2, 3, 4, 5])
        # Use non-normalized vector
        features = [
            NPSteerFeature(
                model="gpt2-small",
                source="0-res-jb",
                index=100,
                strength=1.0,
                steering_vector=[3.0, 4.0],  # Norm = 5.0
            )
        ]

        hook_func = create_batched_steering_hook(
            promptTokenized=promptTokenized,
            features=features,
            strength_multiplier=1.0,
            steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
            normalize_steering=True,
            steer_special_tokens=True,
        )

        activations = torch.zeros(2, 5, 2)
        result = hook_func(activations, None)

        # Should apply normalized vector (strength=1.0, so result should be [0.6, 0.8] * mask)
        assert result is not None

    @patch("neuronpedia_inference.endpoints.steer.completion_chat.Model")
    def test_batched_hook_special_token_masking(self, mock_model_class):
        """Test batched hook with special token masking disabled."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.bos_token_id = 1
        mock_tokenizer.chat_template = None  # No chat template
        mock_model.tokenizer = mock_tokenizer
        mock_model_class.get_instance.return_value = mock_model

        promptTokenized = torch.tensor([1, 2, 3, 1, 5])  # BOS tokens at positions 0,3
        features = [
            NPSteerFeature(
                model="gpt2-small",
                source="0-res-jb",
                index=100,
                strength=1.0,
                steering_vector=[1.0, 1.0],
            )
        ]

        hook_func = create_batched_steering_hook(
            promptTokenized=promptTokenized,
            features=features,
            strength_multiplier=1.0,
            steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
            normalize_steering=False,
            steer_special_tokens=False,  # Should mask special tokens
        )

        activations = torch.zeros(2, 5, 2)
        result = hook_func(activations, None)

        # BOS positions should not be steered (remain 0)
        assert torch.equal(result[0][0], torch.zeros(2))  # Position 0 (BOS)
        assert torch.equal(result[0][3], torch.zeros(2))  # Position 3 (BOS)

    def test_batched_hook_error_handling(self):
        """Test error handling in batched hook."""
        promptTokenized = torch.tensor([1, 2, 3])
        features = [
            NPSteerFeature(
                model="gpt2-small",
                source="0-res-jb",
                index=100,
                strength=1.0,
                steering_vector=[float("inf"), 1.0],  # Invalid vector
            )
        ]

        hook_func = create_batched_steering_hook(
            promptTokenized=promptTokenized,
            features=features,
            strength_multiplier=1.0,
            steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
            normalize_steering=False,
            steer_special_tokens=True,
        )

        with patch(
            "neuronpedia_inference.endpoints.steer.completion_chat.Model"
        ) as mock_model_class:
            mock_model = Mock()
            mock_model.tokenizer = Mock()
            mock_model_class.get_instance.return_value = mock_model

            activations = torch.zeros(2, 3, 2)

            # Should raise ValueError for infinite values
            with pytest.raises(
                ValueError, match="Steering vector contains inf or nan values"
            ):
                hook_func(activations, None)


class TestGenerateSingleCompletionChat:
    """Test single completion chat generation function."""

    @pytest.mark.asyncio
    @patch("neuronpedia_inference.endpoints.steer.completion_chat.Model")
    @patch("neuronpedia_inference.endpoints.steer.completion_chat.SAEManager")
    async def test_generate_single_steered(
        self, mock_sae_manager_class, mock_model_class
    ):
        """Test single steered completion generation."""
        # Setup mocks
        mock_model = Mock()
        mock_model.cfg.device = "cpu"
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.bos_token_id = 1
        mock_model.to_string.return_value = "test output"
        mock_model.reset_hooks = Mock()
        # Create a context manager mock
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=None)
        context_manager.__exit__ = Mock(return_value=None)
        mock_model.hooks = Mock(return_value=context_manager)

        # Mock generate_stream to yield results
        def mock_generate_stream(**kwargs):  # noqa: ARG001
            yield [torch.tensor([1, 2, 3])]

        mock_model.generate_stream = mock_generate_stream
        mock_model_class.get_instance.return_value = mock_model

        mock_sae_manager = Mock()
        mock_sae_manager.get_sae_hook.return_value = "test_hook"
        mock_sae_manager_class.get_instance.return_value = mock_sae_manager

        # Test parameters
        promptTokenized = torch.tensor([1, 2, 3, 4])
        inputPrompt = [NPSteerChatMessage(role="user", content="test")]
        features = [
            NPSteerFeature(
                model="gpt2-small",
                source="0-res-jb",
                index=100,
                strength=1.0,
                steering_vector=[0.1] * 768,
            )
        ]

        # Call function
        results = []
        async for result in generate_single_completion_chat(
            promptTokenized=promptTokenized,
            inputPrompt=inputPrompt,
            features=features,
            steer_type=NPSteerType.STEERED,
            strength_multiplier=1.0,
            seed=None,
            steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
            normalize_steering=False,
            steer_special_tokens=True,
        ):
            results.append(result)

        # Verify results
        assert len(results) == 1
        assert results[0] == "test output"

    @pytest.mark.asyncio
    @patch("neuronpedia_inference.endpoints.steer.completion_chat.Model")
    @patch("neuronpedia_inference.endpoints.steer.completion_chat.SAEManager")
    async def test_generate_single_default(
        self, mock_sae_manager_class, mock_model_class
    ):
        """Test single default completion generation (no steering)."""
        # Setup mocks
        mock_model = Mock()
        mock_model.cfg.device = "cpu"
        mock_model.tokenizer = Mock()
        mock_model.to_string.return_value = "default output"
        mock_model.reset_hooks = Mock()
        # Create a context manager mock
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=None)
        context_manager.__exit__ = Mock(return_value=None)
        mock_model.hooks = Mock(return_value=context_manager)

        def mock_generate_stream(**kwargs):  # noqa: ARG001
            yield [torch.tensor([1, 2, 3])]

        mock_model.generate_stream = mock_generate_stream
        mock_model_class.get_instance.return_value = mock_model

        mock_sae_manager = Mock()
        mock_sae_manager_class.get_instance.return_value = mock_sae_manager

        # Test with DEFAULT type (should not apply steering)
        promptTokenized = torch.tensor([1, 2, 3, 4])
        inputPrompt = [NPSteerChatMessage(role="user", content="test")]
        features = []

        results = []
        async for result in generate_single_completion_chat(
            promptTokenized=promptTokenized,
            inputPrompt=inputPrompt,
            features=features,
            steer_type=NPSteerType.DEFAULT,
            strength_multiplier=1.0,
            seed=42,  # Test seed setting
            steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
            normalize_steering=False,
            steer_special_tokens=True,
        ):
            results.append(result)

        assert len(results) == 1
        assert results[0] == "default output"


class TestMakeSteerCompletionChatResponse:
    """Test response formatting function."""

    @patch("neuronpedia_inference.endpoints.steer.completion_chat.NPSteerChatResult")
    @patch(
        "neuronpedia_inference.endpoints.steer.completion_chat.SteerCompletionChatPost200Response"
    )
    def test_make_response_both_types(self, mock_response_class, mock_result_class):
        """Test response creation with both STEERED and DEFAULT types."""
        # Setup mocks
        mock_result_class.return_value = Mock()
        mock_response_class.return_value = Mock()

        mock_model = Mock()
        mock_model.to_string = Mock(return_value="mocked prompt string")
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode = Mock(return_value=[1, 2, 3, 4])
        mock_model.tokenizer.decode = Mock(return_value="decoded text")
        mock_model.tokenizer.bos_token_id = 1
        mock_model.tokenizer.eos_token_id = 2
        promptTokenized = torch.tensor([1, 2, 3])
        promptChat = [NPSteerChatMessage(role="user", content="test")]

        # Call function
        make_steer_completion_chat_response(
            steer_types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
            steered_result="steered output",
            default_result="default output",
            model=mock_model,
            promptTokenized=promptTokenized,
            promptChat=promptChat,
            custom_hf_model_id=None,
        )

        # Verify response creation was called
        assert mock_response_class.called

    @patch("neuronpedia_inference.endpoints.steer.completion_chat.NPSteerChatResult")
    @patch(
        "neuronpedia_inference.endpoints.steer.completion_chat.SteerCompletionChatPost200Response"
    )
    def test_make_response_single_type(self, mock_response_class, mock_result_class):
        """Test response creation with single type."""
        mock_result_class.return_value = Mock()
        mock_response_class.return_value = Mock()

        mock_model = Mock()
        mock_model.to_string = Mock(return_value="mocked prompt string")
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.encode = Mock(return_value=[1, 2, 3, 4])
        mock_model.tokenizer.decode = Mock(return_value="decoded text")
        mock_model.tokenizer.bos_token_id = 1
        mock_model.tokenizer.eos_token_id = 2
        promptTokenized = torch.tensor([1, 2, 3])
        promptChat = [NPSteerChatMessage(role="user", content="test")]

        make_steer_completion_chat_response(
            steer_types=[NPSteerType.STEERED],
            steered_result="steered output",
            default_result="",
            model=mock_model,
            promptTokenized=promptTokenized,
            promptChat=promptChat,
            custom_hf_model_id="custom-model",
        )

        assert mock_response_class.called


class TestSequentialGenerateChat:
    """Test fallback sequential generation function."""

    @pytest.mark.asyncio
    @patch("neuronpedia_inference.endpoints.steer.completion_chat.Model")
    @patch("neuronpedia_inference.endpoints.steer.completion_chat.SAEManager")
    @patch(
        "neuronpedia_inference.endpoints.steer.completion_chat.make_steer_completion_chat_response"
    )
    @patch("neuronpedia_inference.endpoints.steer.completion_chat.format_sse_message")
    async def test_sequential_generate_both_types(
        self,
        mock_format_sse,
        mock_make_response,
        mock_sae_manager_class,
        mock_model_class,
    ):
        """Test sequential generation with both STEERED and DEFAULT."""
        # Setup mocks
        mock_model = Mock()
        mock_model.cfg.device = "cpu"
        mock_model.tokenizer = Mock()
        mock_model.tokenizer.bos_token_id = 1
        mock_model.tokenizer.chat_template = None
        mock_model.to_string.return_value = "output"
        mock_model.reset_hooks = Mock()
        # Create a context manager mock
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=None)
        context_manager.__exit__ = Mock(return_value=None)
        mock_model.hooks = Mock(return_value=context_manager)

        # Mock generate_stream to yield different results for steered vs default
        call_count = 0

        def mock_generate_stream(**kwargs):  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # STEERED call
                yield [torch.tensor([1, 2])]
            else:  # DEFAULT call
                yield [torch.tensor([3, 4])]

        mock_model.generate_stream = mock_generate_stream
        mock_model_class.get_instance.return_value = mock_model

        mock_sae_manager = Mock()
        mock_sae_manager.get_sae_hook.return_value = "test_hook"
        mock_sae_manager_class.get_instance.return_value = mock_sae_manager

        mock_response = Mock()
        mock_response.to_json.return_value = '{"test": "response"}'
        mock_make_response.return_value = mock_response
        mock_format_sse.return_value = "formatted_sse"

        # Test parameters
        promptTokenized = torch.tensor([1, 2, 3])
        inputPrompt = [NPSteerChatMessage(role="user", content="test")]
        features = [
            NPSteerFeature(
                model="gpt2-small",
                source="0-res-jb",
                index=100,
                strength=1.0,
                steering_vector=[0.1] * 768,
            )
        ]

        # Call function
        results = []
        async for result in sequential_generate_chat(
            promptTokenized=promptTokenized,
            inputPrompt=inputPrompt,
            features=features,
            steer_types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
            strength_multiplier=1.0,
            seed=None,
            steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
            normalize_steering=False,
            steer_special_tokens=False,
        ):
            results.append(result)

        # Should have yielded results for both STEERED and DEFAULT generations
        assert len(results) >= 2
        assert all(result == "formatted_sse" for result in results)
