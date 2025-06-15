# ABOUTME: Integration tests for completion_chat endpoint optimization functionality
# ABOUTME: Tests real API behavior with actual models to ensure optimization works correctly

import json

import pytest
from fastapi.testclient import TestClient
from neuronpedia_inference_client.models.np_steer_chat_message import NPSteerChatMessage
from neuronpedia_inference_client.models.np_steer_feature import NPSteerFeature
from neuronpedia_inference_client.models.np_steer_method import NPSteerMethod
from neuronpedia_inference_client.models.np_steer_type import NPSteerType
from neuronpedia_inference_client.models.steer_completion_chat_post_request import (
    SteerCompletionChatPostRequest,
)

from tests.conftest import X_SECRET_KEY


class TestCompletionChatOptimization:
    """Integration tests for completion_chat endpoint optimization."""

    def test_steer_completion_chat_both_types_basic(self, client: TestClient):
        """Test basic completion_chat with both STEERED and DEFAULT types."""
        # Basic chat with single feature
        messages = [
            NPSteerChatMessage(role="user", content="What is the weather like?")
        ]

        features = [
            NPSteerFeature(
                source="0-res-jb",  # Use layer 0 for gpt2-small
                index=100,
                strength=1.0,
                steering_vector=[0.1] * 768,  # GPT-2 small dimension
            )
        ]

        request = SteerCompletionChatPostRequest(
            messages=messages,
            features=features,
            types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
            strength_multiplier=1.0,
            steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
            normalize_steering=False,
            steer_special_tokens=True,
            n_completion_tokens=10,
            model="gpt2-small",
        )

        response = client.post(
            "/v1/steer/completion-chat",
            json=request.model_dump(),
            headers={"X-SECRET-KEY": X_SECRET_KEY},
        )

        assert response.status_code == 200

        # Parse SSE response
        lines = response.text.strip().split("\n")
        sse_data_lines = [line for line in lines if line.startswith("data: ")]

        assert len(sse_data_lines) > 0, "No SSE data received"

        # Parse final response
        final_data = sse_data_lines[-1].replace("data: ", "")
        response_data = json.loads(final_data)

        # Verify response structure
        assert "results" in response_data
        assert len(response_data["results"]) == 2  # Both STEERED and DEFAULT

        # Verify types are present
        result_types = [result["type"] for result in response_data["results"]]
        assert "STEERED" in result_types
        assert "DEFAULT" in result_types

    def test_steer_completion_chat_steered_only(self, client: TestClient):
        """Test completion_chat with only STEERED type."""
        messages = [NPSteerChatMessage(role="user", content="Tell me about the sky.")]

        features = [
            NPSteerFeature(
                source="1-res-jb", index=50, strength=2.0, steering_vector=[0.05] * 768
            )
        ]

        request = SteerCompletionChatPostRequest(
            messages=messages,
            features=features,
            types=[NPSteerType.STEERED],  # Only STEERED
            strength_multiplier=1.5,
            steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
            normalize_steering=True,
            steer_special_tokens=False,
            n_completion_tokens=15,
            model="gpt2-small",
        )

        response = client.post(
            "/v1/steer/completion-chat",
            json=request.model_dump(),
            headers={"X-SECRET-KEY": X_SECRET_KEY},
        )

        assert response.status_code == 200

        # Parse SSE response
        lines = response.text.strip().split("\n")
        sse_data_lines = [line for line in lines if line.startswith("data: ")]

        # Parse final response
        final_data = sse_data_lines[-1].replace("data: ", "")
        response_data = json.loads(final_data)

        # Should only have STEERED result
        assert len(response_data["results"]) == 1
        assert response_data["results"][0]["type"] == "STEERED"

    def test_steer_completion_chat_default_only(self, client: TestClient):
        """Test completion_chat with only DEFAULT type."""
        messages = [NPSteerChatMessage(role="user", content="Describe a tree.")]

        # Empty features for DEFAULT-only
        features = []

        request = SteerCompletionChatPostRequest(
            messages=messages,
            features=features,
            types=[NPSteerType.DEFAULT],  # Only DEFAULT
            strength_multiplier=1.0,
            steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
            normalize_steering=False,
            steer_special_tokens=True,
            n_completion_tokens=12,
            model="gpt2-small",
        )

        response = client.post(
            "/v1/steer/completion-chat",
            json=request.model_dump(),
            headers={"X-SECRET-KEY": X_SECRET_KEY},
        )

        assert response.status_code == 200

        # Parse SSE response
        lines = response.text.strip().split("\n")
        sse_data_lines = [line for line in lines if line.startswith("data: ")]

        # Parse final response
        final_data = sse_data_lines[-1].replace("data: ", "")
        response_data = json.loads(final_data)

        # Should only have DEFAULT result
        assert len(response_data["results"]) == 1
        assert response_data["results"][0]["type"] == "DEFAULT"

    def test_steer_completion_chat_multiple_features(self, client: TestClient):
        """Test completion_chat with multiple steering features."""
        messages = [
            NPSteerChatMessage(role="user", content="What is artificial intelligence?")
        ]

        # Multiple features with different strengths
        features = [
            NPSteerFeature(
                source="0-res-jb", index=100, strength=1.0, steering_vector=[0.1] * 768
            ),
            NPSteerFeature(
                source="1-res-jb",
                index=200,
                strength=0.5,
                steering_vector=[-0.05] * 768,
            ),
        ]

        request = SteerCompletionChatPostRequest(
            messages=messages,
            features=features,
            types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
            strength_multiplier=2.0,
            steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
            normalize_steering=True,
            steer_special_tokens=False,
            n_completion_tokens=20,
            model="gpt2-small",
        )

        response = client.post(
            "/v1/steer/completion-chat",
            json=request.model_dump(),
            headers={"X-SECRET-KEY": X_SECRET_KEY},
        )

        assert response.status_code == 200

        # Parse and verify response structure
        lines = response.text.strip().split("\n")
        sse_data_lines = [line for line in lines if line.startswith("data: ")]

        final_data = sse_data_lines[-1].replace("data: ", "")
        response_data = json.loads(final_data)

        assert len(response_data["results"]) == 2
        result_types = [result["type"] for result in response_data["results"]]
        assert "STEERED" in result_types
        assert "DEFAULT" in result_types

    def test_steer_completion_chat_long_conversation(self, client: TestClient):
        """Test completion_chat with longer conversation history."""
        messages = [
            NPSteerChatMessage(role="user", content="Hello, how are you today?"),
            NPSteerChatMessage(
                role="assistant", content="I'm doing well, thank you for asking!"
            ),
            NPSteerChatMessage(
                role="user", content="Can you tell me about machine learning?"
            ),
            NPSteerChatMessage(
                role="assistant",
                content="Machine learning is a subset of artificial intelligence.",
            ),
            NPSteerChatMessage(
                role="user", content="What are the main types of machine learning?"
            ),
        ]

        features = [
            NPSteerFeature(
                source="2-res-jb", index=150, strength=1.5, steering_vector=[0.08] * 768
            )
        ]

        request = SteerCompletionChatPostRequest(
            messages=messages,
            features=features,
            types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
            strength_multiplier=1.0,
            steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
            normalize_steering=False,
            steer_special_tokens=True,
            n_completion_tokens=25,
            model="gpt2-small",
        )

        response = client.post(
            "/v1/steer/completion-chat",
            json=request.model_dump(),
            headers={"X-SECRET-KEY": X_SECRET_KEY},
        )

        assert response.status_code == 200

        # Verify response handles longer context properly
        lines = response.text.strip().split("\n")
        sse_data_lines = [line for line in lines if line.startswith("data: ")]

        assert len(sse_data_lines) > 0

        final_data = sse_data_lines[-1].replace("data: ", "")
        response_data = json.loads(final_data)

        assert len(response_data["results"]) == 2

    def test_steer_completion_chat_streaming_consistency(self, client: TestClient):
        """Test that streaming responses are consistent and properly formatted."""
        messages = [NPSteerChatMessage(role="user", content="Count to five.")]

        features = [
            NPSteerFeature(
                source="0-res-jb", index=75, strength=1.0, steering_vector=[0.12] * 768
            )
        ]

        request = SteerCompletionChatPostRequest(
            messages=messages,
            features=features,
            types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
            strength_multiplier=1.0,
            steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
            normalize_steering=False,
            steer_special_tokens=True,
            n_completion_tokens=30,
            model="gpt2-small",
        )

        response = client.post(
            "/v1/steer/completion-chat",
            json=request.model_dump(),
            headers={"X-SECRET-KEY": X_SECRET_KEY},
        )

        assert response.status_code == 200

        # Parse all SSE data lines
        lines = response.text.strip().split("\n")
        sse_data_lines = [line for line in lines if line.startswith("data: ")]

        # Verify we get incremental updates
        assert len(sse_data_lines) > 1, "Should receive multiple streaming updates"

        # Verify each SSE message is valid JSON
        for sse_line in sse_data_lines:
            data_part = sse_line.replace("data: ", "")
            try:
                parsed_data = json.loads(data_part)
                assert "results" in parsed_data
                # Early messages might have partial content, final should have both types
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON in SSE message: {data_part}")

        # Final message should have complete results
        final_data = sse_data_lines[-1].replace("data: ", "")
        final_response = json.loads(final_data)
        assert len(final_response["results"]) == 2

    def test_steer_completion_chat_error_handling(self, client: TestClient):
        """Test error handling with invalid parameters."""
        # Test with invalid source
        messages = [NPSteerChatMessage(role="user", content="Test message.")]

        features = [
            NPSteerFeature(
                source="invalid-source",  # Invalid source
                index=100,
                strength=1.0,
                steering_vector=[0.1] * 768,
            )
        ]

        request = SteerCompletionChatPostRequest(
            messages=messages,
            features=features,
            types=[NPSteerType.STEERED],
            strength_multiplier=1.0,
            steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
            normalize_steering=False,
            steer_special_tokens=True,
            n_completion_tokens=10,
            model="gpt2-small",
        )

        response = client.post(
            "/v1/steer/completion-chat",
            json=request.model_dump(),
            headers={"X-SECRET-KEY": X_SECRET_KEY},
        )

        # Should return an error status
        assert response.status_code != 200

    def test_steer_completion_chat_empty_messages(self, client: TestClient):
        """Test completion_chat with empty message list."""
        request = SteerCompletionChatPostRequest(
            messages=[],  # Empty messages
            features=[],
            types=[NPSteerType.DEFAULT],
            strength_multiplier=1.0,
            steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
            normalize_steering=False,
            steer_special_tokens=True,
            n_completion_tokens=10,
            model="gpt2-small",
        )

        response = client.post(
            "/v1/steer/completion-chat",
            json=request.model_dump(),
            headers={"X-SECRET-KEY": X_SECRET_KEY},
        )

        # Should handle empty messages gracefully or return appropriate error
        # Exact behavior depends on implementation - just verify it doesn't crash
        assert response.status_code in [200, 400, 422]
