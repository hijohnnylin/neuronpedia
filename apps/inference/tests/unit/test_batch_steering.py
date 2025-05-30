# ABOUTME: Simple unit tests for batch generation functionality in steering endpoint
# ABOUTME: Tests core response formatting and batch logic without model dependencies

import json

import pytest
from neuronpedia_inference_client.models.np_steer_completion_response_inner import (
    NPSteerCompletionResponseInner,
)
from neuronpedia_inference_client.models.np_steer_type import NPSteerType

from neuronpedia_inference.endpoints.steer.completion import (
    make_steer_completion_response,
)


class TestSteerCompletion:
    """Test steering completion functionality."""

    def test_make_steer_completion_response_both_types(self):
        """Test response formatting with both steered and default types."""
        steer_types = [NPSteerType.STEERED, NPSteerType.DEFAULT]
        steered_result = "The weather today is sunny and warm."
        default_result = "The weather today is cloudy and cool."

        response = make_steer_completion_response(
            steer_types, steered_result, default_result
        )

        # Validate response structure
        assert hasattr(response, "outputs")
        assert len(response.outputs) == 2

        # Check steered output
        steered_output = next(
            output for output in response.outputs if output.type == NPSteerType.STEERED
        )
        assert steered_output.output == steered_result

        # Check default output  
        default_output = next(
            output for output in response.outputs if output.type == NPSteerType.DEFAULT
        )
        assert default_output.output == default_result

    def test_make_steer_completion_response_steered_only(self):
        """Test response formatting with only steered type."""
        steer_types = [NPSteerType.STEERED]
        result = "The weather today is sunny and warm."

        response = make_steer_completion_response(steer_types, result, result)

        assert len(response.outputs) == 1
        assert response.outputs[0].type == NPSteerType.STEERED
        assert response.outputs[0].output == result

    def test_make_steer_completion_response_default_only(self):
        """Test response formatting with only default type."""
        steer_types = [NPSteerType.DEFAULT]
        result = "The weather today is cloudy and cool."

        response = make_steer_completion_response(steer_types, result, result)

        assert len(response.outputs) == 1
        assert response.outputs[0].type == NPSteerType.DEFAULT
        assert response.outputs[0].output == result

    def test_make_steer_completion_response_json_serialization(self):
        """Test that response can be serialized to JSON."""
        steer_types = [NPSteerType.STEERED, NPSteerType.DEFAULT]
        steered_result = "Steered: The weather is fantastic!"
        default_result = "Default: The weather is okay."

        response = make_steer_completion_response(
            steer_types, steered_result, default_result
        )

        # Convert to JSON and back
        json_str = response.to_json()
        parsed = json.loads(json_str)

        # Validate JSON structure
        assert "outputs" in parsed
        assert len(parsed["outputs"]) == 2
        
        # Check that both outputs are present
        output_types = [output["type"] for output in parsed["outputs"]]
        assert "STEERED" in output_types
        assert "DEFAULT" in output_types

        # Check output content
        steered_json = next(
            output for output in parsed["outputs"] if output["type"] == "STEERED"
        )
        default_json = next(
            output for output in parsed["outputs"] if output["type"] == "DEFAULT"
        )
        
        assert steered_json["output"] == steered_result
        assert default_json["output"] == default_result

    def test_response_ordering(self):
        """Test that response maintains correct ordering of types."""
        # Test STEERED first, then DEFAULT
        steer_types = [NPSteerType.STEERED, NPSteerType.DEFAULT]
        response = make_steer_completion_response(
            steer_types, "steered text", "default text"
        )
        
        assert response.outputs[0].type == NPSteerType.STEERED
        assert response.outputs[1].type == NPSteerType.DEFAULT

        # Test DEFAULT first, then STEERED  
        steer_types = [NPSteerType.DEFAULT, NPSteerType.STEERED]
        response = make_steer_completion_response(
            steer_types, "steered text", "default text"
        )
        
        assert response.outputs[0].type == NPSteerType.DEFAULT
        assert response.outputs[1].type == NPSteerType.STEERED

    def test_empty_outputs(self):
        """Test handling of empty output strings."""
        steer_types = [NPSteerType.STEERED, NPSteerType.DEFAULT]
        response = make_steer_completion_response(steer_types, "", "")

        assert len(response.outputs) == 2
        assert response.outputs[0].output == ""
        assert response.outputs[1].output == ""

    def test_long_outputs(self):
        """Test handling of long output strings."""
        steer_types = [NPSteerType.STEERED, NPSteerType.DEFAULT]
        long_text = "This is a very long text " * 100  # 2500+ characters
        
        response = make_steer_completion_response(steer_types, long_text, long_text)

        assert len(response.outputs) == 2
        assert len(response.outputs[0].output) > 2000
        assert len(response.outputs[1].output) > 2000
        assert response.outputs[0].output == long_text
        assert response.outputs[1].output == long_text

    def test_special_characters_in_outputs(self):
        """Test handling of special characters in outputs."""
        steer_types = [NPSteerType.STEERED, NPSteerType.DEFAULT]
        special_text = 'Text with "quotes", newlines\n, and unicode: 🌟'
        
        response = make_steer_completion_response(steer_types, special_text, special_text)

        # Should handle special characters without issues
        json_str = response.to_json()
        parsed = json.loads(json_str)
        
        for output in parsed["outputs"]:
            assert output["output"] == special_text

    def test_different_content_per_type(self):
        """Test that different content is properly assigned to each type."""
        steer_types = [NPSteerType.STEERED, NPSteerType.DEFAULT]
        steered_content = "Steered response with specific content A"
        default_content = "Default response with specific content B"
        
        response = make_steer_completion_response(
            steer_types, steered_content, default_content
        )

        steered_output = next(
            output for output in response.outputs if output.type == NPSteerType.STEERED
        )
        default_output = next(
            output for output in response.outputs if output.type == NPSteerType.DEFAULT
        )

        assert steered_output.output == steered_content
        assert default_output.output == default_content
        assert steered_output.output != default_output.output