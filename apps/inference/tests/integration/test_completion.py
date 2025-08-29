import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from neuronpedia_inference_client.models.np_steer_feature import NPSteerFeature
from neuronpedia_inference_client.models.np_steer_method import NPSteerMethod
from neuronpedia_inference_client.models.np_steer_type import NPSteerType
from neuronpedia_inference_client.models.np_steer_vector import NPSteerVector
from neuronpedia_inference_client.models.steer_completion_post200_response import (
    SteerCompletionPost200Response,
)
from neuronpedia_inference_client.models.steer_completion_request import (
    SteerCompletionRequest,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from tests.conftest import (
    ABS_TOLERANCE,
    BOS_TOKEN_STR,
    FREQ_PENALTY,
    MODEL_ID,
    N_COMPLETION_TOKENS,
    SAE_SELECTED_SOURCES,
    SEED,
    STEER_FEATURE_INDEX,
    STRENGTH,
    STRENGTH_MULTIPLIER,
    TEMPERATURE,
    TEST_PROMPT,
    X_SECRET_KEY,
)

ENDPOINT = "/v1/steer/completion"


def test_completion_steered_with_features_additive(client: TestClient):
    """
    Test steering using features with additive method.
    """
    request = SteerCompletionRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
        features=[
            NPSteerFeature(
                model=MODEL_ID,
                source=SAE_SELECTED_SOURCES[0],
                index=STEER_FEATURE_INDEX,
                strength=STRENGTH,
            )
        ],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    response_model = SteerCompletionPost200Response(**data)

    # Create a mapping of output type to output text
    outputs_by_type = {output.type: output.output for output in response_model.outputs}

    # Test basic API contract
    assert len(outputs_by_type) == 2
    assert NPSteerType.STEERED in outputs_by_type
    assert NPSteerType.DEFAULT in outputs_by_type

    # Both outputs should start with the original prompt
    assert outputs_by_type[NPSteerType.STEERED].startswith(TEST_PROMPT)
    assert outputs_by_type[NPSteerType.DEFAULT].startswith(TEST_PROMPT)

    # Both outputs should be longer than the prompt (completion was generated)
    assert len(outputs_by_type[NPSteerType.STEERED]) > len(TEST_PROMPT)
    assert len(outputs_by_type[NPSteerType.DEFAULT]) > len(TEST_PROMPT)

    # Steered output should be different from default output
    assert outputs_by_type[NPSteerType.STEERED] != outputs_by_type[NPSteerType.DEFAULT]

    expected_steered_output = "Hello, world! the world, the world, the world, the"
    expected_default_output = "Hello, world!\n\nI'm a programmer and I'm a"

    assert outputs_by_type[NPSteerType.STEERED] == expected_steered_output
    assert outputs_by_type[NPSteerType.DEFAULT] == expected_default_output


def test_completion_steered_with_vectors_additive(client: TestClient):
    """
    Test steering using vectors with additive method.
    """
    request = SteerCompletionRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
        vectors=[
            NPSteerVector(
                steering_vector=[1000.0]
                * 768,  # We utilize a large vector to ensure the steering vector is impactful
                strength=STRENGTH,
                hook="blocks.7.hook_resid_post",
            )
        ],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 200

    data = response.json()
    response_model = SteerCompletionPost200Response(**data)

    # Create a mapping of output type to output text
    outputs_by_type = {output.type: output.output for output in response_model.outputs}

    # Test basic API contract
    assert len(outputs_by_type) == 2
    assert NPSteerType.STEERED in outputs_by_type
    assert NPSteerType.DEFAULT in outputs_by_type

    # Both outputs should start with the original prompt
    assert outputs_by_type[NPSteerType.STEERED].startswith(TEST_PROMPT)
    assert outputs_by_type[NPSteerType.DEFAULT].startswith(TEST_PROMPT)

    # Both outputs should be longer than the prompt (completion was generated)
    assert len(outputs_by_type[NPSteerType.STEERED]) > len(TEST_PROMPT)
    assert len(outputs_by_type[NPSteerType.DEFAULT]) > len(TEST_PROMPT)

    # Steered output should be different from default output
    assert outputs_by_type[NPSteerType.STEERED] != outputs_by_type[NPSteerType.DEFAULT]

    expected_steered_output = "Hello, world!!!!!!!!!!!"
    expected_default_output = "Hello, world!\n\nI'm a programmer and I'm a"
        
    assert outputs_by_type[NPSteerType.STEERED] == expected_steered_output
    assert outputs_by_type[NPSteerType.DEFAULT] == expected_default_output


def test_completion_steered_token_limit_exceeded(client: TestClient):
    """
    Test handling of a prompt that exceeds the token limit.
    """
    long_prompt = "This is a test prompt. " * 1000
    request = SteerCompletionRequest(
        prompt=long_prompt,
        model=MODEL_ID,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=[NPSteerType.STEERED],
        features=[
            NPSteerFeature(
                model=MODEL_ID,
                source=SAE_SELECTED_SOURCES[0],
                index=0,
                strength=STRENGTH,
            )
        ],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 400

    data = response.json()
    expected_error_message = "Text too long: 6001 tokens, max is 500"
    assert data["error"] == expected_error_message


def test_completion_steered_with_features_orthogonal(client: TestClient):
    """
    Test steering using features with orthogonal decomposition method.
    """
    request = SteerCompletionRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        steer_method=NPSteerMethod.ORTHOGONAL_DECOMP,
        normalize_steering=False,
        types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
        features=[
            NPSteerFeature(
                model=MODEL_ID,
                source=SAE_SELECTED_SOURCES[0],
                index=STEER_FEATURE_INDEX,
                strength=STRENGTH,
            )
        ],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    response_model = SteerCompletionPost200Response(**data)

    # Create a mapping of output type to output text
    outputs_by_type = {output.type: output.output for output in response_model.outputs}

    # Test basic API contract
    assert len(outputs_by_type) == 2
    assert NPSteerType.STEERED in outputs_by_type
    assert NPSteerType.DEFAULT in outputs_by_type

    # Both outputs should start with the original prompt
    assert outputs_by_type[NPSteerType.STEERED].startswith(TEST_PROMPT)
    assert outputs_by_type[NPSteerType.DEFAULT].startswith(TEST_PROMPT)

    # Both outputs should be longer than the prompt (completion was generated)
    assert len(outputs_by_type[NPSteerType.STEERED]) > len(TEST_PROMPT)
    assert len(outputs_by_type[NPSteerType.DEFAULT]) > len(TEST_PROMPT)

    # Steered output should be different from default output
    assert outputs_by_type[NPSteerType.STEERED] != outputs_by_type[NPSteerType.DEFAULT]

    expected_steered_output = "Hello, world! Hy Hy Hy Hy Hy Hy Hy Hy Hy Hy"
    expected_default_output = "Hello, world!\n\nI'm a programmer and I'm a"

    assert outputs_by_type[NPSteerType.STEERED] == expected_steered_output
    assert outputs_by_type[NPSteerType.DEFAULT] == expected_default_output


def test_completion_steered_with_vectors_orthogonal(client: TestClient):
    """
    Test steering using vectors with orthogonal decomposition.
    """
    request = SteerCompletionRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        steer_method=NPSteerMethod.ORTHOGONAL_DECOMP,
        normalize_steering=False,
        types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
        vectors=[
            NPSteerVector(
                steering_vector=[1000.0]
                * 768,  # We utilize a large vector to ensure the steering vector is impactful
                strength=STRENGTH,
                hook="blocks.7.hook_resid_post",
            )
        ],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 200

    data = response.json()
    response_model = SteerCompletionPost200Response(**data)

    # Create a mapping of output type to output text
    outputs_by_type = {output.type: output.output for output in response_model.outputs}

    # Test basic API contract
    assert len(outputs_by_type) == 2
    assert NPSteerType.STEERED in outputs_by_type
    assert NPSteerType.DEFAULT in outputs_by_type

    # Both outputs should start with the original prompt
    assert outputs_by_type[NPSteerType.STEERED].startswith(TEST_PROMPT)
    assert outputs_by_type[NPSteerType.DEFAULT].startswith(TEST_PROMPT)

    # Both outputs should be longer than the prompt (completion was generated)
    assert len(outputs_by_type[NPSteerType.STEERED]) > len(TEST_PROMPT)
    assert len(outputs_by_type[NPSteerType.DEFAULT]) > len(TEST_PROMPT)

    # Steered output should be different from default output
    assert outputs_by_type[NPSteerType.STEERED] != outputs_by_type[NPSteerType.DEFAULT]

    expected_steered_output = "Hello, world!!!!!!!!!!!"
    expected_default_output = "Hello, world!\n\nI'm a programmer and I'm a"

    assert outputs_by_type[NPSteerType.STEERED] == expected_steered_output
    assert outputs_by_type[NPSteerType.DEFAULT] == expected_default_output


def test_completion_invalid_request_no_features_or_vectors(client: TestClient):
    """
    Test error handling when neither features nor vectors are provided.
    """
    request = SteerCompletionRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=[NPSteerType.STEERED],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 400
    data = response.json()
    assert "exactly one of features or vectors must be provided" in data["error"]


def test_completion_invalid_request_both_features_and_vectors(client: TestClient):
    """
    Test error handling when both features and vectors are provided.
    """
    request = SteerCompletionRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=[NPSteerType.STEERED],
        features=[
            NPSteerFeature(
                model=MODEL_ID,
                source=SAE_SELECTED_SOURCES[0],
                index=STEER_FEATURE_INDEX,
                strength=STRENGTH,
            )
        ],
        vectors=[
            NPSteerVector(
                steering_vector=[1.0] * 768,
                strength=STRENGTH,
                hook="blocks.7.hook_resid_post",
            )
        ],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )
    assert response.status_code == 400
    data = response.json()
    assert "exactly one of features or vectors must be provided" in data["error"]


def test_completion_logprobs(client: TestClient):
    """Test that logprobs are returned for both STEERED and DEFAULT types."""
    request = SteerCompletionRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
        features=[
            NPSteerFeature(
                model=MODEL_ID,
                source=SAE_SELECTED_SOURCES[0],
                index=STEER_FEATURE_INDEX,
                strength=STRENGTH,
            )
        ],
        n_completion_tokens=5,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
        n_logprobs=2,
    )

    response = client.post(
        ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY}
    )

    assert response.status_code == 200

    data = response.json()
    response_model = SteerCompletionPost200Response(**data)

    assert len(response_model.outputs) == 2

    # find steered and default outputs
    steered_output = None
    default_output = None
    for output in response_model.outputs:
        if output.type == NPSteerType.STEERED:
            steered_output = output
        elif output.type == NPSteerType.DEFAULT:
            default_output = output

    assert steered_output is not None
    assert default_output is not None

    # both should have logprobs
    for output in [steered_output, default_output]:
        assert (
            output.logprobs is not None
        ), f"logprobs should not be None for {output.type}"
        assert (
            len(output.logprobs) > 0
        ), f"logprobs should contain items for {output.type}"

        for logprob_item in output.logprobs:
            assert logprob_item.token is not None
            assert logprob_item.logprob is not None
            assert logprob_item.top_logprobs is not None
            assert len(logprob_item.top_logprobs) == 2  # Should match n_logprobs

    steered_logprobs = steered_output.logprobs
    default_logprobs = default_output.logprobs

    # get pyright checks to pass
    assert steered_logprobs is not None, "Steered logprobs should not be None"
    assert default_logprobs is not None, "Default logprobs should not be None"
    assert len(steered_logprobs) > 0, "Steered logprobs should not be empty"
    assert len(default_logprobs) > 0, "Default logprobs should not be empty"

    expected_steered_logprobs = [
        (" the", -2.78125),
        (" world", -3.552734375),
        (",", -1.28515625),
        (" the", -1.271484375),
        (" world", -1.42578125),
    ]

    expected_default_logprobs = [
        ("\n", -1.390625),
        ("\n", -0.01122283935546875),
        ("I", -1.7119140625),
        ("'m", -1.2666015625),
        (" a", -2.939453125),
    ]

    # verify all steered logprobs
    for i, (expected_token, expected_logprob) in enumerate(expected_steered_logprobs):
        assert (
            steered_logprobs[i].token == expected_token
        ), f"Steered token mismatch at {i}: expected '{expected_token}', got '{steered_logprobs[i].token}'"
        assert (
            steered_logprobs[i].logprob
            == pytest.approx(expected_logprob, abs=ABS_TOLERANCE)
        ), f"Steered logprob mismatch at {i}: expected {expected_logprob}, got {steered_logprobs[i].logprob}"

    # verify all default logprobs
    for i, (expected_token, expected_logprob) in enumerate(expected_default_logprobs):
        assert (
            default_logprobs[i].token == expected_token
        ), f"Default token mismatch at {i}: expected '{expected_token}', got '{default_logprobs[i].token}'"
        assert (
            default_logprobs[i].logprob
            == pytest.approx(expected_logprob, abs=ABS_TOLERANCE)
        ), f"Default logprob mismatch at {i}: expected {expected_logprob}, got {default_logprobs[i].logprob}"


def test_completion_logprobs_match_hugging_face(client: TestClient):
    """
    Compare the API's returned logprobs with Hugging Face using the same parameters as test_completion_logprobs.
    This provides a direct comparison to see if TransformerLens logprobs are close to HF reference.
    """
    hf_model_id = "gpt2"
    model_id = "gpt2-small"

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)
    tokenizer_kwargs = dict(return_tensors="pt", add_special_tokens=False)
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_id, torch_dtype=torch.float32
    )
    hf_model.eval()

    request = SteerCompletionRequest(
        prompt=TEST_PROMPT,
        model=model_id,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
        features=[
            NPSteerFeature(
                model=model_id,
                source=SAE_SELECTED_SOURCES[0],
                index=STEER_FEATURE_INDEX,
                strength=STRENGTH,
            )
        ],
        n_completion_tokens=5,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
        n_logprobs=2,
    )

    response = client.post(
        ENDPOINT,
        json=request.model_dump(),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert (
        response.status_code == 200
    ), f"API call failed: {response.status_code} - {response.text}"
    data = response.json()

    default_output = next((o for o in data["outputs"] if o["type"] == "DEFAULT"), None)
    assert default_output is not None, "DEFAULT output not found"

    str_tokens = [t["token"] for t in default_output["logprobs"]]
    logprobs = [float(t["logprob"]) for t in default_output["logprobs"]]
    assert len(str_tokens) == 5
    assert len(logprobs) == 5

    text = "".join(str_tokens)
    # HF doesn't prepend BOS to gpt2-small by default, so we need to do it manually
    HF_PROMPT = BOS_TOKEN_STR + TEST_PROMPT
    prompt_ids = tokenizer(HF_PROMPT, **tokenizer_kwargs)["input_ids"][0]
    combined_ids = tokenizer(HF_PROMPT + text, **tokenizer_kwargs)["input_ids"][0]
    ids = combined_ids[len(prompt_ids) :]
    assert len(ids) == len(logprobs), "length mismatch after retokenizing API text"

    # HF logprobs at the same positions
    with torch.no_grad():
        logits = hf_model(combined_ids.unsqueeze(0)).logits[0]  # [seq, vocab]
        hf_logprobs = torch.log_softmax(logits, dim=-1)

    start = len(prompt_ids)
    hf_reference_logprobs = [
        hf_logprobs[start + i - 1, tid].item() for i, tid in enumerate(ids)
    ]

    # numerical check, allowing for implementation differences between TransformerLens and HuggingFace
    assert np.allclose(
        logprobs, hf_reference_logprobs, rtol=0.001, atol=0.07
    ), f"logprob mismatch.\nAPI: {logprobs}\nHF:  {hf_reference_logprobs}"


def test_completion_logprobs_streaming(client: TestClient):
    """Test that logprobs work correctly in streaming mode."""
    prompt = "The cat sat"
    n_tokens = 3
    n_logprobs = 2

    request = SteerCompletionRequest(
        prompt=prompt,
        model=MODEL_ID,
        n_completion_tokens=n_tokens,
        n_logprobs=n_logprobs,
        temperature=0,
        types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
        features=[
            NPSteerFeature(
                model=MODEL_ID,
                source=SAE_SELECTED_SOURCES[0],
                index=0,
                strength=0.0,
            )
        ],
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        strength_multiplier=0.0,
        freq_penalty=0.0,
        seed=42,
        stream=True,
    )

    response = client.post(
        ENDPOINT,
        json=request.model_dump(),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    chunks = response.content.decode().strip().split("\n\n")
    final_chunk = None
    for chunk in reversed(chunks):
        if chunk.startswith("data: ") and not chunk.endswith("[DONE]"):
            final_chunk = chunk[6:]  # remove 'data: ' prefix
            break

    assert final_chunk is not None, "No valid streaming chunk found"
    resp = json.loads(final_chunk)

    # verify response structure
    assert "outputs" in resp
    assert len(resp["outputs"]) == 2

    # check both outputs have logprobs
    for output in resp["outputs"]:
        assert output["type"] in ["STEERED", "DEFAULT"]
        assert "logprobs" in output
        assert output["logprobs"] is not None
        assert (
            len(output["logprobs"]) > 0
        ), f"Expected logprobs, got {len(output['logprobs'])}"

        # verify logprobs structure
        for logprob_item in output["logprobs"]:
            assert "token" in logprob_item
            assert "logprob" in logprob_item
            assert "top_logprobs" in logprob_item
            assert isinstance(logprob_item["token"], str)
            assert isinstance(logprob_item["logprob"], (int, float))
            assert len(logprob_item["top_logprobs"]) == n_logprobs

            # verify top_logprobs structure
            for top_logprob in logprob_item["top_logprobs"]:
                assert "token" in top_logprob
                assert "logprob" in top_logprob
                assert isinstance(top_logprob["token"], str)
                assert isinstance(top_logprob["logprob"], (int, float))
