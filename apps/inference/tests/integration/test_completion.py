"""Integration tests for /v1/steer/completion.

Generation is greedy and seeded, so the completions below are asserted as exact strings.
That is deliberate: a change in the steering maths shows up here as a readable diff of
what the model actually said, which a "steered differs from default" assertion would
quietly let through.

Only the fields a test cares about are named at each call site; `completion_request`
supplies the rest, since a steer request has a dozen required fields that are the same
every time.
"""

import json
from typing import Any

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from httpx import Response
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuronpedia_inference.schemas import (
    NPSteerFeature,
    NPSteerMethod,
    NPSteerType,
    NPSteerVector,
    SteerCompletionRequest,
    SteerCompletionResponse,
)
from tests.conftest import (
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

# gpt2-small's residual width; steering vectors have to match it exactly.
D_MODEL = 768

# What the model says with no steering applied. Shared by every case below, which is
# what makes an unchanged DEFAULT a meaningful control on the steered result.
DEFAULT_COMPLETION = "\n\nI'm a programmer and I'm a"


def completion_request(**overrides: Any) -> SteerCompletionRequest:
    settings: dict[str, Any] = {
        "prompt": TEST_PROMPT,
        "model": MODEL_ID,
        "steer_method": NPSteerMethod.SIMPLE_ADDITIVE,
        "normalize_steering": False,
        "types": [NPSteerType.STEERED, NPSteerType.DEFAULT],
        "n_completion_tokens": N_COMPLETION_TOKENS,
        "temperature": TEMPERATURE,
        "strength_multiplier": STRENGTH_MULTIPLIER,
        "freq_penalty": FREQ_PENALTY,
        "seed": SEED,
    }
    settings.update(overrides)
    return SteerCompletionRequest(**settings)


def post(client: TestClient, request: SteerCompletionRequest) -> Response:
    return client.post(ENDPOINT, json=request.model_dump(), headers={"X-SECRET-KEY": X_SECRET_KEY})


def completions(response: Response) -> dict[NPSteerType, str]:
    """The generated text for each steer type in the response."""
    parsed = SteerCompletionResponse(**response.json())
    return {output.type: output.output for output in parsed.outputs}


def steer_feature(index: int = STEER_FEATURE_INDEX) -> NPSteerFeature:
    return NPSteerFeature(
        model=MODEL_ID,
        source=SAE_SELECTED_SOURCES[0],
        index=index,
        strength=STRENGTH,
    )


def steer_vector(magnitude: float = 1000.0) -> NPSteerVector:
    # Deliberately large and uniform: big enough to dominate the residual stream, so a
    # steering method that silently did nothing could not produce the outputs below.
    return NPSteerVector(
        steering_vector=[magnitude] * D_MODEL,
        strength=STRENGTH,
        hook="blocks.7.hook_resid_post",
    )


@pytest.mark.parametrize(
    ("steer_method", "steer_with", "expected_steered"),
    [
        pytest.param(
            NPSteerMethod.SIMPLE_ADDITIVE,
            {"features": [steer_feature()]},
            " the world, the world, the world, the",
            id="additive-with-feature",
        ),
        pytest.param(
            NPSteerMethod.SIMPLE_ADDITIVE,
            {"vectors": [steer_vector()]},
            "!!!!!!!!!!",
            id="additive-with-vector",
        ),
        pytest.param(
            NPSteerMethod.ORTHOGONAL_DECOMP,
            {"features": [steer_feature()]},
            " Hy Hy Hy Hy Hy Hy Hy Hy Hy Hy",
            id="orthogonal-with-feature",
        ),
        pytest.param(
            NPSteerMethod.ORTHOGONAL_DECOMP,
            {"vectors": [steer_vector()]},
            # Projecting out this direction happens not to change gpt2's greedy decode,
            # so STEERED matching DEFAULT is the expected result. The case earns its
            # keep as a regression guard: large-magnitude vectors used to overflow to
            # NaN in fp16 while `OrthogonalProjector` still built a projection matrix,
            # which this would catch.
            DEFAULT_COMPLETION,
            id="orthogonal-with-vector",
        ),
    ],
)
def test_steering_produces_expected_completion(
    client: TestClient,
    steer_method: NPSteerMethod,
    steer_with: dict[str, Any],
    expected_steered: str,
):
    response = post(client, completion_request(steer_method=steer_method, **steer_with))
    assert response.status_code == 200

    output = completions(response)
    assert set(output) == {NPSteerType.STEERED, NPSteerType.DEFAULT}
    assert output[NPSteerType.STEERED] == expected_steered
    assert output[NPSteerType.DEFAULT] == DEFAULT_COMPLETION


class TestRejectedRequests:
    def test_prompt_over_the_token_limit(self, client: TestClient):
        response = post(
            client,
            completion_request(
                prompt="This is a test prompt. " * 1000,
                types=[NPSteerType.STEERED],
                features=[steer_feature(index=0)],
            ),
        )

        assert response.status_code == 400
        assert response.json()["error"] == "Text too long: 6002 tokens, max is 500"

    @pytest.mark.parametrize(
        "steer_with",
        [
            pytest.param({}, id="neither-features-nor-vectors"),
            pytest.param(
                {
                    "features": [steer_feature()],
                    "vectors": [steer_vector(magnitude=1.0)],
                },
                id="both-features-and-vectors",
            ),
        ],
    )
    def test_requires_exactly_one_of_features_or_vectors(self, client: TestClient, steer_with: dict[str, Any]):
        response = post(client, completion_request(types=[NPSteerType.STEERED], **steer_with))

        assert response.status_code == 400
        assert "exactly one of features or vectors must be provided" in response.json()["error"]


# Nothing in the server constructs an NPLogprob, on either the engine or the vLLM path,
# so `logprobs` never comes back and these describe a contract rather than a behaviour.
# They are kept as the specification for whoever wires it up; non-strict so that they
# start reporting xpass rather than failing the suite on the day that lands.
logprobs_not_implemented = pytest.mark.xfail(
    reason="No generation backend reports per-token scores yet, so the response's logprobs field is always absent.",
    strict=False,
)

N_LOGPROBS = 2
N_LOGPROB_TOKENS = 5


@logprobs_not_implemented
def test_logprobs_returned_for_each_steer_type(client: TestClient):
    response = post(
        client,
        completion_request(
            features=[steer_feature()],
            n_completion_tokens=N_LOGPROB_TOKENS,
            n_logprobs=N_LOGPROBS,
        ),
    )
    assert response.status_code == 200

    parsed = SteerCompletionResponse(**response.json())
    by_type = {output.type: output for output in parsed.outputs}
    assert set(by_type) == {NPSteerType.STEERED, NPSteerType.DEFAULT}

    for steer_type, output in by_type.items():
        assert output.logprobs is not None, f"no logprobs for {steer_type}"
        assert len(output.logprobs) == N_LOGPROB_TOKENS
        for entry in output.logprobs:
            assert entry.token is not None
            assert entry.logprob is not None
            # One entry per candidate the caller asked to see.
            assert len(entry.top_logprobs) == N_LOGPROBS


@logprobs_not_implemented
def test_logprobs_agree_with_hugging_face(client: TestClient):
    """Cross-check the unsteered logprobs against a plain HF forward pass.

    The reference is computed here rather than frozen into the test, so this stays
    meaningful across model or tokenizer updates instead of pinning numbers that were
    only ever true of one implementation.
    """
    response = post(
        client,
        completion_request(
            features=[steer_feature()],
            n_completion_tokens=N_LOGPROB_TOKENS,
            n_logprobs=N_LOGPROBS,
        ),
    )
    assert response.status_code == 200, f"API call failed: {response.text}"

    default = next((o for o in response.json()["outputs"] if o["type"] == "DEFAULT"), None)
    assert default is not None, "DEFAULT output not found"
    # Checked before the model is loaded below: while logprobs are unimplemented this
    # is where the test stops, and loading gpt2 to get here would be wasted work.
    assert default["logprobs"], "no logprobs to compare against"

    api_tokens = [entry["token"] for entry in default["logprobs"]]
    api_logprobs = [float(entry["logprob"]) for entry in default["logprobs"]]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    hf_model.eval()

    # HF does not prepend gpt2's BOS, so add it to match what the server tokenized.
    tokenizer_kwargs = {"return_tensors": "pt", "add_special_tokens": False}
    prompt = BOS_TOKEN_STR + TEST_PROMPT
    prompt_ids = tokenizer(prompt, **tokenizer_kwargs)["input_ids"][0]
    combined_ids = tokenizer(prompt + "".join(api_tokens), **tokenizer_kwargs)["input_ids"][0]
    generated_ids = combined_ids[len(prompt_ids) :]
    assert len(generated_ids) == len(api_logprobs), (
        "retokenizing the API's text did not round-trip to the same token count"
    )

    with torch.no_grad():
        logits = hf_model(combined_ids.unsqueeze(0)).logits[0]
    hf_logprobs = torch.log_softmax(logits, dim=-1)

    # Position i of the generation is predicted by the logits at the preceding index.
    start = len(prompt_ids)
    reference = [hf_logprobs[start + i - 1, token_id].item() for i, token_id in enumerate(generated_ids)]

    # Loose tolerances: the serving path and HF differ in kernels and accumulation order.
    assert np.allclose(api_logprobs, reference, rtol=0.001, atol=0.07), (
        f"logprob mismatch.\nAPI: {api_logprobs}\nHF:  {reference}"
    )


@logprobs_not_implemented
def test_logprobs_returned_when_streaming(client: TestClient):
    n_tokens = 3
    response = post(
        client,
        completion_request(
            prompt="The cat sat",
            features=[steer_feature(index=0)],
            n_completion_tokens=n_tokens,
            n_logprobs=N_LOGPROBS,
            temperature=0,
            strength_multiplier=0.0,
            freq_penalty=0.0,
            stream=True,
        ),
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    # Frames are cumulative, so the last one carries the complete result.
    frames = [
        chunk[len("data: ") :]
        for chunk in response.content.decode().strip().split("\n\n")
        if chunk.startswith("data: ") and not chunk.endswith("[DONE]")
    ]
    assert frames, "no streaming frames found"
    final = json.loads(frames[-1])

    assert len(final["outputs"]) == 2
    for output in final["outputs"]:
        assert output["type"] in {"STEERED", "DEFAULT"}
        assert output["logprobs"], f"no logprobs for {output['type']}"
        for entry in output["logprobs"]:
            assert isinstance(entry["token"], str)
            assert isinstance(entry["logprob"], int | float)
            assert len(entry["topLogprobs"]) == N_LOGPROBS
            for candidate in entry["topLogprobs"]:
                assert isinstance(candidate["token"], str)
                assert isinstance(candidate["logprob"], int | float)
