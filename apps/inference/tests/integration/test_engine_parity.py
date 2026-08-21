"""EagerModel vs vLLM numeric parity.

This lives in its own module on purpose. ``test_engine_matrix.py`` holds its vLLM server
in a *module*-scoped fixture, and pytest only finalizes module-scoped fixtures when it
leaves the module -- so a parity test sitting in that file would boot a second
EngineCore while the first still owns the GPU, and block. A separate module guarantees
the matrix engine is torn down before this one starts.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.harness import (
    EAGER,
    GPT2,
    VLLM,
    X_SECRET_KEY,
    cuda_available,
    initialized_server,
    try_initialized_server,
    vllm_available,
)

PROMPT = "Hello, world!"


def _single_activation(client: TestClient) -> list[float]:
    resp = client.post(
        "/v1/activation/single",
        json={
            "model": GPT2.model_id,
            "prompt": PROMPT,
            "source": "7-res-jb",
            "index": "0",
        },
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["activation"]["values"]


@pytest.mark.cuda
@pytest.mark.vllm
def test_activation_single_parity_across_engines():
    """Both engines must agree on SAE feature activations for the same prompt/source.

    Eager runs on CPU so the two engines never contend for CUDA memory.
    """
    if not (cuda_available() and vllm_available()):
        pytest.skip("cross-engine parity needs a CUDA + vLLM box")

    with initialized_server(GPT2, engine=EAGER, device="cpu") as eager_client:
        eager_values = _single_activation(eager_client)

    with try_initialized_server(GPT2, engine=VLLM) as vllm_client:
        vllm_values = _single_activation(vllm_client)

    assert len(eager_values) == len(vllm_values) == 4
    for e, v in zip(eager_values, vllm_values, strict=True):
        assert e == pytest.approx(v, abs=0.5), f"engine activation mismatch: eager={eager_values} vllm={vllm_values}"
