"""Per-model endpoint expectations, driven by tests/model_expectations.yaml.

The counterpart to the engine's own tests/test_model_expectations.py, split along the line of what
each layer can assert. The engine owns the read-out and the attention rows, which need only
weights. Everything here needs the server: SAEManager for a curated feature's activation, and
a real steer request whose continuation has to change in a specific way.

This is the layer the RUNBOOK's calibration tables were really about. A steer strength is a
property of a model *and* an SAE family *and* a backend -- the same concept needs 45 on
gpt2's res-jb and 1000 on gemmascope-2 -- so those numbers only mean anything measured
against a live server, which is what this file does.

Model-specific values live in the YAML, so covering a new model is a block there. Every
capability block is optional: a row with no `saeSource` gets lens coverage only, which is the
honest state for a model with no published SAEs.

    pytest tests/integration/test_model_expectations.py -m "not gated and not thinking"
    pytest tests/integration/test_model_expectations.py -m xl    # the multi-GB rows
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from tests.harness import MODELS, X_SECRET_KEY, ModelSpec, try_initialized_server

_YAML = Path(__file__).parent.parent / "model_expectations.yaml"

HEADERS = {"X-SECRET-KEY": X_SECRET_KEY}


@dataclass(frozen=True)
class Expectation:
    """One row of model_expectations.yaml, with the file-wide bindings already merged in."""

    key: str
    spec: ModelSpec
    values: dict[str, Any]

    def __getitem__(self, name: str) -> Any:
        return self.values[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    def require(self, name: str) -> Any:
        """Read an optional capability field, skipping when the row does not carry it."""
        if name not in self.values:
            pytest.skip(f"{self.key} has no {name} in model_expectations.yaml")
        return self.values[name]


def _load_rows() -> list[Expectation]:
    """Merge each model block over the file-wide bindings.

    A key naming a row of the harness matrix reuses that ModelSpec, so the CI model list and
    its SAE configuration stay single-sourced in harness.py; the row may still override the
    SAE fields, which is how a matrix model gains SAE coverage it did not ship with. A key
    that is not in the matrix (the `manual` tier, whose weights CI must never download) builds
    its spec from the row's own fields.
    """
    doc = yaml.safe_load(_YAML.read_text(encoding="utf-8"))
    bindings = doc.get("bindings", {})

    rows: list[Expectation] = []
    for key, overrides in doc["models"].items():
        values = {**bindings, **(overrides or {})}
        spec = MODELS.get(key)
        if spec is None:
            hf_id = values.get("hfId")
            if not hf_id:
                raise ValueError(f"model_expectations.yaml: '{key}' is not in the harness matrix, so it needs an hfId")
            spec = ModelSpec(
                key=key,
                model_id=hf_id,
                dtype=values.get("dtype", "float32"),
                is_chat=bool(values.get("chat", False)),
                is_thinking=bool(values.get("thinking", False)),
                is_gated=bool(values.get("gated", False)),
            )
        if "saeSets" in values or "includeSae" in values:
            spec = replace(
                spec,
                sae_sets=values.get("saeSets", spec.sae_sets),
                include_sae=values.get("includeSae", spec.include_sae),
            )
        rows.append(Expectation(key=key, spec=spec, values=values))
    return rows


ROWS = _load_rows()


def _params() -> list[Any]:
    """One pytest.param per row, carrying the marks its tier and model imply.

    The same marks the rest of the suite uses, so these rows route themselves: the CPU job
    runs `-m "not cuda and not vllm and not thinking and not gated"` and so picks up gpt2
    alone, while the gated and thinking rows land on the GPU job without either workflow
    needing to know this file exists.
    """
    params = []
    for row in ROWS:
        marks = []
        if row.spec.is_gated:
            marks.append(pytest.mark.gated)
        if row.spec.is_thinking:
            marks.append(pytest.mark.thinking)
        if row.spec.is_chat:
            marks.append(pytest.mark.chat)
        if row.get("tier") == "manual":
            marks.append(pytest.mark.xl)
        params.append(pytest.param(row, id=row.key, marks=marks))
    return params


PARAMS = _params()


def _server(row: Expectation):
    """A live server for the row. Two SAE slots so a row's feature never evicts the default."""
    return try_initialized_server(row.spec, max_loaded_saes=2)


def _post(client: TestClient, path: str, body: dict[str, Any]) -> dict[str, Any]:
    response = client.post(path, json=body, headers=HEADERS)
    assert response.status_code == 200, f"{path} -> {response.status_code}: {response.text[:400]}"
    return response.json()


# --- lens read-out ----------------------------------------------------------


def _lens_readouts(row: Expectation, client: TestClient) -> dict[str, tuple[list[list[str]], list[list[float]]]]:
    """Per-layer top tokens and probabilities for the LAST prompt position, by lens type.

    The response nests one `results` entry per requested type, each holding a per-layer list.
    Returning all of them keyed by type is what lets a row that asks for a fitted lens get the
    same assertions applied to it as the logit lens, rather than only to the first type.
    """
    body = _post(
        client,
        "/v1/lens/prompt",
        {
            "model": row.spec.model_id,
            "prompt": row["lensPrompt"],
            "type": row["lensTypes"],
            "top_n": row["lensTopK"],
            "num_completion_tokens": 0,
            "temperature": 0,
            "stream": False,
        },
    )
    results = body["tokens"][-1]["results"]
    readouts: dict[str, tuple[list[list[str]], list[list[float]]]] = {}
    for lens_type in row["lensTypes"]:
        entry = next((r for r in results if r["type"] == lens_type), None)
        assert entry is not None, (
            f"{row.key}: asked for {lens_type} but the response carries {[r['type'] for r in results]}"
        )
        readouts[lens_type] = (entry["top_tokens"], entry["top_probs"])
    return readouts


def _answer_layers(top_tokens: list[list[str]], answer: str) -> list[int]:
    """Indices of the layers whose top-k contains `answer`, compared whitespace-stripped."""
    return [index for index, layer in enumerate(top_tokens) if answer in [token.strip() for token in layer]]


@pytest.mark.parametrize("row", PARAMS)
def test_lens_endpoint_recovers_the_answer(row: Expectation):
    """The token the model obviously predicts must survive the read-out, through HTTP.

    The engine asserts this against `layer_logits` directly; here the same claim has to hold
    end to end, which additionally covers the endpoint's layer selection, its top-k, and its
    serialization. Tokens are compared whitespace-stripped so a row need not know the model's
    leading-space convention.
    """
    with _server(row) as client:
        answer = row["lensAnswer"]
        for lens_type, (top_tokens, _) in _lens_readouts(row, client).items():
            final = [token.strip() for token in top_tokens[-1]]
            assert answer in final, (
                f"{row.key}/{lens_type}: {answer!r} missing from the final-layer "
                f"top-{row['lensTopK']} for {row['lensPrompt']!r}; got {final}"
            )

            hits = _answer_layers(top_tokens, answer)
            assert len(hits) >= row["minLensLayers"], (
                f"{row.key}/{lens_type}: {answer!r} reached the top-k at {len(hits)} "
                f"layers ({hits}), expected at least {row['minLensLayers']}"
            )


@pytest.mark.parametrize("row", PARAMS)
def test_lens_endpoint_returns_probabilities(row: Expectation):
    """Finite and in [0, 1] at every layer, and sorted descending within a layer.

    NaNs render as blank cells in the UI rather than as an error, so nothing else reports
    them; an unsorted layer means the top-k was taken before the softmax.
    """
    with _server(row) as client:
        for lens_type, (top_tokens, top_probs) in _lens_readouts(row, client).items():
            assert len(top_probs) == len(top_tokens), f"{row.key}/{lens_type}: probs and tokens disagree on layer count"
            for index, layer in enumerate(top_probs):
                assert len(layer) == row["lensTopK"], (
                    f"{row.key}/{lens_type}: layer {index} returned {len(layer)} probs"
                )
                for prob in layer:
                    assert (  # noqa: PLR0124
                        prob == prob
                    ), f"{row.key}/{lens_type}: NaN probability at layer {index}"
                    assert 0.0 <= prob <= 1.0, (
                        f"{row.key}/{lens_type}: probability {prob} outside [0, 1] at layer {index}"
                    )
                assert layer == sorted(layer, reverse=True), (
                    f"{row.key}/{lens_type}: layer {index} probs are not descending"
                )


@pytest.mark.parametrize("row", PARAMS)
def test_fitted_lens_transports_and_reads_out_no_later(row: Expectation):
    """A JACOBIAN_LENS row must show the fitted transport actually doing something.

    Two ways this path can be broken while still returning a well-formed response, so both are
    checked here:

    The transport is skipped. `prompt.py` applies `J_bar` only where `lens is not None` and the
    layer is fitted, so a lens that resolved to the wrong model, or a fitted-layer set that does
    not line up with the layers being read out, degrades silently into a plain logit lens. That
    response passes every other assertion in this file, so the two read-outs are required to
    differ somewhere.

    The transport is applied but wrong. A lens fitted for a different checkpoint is still a
    stack of correctly-shaped [d_model, d_model] matmuls; what it stops doing is helping. So the
    answer must surface at least as early under the fitted lens as under the raw read-out, which
    is the entire point of fitting one.
    """
    if "JACOBIAN_LENS" not in row["lensTypes"]:
        pytest.skip("row does not bind a fitted lens")

    with _server(row) as client:
        readouts = _lens_readouts(row, client)
        logit_tokens, _ = readouts["LOGIT_LENS"]
        fitted_tokens, _ = readouts["JACOBIAN_LENS"]

        assert fitted_tokens != logit_tokens, (
            f"{row.key}: the JACOBIAN_LENS read-out is identical to LOGIT_LENS at every "
            f"layer, so the fitted transport was not applied"
        )

        answer = row["lensAnswer"]
        logit_hits = _answer_layers(logit_tokens, answer)
        fitted_hits = _answer_layers(fitted_tokens, answer)
        assert fitted_hits and logit_hits, (
            f"{row.key}: {answer!r} never reaches the top-k (logit={logit_hits}, fitted={fitted_hits})"
        )
        assert fitted_hits[0] <= logit_hits[0], (
            f"{row.key}: the fitted lens first recovers {answer!r} at layer "
            f"{fitted_hits[0]}, later than the raw read-out at layer {logit_hits[0]}"
        )


# --- thinking template ------------------------------------------------------


def _chat_lens(row: Expectation, client: TestClient, *, thinking: bool) -> dict[str, Any]:
    return _post(
        client,
        "/v1/lens/prompt",
        {
            "model": row.spec.model_id,
            "chat": [{"role": "user", "content": row["chatPrompt"]}],
            "type": row["lensTypes"],
            "top_n": row["lensTopK"],
            "num_completion_tokens": row["chatTokens"],
            "temperature": 0,
            "stream": False,
            "enable_thinking": thinking,
        },
    )


@pytest.mark.parametrize("row", PARAMS)
def test_thinking_toggle_reaches_generation(row: Expectation):
    """`enable_thinking` must change what the model GENERATES, not just how it is templated.

    The existing coverage asserts the toggle changes the rendered token ids. That passes for a
    flag threaded into the chat template and then dropped before generation -- which looks
    fine in the prompt and produces the wrong output. Asserting the channel of the generated
    tokens is what closes that gap.
    """
    row.require("chatPrompt")
    channel = row["thinkingChannel"]
    with _server(row) as client:
        on = _chat_lens(row, client, thinking=True)
        generated = [token for token in on["tokens"] if token.get("is_generated")]
        assert len(generated) == row["chatTokens"], (
            f"{row.key}: asked for {row['chatTokens']} tokens, got {len(generated)} generated"
        )
        assert all(token.get("channel") == channel for token in generated), (
            f"{row.key}: thinking-on generation should be on the {channel!r} channel, "
            f"got {sorted({str(t.get('channel')) for t in generated})}"
        )

        off = _chat_lens(row, client, thinking=False)
        off_generated = [token for token in off["tokens"] if token.get("is_generated")]
        assert not any(token.get("channel") == channel for token in off_generated), (
            f"{row.key}: thinking-off generation should not be on the {channel!r} channel"
        )
        # Thinking off closes the block in the template, so the model answers immediately
        # instead of reasoning -- the visible difference the toggle exists to produce.
        answer = "".join(token["token"] for token in off_generated)
        assert re.search(row["thinkingOffPattern"], answer), (
            f"{row.key}: thinking-off answer {answer!r} does not match {row['thinkingOffPattern']!r}"
        )


# --- SAE activation ---------------------------------------------------------


@pytest.mark.parametrize("row", PARAMS)
def test_curated_feature_fires_on_its_own_token(row: Expectation):
    """The feature must peak on the token it is supposed to be about.

    Asserting the position and not just the magnitude is what separates "the SAE encode works"
    from "a number came back": a wrong layer, a wrong feature index, or an encode applied to
    the wrong hook point all still return a plausible-looking array of floats.
    """
    source = row.require("saeSource")
    with _server(row) as client:
        body = _post(
            client,
            "/v1/activation/single",
            {
                "prompt": row["activationText"],
                "model": row.spec.model_id,
                "source": source,
                "index": str(row["saeIndex"]),
            },
        )
        activation, tokens = body["activation"], body["tokens"]
        peak = tokens[activation["maxValueIndex"]]
        assert peak == row["activationMaxToken"], (
            f"{row.key}: {source}/{row['saeIndex']} peaked on {peak!r}, "
            f"expected {row['activationMaxToken']!r} (tokens: {tokens})"
        )

        low, high = row["activationMaxRange"]
        assert low <= activation["maxValue"] <= high, (
            f"{row.key}: peak activation {activation['maxValue']} outside the measured band [{low}, {high}]"
        )


# --- steering ---------------------------------------------------------------


@pytest.mark.parametrize("row", PARAMS)
def test_steering_redirects_the_continuation(row: Expectation):
    """Steering at the calibrated strength must write the concept -- and the unsteered run must not.

    Both halves matter. A steer case whose DEFAULT output already matches the pattern passes
    with the steering vector removed entirely, which is the easiest way to write a steer test
    that proves nothing; the RUNBOOK's standing instruction to read the unsteered output first
    is that check, made automatic. Both continuations come from one request, so they share a
    seed and a backend and differ only in the steering.
    """
    strength = row.require("steerStrength")
    source = row.require("saeSource")
    with _server(row) as client:
        body = _post(
            client,
            "/v1/steer/completion",
            {
                "prompt": row["steerPrompt"],
                "model": row.spec.model_id,
                "steer_method": "SIMPLE_ADDITIVE",
                "normalize_steering": False,
                "types": ["STEERED", "DEFAULT"],
                "features": [
                    {
                        "model": row.spec.model_id,
                        "source": source,
                        "index": row["saeIndex"],
                        "strength": strength,
                    }
                ],
                "n_completion_tokens": row["steerTokens"],
                "temperature": 0,
                "strength_multiplier": 1.0,
                "freq_penalty": 0.0,
                "seed": row["steerSeed"],
            },
        )
        outputs = {entry["type"]: entry["output"] for entry in body["outputs"]}
        steered, default = outputs["STEERED"], outputs["DEFAULT"]
        pattern = row["steerPattern"]

        assert re.search(pattern, steered), (
            f"{row.key}: steering {source}/{row['saeIndex']} at {strength} produced {steered!r}, "
            f"which does not match {pattern!r}"
        )
        assert not re.search(pattern, default), (
            f"{row.key}: the UNSTEERED continuation {default!r} already matches {pattern!r}, "
            "so this case would pass with the steering removed -- pick a different prompt or feature"
        )
        assert re.search(row["unsteeredPattern"], default), (
            f"{row.key}: unsteered continuation {default!r} does not match the measured {row['unsteeredPattern']!r}"
        )


# --- the data file itself ---------------------------------------------------


def test_every_matrix_model_has_expectations():
    """The YAML and the harness matrix must not drift apart.

    Without this, adding a model to harness.py silently buys it no behavioral coverage -- CI
    would boot a server for it, run the structural tests, and report green.
    """
    missing = sorted(set(MODELS) - {row.key for row in ROWS})
    assert not missing, f"models in harness.MODELS with no model_expectations.yaml row: {missing}"


def test_steer_rows_are_complete():
    """A row that steers needs the feature, the strength, and both patterns.

    Checked as data rather than discovered as a KeyError inside a test that has already paid
    to boot a server.
    """
    for row in ROWS:
        if "steerStrength" not in row.values:
            continue
        for field in (
            "saeSource",
            "saeIndex",
            "steerPattern",
            "unsteeredPattern",
            "steerPrompt",
        ):
            assert field in row.values, f"{row.key} sets steerStrength but not {field}"


def test_manual_rows_are_not_in_the_ci_matrix():
    """A `manual` row naming a matrix model would be marked xl and so run nowhere."""
    for row in ROWS:
        if row.get("tier") == "manual":
            assert row.key not in MODELS, f"{row.key} is in the CI matrix but marked tier: manual"
