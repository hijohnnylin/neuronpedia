"""Guards the committed ``openapi.json`` against the models it is derived from.

Deriving the spec from pydantic only helps if the committed artifact is regenerated; nothing
about editing ``schemas.py`` forces you to run ``make openapi``. This turns forgetting into a
failed test rather than a spec that quietly describes the wrong server.

The spec-to-TypeScript half of the chain is gated separately, by the openapi-drift workflow.
"""

import importlib.util
import json
import os

import pytest

# Importing the server needs an attribution backend, so a machine without one cannot run this
# file at all. Locally that is a skip. Under CI it is a failure instead: the graph job installs
# the extra, and `importorskip` here would turn a job that stopped installing it into a green run
# with these four tests silently absent -- including the only check that the committed spec still
# matches the models.
if importlib.util.find_spec("circuit_tracer") is None:
    if os.environ.get("CI"):
        raise RuntimeError(
            "circuit_tracer is not installed but CI is set, so these tests would skip instead of run. "
            "The graph job must sync with `--extra circuit-tracer`."
        )
    pytest.skip("needs an attribution extra: uv sync --extra circuit-tracer", allow_module_level=True)

from dump_openapi import OUTPUT_PATH, render  # noqa: E402
from neuronpedia_graph.schemas import GraphSchema, SteerResponse  # noqa: E402


def test_committed_openapi_is_current():
    assert OUTPUT_PATH.read_text() == render(), (
        "openapi.json no longer matches the models it is generated from. "
        "Run `make openapi` in apps/graph and commit the result."
    )


def test_every_endpoint_documents_its_response_body():
    """A handler with no documented response still serves fine but documents ``{}``.

    That is the state this server was in before: five endpoints returning hand-built dicts,
    three of them not even binding their request through FastAPI, all mirrored by hand in
    ``apps/webapp/lib/utils/graph.ts``.
    """
    schema = json.loads(render())
    undocumented = {
        f"{method.upper()} {path}"
        for path, operations in schema["paths"].items()
        for method, operation in operations.items()
        if not operation["responses"].get("200", {}).get("content", {}).get("application/json", {}).get("schema")
    }
    assert undocumented == set(), f"undocumented response body: {sorted(undocumented)}"


def test_every_request_body_is_bound_through_fastapi():
    """Three handlers used to read ``await req.json()`` and validate by hand.

    FastAPI cannot see a body read that way, so those endpoints reached the spec with no
    request schema at all and the webapp had nothing to type its payloads against.
    """
    schema = json.loads(render())
    unbound = sorted(
        f"{method.upper()} {path}"
        for path, operations in schema["paths"].items()
        for method, operation in operations.items()
        if method == "post" and "requestBody" not in operation
    )
    assert unbound == [], f"POST endpoints with no request schema: {unbound}"


def test_field_names_stay_snake_case():
    """This server's names are a public contract; see the note in schemas.py.

    ``/api/graph/tokenize`` and ``/api/steer-logits`` forward these responses nearly verbatim,
    and the latter publishes :class:`SteerFeature`'s own field names in its documented request.
    Adding an alias generator to :class:`GraphSchema` -- by reflex, to match inference and
    autointerp -- would rename all of them, and nothing else here would fail.
    """
    assert GraphSchema.model_config.get("alias_generator") is None

    # The steer response is the one place with deliberate aliases, and they are louder than
    # camelCase rather than a casing convention, so they are checked by name instead.
    expected_steer_keys = {
        "DEFAULT_GENERATION",
        "STEERED_GENERATION",
        "DEFAULT_LOGITS_BY_TOKEN",
        "STEERED_LOGITS_BY_TOKEN",
    }
    assert set(SteerResponse.model_json_schema()["properties"]) == expected_steer_keys

    schema = json.loads(render())
    camel_cased = sorted(
        f"{name}.{field}"
        for name, definition in schema["components"]["schemas"].items()
        if name != "SteerResponse"
        for field in definition.get("properties", {})
        if any(character.isupper() for character in field)
    )
    assert camel_cased == [], f"expected snake_case on the wire, found: {camel_cased}"
