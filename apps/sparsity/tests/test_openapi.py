"""Guards the committed ``openapi.json`` against the models it is derived from.

Deriving the spec from pydantic only helps if the committed artifact is regenerated; nothing
about editing ``schemas.py`` forces you to run ``make openapi``. This turns forgetting into a
failed test rather than a spec that quietly describes the wrong server.

Unlike the inference and autointerp equivalents, this file is not run in CI -- importing the
server pulls torch and transformers, and there is no sparsity test job to hang that off. Run
it locally (``make test`` in this directory) after touching a model. The spec-to-TypeScript
half of the chain *is* CI-gated, by the openapi-drift workflow.
"""

import json

from dump_openapi import OUTPUT_PATH, render
from schemas import SparsitySchema


def test_committed_openapi_is_current():
    assert OUTPUT_PATH.read_text() == render(), (
        "openapi.json no longer matches the models it is generated from. "
        "Run `make openapi` in apps/sparsity and commit the result."
    )


def test_every_endpoint_documents_its_response_body():
    """A handler with no documented response still serves fine but documents ``{}``.

    That is the state this server was in before: three endpoints returning hand-built dicts,
    mirrored by hand in two webapp files that nothing checked.
    """
    schema = json.loads(render())
    undocumented = {
        f"{method.upper()} {path}"
        for path, operations in schema["paths"].items()
        for method, operation in operations.items()
        if not operation["responses"].get("200", {}).get("content", {}).get("application/json", {}).get("schema")
    }
    assert undocumented == set(), f"undocumented response body: {sorted(undocumented)}"


def test_field_names_stay_snake_case():
    """This server's names are a public contract; see the note in schemas.py.

    ``/api/sparsity/connected-neurons`` forwards ``trace_forward`` and ``trace_backward``
    nearly verbatim, and two webapp files read the nested ``read_weight`` / ``via_channel``
    fields. Adding an alias generator to ``SparsitySchema`` -- by reflex, to match inference
    and autointerp -- would rename all of them, and nothing else here would fail.
    """
    assert SparsitySchema.model_config.get("alias_generator") is None

    schema = json.loads(render())
    camel_cased = sorted(
        f"{name}.{field}"
        for name, definition in schema["components"]["schemas"].items()
        for field in definition.get("properties", {})
        if any(character.isupper() for character in field)
    )
    assert camel_cased == [], f"expected snake_case on the wire, found: {camel_cased}"
