"""Guards the committed ``openapi.json`` against the models it is derived from.

Deriving the spec from pydantic only helps if the committed artifact is regenerated; nothing
about editing a model forces you to run ``make openapi``. This turns forgetting into a failed
test rather than a spec that quietly describes the wrong server.

Not run in CI: importing the server needs torch and vLLM, and there is no nla test job with
them installed. Run it locally after touching a model. The spec-to-TypeScript half of the chain
*is* CI-gated, by the openapi-drift workflow.
"""

import json

from dump_openapi import OUTPUT_PATH, render
from server import NlaSchema


def test_committed_openapi_is_current():
    assert OUTPUT_PATH.read_text() == render(), (
        "openapi.json no longer matches the models it is generated from. "
        "Run `make openapi` in apps/nla and commit the result."
    )


def test_every_endpoint_documents_its_response_body():
    """A handler with no documented response still serves fine but documents ``{}``."""
    schema = json.loads(render())
    undocumented = {
        f"{method.upper()} {path}"
        for path, operations in schema["paths"].items()
        for method, operation in operations.items()
        if not operation["responses"].get("200", {}).get("content", {}).get("application/json", {}).get("schema")
    }
    assert undocumented == set(), f"undocumented response body: {sorted(undocumented)}"


def test_field_names_stay_snake_case():
    """These names are persisted and published; see the note above the models in server.py.

    ``NlaExplainCache.resultJson`` stores ``ExplainResult`` records verbatim and those rows back
    permanent public share URLs, so an alias generator added here -- by reflex, to match
    inference and autointerp -- would split that column into two casings with no discriminator
    and no migration path.
    """
    assert NlaSchema.model_config.get("alias_generator") is None

    schema = json.loads(render())
    camel_cased = sorted(
        f"{name}.{field}"
        for name, definition in schema["components"]["schemas"].items()
        for field in definition.get("properties", {})
        if any(character.isupper() for character in field)
    )
    assert camel_cased == [], f"expected snake_case on the wire, found: {camel_cased}"
