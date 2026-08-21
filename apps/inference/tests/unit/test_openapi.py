"""Guards the committed ``openapi.json`` against the models it is derived from.

The whole point of deriving the spec from pydantic rather than hand-writing it is that
the two cannot disagree -- but that only holds if the committed artifact is regenerated.
Nothing about editing a schema forces you to run ``make openapi``, so this is what turns
forgetting into a failed test instead of a spec that quietly describes the wrong server.
"""

import json
import re

from dump_openapi import OUTPUT_PATH, render

# Endpoints whose 200 body genuinely has no JSON schema to document. Each entry needs a
# reason, because the default for a new endpoint must be "documented" -- an unexplained
# exemption here is how the previous spec ended up with zero typed responses.
UNDOCUMENTED_BY_DESIGN = {
    # Returns None; it exists to trigger loading, not to carry a payload.
    "POST /initialize",
    # Emits NDJSON, one frame per line, so there is no single response body. The frame
    # models live in endpoints/lens/prompt.py.
    "POST /v1/lens/prompt",
    # Operational introspection whose shape tracks the backend rather than a wire contract:
    # both splice in whatever the loaded engine reports.
    "GET /v1/capabilities",
    "GET /v1/steer/health",
}


def test_committed_openapi_is_current():
    assert OUTPUT_PATH.read_text() == render(), (
        "openapi.json no longer matches the models it is generated from. "
        "Run `make openapi` in apps/inference and commit the result."
    )


def test_every_endpoint_documents_its_response_body():
    """A handler with no documented response still serves fine but documents ``{}``.

    That is how the previous hand-written spec drifted: the generated clients ended up
    with untyped responses and nobody noticed, because nothing fails when a response
    schema is missing.
    """
    schema = json.loads(render())
    undocumented = {
        f"{method.upper()} {path}"
        for path, operations in schema["paths"].items()
        for method, operation in operations.items()
        if not operation["responses"].get("200", {}).get("content", {}).get("application/json", {}).get("schema")
    }
    assert undocumented == UNDOCUMENTED_BY_DESIGN, (
        f"newly undocumented: {sorted(undocumented - UNDOCUMENTED_BY_DESIGN)}, "
        f"now documented (drop from the exemption list): {sorted(UNDOCUMENTED_BY_DESIGN - undocumented)}"
    )


def test_operation_ids_are_sdk_shaped():
    """Operation ids are the published SDK's method names, and nothing else reads them.

    FastAPI's default is handler name + path + verb, which yields
    ``activation_all_v1_activation_all_post`` and an SDK method to match. ``sdk_operation_id``
    replaces that with the path-plus-verb naming the hand-written spec used to give those
    clients. Dropping the ``generate_unique_id_function`` would rename every method in the SDK
    while changing nothing the webapp compiles against, so nothing else would catch it.
    """
    schema = json.loads(render())
    ids = [operation["operationId"] for path in schema["paths"].values() for operation in path.values()]

    assert len(ids) == len(set(ids)), f"duplicate operation ids: {sorted({i for i in ids if ids.count(i) > 1})}"
    assert all(re.fullmatch(r"[a-z][A-Za-z0-9]*", i) for i in ids), (
        f"not camelCase, so the generate_unique_id_function is probably gone: "
        f"{sorted(i for i in ids if not re.fullmatch(r'[a-z][A-Za-z0-9]*', i))}"
    )
    # A sample of the names the currently published SDK already exposes.
    assert {"activationAllPost", "steerCompletionPost", "utilSaeTopkByDecoderCossimPost"} <= set(ids)


def test_no_number_or_integer_unions_reach_the_spec():
    """``float | int`` generates a junk wrapper class in every published SDK.

    A field written ``StrictFloat | StrictInt`` becomes ``anyOf: [number, integer]``, and
    openapi-generator turns each one into a whole dispatcher class named after the field --
    ``Probability``, ``Maxvalue``, ``Seed`` -- so ``list[float]`` reaches users as
    ``List[ValuesInner]``. ``StrictFloat`` alone already accepts an int and emits a plain
    ``number``, so the union buys nothing and costs that.

    It is an easy mistake to reintroduce, because the Python generator *emits*
    ``Union[StrictFloat, StrictInt]`` for ``type: number``. That is how these got here in the
    first place: they were copied in from the generated client when the models moved into this
    server. Nothing downstream complains, because openapi-typescript collapses the union back
    to ``number`` and the webapp never sees it.
    """
    schema = json.loads(render())

    def offenders(node: object, path: str = "") -> list[str]:
        found: list[str] = []
        if isinstance(node, dict):
            any_of = node.get("anyOf")
            if isinstance(any_of, list):
                types = {branch.get("type") for branch in any_of if isinstance(branch, dict)}
                if types == {"number", "integer"}:
                    found.append(path)
            for key, value in node.items():
                found += offenders(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                found += offenders(value, f"{path}[{i}]")
        return found

    assert not offenders(schema), (
        f"number|integer unions at: {offenders(schema)}. Use StrictFloat on its own -- it "
        "accepts an int too, and produces a plain `number`."
    )


def test_secret_key_auth_is_documented():
    """The X-SECRET-KEY check is middleware, which FastAPI cannot see by itself.

    Without the manual injection in server.py the spec would claim every endpoint is open,
    and the generated SDKs would believe it.
    """
    schema = json.loads(render())
    assert schema["components"]["securitySchemes"]["SimpleSecretAuth"]["name"] == "X-SECRET-KEY"
    assert schema["security"] == [{"SimpleSecretAuth": []}]
    # /health is exempt in the middleware, so it must be exempt in the spec too.
    assert schema["paths"]["/health"]["get"]["security"] == []
