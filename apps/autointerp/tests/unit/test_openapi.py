"""Guards the committed ``openapi.json`` against the models it is derived from.

The whole point of deriving the spec from pydantic rather than hand-writing it is that
the two cannot disagree -- but that only holds if the committed artifact is regenerated.
Nothing about editing a schema forces you to run ``make openapi``, so this is what turns
forgetting into a failed test instead of a spec that quietly describes the wrong server.
"""

import json
import re

from dump_openapi import OUTPUT_PATH, render


def test_committed_openapi_is_current():
    assert OUTPUT_PATH.read_text() == render(), (
        "openapi.json no longer matches the models it is generated from. "
        "Run `make openapi` in apps/autointerp and commit the result."
    )


def test_every_endpoint_documents_its_response_body():
    """A handler with no return annotation still serves fine but documents ``{}``.

    That is how the previous hand-written spec drifted: the generated clients ended up
    with untyped responses and nobody noticed, because nothing fails when a response
    schema is missing.
    """
    schema = json.loads(render())
    undocumented = [
        f"{method.upper()} {path}"
        for path, operations in schema["paths"].items()
        for method, operation in operations.items()
        if not operation["responses"].get("200", {}).get("content", {}).get("application/json", {}).get("schema")
    ]
    assert not undocumented, f"endpoints with no documented 200 body: {undocumented}"


def test_operation_ids_are_sdk_shaped():
    """Operation ids are the published SDK's method names, and nothing else reads them.

    FastAPI's default is handler name + path + verb, giving
    ``explanation_endpoint_v1_explain_default_post`` and an SDK method to match.
    ``sdk_operation_id`` restores the path-plus-verb naming the hand-written spec used to give
    those clients. Losing it would rename every method in the SDK while changing nothing the
    webapp compiles against, so nothing else would catch it.
    """
    schema = json.loads(render())
    ids = [operation["operationId"] for path in schema["paths"].values() for operation in path.values()]

    assert len(ids) == len(set(ids)), f"duplicate operation ids: {sorted({i for i in ids if ids.count(i) > 1})}"
    assert set(ids) == {"explainDefaultPost", "scoreEmbeddingPost", "scoreFuzzDetectionPost"}, (
        f"operation ids changed, which renames every method in the published SDK: {sorted(ids)}"
    )
    assert all(re.fullmatch(r"[a-z][A-Za-z0-9]*", i) for i in ids)


def test_no_number_or_integer_unions_reach_the_spec():
    """``float | int`` generates a junk wrapper class in every published SDK.

    A field written ``StrictFloat | StrictInt`` becomes ``anyOf: [number, integer]``, and
    openapi-generator turns each one into a dispatcher class named after the field, so
    ``list[float]`` reached users as ``List[ValuesInner]``. ``StrictFloat`` alone already
    accepts an int and emits a plain ``number``.

    Easy to reintroduce, because the Python generator *emits* ``Union[StrictFloat, StrictInt]``
    for ``type: number`` -- which is how these arrived, copied in from the generated client when
    the models moved into this server. Nothing downstream complains: openapi-typescript
    collapses the union back to ``number``, so the webapp never sees it.
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
