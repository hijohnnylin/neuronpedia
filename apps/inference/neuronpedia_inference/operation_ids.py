"""Name operations the way the published SDK reads best.

FastAPI derives ``operation_id`` from the handler name plus the path plus the verb, which is
reliably unique and reliably unreadable: it turns ``POST /v1/explain/default`` into
``explanation_endpoint_v1_explain_default_post``, and openapi-generator turns that straight into
the client's method name.

Nothing in this repo reads operation ids -- the webapp's generated types key off paths, and the
``operations`` map in the ``.d.ts`` is only referenced from inside that generated file -- so they
exist for exactly one audience: whoever calls `neuronpedia-inference-client`.

The rule below reproduces the names that audience already has. The hand-written spec this server
replaced declared ``servers: [{url: /v1}]`` and carried no operation ids at all, so the generator
fell back to naming from path plus verb, with the ``/v1`` living in the server URL rather than the
path: ``/explain/default`` + POST became ``explainDefaultPost`` in TypeScript and
``explain_default_post`` in Python. Matching that means moving to a FastAPI-derived spec does not
rename every method in the SDK.

An identical copy of this lives in ``apps/autointerp``. The two servers are separate projects with
no shared package, so the rule is duplicated on purpose rather than imported across the boundary.
"""

from fastapi.routing import APIRoute

# Carried in the old spec's server URL rather than in its paths, so it never reached a method name.
_VERSION_PREFIX = "/v1"


def sdk_operation_id(route: APIRoute) -> str:
    """Build ``explainDefaultPost`` from ``POST /v1/explain/default``."""
    path = route.path_format
    if path.startswith(f"{_VERSION_PREFIX}/"):
        path = path[len(_VERSION_PREFIX) :]

    words = [
        word
        for segment in path.strip("/").split("/")
        # No route here is parameterized, but stripping braces keeps a future `/{id}` readable.
        for word in segment.strip("{}").split("-")
        if word
    ]
    # Every route in this app declares a single method; sorting only makes the pick deterministic.
    words.append(sorted(route.methods)[0].lower())
    return words[0] + "".join(word.capitalize() for word in words[1:])
