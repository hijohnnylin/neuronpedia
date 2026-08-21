#!/usr/bin/env python3
"""Fail if a committed openapi.json breaks one of the wire-format rules in AGENTS.md.

The committed specs are what the webapp's `lib/api/*.d.ts` and the two published SDKs are built
from, so a rule broken here reaches consumers whether or not the owning service is re-tested.

Each rule this checks is also asserted by the owning app's `test_openapi.py`, against the freshly
rendered spec rather than the committed one. That is deliberate rather than redundant: those suites
need the app installed -- torch at minimum, plus an attribution extra for graph -- so their jobs
carry `paths:` filters and do not run for a change that touches only a spec, only the webapp, or
only this workflow. This script needs no dependencies at all, so it can run on everything, every
time, and catch a hand-edited or half-regenerated artifact that the service jobs never look at.

Run it from the repo root:

    python3 .github/scripts/check_openapi_specs.py
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

# The two services whose models alias to camelCase, and which publish an SDK built from this spec.
CAMEL_CASE_APPS = ("inference", "autointerp")

# The three that stay snake_case on purpose: their field names are already public, through graph's
# S3 blobs, nla's stored NlaExplainCache rows, and sparsity's /api/sparsity/connected-neurons.
SNAKE_CASE_APPS = ("graph", "nla", "sparsity")

ALL_APPS = CAMEL_CASE_APPS + SNAKE_CASE_APPS

# A lowercase character followed by an uppercase one: `maxValue`, but not `MAX_VALUE`. Written to
# spot camelCase specifically rather than "contains a capital", so graph's deliberately shouted
# SteerResponse aliases (DEFAULT_GENERATION and friends) pass without needing an exception here.
CAMEL_CASE = re.compile(r"[a-z0-9][A-Z]")

# What openapi-generator turns into a client method name. FastAPI's default is handler name + path
# + verb (`explanation_endpoint_v1_explain_default_post`), which ships to SDK users as-is; the
# `sdk_operation_id` helper in each app restores the path-plus-verb form the hand-written spec gave
# them. Underscores are the tell that an app lost its `generate_unique_id_function`.
SDK_OPERATION_ID = re.compile(r"^[a-z][A-Za-z0-9]*$")


def spec_path(app: str) -> Path:
    return REPO_ROOT / "apps" / app / "openapi.json"


def iter_property_names(node: Any) -> Iterator[str]:
    """Yield every property name under a components.schemas subtree, nested ones included."""
    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict):
            yield from properties
        for value in node.values():
            yield from iter_property_names(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_property_names(value)


def iter_subschemas(node: Any) -> Iterator[dict[str, Any]]:
    """Yield every dict in the spec, so a union can be found wherever it is nested."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from iter_subschemas(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_subschemas(value)


def check_casing(app: str, spec: dict[str, Any], errors: list[str]) -> None:
    """Each service is either aliased to camelCase or deliberately not; neither may leak."""
    schemas = spec.get("components", {}).get("schemas", {})
    names = sorted(set(iter_property_names(schemas)))

    if app in CAMEL_CASE_APPS:
        offenders = [name for name in names if "_" in name]
        if offenders:
            errors.append(
                f"{app}: {len(offenders)} snake_case field name(s) reached the spec "
                f"({', '.join(offenders[:5])}). Every model must subclass BaseSchema so "
                f"alias_generator=to_camel applies; a hand-built response dict bypasses it."
            )
    else:
        offenders = [name for name in names if CAMEL_CASE.search(name)]
        if offenders:
            errors.append(
                f"{app}: {len(offenders)} camelCase field name(s) reached the spec "
                f"({', '.join(offenders[:5])}). This service is snake_case on the wire because "
                f"its field names are already public; see test_field_names_stay_snake_case."
            )


def check_number_unions(app: str, spec: dict[str, Any], errors: list[str]) -> None:
    """`float | int` becomes anyOf: [number, integer], which the SDK generator mangles.

    openapi-generator materializes one dispatcher class per such field -- Probability, Maxvalue,
    Seed -- so a `list[float]` reaches users as `List[ValuesInner]`. StrictFloat alone already
    accepts an int and emits a plain `number`.
    """
    offenders = []
    for subschema in iter_subschemas(spec):
        for keyword in ("anyOf", "oneOf"):
            members = subschema.get(keyword)
            if not isinstance(members, list):
                continue
            types = {
                member.get("type")
                for member in members
                if isinstance(member, dict) and member.get("type") != "null"
            }
            if {"number", "integer"} <= types:
                offenders.append(subschema.get("title") or keyword)

    if offenders:
        errors.append(
            f"{app}: {len(offenders)} number/integer union(s) in the spec "
            f"({', '.join(sorted(set(offenders))[:5])}). Write StrictFloat, not "
            f"`StrictFloat | StrictInt`; see 'The published SDKs' in AGENTS.md."
        )


def check_operation_ids(app: str, spec: dict[str, Any], errors: list[str]) -> None:
    """Only the two SDK-publishing services need readable method names."""
    if app not in CAMEL_CASE_APPS:
        return

    operation_ids = [
        operation.get("operationId")
        for operations in spec.get("paths", {}).values()
        for operation in operations.values()
        if isinstance(operation, dict)
    ]

    malformed = [
        operation_id
        for operation_id in operation_ids
        if not operation_id or not SDK_OPERATION_ID.match(operation_id)
    ]
    if malformed:
        errors.append(
            f"{app}: {len(malformed)} operationId(s) would become unreadable SDK methods "
            f"({', '.join(str(name) for name in malformed[:3])}). The app must pass "
            f"generate_unique_id_function=sdk_operation_id to FastAPI()."
        )

    duplicates = sorted(
        {name for name in operation_ids if operation_ids.count(name) > 1}
    )
    if duplicates:
        errors.append(
            f"{app}: duplicate operationId(s) ({', '.join(str(n) for n in duplicates)}); "
            f"the generated clients would lose a method."
        )


def main() -> int:
    errors: list[str] = []

    for app in ALL_APPS:
        path = spec_path(app)
        if not path.is_file():
            errors.append(f"{app}: no committed spec at {path.relative_to(REPO_ROOT)}.")
            continue
        try:
            spec = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"{app}: openapi.json is not valid JSON ({exc}).")
            continue

        check_casing(app, spec, errors)
        check_number_unions(app, spec, errors)
        check_operation_ids(app, spec, errors)

    if errors:
        print("Committed OpenAPI specs break the wire-format rules:\n", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(
            "\nSee the 'Cross-server APIs' section of AGENTS.md. If a spec merely looks "
            "stale, regenerate it with `make <app>-openapi`.",
            file=sys.stderr,
        )
        return 1

    print(f"Committed OpenAPI specs look right: {', '.join(ALL_APPS)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
