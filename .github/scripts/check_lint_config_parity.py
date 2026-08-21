#!/usr/bin/env python3
"""Fail if the shared ruff/pyright config has drifted between the Python projects.

Every project under `apps/` carries the same lint and type-check block, copied verbatim, because
neither ruff's nor pyright's pyproject config can be inherited from a file outside the project
directory. Duplication is the price of that, and this script is what keeps the copies honest: it
compares the rule-shaped keys and ignores the path-shaped ones, which are legitimately per-project
(`exclude`, `include`, `extraPaths`, `reportMissingImports`).

`interp-engine` used to be in this list. It is now its own repository
(decoderesearch/interp-engine) and keeps a copy of the same block, checked by a parity job of its
own. Nothing here can reach it, so the two are kept in step by convention rather than by this gate.

Run it from the repo root: `python3 .github/scripts/check_lint_config_parity.py`
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]

PROJECTS = [
    "apps/autointerp",
    "apps/graph",
    "apps/inference",
    "apps/nla",
    "apps/sparsity",
]

# The reference the others are compared against. Nothing special about it beyond being the
# largest Python project here, so it is the one whose config gets read the most.
REFERENCE = "apps/inference"

# Dotted paths into pyproject.toml that must be identical everywhere. Deliberately excluded:
# `tool.pyright.exclude` / `include` / `extraPaths` / `reportMissingImports` and
# `tool.ruff.extend-exclude`, which name per-project directories and optional dependencies.
SHARED_KEYS = [
    "tool.ruff.line-length",
    "tool.ruff.lint.select",
    "tool.ruff.lint.ignore",
    "tool.ruff.lint.flake8-bugbear.extend-immutable-calls",
    "tool.ruff.lint.flake8-tidy-imports.banned-api",
    "tool.pyright.typeCheckingMode",
    "tool.pyright.reportMissingTypeStubs",
]

MISSING = object()


def dig(table: dict[str, Any], dotted: str) -> Any:
    node: Any = table
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return MISSING
        node = node[part]
    return node


def main() -> int:
    configs = {
        p: tomllib.loads((REPO_ROOT / p / "pyproject.toml").read_text())
        for p in PROJECTS
    }

    problems: list[str] = []
    for key in SHARED_KEYS:
        expected = dig(configs[REFERENCE], key)
        if expected is MISSING:
            problems.append(f"{REFERENCE} is the reference but has no `{key}`")
            continue
        for project in PROJECTS:
            if project == REFERENCE:
                continue
            actual = dig(configs[project], key)
            if actual is MISSING:
                problems.append(f"{project} is missing `{key}` (expected {expected!r})")
            elif actual != expected:
                problems.append(
                    f"{project} has `{key}` = {actual!r}, expected {expected!r}"
                )

    if problems:
        print("Shared lint/type config has drifted between projects:\n")
        for problem in problems:
            print(f"  - {problem}")
        print(
            f"\nCopy the block from {REFERENCE}/pyproject.toml, or update every project together."
        )
        return 1

    print(f"Shared lint/type config is identical across {len(PROJECTS)} projects.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
