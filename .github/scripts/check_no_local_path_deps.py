#!/usr/bin/env python3
"""Reject local filesystem dependencies in any committed pyproject.toml or uv.lock.

Every app here depends on interp-engine as a pinned release from PyPI, but the engine is worked on
in its own repository (decoderesearch/interp-engine), so pointing an app at a local checkout is a
routine thing to do mid-change. `make engine-link` does it through an editable install and a
gitignored marker file, touching nothing that is committed. Doing it by hand -- editing
`[tool.uv.sources]` or letting `uv lock` write the result -- produces a diff that looks harmless and
is not: a path source resolves against one person's home directory, so every other checkout and
every CI job fails at install time with `Distribution not found at: file:///...`, and the version
that ships is whatever happened to be in that directory rather than a release anyone can fetch.

This script is what makes that mistake loud at review time instead. It reads the committed files
only -- no venv, no network, no install -- so it is cheap enough to run on every push.

What is allowed:
  - `virtual = "."` and `editable = "."`, which is how uv records the project being locked itself.
  - git and registry sources, including the pinned circuit-tracer fork in apps/graph.

Run with no arguments from the repo root. Exits 1 and prints file, line and remedy on a violation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]

# `virtual`/`editable` pointing at the project's own directory is uv describing the thing it just
# locked, present in all five locks and in nothing anyone edits. Anything else is a real path.
SELF_REFERENTIAL = {".", "./"}

# Matches the `source = { <kind> = "<value>" }` line uv writes for each locked package.
LOCK_SOURCE_RE = re.compile(r'source = \{\s*(editable|directory|path)\s*=\s*"([^"]*)"')


def check_pyproject(path: Path) -> list[str]:
    """Return one message per local path source declared in `path`'s [tool.uv.sources]."""
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        # Reported rather than raised: a hand-edited source is the likeliest way this file stops
        # parsing, so a reader here wants the filename and the remedy below, not a traceback.
        return [f"{path.relative_to(REPO_ROOT)}: not valid TOML ({error})."]
    sources = data.get("tool", {}).get("uv", {}).get("sources", {})
    if not isinstance(sources, dict):
        return []

    problems: list[str] = []
    for name, spec in sources.items():
        # A source may be a list of specs, each gated by a marker. Normalize to a list.
        for entry in spec if isinstance(spec, list) else [spec]:
            if not isinstance(entry, dict):
                continue
            for key in ("path", "directory"):
                value = entry.get(key)
                if value is None or str(value) in SELF_REFERENTIAL:
                    continue
                problems.append(
                    f"{path.relative_to(REPO_ROOT)}: [tool.uv.sources] {name} = {{ {key} = "
                    f'"{value}" }} points at a local directory.'
                )
    return problems


def check_lock(path: Path) -> list[str]:
    """Return one message per local path source recorded in the lockfile at `path`."""
    problems: list[str] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        match = LOCK_SOURCE_RE.search(line)
        if match is None:
            continue
        kind, value = match.groups()
        if value in SELF_REFERENTIAL:
            continue
        problems.append(
            f"{path.relative_to(REPO_ROOT)}:{lineno}: locked against a local directory "
            f'({kind} = "{value}").'
        )
    return problems


def main() -> int:
    problems: list[str] = []
    for app in sorted((REPO_ROOT / "apps").iterdir()):
        pyproject = app / "pyproject.toml"
        if pyproject.is_file():
            problems += check_pyproject(pyproject)
        lock = app / "uv.lock"
        if lock.is_file():
            problems += check_lock(lock)

    if not problems:
        print("No local path dependencies in any committed pyproject.toml or uv.lock.")
        return 0

    print("Local path dependencies found in committed files:\n", file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    print(
        "\nThese resolve against one machine's filesystem, so every other checkout and every CI\n"
        "job fails to install. To develop against a local interp-engine checkout, revert these\n"
        "files and use the marker-file workflow instead, which commits nothing:\n"
        "\n"
        "  git checkout -- apps/*/pyproject.toml apps/*/uv.lock\n"
        "  make engine-link APP=<app> [ENGINE_SRC=path/to/interp-engine]\n"
        "  make engine-status      # which apps are linked\n"
        "  make engine-unlink APP=<app>\n"
        "\n"
        "To ship an engine change instead, release it from decoderesearch/interp-engine and bump\n"
        "the `interp-engine==` pin in the app's pyproject.toml.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
