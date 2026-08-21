#!/usr/bin/env python3
"""Generate the public SDK packages from a committed openapi.json and publish them.

The SDKs are the one generated artifact in this repo that nothing in this repo consumes. The
webapp compiles against `apps/webapp/lib/api/<app>.d.ts` and the servers own their pydantic
models, so `neuronpedia-{inference,autointerp}-client` exists purely for callers outside the
repo. That is why they are built in CI and committed nowhere -- see the "Cross-server APIs"
section of AGENTS.md.

Because nothing here depends on the output, a path filter on this job would go silent in exactly
the case that matters, and nobody would notice for months. So the workflow runs on every push to
main and this script decides for itself whether there is anything to do: it compares a hash of
the spec against the `openapiHash` field stamped into the last published npm package. Asking the
registry rather than a git diff makes the check self-healing -- a publish missed for any reason
(expired token, registry outage, a filter that would not have matched) is simply retried on the
next push, and no scheduled drift job is needed.

Publishing is opt-in: without `--publish` this does everything except the two upload commands.
That is what makes it safe to run locally, and it is what CI does until the repository variable
OPENAPI_PUBLISH_ENABLED is set to "true".

Ordering note: PyPI is uploaded before npm, and only the npm package carries `openapiHash`. That
is deliberate. The hash is the "everything succeeded" marker, so it must be written last; if npm
fails after PyPI succeeded, the next push retries both and twine's --skip-existing makes the
repeat upload a no-op. Stamping the hash first would strand a half-published version forever.

Usage:
    python3 .github/scripts/publish_sdk.py --service autointerp            # dry run
    python3 .github/scripts/publish_sdk.py --service inference --publish   # for CI
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Pinned so a generator release cannot silently reshape a public API. This is the version that
# built the packages currently on the registries.
OPENAPI_GENERATOR_VERSION = "7.24.0"

# Only these two servers have published clients. graph, nla and sparsity produce a spec and
# webapp types like everybody else, but nothing outside the repo calls them, so they get no SDK.
SERVICES = ("inference", "autointerp")

REPOSITORY_URL = "https://github.com/hijohnnylin/neuronpedia"
HOMEPAGE = "https://neuronpedia.org"

# The generators leave these placeholders in the package metadata when the spec does not carry a
# repository, and the packages currently on the registries were published with them intact.
PLACEHOLDER_REPO = "https://github.com/GIT_USER_ID/GIT_REPO_ID"


class Failure(Exception):
    """A condition the operator needs to fix; reported without a traceback."""


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command, echoing it first so CI logs show exactly what happened."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env={**os.environ, **(env or {})},
        capture_output=capture,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        if capture:
            sys.stderr.write(proc.stdout or "")
            sys.stderr.write(proc.stderr or "")
        raise Failure(f"command failed with exit {proc.returncode}: {' '.join(cmd)}")
    return proc


# --------------------------------------------------------------------------------------------
# Version resolution
# --------------------------------------------------------------------------------------------


def version_key(version: str) -> tuple[int, int, int] | None:
    """Parse `x.y.z` into a comparable tuple, or None if it is not a plain release version.

    Pre-releases and anything else non-numeric return None so they are skipped as bump
    candidates rather than crashing or, worse, sorting as zero.
    """
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version.strip())
    if not match:
        return None
    return (int(match[1]), int(match[2]), int(match[3]))


def resolve_version(
    npm_version: str | None, pypi_version: str | None, spec_version: str
) -> str:
    """Pick the version to publish: a patch bump of the registries, or the spec, whichever is higher.

    Patch bumps happen automatically because the common case is a wire-format tweak. For a minor
    or major, raise `FastAPI(version=...)` in the server by hand and this takes the higher of the
    two, so the manual bump wins without needing a separate input.

    Both registries are considered so the two cannot drift apart: if a previous run published to
    PyPI and then failed on npm, bumping from npm alone would reuse a version PyPI already has.
    """
    candidates: dict[str, tuple[int, int, int]] = {}

    registry_versions = [v for v in (npm_version, pypi_version) if v and version_key(v)]
    if registry_versions:
        highest = max(registry_versions, key=lambda v: version_key(v))  # type: ignore[arg-type]
        major, minor, patch = version_key(highest)  # type: ignore[misc]
        bumped = f"{major}.{minor}.{patch + 1}"
        candidates[bumped] = (major, minor, patch + 1)

    spec_key = version_key(spec_version)
    if spec_key:
        candidates[spec_version] = spec_key

    if not candidates:
        raise Failure(
            f"cannot resolve a version: registries gave {npm_version!r}/{pypi_version!r} and the "
            f"spec gave {spec_version!r}, none of which is a plain x.y.z"
        )
    return max(candidates, key=lambda v: candidates[v])


# --------------------------------------------------------------------------------------------
# Registry queries
# --------------------------------------------------------------------------------------------


def npm_metadata(package: str) -> dict[str, object]:
    """Return the latest published package.json from npm, or {} if never published."""
    proc = run(["npm", "view", package, "--json"], capture=True, check=False)
    if proc.returncode != 0 or not (proc.stdout or "").strip():
        return {}
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise Failure(f"could not parse `npm view {package} --json`: {exc}") from exc
    return data if isinstance(data, dict) else {}


def pypi_version(package: str) -> str | None:
    """Return the latest version on PyPI, or None if the project does not exist."""
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        # Fixed https URL, not attacker-controlled: `package` is built from the service name.
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)["info"]["version"]
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


# --------------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------------


def generate(
    spec: Path, out_dir: Path, generator: str, extra: list[str], *, cwd: Path
) -> None:
    """Run the generator, with everything it touches confined to `cwd`.

    `cwd` matters: openapi-generator-cli writes a default `openapitools.json` into the working
    directory when it does not find one, so running this from the repo root silently re-creates
    the config file that was deliberately deleted with the rest of the old pipeline. Pointing it
    at the temp directory keeps the checkout clean. The generator version is pinned through the
    environment instead, which is why that file is not needed.
    """
    env = {"OPENAPI_GENERATOR_VERSION": OPENAPI_GENERATOR_VERSION}
    run(
        [
            "npx",
            "--yes",
            "@openapitools/openapi-generator-cli",
            "generate",
            "-i",
            str(spec),
            "-g",
            generator,
            "-o",
            str(out_dir),
            *extra,
        ],
        cwd=cwd,
        env=env,
    )


def stamp_typescript(
    pkg_dir: Path, *, version: str, spec_hash: str, title: str
) -> None:
    """Write the fields the generator cannot know, including the hash the next run reads back."""
    manifest_path = pkg_dir / "package.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = version
    manifest["description"] = f"Generated TypeScript client for the {title}."
    manifest["homepage"] = HOMEPAGE
    # The `git+` prefix is the form npm normalizes to; writing it directly avoids a publish warning.
    manifest["repository"] = {"type": "git", "url": f"git+{REPOSITORY_URL}.git"}
    manifest["author"] = "Neuronpedia"
    # Read by the next run of this script to decide whether anything changed. Keep it last so it
    # is obvious in a diff that this is metadata rather than something npm interprets.
    manifest["openapiHash"] = spec_hash
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def distributions(dist_dir: Path) -> list[str]:
    """The wheel and sdist, and nothing else.

    `uv build` also writes a `.gitignore` into the output directory, which twine rejects as an
    unknown distribution format, so the directory cannot be passed wholesale.
    """
    found = sorted(
        str(path)
        for path in dist_dir.iterdir()
        if path.suffix == ".whl" or path.name.endswith(".tar.gz")
    )
    if not found:
        raise Failure(f"no wheel or sdist was produced in {dist_dir}")
    return found


def stamp_python(pkg_dir: Path, *, spec_hash: str) -> None:
    """Fix the placeholder repository URL and record the hash for traceability.

    Unlike the npm side this hash is never read back -- PyPI has no equivalent of `npm view` for
    arbitrary metadata -- but having it in the sdist makes it possible to tell which spec a given
    release came from.
    """
    pyproject = pkg_dir / "pyproject.toml"
    text = pyproject.read_text().replace(PLACEHOLDER_REPO, REPOSITORY_URL)
    text += f'\n[tool.neuronpedia]\nopenapi_hash = "{spec_hash}"\n'
    pyproject.write_text(text)


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------


def release_needed(service: str, *, force: bool) -> bool:
    """Answer "is there anything to publish?" using only the spec and one `npm view`.

    `build_and_publish` makes the same comparison and stops early, but by then the workflow has
    already installed a JDK, uv and a Python. This is the same question asked before any of that,
    so the common case -- a push to main that touches no spec -- costs a checkout and a registry
    lookup instead of minutes of setup. It is a cheaper gate, not a different one, and it is
    deliberately not a `paths:` filter: the answer comes from the registry, so a release missed
    for any reason is still retried on the next push.

    Anything unexpected answers True. A wrong "yes" wastes a few CI minutes and then no-ops in
    the hash check downstream; a wrong "no" is a release that silently never happens.
    """
    if force:
        return True
    spec_path = REPO_ROOT / "apps" / service / "openapi.json"
    if not spec_path.is_file():
        return True
    try:
        spec_hash = hashlib.sha256(spec_path.read_bytes()).hexdigest()
        published = npm_metadata(f"neuronpedia-{service}-client")
    except (Failure, OSError) as exc:
        print(f"gate check failed ({exc}); assuming a release is needed")
        return True
    return published.get("openapiHash") != spec_hash


def build_and_publish(
    service: str, *, publish: bool, force: bool, keep: Path | None
) -> int:
    spec_path = REPO_ROOT / "apps" / service / "openapi.json"
    if not spec_path.is_file():
        raise Failure(f"no spec at {spec_path}; run `make {service}-openapi` first")

    if publish:
        # Checked before anything is generated: a missing token is a five-second failure, not a
        # five-minute one.
        require_tokens()

    spec_bytes = spec_path.read_bytes()
    spec_hash = hashlib.sha256(spec_bytes).hexdigest()
    spec = json.loads(spec_bytes)
    title = spec.get("info", {}).get("title", f"Neuronpedia {service} server")
    spec_version = str(spec.get("info", {}).get("version", "0.0.0"))

    package = f"neuronpedia-{service}-client"
    print(f"\n=== {package} ===")
    print(f"spec        {spec_path.relative_to(REPO_ROOT)}")
    print(f"spec hash   {spec_hash}")

    published = npm_metadata(package)
    published_hash = published.get("openapiHash")
    npm_version = published.get("version")
    print(
        f"npm         {npm_version or '(unpublished)'} hash={published_hash or '(none)'}"
    )

    if published_hash == spec_hash and not force:
        print(
            "up to date: the published SDK was built from this exact spec, nothing to do"
        )
        return 0

    pypi = pypi_version(package)
    print(f"pypi        {pypi or '(unpublished)'}")

    version = resolve_version(
        npm_version if isinstance(npm_version, str) else None, pypi, spec_version
    )
    print(f"spec info.version {spec_version} -> publishing {version}")

    work = Path(tempfile.mkdtemp(prefix=f"sdk-{service}-"))
    py_dir = work / "python"
    ts_dir = work / "typescript"
    try:
        print("\n-- generating python client")
        generate(
            spec_path,
            py_dir,
            "python",
            [
                "--package-name",
                f"neuronpedia_{service}_client",
                f"--additional-properties=packageVersion={version}",
            ],
            cwd=work,
        )
        print("-- generating typescript client")
        generate(
            spec_path,
            ts_dir,
            "typescript-fetch",
            [
                "-p",
                f"npmName={package},npmVersion={version},licenseName=Apache-2.0",
            ],
            cwd=work,
        )

        # Neither generator emits a LICENSE, and Apache 2.0 section 4(a) requires that whoever
        # receives the package receives the license with it. These ship standalone, so the repo
        # root license is copied in rather than left to the repo to supply.
        for target in (py_dir, ts_dir):
            shutil.copy(REPO_ROOT / "LICENSE", target / "LICENSE")
        stamp_python(py_dir, spec_hash=spec_hash)
        stamp_typescript(ts_dir, version=version, spec_hash=spec_hash, title=title)

        print("\n-- building python distributions")
        dist = work / "dist"
        run(["uv", "build", "--sdist", "--wheel", "--out-dir", str(dist)], cwd=py_dir)

        print("\n-- building typescript package")
        run(["npm", "install", "--no-audit", "--no-fund"], cwd=ts_dir)
        run(["npm", "run", "build"], cwd=ts_dir)

        if not publish:
            print("\n-- dry run: validating the artifacts instead of uploading")
            run(["uvx", "twine", "check", *distributions(dist)])
            run(["npm", "publish", "--dry-run"], cwd=ts_dir)
            summarize(
                f"**{package}** would publish `{version}` to npm and PyPI "
                f"(spec hash `{spec_hash[:12]}`). Dry run: nothing was uploaded."
            )
            print(f"\nwould publish {package} {version} to npm and PyPI")
            return 0

        # PyPI first, npm last: see the ordering note in the module docstring.
        print("\n-- uploading to PyPI")
        run(["uvx", "twine", "upload", "--skip-existing", *distributions(dist)])
        print("\n-- publishing to npm")
        run(["npm", "publish", "--access", "public"], cwd=ts_dir)
        summarize(f"**{package}** published `{version}` to npm and PyPI.")
        print(f"\npublished {package} {version}")
        return 0
    finally:
        if keep:
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                work,
                keep / service,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("node_modules"),
            )
            print(f"\nkept generated output at {keep / service}")
        shutil.rmtree(work, ignore_errors=True)


def require_tokens() -> None:
    missing = [
        name
        for name in ("NODE_AUTH_TOKEN", "TWINE_PASSWORD")
        if not os.environ.get(name)
    ]
    if missing:
        raise Failure(
            f"--publish needs {', '.join(missing)} in the environment. "
            "Set NPM_TOKEN and PYPI_API_TOKEN as repository secrets."
        )


def summarize(markdown: str) -> None:
    """Append a line to the GitHub Actions run summary, if we are in Actions."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(markdown + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--service", required=True, choices=SERVICES)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="actually upload. Without this the script stops after validating the artifacts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even when the published openapiHash already matches the spec.",
    )
    parser.add_argument(
        "--keep",
        type=Path,
        help="copy the generated packages here before cleaning up, for inspection.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="print needed=true|false and exit, without generating anything.",
    )
    args = parser.parse_args()

    if args.check_only:
        needed = "true" if release_needed(args.service, force=args.force) else "false"
        print(f"needed={needed}")
        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a", encoding="utf-8") as handle:
                handle.write(f"needed={needed}\n")
        return 0

    try:
        return build_and_publish(
            args.service, publish=args.publish, force=args.force, keep=args.keep
        )
    except Failure as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
