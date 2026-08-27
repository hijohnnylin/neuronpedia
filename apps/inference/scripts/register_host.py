"""Register a GPU server with a deployed Neuronpedia webapp.

The webapp routes to GPU servers by reading the ComputeHost table, so a server
is invisible to it until something writes that row -- a pod that is up, warm and
serving is still dead weight until then.

Local development writes to Postgres directly (`make host-add`). A deployment
cannot: the database is not reachable from a laptop, and it should not be. So
registration goes through the admin API instead, which is what this does. It is
both an importable helper (new_pod.py calls it once a pod answers) and a CLI
(`make host-register`, for pods brought up by hand).

A row means "ready to serve", so only register once the model is actually
loaded, and deregister before taking a pod down.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from typing import Any

DEFAULT_WEBAPP = "https://neuronpedia.org"
DEFAULT_ENVIRONMENT = "prod"
API_PATH = "/api/compute-host/register"

SERVICES = ("INFERENCE", "GRAPH", "NLA", "AUTOINTERP", "SPARSITY")

# `app:` in pods.yaml names the server that runs on the pod; the registry names
# the service it provides. Inference is the default because it is the original
# app and its configs predate the key.
APP_TO_SERVICE = {
    None: "INFERENCE",
    "inference": "INFERENCE",
    "graph": "GRAPH",
    "nla": "NLA",
    "autointerp": "AUTOINTERP",
    "sparsity": "SPARSITY",
}


class RegistrationError(RuntimeError):
    """The webapp refused the registration, or could not be reached."""


def api_key_from_env() -> str:
    key = os.environ.get("NEURONPEDIA_ADMIN_API_KEY") or os.environ.get("NEURONPEDIA_API_KEY")
    if not key:
        raise RegistrationError(
            "no admin API key: set NEURONPEDIA_ADMIN_API_KEY (Settings on neuronpedia.org). "
            "The key must belong to an admin user."
        )
    return key


def _request(webapp: str, api_key: str, method: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    url = f"{webapp.rstrip('/')}{API_PATH}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "x-api-key": api_key},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as err:
        detail = err.read().decode(errors="replace").strip()
        # 409 is the environment guard: worth naming, because the fix is a
        # setting rather than a retry.
        if err.code == 409:
            raise RegistrationError(
                f"environment mismatch ({detail}). The `environment` sent must match the "
                f"webapp's NEURONPEDIA_ENVIRONMENT."
            ) from err
        raise RegistrationError(f"{url} returned {err.code}: {detail}") from err
    except urllib.error.URLError as err:
        raise RegistrationError(f"could not reach {url}: {err.reason}") from err


def register_host(
    *,
    host_url: str,
    service: str,
    model_id: str,
    name: str,
    webapp: str = DEFAULT_WEBAPP,
    environment: str = DEFAULT_ENVIRONMENT,
    api_key: str | None = None,
    source_ids: Iterable[str] = (),
    source_set_names: Iterable[str] = (),
    nla_source_id: str | None = None,
    provider: str | None = None,
    provider_ref: str | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Add or update this host's row. Returns the webapp's JSON response.

    Registration is declarative: the sources sent replace whatever the host was
    previously recorded as serving, so re-registering with a shorter list stops
    traffic for what was dropped rather than adding to it.
    """
    if service not in SERVICES:
        raise RegistrationError(f"unknown service {service!r}; expected one of {', '.join(SERVICES)}")
    if service == "NLA" and not nla_source_id:
        raise RegistrationError("NLA hosts serve exactly one source: pass nla_source_id")
    if service != "NLA" and nla_source_id:
        raise RegistrationError(f"nla_source_id is only meaningful for NLA hosts, not {service}")

    payload: dict[str, Any] = {
        "name": name,
        "hostUrl": host_url.rstrip("/"),
        "service": service,
        "environment": environment,
        "modelId": model_id,
        "sourceIds": list(source_ids),
        "sourceSetNames": list(source_set_names),
    }
    if nla_source_id:
        payload["nlaSourceId"] = nla_source_id
    if provider:
        payload["provider"] = provider
    if provider_ref:
        payload["providerRef"] = provider_ref

    return _request(webapp, api_key or api_key_from_env(), "POST", payload, timeout)


def deregister_host(
    *,
    host_url: str,
    service: str,
    webapp: str = DEFAULT_WEBAPP,
    api_key: str | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Remove this host's row, stopping traffic to it."""
    payload = {"hostUrl": host_url.rstrip("/"), "service": service}
    return _request(webapp, api_key or api_key_from_env(), "DELETE", payload, timeout)


def _split(values: Sequence[str] | None) -> list[str]:
    """Accept both repeated flags and comma/space separated lists."""
    out: list[str] = []
    for value in values or ():
        out.extend(part for part in value.replace(",", " ").split() if part)
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="register_host.py",
        description="Register or deregister a GPU server with a deployed Neuronpedia webapp.",
    )
    parser.add_argument("action", choices=("register", "deregister"))
    parser.add_argument("--url", required=True, help="base URL of the server, no trailing slash")
    parser.add_argument("--service", required=True, choices=SERVICES)
    parser.add_argument("--model", help="Neuronpedia model id (e.g. gpt2-small), not the HuggingFace id")
    parser.add_argument("--name", help="fleet profile that launched it; defaults to the pods.yaml config name")
    parser.add_argument(
        "--sources", action="append", help="source ids this host serves (repeatable or comma separated)"
    )
    parser.add_argument("--source-sets", action="append", help="source set names (repeatable or comma separated)")
    parser.add_argument("--nla-source", help="the single NlaSource an NLA host serves")
    parser.add_argument("--provider", help="e.g. runpod, local-ssh")
    parser.add_argument("--provider-ref", help="provider-side instance id, for automated teardown")
    parser.add_argument("--webapp", default=os.environ.get("NEURONPEDIA_WEBAPP_URL", DEFAULT_WEBAPP))
    parser.add_argument("--environment", default=os.environ.get("NEURONPEDIA_ENVIRONMENT", DEFAULT_ENVIRONMENT))
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args(argv)

    try:
        if args.action == "deregister":
            deregister_host(host_url=args.url, service=args.service, webapp=args.webapp, timeout=args.timeout)
            print(f"deregistered {args.url} ({args.service}) from {args.webapp}")
            return 0

        if not args.model:
            parser.error("--model is required to register")
        result = register_host(
            host_url=args.url,
            service=args.service,
            model_id=args.model,
            name=args.name or f"{args.service.lower()}-{args.model}",
            webapp=args.webapp,
            environment=args.environment,
            source_ids=_split(args.sources),
            source_set_names=_split(args.source_sets),
            nla_source_id=args.nla_source,
            provider=args.provider,
            provider_ref=args.provider_ref,
            timeout=args.timeout,
        )
    except RegistrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"registered {args.url} ({args.service}, {args.model}) with {args.webapp}")
    served = result.get("sourceIds") or result.get("sources")
    if served:
        print(f"  serving {len(served)} source(s): {', '.join(sorted(served)[:8])}{' ...' if len(served) > 8 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
