import gc
import gzip
import json
import os
import threading
import time
import tomllib
from contextlib import asynccontextmanager
from functools import cache
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import psutil
import requests
import sentry_sdk
import torch
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from transformers import AutoTokenizer

from neuronpedia_graph.chat_prompt import (
    learn_turn_delimiters,
    parse_chat_prompt,
    render_prompt_from_messages,
    strip_leading_bos,
    unsteerable_token_positions,
)
from neuronpedia_graph.model_ids import hf_model_id_to_np_model_id
from neuronpedia_graph.runtime_env import get_device, get_model_dtype, get_model_engine
from neuronpedia_graph.schemas import (
    CheckBusyResponse,
    ForwardPassRequest,
    ForwardPassResponse,
    GraphChatMessage,
    GraphGenerationRequest,
    GraphGenerationResponse,
    LogitsByToken,
    ParseChatPromptRequest,
    ParseChatPromptResponse,
    SalientLogit,
    SteerRequest,
    SteerResponse,
    TopLogit,
)
from neuronpedia_graph.steer_generation import generate_default, generate_steered

load_dotenv()

if TYPE_CHECKING:
    # Aliased so the names stay distinct from the backend symbols the block further down declares.
    from sentry_sdk._types import Event as SentryEvent
    from sentry_sdk._types import Hint as SentryHint

# Callers authenticate with a shared secret in `x-secret-key`, and it reaches Sentry by two
# routes. The obvious one is `request.headers`: the SDK redacts Authorization, Cookie and
# X-Api-Key, but its list does not include this header. The other is stack-trace locals -- an
# error inside a route serializes every frame's variables, and the raw ASGI `scope` (headers
# included) is a local in roughly twenty of them, so scrubbing by header name alone still ships
# the secret about twenty times over. The value sweep is what covers that; the key passes above
# are still worth keeping, since they redact the header on transactions, which carry no locals.
SENTRY_SCRUBBED_HEADERS = frozenset({"x-secret-key"})
_SENTRY_HEADER_ATTR_PREFIX = "http.request.header."
_SENTRY_FILTERED = "[Filtered]"


def _redact_sentry_value(node: "Any", secret: str) -> None:
    """Replace every occurrence of `secret` in a nested event payload, in place."""
    if isinstance(node, dict):
        pairs = list(node.items())
    elif isinstance(node, list):
        pairs = list(enumerate(node))
    else:
        return
    for key, value in pairs:
        if isinstance(value, str) and secret in value:
            node[key] = value.replace(secret, _SENTRY_FILTERED)
        else:
            _redact_sentry_value(value, secret)


def scrub_sentry_event(event: "SentryEvent", _hint: "SentryHint") -> "SentryEvent":
    """Redact the auth header wherever it reaches an event: headers, span attributes, locals."""
    request: Any = event.get("request")
    headers = request.get("headers") if isinstance(request, dict) else None
    if isinstance(headers, dict):
        for key in headers:
            if key.lower() in SENTRY_SCRUBBED_HEADERS:
                headers[key] = _SENTRY_FILTERED

    # Transactions carry the same headers again as span attributes, so scrubbing only `request`
    # would still ship the secret on the sampled quarter of traffic.
    contexts: Any = event.get("contexts") or {}
    trace: Any = contexts.get("trace") or {}
    spans: Any = event.get("spans") or []
    for data in [trace.get("data"), *(span.get("data") for span in spans)]:
        if not isinstance(data, dict):
            continue
        for key in data:
            if key.lower().removeprefix(_SENTRY_HEADER_ATTR_PREFIX) in SENTRY_SCRUBBED_HEADERS:
                data[key] = _SENTRY_FILTERED

    # Guarded on non-empty: `"".replace` would rewrite every string in the event.
    secret = os.environ.get("SECRET")
    if secret:
        _redact_sentry_value(event, secret)
    return event


# Error reporting. Off entirely without a DSN, so a local run needs no Sentry account. This has
# to precede the `FastAPI(...)` below: the integration wraps route handlers as they are
# registered, so an app built before init reports nothing.
if os.getenv("SENTRY_DSN"):
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        environment=os.getenv("SENTRY_ENVIRONMENT", "development"),
        release=os.getenv("SENTRY_RELEASE"),
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.25")),
        profile_session_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.25")),
        profile_lifecycle="trace",
        before_send=scrub_sentry_event,
        before_send_transaction=scrub_sentry_event,
    )

# Which attribution algorithm builds the graph, and so what the graph contains. The two share no
# code, which is why their dependencies are separate extras (see pyproject.toml) and why every
# import below is gated.
ATTRIBUTION_ENGINE = os.getenv("ATTRIBUTION_ENGINE", "circuit-tracer")

# Which implementation executes the model. Orthogonal to the above: it changes nothing about the
# graph's contents, only which architectures are reachable. `nnsight` is circuit-tracer only.
# Resolved here rather than at first use so an invalid value fails at startup.
MODEL_ENGINE = get_model_engine()

# Load transcoder encoder weights lazily (VRAM for speed). A property of the transcoder loader, not
# of the model engine, despite having once been decided by the same flag.
LAZY_ENCODER = os.getenv("LAZY_ENCODER", "").lower() in ("1", "true", "yes")

# The env vars start.py writes from its CLI arguments, i.e. this pod's start args. Listed
# explicitly rather than dumped from os.environ, which also holds HF_TOKEN and SECRET; a new flag
# in start.py needs a line here to show up in Sentry.
SENTRY_START_ARG_ENV_VARS = (
    "ATTRIBUTION_ENGINE",
    "MODEL_ENGINE",
    "MODEL_ID",
    "MODEL_DTYPE",
    "TRANSCODER_SET",
    "LAZY_ENCODER",
    "DEVICE",
    "TOKEN_LIMIT",
    "MAX_FEATURE_NODES",
    "UPDATE_INTERVAL",
    "NP_MODEL_ID",
    "SAE_REPO",
    "SAE_EXPANSION",
    "SAE_TOPK",
    "NP_TRANSCODER_SOURCE_SET",
    "NP_LORSA_SOURCE_SET",
)

# Deliberately below the config above rather than inside the init block: what a graph pod is
# serving is only known once these resolve, but init has to happen first so that a bad
# MODEL_ENGINE or ATTRIBUTION_ENGINE still crashes into a Sentry that is listening. Only the axes
# worth filtering a pod by are tags ("which graph pods are on gemma-2?"); the rest is a context,
# read once an issue is already open.
if sentry_sdk.is_initialized():
    sentry_sdk.set_tag("model_id", os.getenv("MODEL_ID") or "unset")
    sentry_sdk.set_tag("transcoder_set", os.getenv("TRANSCODER_SET") or "none")
    sentry_sdk.set_tag("attribution_engine", ATTRIBUTION_ENGINE)
    sentry_sdk.set_tag("model_engine", MODEL_ENGINE)
    sentry_sdk.set_context(
        "start_args",
        {name: os.environ[name] for name in SENTRY_START_ARG_ENV_VARS if name in os.environ},
    )

if TYPE_CHECKING:
    # Both backends' symbols, declared unconditionally so a type checker sees them as always bound.
    # It cannot follow an environment variable into an import, so without this every use below --
    # each one already inside a matching `ATTRIBUTION_ENGINE` guard -- reads as possibly unbound.
    # At runtime only one of the two blocks below actually executes.
    from circuit_tracer import attribute
    from circuit_tracer.graph import prune_graph
    from circuit_tracer.replacement_model import ReplacementModel
    from circuit_tracer.utils.create_graph_files import (
        build_model,
        create_nodes,
        create_used_nodes_and_edges,
    )
    from circuit_tracer.utils.salient_logits import compute_salient_logits

    from neuronpedia_graph.crm_backend import (
        forward_pass_crm,
        generate_graph_crm,
        load_crm_model,
    )

if ATTRIBUTION_ENGINE == "circuit-tracer":
    from circuit_tracer import attribute
    from circuit_tracer.graph import prune_graph
    from circuit_tracer.replacement_model import ReplacementModel
    from circuit_tracer.utils.create_graph_files import (
        build_model,
        create_nodes,
        create_used_nodes_and_edges,
    )
    from circuit_tracer.utils.salient_logits import compute_salient_logits
elif ATTRIBUTION_ENGINE == "lm-saes-crm":
    from neuronpedia_graph.crm_backend import (
        forward_pass_crm,
        generate_graph_crm,
        load_crm_model,
    )
else:
    raise ValueError(f"ATTRIBUTION_ENGINE must be 'circuit-tracer' or 'lm-saes-crm', got {ATTRIBUTION_ENGINE!r}")


LIMIT_TOKENS = int(os.getenv("TOKEN_LIMIT", "64"))
OFFLOAD = None
UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", "1000"))

SECRET_KEY = os.getenv("SECRET")
if not SECRET_KEY:
    raise ValueError("SECRET environment variable not set. Please create a .env file with SECRET=<your_secret_key>")

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN environment variable not set. Please create a .env file with HF_TOKEN=<your_huggingface_token>"
    )


transcoders: Any = None
model: Any = None
request_lock = threading.Lock()

TRANSCODER_SET_TO_SOURCE_URL_ARRAYS = {
    "gemma": [
        "https://neuronpedia.org/gemma-2-2b/gemmascope-transcoder-16k",
        "https://huggingface.co/google/gemma-scope-2b-pt-transcoders",
    ],
    "mwhanna/qwen3-4b-transcoders": [
        "https://neuronpedia.org/qwen3-4b/transcoder-hp",
        "https://huggingface.co/mwhanna/qwen3-4b-transcoders",
    ],
    "mntss/clt-gemma-2-2b-2.5M": [
        "https://neuronpedia.org/gemma-2-2b/clt-hp",
        "https://huggingface.co/mntss/clt-gemma-2-2b-2.5M",
    ],
    "mwhanna/gemma-scope-2-4b-it/transcoder_all/width_262k_l0_small_affine": [
        "https://neuronpedia.org/gemma-3-4b-it/gemmascope-transcoder-262k",
        "https://huggingface.co/mwhanna/gemma-scope-2-4b-it/transcoder_all/width_262k_l0_small_affine",
        "https://huggingface.co/google/gemma-scope-2-4b-it/transcoder_all",
    ],
}


# HuggingFace repo id -> the Neuronpedia model id a graph is labelled with. See model_ids.py; it
# is reversed from np_model_to_hf.json rather than hand-kept here, which is what the dict this
# replaced got wrong.


def _circuit_tracer_version() -> str:
    """Report the pinned circuit-tracer version from uv.lock.

    circuit-tracer is pinned to a git reference in pyproject.toml, and that reference is the
    meaningful version to surface -- more so than importlib metadata, since a package's
    self-reported version can lag its release tags (a branch build reports something like
    `0.0.1.dev48+g6ca66b3f1`, which says nothing about what it contains).

    uv.lock records the resolved source as `...?tag=v0.5.1#<commit>` for a tag pin and
    `...?branch=<name>#<commit>` for a branch pin. A tag reports as-is; a branch reports as
    `<branch>@<short commit>`, which stays honest that it is not a release while still naming
    exactly what is installed; a rev pin reports the short commit alone. Falls back to metadata
    if the lock cannot be read.
    """
    lock_path = Path(__file__).resolve().parent.parent / "uv.lock"
    try:
        with lock_path.open("rb") as f:
            lock = tomllib.load(f)
        for pkg in lock.get("package", []):
            if pkg.get("name") == "circuit-tracer":
                git_url = (pkg.get("source") or {}).get("git", "")
                parsed = urlparse(git_url)
                query = parse_qs(parsed.query)
                tag = query.get("tag", [None])[0]
                if tag:
                    return tag
                commit = parsed.fragment[:8]
                branch = query.get("branch", [None])[0]
                if branch:
                    return f"{branch}@{commit}" if commit else branch
                rev = query.get("rev", [None])[0]
                if rev:
                    # Already a commit, so a short form of it says everything.
                    return commit or rev[:8]
    except (OSError, tomllib.TOMLDecodeError):
        pass
    return pkg_version("circuit-tracer")


@cache
def generator_info() -> dict[str, str]:
    """Attribution-library provenance for the circuit-tracer backend's graph metadata.

    Deliberately a function, not a module-level constant: it reads circuit-tracer's installed
    version, and the CRM backend imports none of circuit-tracer. Evaluating this at import time
    would make an optional dependency mandatory for a server that never touches it. The CRM
    backend reports its own generator, in `format_converter`.
    """
    return {
        "name": "circuit-tracer by Hanna & Piotrowski",
        "version": _circuit_tracer_version(),
        "url": "https://github.com/decoderesearch/circuit-tracer",
    }


def _serving_url() -> str:
    """The address uvicorn was told to bind to (set in start.py / env)."""
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = os.getenv("SERVER_PORT", "5004")
    display_host = "localhost" if host in ("0.0.0.0", "::", "") else host
    return f"http://{display_host}:{port}"


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


def _print_banner(title: str, lines: list[str]) -> None:
    """One visually obvious block, so the end of a long noisy startup is findable."""
    bar = "=" * 100
    body = "\n".join(f"  {line}" for line in lines)
    print(f"\n{bar}\n  {title}\n{'-' * 100}\n{body}\n{bar}\n", flush=True)


def _print_ready_banner(
    elapsed_seconds: float,
    *,
    model_id: str,
    attribution_engine: str,
    model_engine: str,
    device: torch.device,
    model_dtype: torch.dtype,
    transcoder_set: str | None,
) -> None:
    dtype_label = str(model_dtype).removeprefix("torch.")
    lines = [
        f"model: {model_id}",
        f"attribution engine: {attribution_engine} | model engine: {model_engine}",
        f"device: {device} | dtype: {dtype_label}",
    ]
    if transcoder_set:
        lines.append(f"transcoder set: {transcoder_set}")
    lines.append(f"token limit: {LIMIT_TOKENS}")
    lines.append(f"startup took {_format_duration(elapsed_seconds)}")
    _print_banner(
        f"==== LOADING COMPLETE - SERVING ON {_serving_url()} ====",
        lines,
    )


# Assigned by the lifespan below rather than at import. Annotated without a value so the
# handlers still see `str` rather than `str | None`; reading it before startup would be a
# NameError, which is the right outcome if it ever happens.
loaded_model_arg: str

# CRM backend: lm-saes with Lorsa + Transcoders
crm_model: Any = None
crm_replacement_modules: Any = None
crm_sae_metadata: Any = None
transcoder_set: str | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the model before the server accepts traffic.

    This ran at module scope until the wire models were introduced, which made importing the
    module equivalent to booting the server -- so ``dump_openapi.py`` could not read the route
    table without a GPU and a set of weights, defeating the point of committing the spec. A
    lifespan still completes before uvicorn serves its first request, so readiness is
    unchanged; only importing is cheap now.
    """
    global loaded_model_arg, model, crm_model, crm_replacement_modules, crm_sae_metadata, transcoder_set

    model_id_env = os.getenv("MODEL_ID")
    print(f"Model: {model_id_env}")
    if not model_id_env:
        raise ValueError(
            "MODEL_ID is required. Pass --model_id, or set MODEL_ID. Models a graph can be labelled "
            "with: " + ", ".join(sorted(hf_model_id_to_np_model_id()))
        )
    loaded_model_arg = model_id_env

    # Refuse here rather than at build_model. The circuit-tracer path labels a graph with this, and
    # an unlisted model used to raise a KeyError only once attribution had finished -- minutes of
    # GPU work spent on a pod that was never going to be able to return a graph. The lm-saes-crm
    # path is exempt because it labels from NP_MODEL_ID and never reads this map.
    if ATTRIBUTION_ENGINE != "lm-saes-crm" and loaded_model_arg not in hf_model_id_to_np_model_id():
        raise ValueError(
            f"No Neuronpedia model id for MODEL_ID '{loaded_model_arg}'. Add it to "
            "np_model_to_hf.json at the repo root. Known: " + ", ".join(sorted(hf_model_id_to_np_model_id()))
        )

    device = get_device()
    model_dtype = get_model_dtype()
    startup_started_at = time.monotonic()

    if ATTRIBUTION_ENGINE == "lm-saes-crm":
        print(f"[CRM] Loading CRM backend for model: {loaded_model_arg}")
        crm_model, crm_replacement_modules, crm_sae_metadata = load_crm_model()
        model = crm_model
    else:
        # Circuit-tracer backend (default)
        transcoder_set = os.getenv("TRANSCODER_SET")
        print(f"Transcoder set: {transcoder_set}")
        if not transcoder_set:
            raise ValueError("Transcoder set is required. Please specify a transcoders set.")

        model = ReplacementModel.from_pretrained(
            loaded_model_arg,
            transcoder_set,
            device=device,
            dtype=model_dtype,
            lazy_encoder=LAZY_ENCODER,
            lazy_decoder=True,
            backend=MODEL_ENGINE,
        )

    _print_ready_banner(
        time.monotonic() - startup_started_at,
        model_id=loaded_model_arg,
        attribution_engine=ATTRIBUTION_ENGINE,
        model_engine=MODEL_ENGINE,
        device=device,
        model_dtype=model_dtype,
        transcoder_set=transcoder_set,
    )

    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)


def resolve_prompt(prompt: str, messages: list[GraphChatMessage] | None) -> str:
    """Prefer structured ``messages`` (rendered server-side); else the raw string.

    Dumped back to plain dicts here because the renderer hands them to the tokenizer's chat
    template, which is dict-based; ``chat_prompt`` stays free of our wire models.
    """
    if messages:
        return render_prompt_from_messages(model.tokenizer, [message.model_dump() for message in messages])
    return prompt


# The process serves exactly one model, so the probed delimiters never change.
_turn_delimiters_cache: dict[str, tuple[str, str]] | None = None


def get_turn_delimiters() -> dict[str, tuple[str, str]]:
    global _turn_delimiters_cache
    if _turn_delimiters_cache is None:
        _turn_delimiters_cache = learn_turn_delimiters(model.tokenizer)
    return _turn_delimiters_cache


def ensure_bos_for_circuit_tracer(prompt: str) -> str:
    """Prepend <bos> to gemma-3-it prompts for the circuit-tracer backend.

    circuit-tracer's ReplacementModel.ensure_tokenized() tokenizes with
    add_special_tokens=False and, for gemma-3 instruct models, asserts the
    tokens already start with <bos><start_of_turn>user (unlike base models, it
    does NOT auto-prepend BOS). The webapp sends a chat-templated prompt without
    a leading <bos>, so add it here to satisfy that check.
    """
    is_gemma_3_it = "gemma-3" in loaded_model_arg and loaded_model_arg.endswith("-it")
    if not is_gemma_3_it:
        return prompt
    tokens = model.tokenizer.encode(prompt, add_special_tokens=False)
    if tokens and tokens[0] == model.tokenizer.bos_token_id:
        return prompt
    return model.tokenizer.bos_token + prompt


def printMemory():
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / (1024**3)
        print(f"GPU memory usage: {current_memory:.2f} GB")
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage_gb = memory_info.rss / (1024**3)
        print(f"CPU memory usage: {memory_usage_gb:.2f} GB")


async def verify_secret_key(x_secret_key: str = Header(None)):
    if not x_secret_key:
        raise HTTPException(status_code=400, detail="x-secret-key header missing")
    if x_secret_key != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid x-secret-key")
    return x_secret_key


@app.get("/check-busy", responses={200: {"model": CheckBusyResponse}})
async def check_busy():
    """Check if the server is currently busy processing a request."""
    return CheckBusyResponse(busy=request_lock.locked())


@app.post(
    "/parse-chat-prompt",
    dependencies=[Depends(verify_secret_key)],
    responses={200: {"model": ParseChatPromptResponse}},
)
async def parse_chat_prompt_handler(req: ParseChatPromptRequest):
    """Recover structured chat turns from a rendered prompt string.

    Deliberately does NOT take ``request_lock``: this is tokenizer-only string
    work with no GPU involvement, and the webapp calls it while opening a modal,
    so a graph generation in flight must not turn it into a 503.

    ``prompt`` echoes back the BOS-stripped input. A non-chat prompt has no
    ``messages`` to carry the strip, and the caller wants to put the string back
    in an editor, so returning it here is what keeps the BOS literal out of the
    frontend.

    ``unsteerable_positions`` is the set ``/steer`` will silently drop features
    at, computed here the same way and from the same normalized prompt. A steer
    UI needs it to know which sliders would do nothing; without it the client has
    to guess from token strings, which is the token knowledge this endpoint
    exists to keep server-side.
    """
    prompt = strip_leading_bos(model.tokenizer, req.prompt)
    messages = parse_chat_prompt(prompt, get_turn_delimiters())
    return ParseChatPromptResponse(
        messages=[GraphChatMessage(**message) for message in messages] if messages is not None else None,
        is_chat=messages is not None,
        prompt=prompt,
        unsteerable_positions=sorted(unsteerable_token_positions(model.tokenizer, prompt)),
    )


def get_topk(logits: torch.Tensor, tokenizer, k: int = 5):
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    topk = torch.topk(probs, k)
    return [(tokenizer.decode([topk.indices[i]]), topk.values[i].item()) for i in range(k)]


@app.post("/steer", dependencies=[Depends(verify_secret_key)], responses={200: {"model": SteerResponse}})
async def steer_handler(req_data: SteerRequest):
    """Handle steer requests"""
    print("========== Steer Start ==========")
    print(f"Thread {threading.get_ident()}: Received request. Attempting to acquire lock.")
    if not request_lock.acquire(blocking=False):
        print(f"Thread {threading.get_ident()}: Lock acquisition failed (busy). Rejecting request.")
        return JSONResponse(content={"error": "Server busy, please try again later."}, status_code=503)

    print(f"Thread {threading.get_ident()}: Lock acquired.")
    try:
        # A graph's stored prompt has BOS baked in for families whose template
        # emits one, and the tokenizer below adds its own. Left in, every token
        # position would shift by one and each feature would be steered at the
        # wrong token, so strip it here rather than making callers do it.
        req_data.prompt = strip_leading_bos(model.tokenizer, resolve_prompt(req_data.prompt, req_data.messages))

        if req_data.model_id != loaded_model_arg:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{req_data.model_id}' is not available. Only '{loaded_model_arg}' is currently loaded.",
            )

        sequence_length = len(model.tokenizer(req_data.prompt).input_ids)

        # Validate that if ablate is True, delta must be None
        for feature in req_data.features:
            if feature.ablate and feature.delta is not None:
                return JSONResponse(
                    content={"error": "When ablate is True, delta must be None"},
                    status_code=400,
                )
            if not feature.ablate and feature.delta is None:
                return JSONResponse(
                    content={"error": "When ablate is False, delta must be provided"},
                    status_code=400,
                )
            if feature.steer_generated_tokens and feature.steer_position is not None:
                return JSONResponse(
                    content={"error": "When steer_generated_tokens is True, position must be None"},
                    status_code=400,
                )
            # Validate that if steer_generated_tokens is False, position must be provided
            if not feature.steer_generated_tokens and feature.steer_position is None:
                return JSONResponse(
                    content={"error": "When steer_generated_tokens is False, position must be provided"},
                    status_code=400,
                )
            # Validate that if position is provided, it's not out of bounds
            if feature.steer_position is not None and (
                feature.steer_position < 0 or feature.steer_position >= sequence_length
            ):
                return JSONResponse(
                    content={"error": "Position is out of bounds"},
                    status_code=400,
                )

        # Enforcing this here makes it a property of the server rather than
        # something every client reimplements. `/parse-chat-prompt` reports the
        # same set from the same function, which is how the UI knows to hide
        # those sliders instead of rendering ones that silently do nothing.
        unsteerable = unsteerable_token_positions(model.tokenizer, req_data.prompt)
        if unsteerable:
            kept = [f for f in req_data.features if f.steer_position not in unsteerable]
            if len(kept) != len(req_data.features):
                print(
                    f"Ignoring {len(req_data.features) - len(kept)} steer feature(s) "
                    f"targeting unsteerable position(s) {sorted(unsteerable)}"
                )
            req_data.features = kept

        print(f"Received steer request: {req_data}")

        _, activations = model.get_activations(req_data.prompt, sparse=True)

        intervention_tuples = []
        for f in req_data.features:
            if f.steer_generated_tokens:
                intervention_tuples.append(
                    (
                        f.layer,
                        # TODO: double check this
                        slice(sequence_length, None, None),
                        f.index,
                        0 if f.ablate else activations[(f.layer, f.token_active_position, f.index)] + f.delta,
                    )
                )
            else:
                intervention_tuples.append(
                    (
                        f.layer,
                        f.steer_position,
                        f.index,
                        0 if f.ablate else activations[(f.layer, f.token_active_position, f.index)] + f.delta,
                    )
                )

        # set the seed
        if req_data.seed is not None:
            torch.manual_seed(req_data.seed)
        default_tokenized = generate_default(
            model,
            req_data.prompt,
            max_new_tokens=req_data.n_tokens,
            temperature=req_data.temperature,
            freq_penalty=req_data.freq_penalty,
        )

        default_tokenized_str_tokens = [model.tokenizer.decode([token]) for token in default_tokenized]

        default_generation = "".join(default_tokenized_str_tokens)

        # reset the seed
        if req_data.seed is not None:
            torch.manual_seed(req_data.seed)
        steered_tokenized, steered_logits = generate_steered(
            model,
            req_data.prompt,
            intervention_tuples,
            # One more than the default run asks for, as this endpoint has always requested. The
            # loops below index the steered tokens by the *default* run's length, so a steered run
            # that is shorter reads off the end.
            max_new_tokens=req_data.n_tokens + 1,
            temperature=req_data.temperature,
            freq_penalty=req_data.freq_penalty,
            freeze_attention=req_data.freeze_attention,
        )

        steered_tokenized_str_tokens = [model.tokenizer.decode([token]) for token in steered_tokenized]
        steered_generation = "".join(steered_tokenized_str_tokens)

        # Cross-layer transcoders return 2D logits (seq, vocab) — normalize to 3D
        if steered_logits.dim() == 2:
            steered_logits = steered_logits.unsqueeze(0)

        # get the logits at each step
        topk_default_by_token: list[LogitsByToken] = []
        topk_steered_by_token: list[LogitsByToken] = []

        with torch.inference_mode():
            # Pass token IDs directly to avoid retokenization (which can
            # prepend a duplicate BOS and shift logit positions by one).
            default_logits = model(default_tokenized.unsqueeze(0))
            if default_logits.dim() == 2:
                default_logits = default_logits.unsqueeze(0)

            # iterate through the tokens and get the logits
            for i in range(len(default_tokenized_str_tokens)):
                # If we're still processing the original prompt tokens (before generation),
                # append a blank item since we're only interested in generated tokens
                if i < sequence_length - 1:
                    topk_default_by_token.append(LogitsByToken(token=default_tokenized_str_tokens[i], top_logits=[]))
                    continue
                # get the topk tokens
                topk_default = get_topk(default_logits[:, : i + 1, :], model.tokenizer, req_data.top_k)
                # each topk default should be an object of token, prob
                topk_default_by_token.append(
                    LogitsByToken(
                        token=default_tokenized_str_tokens[i],
                        top_logits=[TopLogit(token=token, prob=prob) for token, prob in topk_default],
                    )
                )
            # steered_logits only contains generation-step logits (no prompt positions),
            # so we offset the index: position 0 in steered_logits = sequence_length - 1
            # in the full token sequence.
            for i in range(len(default_tokenized_str_tokens)):
                if i < sequence_length - 1:
                    topk_steered_by_token.append(LogitsByToken(token=steered_tokenized_str_tokens[i], top_logits=[]))
                    continue
                gen_idx = i - (sequence_length - 1)
                topk_steered = get_topk(steered_logits[:, : gen_idx + 1, :], model.tokenizer, req_data.top_k)
                topk_steered_by_token.append(
                    LogitsByToken(
                        token=steered_tokenized_str_tokens[i],
                        top_logits=[TopLogit(token=token, prob=prob) for token, prob in topk_steered],
                    )
                )

        print(f"Default generation: {default_generation}")
        print(f"Steered generation: {steered_generation}")

        return SteerResponse(
            DEFAULT_LOGITS_BY_TOKEN=topk_default_by_token,
            STEERED_LOGITS_BY_TOKEN=topk_steered_by_token,
            DEFAULT_GENERATION=default_generation,
            STEERED_GENERATION=steered_generation,
        )

    finally:
        if request_lock.locked():
            print(f"Thread {threading.get_ident()}: Releasing lock in finally block.")
            request_lock.release()
        else:
            print(
                f"Thread {threading.get_ident()}: Lock was not held by current path in finally block (already released or never acquired)."
            )


@app.post(
    "/forward-pass",
    dependencies=[Depends(verify_secret_key)],
    responses={200: {"model": ForwardPassResponse}},
)
async def forward_pass_handler(req_data: ForwardPassRequest):
    """Handle forward pass requests to get salient logits"""
    print("========== Forward Pass Start ==========")

    # Resolved before the lock: it is pure string work, and FastAPI has already validated the
    # body by this point, so there is no longer a failure path between acquiring and the try.
    req_data.prompt = resolve_prompt(req_data.prompt, req_data.messages)

    print(f"Thread {threading.get_ident()}: Received request. Attempting to acquire lock.")
    if not request_lock.acquire(blocking=False):
        print(f"Thread {threading.get_ident()}: Lock acquisition failed (busy). Rejecting request.")
        return JSONResponse(content={"error": "Server busy, please try again later."}, status_code=503)

    print(f"Thread {threading.get_ident()}: Lock acquired.")
    try:
        print(f"Received forward pass request: prompt='{req_data.prompt}'")

        if ATTRIBUTION_ENGINE == "lm-saes-crm":
            return forward_pass_crm(
                req_data.prompt,
                crm_model,
                max_n_logits=req_data.max_n_logits,
                desired_logit_prob=req_data.desired_logit_prob,
            )

        # Circuit-tracer backend
        tokens = model.tokenizer.encode(req_data.prompt, add_special_tokens=False)
        if tokens and tokens[0] != model.tokenizer.bos_token_id:
            tokens = model.tokenizer.encode(req_data.prompt, add_special_tokens=True)
        print(f"Tokens: {tokens}")

        input_ids = torch.tensor([tokens]).to(get_device())

        with torch.no_grad():
            output = model(input_ids)
            if hasattr(output, "logits"):
                output = output.logits

            logits = output[0, -1, :]

            logit_indices, logit_probs, _ = compute_salient_logits(
                logits,
                model.unembed_weight,
                max_n_logits=req_data.max_n_logits,
                desired_logit_prob=req_data.desired_logit_prob,
            )

        results = [
            SalientLogit(token=model.tokenizer.decode([idx]), token_id=idx, probability=float(prob))
            for idx, prob in zip(logit_indices.tolist(), logit_probs.tolist())
        ]

        response = ForwardPassResponse(
            prompt=req_data.prompt,
            input_tokens=[model.tokenizer.decode([token]) for token in tokens],
            salient_logits=results,
            total_salient_tokens=len(results),
            cumulative_probability=float(logit_probs.sum()),
        )

        print(f"Found {len(results)} salient tokens with cumulative prob: {response.cumulative_probability:.4f}")

        return response

    # Endpoint boundary: a forward pass fails in whatever way the model and its backend
    # choose to, and the caller gets an error field either way.
    except Exception as e:  # noqa: BLE001
        print(f"Error in forward pass: {e!s}")
        return {"error": f"Forward pass failed: {e!s}"}

    finally:
        if request_lock.locked():
            print(f"Thread {threading.get_ident()}: Releasing lock in finally block.")
            request_lock.release()
        else:
            print(
                f"Thread {threading.get_ident()}: Lock was not held by current path in finally block (already released or never acquired)."
            )


@app.post(
    "/generate-graph",
    dependencies=[Depends(verify_secret_key)],
    responses={200: {"model": GraphGenerationResponse}},
)
async def generate_graph(req_data: GraphGenerationRequest):
    # Resolved before the lock: pure string work, and FastAPI has already rejected a malformed
    # body with a 422, so the hand-rolled validate-and-release dance this replaces is gone.
    req_data.prompt = resolve_prompt(req_data.prompt, req_data.messages)

    print(f"Thread {threading.get_ident()}: Received request. Attempting to acquire lock.")
    if not request_lock.acquire(blocking=False):
        print(f"Thread {threading.get_ident()}: Lock acquisition failed (busy). Rejecting request.")
        return JSONResponse(content={"error": "Server busy, please try again later."}, status_code=503)

    print(f"Thread {threading.get_ident()}: Lock acquired.")
    try:
        prompt = req_data.prompt
        requested_model_id = req_data.model_id
        if requested_model_id is None or requested_model_id != loaded_model_arg:
            request_lock.release()
            raise HTTPException(
                status_code=400,
                detail=f"Model '{requested_model_id}' is not available. Only '{loaded_model_arg}' is currently loaded.",
            )

        batch_size = req_data.batch_size
        max_n_logits = req_data.max_n_logits
        desired_logit_prob = req_data.desired_logit_prob
        node_threshold = req_data.node_threshold
        edge_threshold = req_data.edge_threshold
        slug_identifier = req_data.slug_identifier or f"generated-{int(time.time())}"
        max_feature_nodes = req_data.max_feature_nodes
        print(f"Thread {threading.get_ident()}: Processing request for prompt: '{prompt[:50]}...' with parameters:")
        print(f"  model_id: {requested_model_id}")
        print(f"  batch_size: {batch_size}")
        print(f"  max_n_logits: {max_n_logits}")
        print(f"  desired_logit_prob: {desired_logit_prob}")
        print(f"  node_threshold: {node_threshold}")
        print(f"  edge_threshold: {edge_threshold}")
        print(f"  slug_identifier: {slug_identifier}")
        print(f"  max_feature_nodes: {max_feature_nodes}")
        print(f"  attribution engine: {ATTRIBUTION_ENGINE}")

        def _blocking_graph_generation_task():
            print(f"Thread {threading.get_ident()} (worker): Starting blocking graph generation.")
            _total_start_time = time.time()

            try:
                tokens = model.tokenizer.encode(prompt, add_special_tokens=False)
                print(f"Thread {threading.get_ident()} (worker): {len(tokens)} Tokens: {tokens}")
                if len(tokens) > LIMIT_TOKENS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Prompt exceeds token limit ({len(tokens)} > {LIMIT_TOKENS})",
                    )
            # The token-limit refusal above is an HTTPException, so it has to pass through:
            # catching it here reported an over-long prompt as a 500 "Failed to tokenize".
            except HTTPException:
                raise
            # Tokenizers raise their own error types, so the fallback stays broad.
            except Exception as e:  # noqa: BLE001
                print(f"Thread {threading.get_ident()} (worker): Tokenization error: {e}")
                raise HTTPException(status_code=500, detail="Failed to tokenize prompt") from e

            if ATTRIBUTION_ENGINE == "lm-saes-crm":
                return generate_graph_crm(
                    prompt,
                    crm_model,
                    crm_replacement_modules,
                    crm_sae_metadata,
                    slug_identifier=slug_identifier,
                    max_n_logits=max_n_logits,
                    desired_logit_prob=desired_logit_prob,
                    batch_size=batch_size,
                    max_feature_nodes=max_feature_nodes,
                    node_threshold=node_threshold,
                    edge_threshold=edge_threshold,
                    signed_url=req_data.signed_url,
                    user_id=req_data.user_id,
                    compress=req_data.compress,
                    enable_qk_tracing=req_data.enable_qk_tracing,
                    qk_top_fraction=req_data.qk_top_fraction,
                    qk_topk=req_data.qk_topk,
                )

            print(f"Thread {threading.get_ident()} (worker): Prompt: '{prompt}'")

            ct_prompt = ensure_bos_for_circuit_tracer(prompt)

            attribution_start = time.time()
            _graph = attribute(
                ct_prompt,
                model,
                max_n_logits=max_n_logits,
                desired_logit_prob=desired_logit_prob,
                batch_size=batch_size,
                max_feature_nodes=req_data.max_feature_nodes,
                offload=OFFLOAD,
                update_interval=UPDATE_INTERVAL,
            )
            attribution_time_ms = (time.time() - attribution_start) * 1000
            print(f"Thread {threading.get_ident()} (worker): Attribution Time: {attribution_time_ms:.2f}ms")

            _graph.to("cuda")

            _node_mask, _edge_mask, _cumulative_scores = (
                el.cpu() for el in prune_graph(_graph, node_threshold, edge_threshold)
            )
            _graph.to("cpu")

            tokenizer = AutoTokenizer.from_pretrained(model.cfg.tokenizer_name)

            _nodes = create_nodes(
                _graph,
                _node_mask,
                tokenizer,
                _cumulative_scores,
            )
            print("nodes created")
            _used_nodes, _used_edges = create_used_nodes_and_edges(_graph, _nodes, _edge_mask)
            print("used nodes and edges created")
            _output_model = build_model(
                _graph,
                _used_nodes,
                _used_edges,
                slug_identifier,
                hf_model_id_to_np_model_id()[requested_model_id],
                node_threshold,
                tokenizer,
            )
            print("output model created")

            # if signed_url is not provided, we don't upload the file, just return the output model
            if req_data.signed_url is None:
                print("No signed url provided, returning output model")
                return _output_model

            # if signed_url is provided, we upload the file and return a success message
            print(f"Uploading file to url: {req_data.signed_url}")
            current_time_ms = int(time.time() * 1000)
            # Convert to dict to add additional fields
            model_dict = _output_model.model_dump()

            # Only the circuit-tracer path reaches here, and it refuses to start without a
            # transcoder set, so this is a startup invariant rather than a runtime check.
            assert transcoder_set is not None
            model_dict["metadata"]["info"] = {
                "creator_name": req_data.user_id if req_data.user_id else "Anonymous (CT)",
                "creator_url": "https://neuronpedia.org",
                "source_urls": TRANSCODER_SET_TO_SOURCE_URL_ARRAYS[transcoder_set],
                "transcoder_set": transcoder_set,
                "generator": generator_info(),
                "create_time_ms": current_time_ms,
            }

            model_dict["metadata"]["generation_settings"] = {
                "max_n_logits": max_n_logits,
                "desired_logit_prob": desired_logit_prob,
                "batch_size": batch_size,
                "max_feature_nodes": max_feature_nodes,
            }

            model_dict["metadata"]["pruning_settings"] = {
                "node_threshold": node_threshold,
                "edge_threshold": edge_threshold,
            }

            # Convert back to JSON string
            model_json = json.dumps(model_dict)

            # Handle compression if requested
            compress_time_ms = 0
            if req_data.compress:
                print("Compressing data with gzip (level 3)...")
                compress_start = time.time()
                data_to_upload = gzip.compress(model_json.encode("utf-8"), compresslevel=3)
                compress_time_ms = (time.time() - compress_start) * 1000
                headers = {
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip",
                }
            else:
                data_to_upload = model_json.encode("utf-8")
                headers = {"Content-Type": "application/json"}

            # Track upload size
            upload_size_bytes = len(data_to_upload)

            # Start upload timing
            upload_start = time.time()
            response = requests.put(
                req_data.signed_url,
                data=data_to_upload,
                headers=headers,
            )
            upload_time_ms = (time.time() - upload_start) * 1000

            print(f"Upload response: {response.status_code}")
            # print(f"Upload response: {response.text}")
            if response.status_code != 200:
                return GraphGenerationResponse(error="Failed to upload file")

            print(f"File: uploaded successfully to url: {req_data.signed_url}")

            _total_time_ms = time.time() - _total_start_time

            # Log timing summary
            timing_parts = [
                f"attribution_ms={attribution_time_ms:.0f}",
                f"upload_ms={upload_time_ms:.0f}",
                f"upload_size_bytes={upload_size_bytes}",
                f"upload_size_mb={upload_size_bytes / (1024 * 1024):.2f}",
                f"total_ms={_total_time_ms:.0f}",
            ]

            if req_data.compress:
                timing_parts.extend(
                    [
                        f"compress_ms={compress_time_ms:.0f}",
                        f"compression_ratio={len(model_json.encode('utf-8')) / upload_size_bytes:.2f}",
                    ]
                )

            print(f"Thread {threading.get_ident()} (worker): Total Time for blocking task: {_total_time_ms=:.2f}s")

            return GraphGenerationResponse(success=f"Graph uploaded successfully to url: {req_data.signed_url}")

        try:
            result = await run_in_threadpool(_blocking_graph_generation_task)
            print(f"Thread {threading.get_ident()}: Blocking task completed.")
            return result
        except HTTPException:
            raise
        # Worker boundary: attribution runs third-party model code, so anything it raises
        # becomes a 500 here rather than escaping as an unhandled task exception.
        except Exception as e:  # noqa: BLE001
            import traceback

            print(f"Thread {threading.get_ident()}: Error during graph generation in worker thread: {e}")
            print("Stack trace:")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Internal server error during graph generation") from e

    finally:
        printMemory()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared CUDA cache")

        gc.collect()
        print("Cleared CPU memory")
        if request_lock.locked():
            print(f"Thread {threading.get_ident()}: Releasing lock in finally block.")
            request_lock.release()
        else:
            print(
                f"Thread {threading.get_ident()}: Lock was not held by current path in finally block (already released or never acquired)."
            )
