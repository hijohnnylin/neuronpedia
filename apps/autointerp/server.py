# ruff: noqa: T201

import os
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import sentry_sdk
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, Body, FastAPI, HTTPException, Request, Response
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer

from neuronpedia_autointerp.operation_ids import sdk_operation_id
from neuronpedia_autointerp.routes.explain.default import explain_default
from neuronpedia_autointerp.routes.score.embedding import generate_score_embedding
from neuronpedia_autointerp.routes.score.fuzz_detection import (
    generate_score_fuzz_detection,
)
from neuronpedia_autointerp.schemas import (
    ExplainDefaultRequest,
    ExplainDefaultResponse,
    ScoreEmbeddingRequest,
    ScoreEmbeddingResponse,
    ScoreFuzzDetectionRequest,
    ScoreFuzzDetectionResponse,
)

if TYPE_CHECKING:
    from sentry_sdk._types import Event, Hint

VERSION_PREFIX_PATH = "/v1"

router = APIRouter(prefix=VERSION_PREFIX_PATH)

# Load environment variables from .env file
load_dotenv()
SECRET = os.getenv("SECRET")

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


def scrub_sentry_event(event: "Event", _hint: "Hint") -> "Event":
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


# only initialize sentry if we have a dsn
if os.getenv("SENTRY_DSN"):
    print("initializing sentry")
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

EMBEDDING_MODEL = "dunzhang/stella_en_400M_v5"

model = None


def _load_embedding_model(device: str, use_xformers: bool) -> SentenceTransformer:
    # stella's remote code runs attention through xformers unless both of these are off;
    # `unpad_inputs` is only implemented for the xformers path, so it has to go too.
    config_kwargs = {} if use_xformers else {"use_memory_efficient_attention": False, "unpad_inputs": False}
    loaded = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device,
        trust_remote_code=True,  # type: ignore[call-arg]
        config_kwargs=config_kwargs,  # type: ignore[call-arg]
    )
    # xformers picks a kernel only once attention actually runs, so encode something here
    # to find out whether this build can serve this GPU at all.
    loaded.encode("warmup")
    return loaded


def initialize_globals():
    print("initializing globals")
    global model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        # xformers is CUDA-only, so there is no point attempting it.
        model = _load_embedding_model(device, use_xformers=False)
    else:
        try:
            model = _load_embedding_model(device, use_xformers=True)
        except Exception as e:  # noqa: BLE001
            # xformers ships no kernel for every GPU -- it has none for sm_120 in fp32, for one.
            print(f"embedding model failed with xformers attention, falling back to standard attention: {e}")
            model = _load_embedding_model(device, use_xformers=False)
    print(f"initialized embedding model on {device}")


@router.post("/explain/default")
async def explanation_endpoint(
    request: ExplainDefaultRequest = Body(
        ...,
        example={
            "activations": [
                {
                    "tokens": ["The", "cat", "sat", "on", "the", "mat"],
                    "values": [0.0, 0.8, 0.0, 0.0, 0.0, 0.0],
                },
                {"tokens": ["I", " like", " felines"], "values": [0, 0, 0.9]},
            ],
            "openrouter_key": "YOUR_OPENROUTER_KEY",
            "model": "openai/gpt-4o-mini",
        },
    ),
) -> ExplainDefaultResponse:
    print("Explain Default Called")
    return await explain_default(request)


@router.post("/score/embedding")
async def score_embedding_endpoint(request: ScoreEmbeddingRequest) -> ScoreEmbeddingResponse:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    print("Score Embedding Called")
    return await generate_score_embedding(request, model)


@router.post("/score/fuzz-detection")
async def score_fuzz_detection_endpoint(request: ScoreFuzzDetectionRequest) -> ScoreFuzzDetectionResponse:
    print("Score Fuzz Detection Called")
    return await generate_score_fuzz_detection(request)


# Metadata carried over from the hand-written spec this server used to be generated from,
# so the published clients keep the same title, version and license. Bump `version` when
# the wire format changes; the publish job reads it.
app = FastAPI(
    title="Neuronpedia - AutoInterp Server",
    version="1.0.0",
    contact={"email": "johnny@neuronpedia.org"},
    license_info={"name": "Apache-2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
    generate_unique_id_function=sdk_operation_id,
)
app.include_router(router)


def _openapi_with_secret_key_auth() -> dict[str, Any]:
    """Document the ``X-SECRET-KEY`` header that ``check_secret_key`` below enforces.

    That check is middleware rather than a route dependency, so FastAPI cannot see it and
    would otherwise emit a spec claiming every endpoint is open -- which the clients
    generated from that spec would then believe.
    """
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        contact=app.contact,
        license_info=app.license_info,
        routes=app.routes,
    )
    schema.setdefault("components", {})["securitySchemes"] = {
        "SimpleSecretAuth": {"type": "apiKey", "in": "header", "name": "X-SECRET-KEY"}
    }
    schema["security"] = [{"SimpleSecretAuth": []}]
    app.openapi_schema = schema
    return schema


app.openapi = _openapi_with_secret_key_auth


@app.on_event("startup")  # type: ignore[deprecated]
async def startup_event():
    initialize_globals()


@app.middleware("http")
async def check_secret_key(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    # if we didn't specify a secret, then just allow the request through
    if SECRET is None:
        return await call_next(request)
    secret_key = request.headers.get("X-SECRET-KEY")
    if not secret_key or secret_key != SECRET:
        return JSONResponse(
            status_code=401,
            content={
                "error": "Invalid secret in X-SECRET-KEY header. Check that it matches the SECRET set in the server .env file."
            },
        )
    response = await call_next(request)
    return response  # noqa: RET504


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5003)
