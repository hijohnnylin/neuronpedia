# ruff: noqa: T201

import os
from collections.abc import Awaitable, Callable

import sentry_sdk
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from neuronpedia_autointerp_client.models.explain_default_post_request import (
    ExplainDefaultPostRequest,
)
from neuronpedia_autointerp_client.models.score_embedding_post_request import (
    ScoreEmbeddingPostRequest,
)
from neuronpedia_autointerp_client.models.score_fuzz_detection_post_request import (
    ScoreFuzzDetectionPostRequest,
)
from sentence_transformers import SentenceTransformer

from neuronpedia_autointerp.routes.explain.default import explain_default
from neuronpedia_autointerp.routes.score.embedding import generate_score_embedding
from neuronpedia_autointerp.routes.score.fuzz_detection import (
    generate_score_fuzz_detection,
)

VERSION_PREFIX_PATH = "/v1"

router = APIRouter(prefix=VERSION_PREFIX_PATH)

# Load environment variables from .env file
load_dotenv()
SECRET = os.getenv("SECRET")

# only initialize sentry if we have a dsn
if os.getenv("SENTRY_DSN"):
    print("initializing sentry")
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for tracing.
        traces_sample_rate=1.0,
        _experiments={
            # Set continuous_profiling_auto_start to True
            # to automatically start the profiler on when
            # possible.
            "continuous_profiling_auto_start": True,
        },
    )

model = None


def initialize_globals():
    print("initializing globals")
    global model
    if torch.cuda.is_available():
        model = SentenceTransformer(
            "dunzhang/stella_en_400M_v5",
            trust_remote_code=True,  # type: ignore[call-arg]
        ).cuda()
        print("initialized embedding model")
    else:
        print("no cuda available, not initializing embedding model")


@router.post("/explain/default")
async def explanation_endpoint(request: ExplainDefaultPostRequest):
    print("Explain Default Called")
    return await explain_default(request)


@router.post("/score/embedding")
async def score_embedding_endpoint(request: ScoreEmbeddingPostRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    print("Score Embedding Called")
    return await generate_score_embedding(request, model)


@router.post("/score/fuzz-detection")
async def score_fuzz_detection_endpoint(request: ScoreFuzzDetectionPostRequest):
    print("Score Fuzz Detection Called")
    return await generate_score_fuzz_detection(request)


app = FastAPI()
app.include_router(router)


@app.on_event("startup")  # type: ignore[deprecated]
async def startup_event():
    initialize_globals()


@app.middleware("http")
async def check_secret_key(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
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
