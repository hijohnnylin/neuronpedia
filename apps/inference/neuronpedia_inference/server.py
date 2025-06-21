import asyncio
import gc
import json
import logging
import os
import sys
import traceback
from collections.abc import Awaitable
from typing import Callable

import sentry_sdk
import torch
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuronpedia_inference.args import list_available_options, parse_env_and_args
from neuronpedia_inference.config import Config, get_saelens_neuronpedia_directory_df
from neuronpedia_inference.endpoints.activation.all import (
    router as activation_all_router,
)
from neuronpedia_inference.endpoints.activation.single import (
    router as activation_single_router,
)
from neuronpedia_inference.endpoints.activation.topk_by_token import (
    router as activation_topk_by_token_router,
)
from neuronpedia_inference.endpoints.steer.completion import (
    router as steer_completion_router,
)
from neuronpedia_inference.endpoints.steer.completion_chat import (
    router as steer_completion_chat_router,
)
from neuronpedia_inference.endpoints.tokenize import router as tokenize_router
from neuronpedia_inference.endpoints.util.sae_topk_by_decoder_cossim import (
    router as sae_topk_by_decoder_cossim_router,
)
from neuronpedia_inference.endpoints.util.sae_vector import router as sae_vector_router
from neuronpedia_inference.logging import initialize_logging
from neuronpedia_inference.sae_manager import SAEManager  # noqa: F401
from neuronpedia_inference.shared import STR_TO_DTYPE, Model  # noqa: F401
from neuronpedia_inference.utils import checkCudaError

# Initialize logging at module level
initialize_logging()

logger = logging.getLogger(__name__)
logger.info("Server module initialized")

load_dotenv()

global initialized
initialized = False

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

args = parse_env_and_args()


# we have to initialize SAE's AFTER server startup, because some infrastructure providers require
# our server to respond to health checks within a few minutes of starting up
@app.on_event("startup")  # pyright: ignore[reportDeprecated]
async def startup_event():
    logger.info("Starting initialization...")
    # Wait briefly to ensure server is ready
    await asyncio.sleep(3)
    # Start initialization in background
    asyncio.create_task(initialize(args.custom_hf_model_id))
    logger.info("Initialization started")


v1_router = APIRouter(prefix="/v1")

v1_router.include_router(activation_all_router)
v1_router.include_router(steer_completion_chat_router)
v1_router.include_router(steer_completion_router)
v1_router.include_router(activation_single_router)
v1_router.include_router(activation_topk_by_token_router)
v1_router.include_router(sae_topk_by_decoder_cossim_router)
v1_router.include_router(sae_vector_router)
v1_router.include_router(tokenize_router)

app.include_router(v1_router)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/initialize")
async def initialize(
    custom_hf_model_id: str | None = None,
):
    logger.info("Initializing...")

    # Move the heavy operations to a separate thread pool to prevent blocking
    def load_model_and_sae():
        # Validate inputs
        df = get_saelens_neuronpedia_directory_df()
        models = df["model"].unique()
        sae_sets = df["neuronpedia_set"].unique()
        if args.model_id not in models:
            logger.error(
                f"Error: Invalid model_id '{args.model_id}'. Use --list_models to see available options."
            )
            exit(1)
        # iterate through sae_sets and split them by spaces
        args_sae_sets = []
        for sae_set in args.sae_sets:
            args_sae_sets.extend(sae_set.split())
        logger.info("SAE sets: %s", args_sae_sets)
        logger.info("Checking for invalid SAE sets...")
        invalid_sae_sets = set(args_sae_sets) - set(sae_sets)
        if invalid_sae_sets:
            logger.error(
                f"Error: Invalid SAE set(s): {', '.join(invalid_sae_sets)}. Use --list_models to see available options."
            )
            exit(1)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.set_grad_enabled(False)
        checkCudaError("cpu")
        # todo: use multiple devices if available
        device_count = 1

        SECRET = os.getenv("SECRET")

        logger.info(f"device in args: {args.device}")
        logger.info(f"device set in args: {args.device}")
        config = Config(
            secret=SECRET,
            model_id=args.model_id,
            custom_hf_model_id=custom_hf_model_id,
            sae_sets=args_sae_sets,
            model_dtype=args.model_dtype,
            sae_dtype=args.sae_dtype,
            token_limit=args.token_limit,
            device=args.device,
            override_model_id=args.override_model_id,
            include_sae=args.include_sae,
            exclude_sae=args.exclude_sae,
            model_from_pretrained_kwargs=args.model_from_pretrained_kwargs,
            max_loaded_saes=args.max_loaded_saes,
            nnsight=args.nnsight,
        )
        Config._instance = config

        logger.info("Loading model...")

        hf_model = None
        hf_tokenizer = None
        if custom_hf_model_id is not None:
            logger.info("Loading custom HF model: %s", custom_hf_model_id)
            hf_model = AutoModelForCausalLM.from_pretrained(
                custom_hf_model_id,
                torch_dtype=STR_TO_DTYPE[config.model_dtype],
            )
            hf_tokenizer = AutoTokenizer.from_pretrained(custom_hf_model_id)

        model = HookedTransformer.from_pretrained_no_processing(
            (config.override_model_id if config.override_model_id else config.model_id),
            device=args.device,
            dtype=STR_TO_DTYPE[config.model_dtype],
            n_devices=device_count,
            hf_model=hf_model,
            **({"hf_config": hf_model.config} if hf_model else {}),
            tokenizer=hf_tokenizer,
            **config.model_kwargs,
        )
        Model._instance = model
        config.set_num_layers(model.cfg.n_layers)

        if model.tokenizer:
            special_token_ids = set(
                [
                    model.tokenizer.bos_token_id,  # type: ignore
                    model.tokenizer.eos_token_id,
                ]
                + model.tokenizer.additional_special_tokens_ids
            )
            special_token_ids = {
                tid
                for tid in special_token_ids
                if tid is not None  # type: ignore
            }
            # cache this one time for steering later use
            config.set_steer_special_token_ids(special_token_ids)  # type: ignore

        logger.info(
            f"Loaded {config.custom_hf_model_id if config.custom_hf_model_id else config.override_model_id} on {args.device}"
        )
        checkCudaError()

        logger.info("Loading SAEs...")
        SAEManager._instance = SAEManager(model.cfg.n_layers, args.device)
        SAEManager._instance.load_saes()

        global initialized
        initialized = True
        logger.info("Initialized: %s", initialized)

    await asyncio.get_event_loop().run_in_executor(None, load_model_and_sae)


@app.middleware("http")
async def check_secret_key(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    if request.url.path == "/health":
        return await call_next(request)

    config = Config.get_instance()
    if config.secret is None:
        return await call_next(request)
    secret_key = request.headers.get("X-SECRET-KEY")
    if not secret_key or secret_key != config.secret:
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid or missing X-SECRET-KEY header"},
        )
    return await call_next(request)


@app.middleware("http")
async def check_model(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    config = Config.get_instance()

    if request.method == "POST":
        try:
            body = await request.json()
            if "model" in body and (
                body["model"] != config.MODEL_ID
                and body["model"] != config.OVERRIDE_MODEL_ID
                and body["model"] != config.CUSTOM_HF_MODEL_ID
            ):
                logger.error("Unsupported model: %s", body["model"])
                return JSONResponse(
                    content={"error": "Unsupported model"}, status_code=400
                )
        except (json.JSONDecodeError, ValueError):
            pass

    return await call_next(request)


@app.middleware("http")
async def log_and_check_cuda_error(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    if not initialized:
        return JSONResponse(
            status_code=500,
            content={"error": "Server not initialized"},
        )
    logger.info("=== Request Info ===")
    logger.info(f"URL: {request.url}")

    try:
        body = await request.body()
        if body:
            logger.info(f"Body: {body.decode()}")
    except Exception as e:
        logger.error(f"Error reading body: {str(e)}")

    return await call_next(request)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):  # noqa: ARG001
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            # Optionally include traceback in development
            "traceback": traceback.format_exc() if app.debug else None,
        },
    )


def main():
    if os.getenv("SENTRY_DSN"):
        logger.info("Initializing Sentry")
        sentry_sdk.init(
            dsn=os.getenv("SENTRY_DSN"),
            traces_sample_rate=1.0,
            _experiments={
                "continuous_profiling_auto_start": True,
            },
            environment=",".join(args.sae_sets),
        )
    else:
        logger.info("SENTRY_DSN not set, skipping Sentry initialization")

    if args.list_models:
        list_available_options()
        sys.exit(0)

    return app
