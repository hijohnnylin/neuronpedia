# currently we only support one transcoder set per model.
# we should augment to support multiple transcoder sets per model

import gc
import gzip
import json
import os
import threading
import time
from typing import Any

import psutil
import requests
import torch
from circuit_tracer import attribute
from circuit_tracer.graph import prune_graph
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.utils.create_graph_files import (
    build_model,
    create_nodes,
    create_used_nodes_and_edges,
)
from circuit_tracer.utils.salient_logits import compute_salient_logits
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from starlette.concurrency import run_in_threadpool
from transformers import AutoTokenizer

load_dotenv()


LIMIT_TOKENS = int(os.getenv("TOKEN_LIMIT", 64))
DEFAULT_MAX_FEATURE_NODES = int(os.getenv("MAX_FEATURE_NODES", 10000))
OFFLOAD = None
UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", 1000))

SECRET_KEY = os.getenv("SECRET")
if not SECRET_KEY:
    raise ValueError(
        "SECRET environment variable not set. Please create a .env file with SECRET=<your_secret_key>"
    )

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN environment variable not set. Please create a .env file with HF_TOKEN=<your_huggingface_token>"
    )


def get_device() -> torch.device:
    """Determine the appropriate device for model loading."""
    device_env = os.environ.get("DEVICE")
    if device_env:
        return torch.device(device_env)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_model_dtype() -> torch.dtype | None:
    """
    Parse MODEL_DTYPE environment variable into torch dtype.
    Default is float32.
    """
    model_dtype_env = os.environ.get("MODEL_DTYPE", "bfloat16")

    dtype_mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    return dtype_mapping.get(model_dtype_env)


app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)

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

TLENS_MODEL_ID_TO_NP_MODEL_ID = {
    "google/gemma-2-2b": "gemma-2-2b",
    "google/gemma-3-4b-it": "gemma-3-4b-it",
    "meta-llama/Llama-3.2-1B": "llama3.1-8b",
    "Qwen/Qwen3-4B": "qwen3-4b",
}

GENERATOR_INFO = {
    "name": "circuit-tracer by Hanna & Piotrowski",
    "version": "0.3.1 | e09b5f3",
    "url": "https://github.com/safety-research/circuit-tracer",
}

loaded_model_arg = os.getenv("MODEL_ID")
print(f"Model: {loaded_model_arg}")
if not loaded_model_arg:
    raise ValueError(
        "TransformerLens model name is required. Please specify a model as a command line argument. Valid models: "
        + ", ".join(TLENS_MODEL_ID_TO_NP_MODEL_ID.keys())
    )

transcoder_set = os.getenv("TRANSCODER_SET")
print(f"Transcoder set: {transcoder_set}")
if not transcoder_set:
    raise ValueError("Transcoder set is required. Please specify a transcoders set.")

device = get_device()
model_dtype = get_model_dtype()


def check_is_nnsight_model(model_id: str) -> bool:
    return model_id.startswith("google/gemma-3-")


is_nnsight_model = check_is_nnsight_model(loaded_model_arg)

model = ReplacementModel.from_pretrained(
    loaded_model_arg,
    transcoder_set,
    device=device,
    dtype=model_dtype,
    lazy_encoder=False if is_nnsight_model else True,
    lazy_decoder=True,
    backend="nnsight" if is_nnsight_model else "transformerlens",
)


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


class GraphGenerationRequest(BaseModel):
    prompt: str
    model_id: str
    batch_size: int = 48
    max_n_logits: int = 10
    desired_logit_prob: float = 0.95
    node_threshold: float = 0.8
    edge_threshold: float = 0.98
    slug_identifier: str
    max_feature_nodes: int = DEFAULT_MAX_FEATURE_NODES
    signed_url: str | None = None
    user_id: str | None = None
    compress: bool = False


class ForwardPassRequest(BaseModel):
    prompt: str
    max_n_logits: int = 10
    desired_logit_prob: float = 0.95


class SteerFeature(BaseModel):
    layer: int
    index: int
    token_active_position: int
    steer_position: int | None = None
    steer_generated_tokens: bool = False
    delta: float | None = None
    ablate: bool = False


class SteerRequest(BaseModel):
    model_id: str
    prompt: str
    features: list[SteerFeature]
    n_tokens: int = 10
    top_k: int = 5
    temperature: float = 0.0
    freq_penalty: float = 0
    seed: int | None = None
    freeze_attention: bool = False


@app.get("/check-busy")
async def check_busy():
    """Check if the server is currently busy processing a request."""
    is_busy = request_lock.locked()
    return {"busy": is_busy}


def get_topk(logits: torch.Tensor, tokenizer, k: int = 5):
    probs = torch.softmax(logits.squeeze()[-1], dim=-1)
    topk = torch.topk(probs, k)
    return [
        (tokenizer.decode([topk.indices[i]]), topk.values[i].item()) for i in range(k)
    ]


@app.post("/steer", dependencies=[Depends(verify_secret_key)])
async def steer_handler(req: Request):
    """Handle steer requests"""
    print("========== Steer Start ==========")
    print(
        f"Thread {threading.get_ident()}: Received request. Attempting to acquire lock."
    )
    if not request_lock.acquire(blocking=False):
        print(
            f"Thread {threading.get_ident()}: Lock acquisition failed (busy). Rejecting request."
        )
        return JSONResponse(
            content={"error": "Server busy, please try again later."}, status_code=503
        )

    print(f"Thread {threading.get_ident()}: Lock acquired.")
    try:
        request_body = await req.json()
        req_data = SteerRequest.model_validate(request_body)

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
                    content={
                        "error": "When steer_generated_tokens is True, position must be None"
                    },
                    status_code=400,
                )
            # Validate that if steer_generated_tokens is False, position must be provided
            if not feature.steer_generated_tokens and feature.steer_position is None:
                return JSONResponse(
                    content={
                        "error": "When steer_generated_tokens is False, position must be provided"
                    },
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
                        0
                        if f.ablate
                        else activations[(f.layer, f.token_active_position, f.index)]
                        + f.delta,
                    )
                )
            else:
                intervention_tuples.append(
                    (
                        f.layer,
                        f.steer_position,
                        f.index,
                        0
                        if f.ablate
                        else activations[(f.layer, f.token_active_position, f.index)]
                        + f.delta,
                    )
                )

        # set the seed
        if req_data.seed is not None:
            torch.manual_seed(req_data.seed)
        default_tokenized = model.generate(
            req_data.prompt,
            do_sample=True,
            use_past_kv_cache=False,
            verbose=False,
            stop_at_eos=True,
            max_new_tokens=req_data.n_tokens,
            temperature=req_data.temperature,
            freq_penalty=req_data.freq_penalty,
            return_type="tokens",
        )[0]

        default_tokenized_str_tokens = [
            model.tokenizer.decode([token]) for token in default_tokenized
        ]

        default_generation = "".join(default_tokenized_str_tokens)

        # reset the seed
        if req_data.seed is not None:
            torch.manual_seed(req_data.seed)
        (steered_tokenized, steered_logits, _) = model.feature_intervention_generate(
            req_data.prompt,
            intervention_tuples,
            freeze_attention=req_data.freeze_attention,
            do_sample=True,
            verbose=False,
            stop_at_eos=True,
            max_new_tokens=req_data.n_tokens + 1,
            temperature=req_data.temperature,
            freq_penalty=req_data.freq_penalty,
            return_type="tokens",
        )

        steered_tokenized = steered_tokenized[0]
        steered_tokenized_str_tokens = [
            model.tokenizer.decode([token]) for token in steered_tokenized
        ]
        steered_generation = "".join(steered_tokenized_str_tokens)

        # get the logits at each step
        topk_default_by_token = []
        topk_steered_by_token = []

        with torch.inference_mode():
            default_logits = model(default_generation)

            # iterate through the tokens and get the logits
            for i in range(len(default_tokenized_str_tokens)):
                # If we're still processing the original prompt tokens (before generation),
                # append a blank item since we're only interested in generated tokens
                if i < sequence_length - 1:
                    topk_default_by_token.append(
                        {"token": default_tokenized_str_tokens[i], "top_logits": []}
                    )
                    continue
                # get the topk tokens
                topk_default = get_topk(
                    default_logits[:, : i + 1, :], model.tokenizer, req_data.top_k
                )
                # each topk default should be an object of token, prob
                topk_default_by_token.append(
                    {
                        "token": default_tokenized_str_tokens[i],
                        "top_logits": [
                            {"token": token, "prob": prob}
                            for token, prob in topk_default
                        ],
                    }
                )
            # we use the default tokenized str length because max_new_tokens is not +1 for default
            # we need +1 on steered because we want the logits for the last token
            for i in range(
                len(default_tokenized_str_tokens)
            ):  # If we're still processing the original prompt tokens (before generation),
                # append a blank item since we're only interested in generated tokens
                if i < sequence_length - 1:
                    topk_steered_by_token.append(
                        {"token": steered_tokenized_str_tokens[i], "top_logits": []}
                    )
                    continue
                topk_steered = get_topk(
                    steered_logits[:, : i + 1, :], model.tokenizer, req_data.top_k
                )
                topk_steered_by_token.append(
                    {
                        "token": steered_tokenized_str_tokens[i],
                        "top_logits": [
                            {"token": token, "prob": prob}
                            for token, prob in topk_steered
                        ],
                    }
                )

        print(f"Default generation: {default_generation}")
        print(f"Steered generation: {steered_generation}")

        response = {
            "DEFAULT_LOGITS_BY_TOKEN": topk_default_by_token,
            "STEERED_LOGITS_BY_TOKEN": topk_steered_by_token,
            "DEFAULT_GENERATION": default_generation,
            "STEERED_GENERATION": steered_generation,
        }

        return response

    finally:
        if request_lock.locked():
            print(f"Thread {threading.get_ident()}: Releasing lock in finally block.")
            request_lock.release()
        else:
            print(
                f"Thread {threading.get_ident()}: Lock was not held by current path in finally block (already released or never acquired)."
            )


@app.post("/forward-pass", dependencies=[Depends(verify_secret_key)])
async def forward_pass_handler(req: Request):
    """Handle forward pass requests to get salient logits"""
    print("========== Forward Pass Start ==========")

    print(
        f"Thread {threading.get_ident()}: Received request. Attempting to acquire lock."
    )
    if not request_lock.acquire(blocking=False):
        print(
            f"Thread {threading.get_ident()}: Lock acquisition failed (busy). Rejecting request."
        )
        return JSONResponse(
            content={"error": "Server busy, please try again later."}, status_code=503
        )

    print(f"Thread {threading.get_ident()}: Lock acquired.")
    try:
        request_body = await req.json()
        req_data = ForwardPassRequest.model_validate(request_body)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid request body", "details": e.errors()},
        )
    finally:
        if request_lock.locked():
            print(
                f"Thread {threading.get_ident()}: Releasing lock in validation finally block."
            )
            request_lock.release()

    try:
        print(f"Received forward pass request: prompt='{req_data.prompt}'")

        # Tokenize prompt
        # Only add special tokens if prompt doesn't already start with BOS
        tokens = model.tokenizer.encode(req_data.prompt, add_special_tokens=False)
        if tokens and tokens[0] != model.tokenizer.bos_token_id:
            tokens = model.tokenizer.encode(req_data.prompt, add_special_tokens=True)
        print(f"Tokens: {tokens}")

        # Convert to tensor and run forward pass
        input_ids = torch.tensor([tokens]).to(get_device())

        with torch.no_grad():
            # Get model output
            output = model(input_ids)
            # Check if output has logits property (nnsight case)
            if hasattr(output, "logits"):
                output = output.logits

            logits = output[0, -1, :]  # Get logits for last token

            # Get unembedding matrix
            # Compute salient logits
            # For models without unembed attribute, use lm_head weight matrix
            if hasattr(model, "unembed"):
                unembed_matrix = model.unembed.W_U
            elif hasattr(model, "lm_head"):
                unembed_matrix = model.lm_head.weight
            else:
                raise AttributeError(
                    "Model has neither 'unembed' nor 'lm_head' attribute"
                )

            logit_indices, logit_probs, _ = compute_salient_logits(
                logits,
                unembed_matrix,
                max_n_logits=req_data.max_n_logits,
                desired_logit_prob=req_data.desired_logit_prob,
            )

        # Decode tokens and create result
        results = []
        for idx, prob in zip(logit_indices.tolist(), logit_probs.tolist()):
            token = model.tokenizer.decode([idx])
            results.append(
                {"token": token, "token_id": idx, "probability": float(prob)}
            )

        # Also include some metadata
        response = {
            "prompt": req_data.prompt,
            "input_tokens": [model.tokenizer.decode([token]) for token in tokens],
            "salient_logits": results,
            "total_salient_tokens": len(results),
            "cumulative_probability": float(logit_probs.sum()),
        }

        print(
            f"Found {len(results)} salient tokens with cumulative prob: {response['cumulative_probability']:.4f}"
        )

        return response

    except Exception as e:
        print(f"Error in forward pass: {str(e)}")
        return {"error": f"Forward pass failed: {str(e)}"}

    finally:
        if request_lock.locked():
            print(f"Thread {threading.get_ident()}: Releasing lock in finally block.")
            request_lock.release()
        else:
            print(
                f"Thread {threading.get_ident()}: Lock was not held by current path in finally block (already released or never acquired)."
            )


@app.post("/generate-graph", dependencies=[Depends(verify_secret_key)])
async def generate_graph(req: Request):
    print(
        f"Thread {threading.get_ident()}: Received request. Attempting to acquire lock."
    )
    if not request_lock.acquire(blocking=False):
        print(
            f"Thread {threading.get_ident()}: Lock acquisition failed (busy). Rejecting request."
        )
        return JSONResponse(
            content={"error": "Server busy, please try again later."}, status_code=503
        )

    print(f"Thread {threading.get_ident()}: Lock acquired.")
    try:
        try:
            request_body = await req.json()
            req_data = GraphGenerationRequest.model_validate(request_body)
        except ValidationError as e:
            print(f"Thread {threading.get_ident()}: Validation error. Releasing lock.")
            request_lock.release()
            raise HTTPException(
                status_code=400,
                detail={"error": "Invalid request body", "details": e.errors()},
            )
        except Exception as e:
            print(
                f"Thread {threading.get_ident()}: JSON parsing error. Releasing lock."
            )
            request_lock.release()
            print(f"Error getting/parsing JSON: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        prompt = req_data.prompt
        tlens_model_id = req_data.model_id
        if tlens_model_id is None or tlens_model_id != loaded_model_arg:
            request_lock.release()
            raise HTTPException(
                status_code=400,
                detail=f"Model '{tlens_model_id}' is not available. Only '{loaded_model_arg}' is currently loaded.",
            )

        batch_size = req_data.batch_size
        max_n_logits = req_data.max_n_logits
        desired_logit_prob = req_data.desired_logit_prob
        node_threshold = req_data.node_threshold
        edge_threshold = req_data.edge_threshold
        slug_identifier = req_data.slug_identifier or f"generated-{int(time.time())}"
        max_feature_nodes = req_data.max_feature_nodes
        print(
            f"Thread {threading.get_ident()}: Processing request for prompt: '{prompt[:50]}...' with parameters:"
        )
        print(f"  model_id: {tlens_model_id}")
        print(f"  batch_size: {batch_size}")
        print(f"  max_n_logits: {max_n_logits}")
        print(f"  desired_logit_prob: {desired_logit_prob}")
        print(f"  node_threshold: {node_threshold}")
        print(f"  edge_threshold: {edge_threshold}")
        print(f"  transcoder_set: {transcoder_set}")
        print(f"  slug_identifier: {slug_identifier}")
        print(f"  max_feature_nodes: {max_feature_nodes}")

        def _blocking_graph_generation_task():
            print(
                f"Thread {threading.get_ident()} (worker): Starting blocking graph generation."
            )
            _total_start_time = time.time()

            try:
                tokens = model.tokenizer.encode(prompt, add_special_tokens=False)
                print(
                    f"Thread {threading.get_ident()} (worker): {len(tokens)} Tokens: {tokens}"
                )
                if len(tokens) > LIMIT_TOKENS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Prompt exceeds token limit ({len(tokens)} > {LIMIT_TOKENS})",
                    )
            except Exception as e:
                print(
                    f"Thread {threading.get_ident()} (worker): Tokenization error: {e}"
                )
                raise HTTPException(status_code=500, detail="Failed to tokenize prompt")

            print(f"Thread {threading.get_ident()} (worker): Prompt: '{prompt}'")

            attribution_start = time.time()
            _graph = attribute(
                prompt,
                model,
                max_n_logits=max_n_logits,
                desired_logit_prob=desired_logit_prob,
                batch_size=batch_size,
                max_feature_nodes=req_data.max_feature_nodes,
                offload=OFFLOAD,
                update_interval=UPDATE_INTERVAL,
            )
            attribution_time_ms = (time.time() - attribution_start) * 1000
            print(
                f"Thread {threading.get_ident()} (worker): Attribution Time: {attribution_time_ms:.2f}ms"
            )

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
            _used_nodes, _used_edges = create_used_nodes_and_edges(
                _graph, _nodes, _edge_mask
            )
            print("used nodes and edges created")
            _output_model = build_model(
                _graph,
                _used_nodes,
                _used_edges,
                slug_identifier,
                TLENS_MODEL_ID_TO_NP_MODEL_ID[tlens_model_id],
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

            # Add additional metadata fields
            model_dict["metadata"]["info"] = {
                "creator_name": req_data.user_id
                if req_data.user_id
                else "Anonymous (CT)",
                "creator_url": "https://neuronpedia.org",
                "source_urls": TRANSCODER_SET_TO_SOURCE_URL_ARRAYS[transcoder_set],
                "transcoder_set": transcoder_set,
                "generator": GENERATOR_INFO,
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
                data_to_upload = gzip.compress(
                    model_json.encode("utf-8"), compresslevel=3
                )
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
                return {"error": "Failed to upload file"}

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

            print(
                f"Thread {threading.get_ident()} (worker): Total Time for blocking task: {_total_time_ms=:.2f}s"
            )

            return {
                "success": f"Graph uploaded successfully to url: {req_data.signed_url}"
            }

        try:
            result = await run_in_threadpool(_blocking_graph_generation_task)
            print(f"Thread {threading.get_ident()}: Blocking task completed.")
            return result
        except HTTPException:
            raise
        except Exception as e:
            import traceback

            print(
                f"Thread {threading.get_ident()}: Error during graph generation in worker thread: {e}"
            )
            print("Stack trace:")
            traceback.print_exc()
            raise HTTPException(
                status_code=500, detail="Internal server error during graph generation"
            )

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
