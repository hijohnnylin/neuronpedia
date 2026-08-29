import asyncio
import gc
import json
import logging
import os
import time
import traceback
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import sentry_sdk
import torch
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from interp_engine import (
    Address,
    VLLMModel,
    check_cuda_driver,
    load_model,
    select_backend,
    to_address,
)

from neuronpedia_inference.args import parse_env_and_args
from neuronpedia_inference.config import Config
from neuronpedia_inference.endpoints.activation.all import (
    router as activation_all_router,
)
from neuronpedia_inference.endpoints.activation.all_batch import (
    router as activation_all_batch_router,
)
from neuronpedia_inference.endpoints.activation.attention import (
    router as activation_attention_router,
)
from neuronpedia_inference.endpoints.activation.raw import (
    router as activation_raw_router,
)
from neuronpedia_inference.endpoints.activation.single import (
    router as activation_single_router,
)
from neuronpedia_inference.endpoints.activation.single_batch import (
    router as activation_single_batch_router,
)
from neuronpedia_inference.endpoints.activation.source import (
    router as activation_source_router,
)
from neuronpedia_inference.endpoints.activation.topk_by_token import (
    router as activation_topk_by_token_router,
)
from neuronpedia_inference.endpoints.activation.topk_by_token_batch import (
    router as activation_topk_by_token_batch_router,
)
from neuronpedia_inference.endpoints.capabilities import router as capabilities_router
from neuronpedia_inference.endpoints.chat_template import (
    router as chat_template_router,
)
from neuronpedia_inference.endpoints.lens.lens_loader import (
    load_jacobian_lens_at_startup,
    place_jacobian_lens_on_device,
    place_jacobian_lens_on_worker,
)
from neuronpedia_inference.endpoints.lens.prompt import (
    router as lens_prompt_router,
)
from neuronpedia_inference.endpoints.lens.prompt import warmup_lens
from neuronpedia_inference.endpoints.lens.residual_spec import block_output_point
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
from neuronpedia_inference.endpoints.util.similarity_matrix_pred import (
    router as similarity_matrix_pred_router,
)
from neuronpedia_inference.logging import initialize_logging
from neuronpedia_inference.operation_ids import sdk_operation_id
from neuronpedia_inference.resilience import (
    is_fatal_cuda_error,
    probe_cuda_or_die,
    terminate_for_restart,
)
from neuronpedia_inference.sae_cache import sae_cache
from neuronpedia_inference.sae_manager import SAEManager  # noqa: F401
from neuronpedia_inference.schemas import HealthResponse
from neuronpedia_inference.shared import (  # noqa: F401
    STR_TO_DTYPE,
    Model,
    RecoverableOutOfMemory,
    RequestTooLarge,
    configure_budget,
    configure_limiter,
)
from neuronpedia_inference.startup_memory import (
    ModelMemoryInfo,
    compute_activation_token_limit,
    compute_serving_limits,
    measure_pinnable_host_bytes,
    measure_transient_budget,
    resolve_sae_gpu_budget_bytes,
)
from neuronpedia_inference.utils import checkCudaError
from neuronpedia_inference.vllm_optional import VLLM_AVAILABLE

# How long to wait before retrying after a HuggingFace 429 (Too Many Requests).
HF_429_RETRY_WAIT_SECONDS = 60


def _vllm_gpu_memory_utilization() -> float:
    """Read ``VLLM_GPU_MEMORY_UTILIZATION`` at call time (default 0.9).

    Must not be a module-level constant: tests (and ops) set the env var after the
    server module is imported, and a frozen import-time value would ignore them.
    """
    return float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))


#: Powers of two through 256. Decode uses 1; a 200-token prompt pads to 256.
#: vLLM's default is every 8 tokens to 256 (~35 graphs). On DeepSeek-V4 that
#: ladder profiles at ~47 GiB and leaves no KV; this list is 9 graphs.
_DEFAULT_CUDAGRAPH_CAPTURE_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _vllm_cudagraph_capture_sizes() -> list[int]:
    """CUDA-graph batch sizes for static / generation-only pods.

    ``VLLM_CUDAGRAPH_CAPTURE_SIZES`` is a comma-separated list. Unset uses
    :data:`_DEFAULT_CUDAGRAPH_CAPTURE_SIZES`. Hooked vLLM ignores this (no graphs).
    """
    raw = os.getenv("VLLM_CUDAGRAPH_CAPTURE_SIZES")
    if raw is None or raw.strip() == "":
        return list(_DEFAULT_CUDAGRAPH_CAPTURE_SIZES)
    try:
        sizes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(
            f"VLLM_CUDAGRAPH_CAPTURE_SIZES={raw!r} must be a comma-separated list of integers "
            "(e.g. 1,2,4,8,16,32,64,128,256)."
        ) from exc
    if not sizes or any(n < 1 for n in sizes):
        raise ValueError(f"VLLM_CUDAGRAPH_CAPTURE_SIZES={raw!r} needs at least one integer >= 1.")
    return sorted(set(sizes))


def _graph_compilation_config() -> dict[str, list[int]]:
    return {"cudagraph_capture_sizes": _vllm_cudagraph_capture_sizes()}


def _vllm_max_num_batched_tokens() -> int | None:
    """``MAX_NUM_BATCHED_TOKENS``: the prefill chunk size. Unset lands on vLLM's 2048, not 8192.

    On a static pod this is also what sizes the static buffers, which are one static row per
    batched token at every static site. The engine already picks the largest value whose buffers
    fit and refuses to start when none do, so most pods want that default.

    Read "the engine's default" carefully, though, because it is only half the story: the fit
    reasons about an ASSUMED 8192 and writes the value back only when it has to LOWER it. A set
    that fits 8192 therefore leaves the kwarg unset and vLLM chooses, and vLLM chooses 2048 --
    ``AsyncLLM.from_engine_args`` defaults ``usage_context`` to ``ENGINE_CONTEXT``, which appears
    in neither branch of vLLM's ``get_batch_defaults``, so it falls through to
    ``SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS``. So the fit over-budgets buffers 4x on
    every pod it does not lower, and pinning is also how a pod stops depending on that fallback.

    What it cannot see is that the fit is GREEDY: it spends the pool on buffers before graphs, so
    where the weights leave little room the largest size that fits can leave CUDA-graph capture
    too little and vLLM then starts with no KV blocks. Pinning it is how such a pod buys graphs
    and KV instead of buffer rows no request reaches. A pinned value is kept when it fits, so this
    only ever lowers the ceiling; it never asks for more than the engine would allow.
    """
    raw = os.getenv("MAX_NUM_BATCHED_TOKENS")
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"MAX_NUM_BATCHED_TOKENS={raw!r} must be an integer.") from exc
    if value < 1:
        raise ValueError(f"MAX_NUM_BATCHED_TOKENS={raw!r} must be an integer >= 1.")
    return value


def _engine_context_len(token_limit: int, lens_token_limit: int) -> int:
    """The context the engine has to be built with, given both per-endpoint prompt caps.

    ``token_limit`` bounds completion/steer/tokenize and ``lens_token_limit`` bounds the lens
    endpoints, and the latter is deliberately allowed to be the higher of the two -- that is the
    whole point of it being a separate knob. The engine, though, has exactly one context, so it
    has to be sized from whichever cap is larger.

    Sizing it from ``token_limit`` alone is what made a lens conversation between the two caps
    fail the wrong way: it passed the endpoint's own length check (against ``lens_token_limit``)
    and then died inside vLLM with a raw "maximum context length is N tokens" error, on a pod
    where the lens limit was 1024 and the engine had been built for 256.
    """
    return max(int(token_limit), int(lens_token_limit))


#: ``STATIC_POINTS`` values whose sites are not known until the SAEs have loaded. Such a pod is
#: built hooked and promoted by ``configure_static`` before warmup, so it reaches the engine
#: constructor with no tap set at all -- see :func:`_vllm_engine_backend`.
_SAE_RESOLVED_MODES = ("sae", "sae+auto")


def _parse_static_points(raw: str | None) -> Any:
    """``STATIC_POINTS``: unset, ``auto``, ``sae``, ``sae+auto``, or a JSON list of addresses.

    ``sae+auto`` is the union of the other two named modes: the SAE hook sites AND the residual
    point at every layer, reads and writes. It exists because those are two different questions and
    one pod needs both answered. An SAE set covers the layers its SAEs were trained at, which for
    `gemmascope-mlp-16k` + `gemmascope-transcoder-16k` is `mlp_out_post` and `resid_mid`
    everywhere and a residual point on 13 layers of 26 -- and a lens read-out asks for the residual
    point at ALL of them, final layer included (``endpoints/lens/prompt._select_layers``). So `sae`
    alone serves the sources and refuses the lens, while `auto` alone serves the lens and takes the
    transcoder and MLP sources down with it. Neither is the pod, and an explicit JSON list is not
    either: a list declares no writes, so it would trade every steer for the lens.

    An empty list used to be how a pod asked for graphs with no taps. That mode now has a name of
    its own -- ``backend="vllm-generate"``, reached through ``GENERATION_ONLY`` -- so the empty list
    is refused rather than quietly routed there, which keeps one spelling per mode.
    """
    if raw is None or raw == "":
        return None
    if raw in ("auto", *_SAE_RESOLVED_MODES):
        return raw
    try:
        points = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"STATIC_POINTS={raw!r} is not 'auto', 'sae', 'sae+auto', or JSON (e.g. 'auto' or [[\"resid_post\", 7]])."
        ) from exc
    if isinstance(points, list) and not points:
        raise ValueError(
            "STATIC_POINTS=[] declares no taps, so it asks for a graph-mode pod that cannot "
            "capture anything. Set GENERATION_ONLY=true for that pod, or name the sites to "
            "declare (STATIC_POINTS=auto, sae, sae+auto, or a JSON list of addresses)."
        )
    return points


def _parse_extra_static_points(raw: str | None) -> list[Address]:
    """``STATIC_POINTS_EXTRA``: a JSON list of addresses to declare beside a resolved set.

    Same spelling as an explicit ``STATIC_POINTS`` list, so a site moves between the two by
    cut-and-paste. Empty and unset both mean nothing extra, which is the common case.
    """
    if raw is None or raw.strip() == "":
        return []
    try:
        points = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"STATIC_POINTS_EXTRA={raw!r} is not JSON (e.g. '[\"resid_post.40\"]' or '[[\"resid_post\", 40]]')."
        ) from exc
    if not isinstance(points, list):
        raise ValueError(f"STATIC_POINTS_EXTRA={raw!r} must be a JSON list of addresses.")
    return [to_address(point) for point in points]


def _with_extra_points(
    reads: list[Address],
    writes: list[Address],
    extra: list[Address],
) -> tuple[list[Address], list[Address]]:
    """Add ``STATIC_POINTS_EXTRA`` to a resolved set, as reads and as writes.

    Both, for the reason ``auto`` implies its writes: a capture site that cannot be written is a
    readout that works and then refuses every steer at the same layer, and steering on a persona
    direction at the layer it was fitted at is a thing this server is asked to do. One layer both
    ways costs two buffers, which is the cheap half of the trade `sae+auto` gets wrong at scale.

    Deduplicated, so naming a site an SAE already covers is free rather than doubled, and appended
    after the resolved set so the startup log reads as "the SAE set, plus what was declared on top".
    """
    read_keys = {str(address) for address in reads}
    write_keys = {str(address) for address in writes}
    merged_reads = list(reads)
    merged_writes = list(writes)
    for address in extra:
        if str(address) not in read_keys:
            merged_reads.append(address)
            read_keys.add(str(address))
        if str(address) not in write_keys:
            merged_writes.append(address)
            write_keys.add(str(address))
    return merged_reads, merged_writes


def _with_residual_set(
    model: Any,
    reads: list[Address],
    writes: list[Address],
    num_layers: int,
) -> tuple[list[Address], list[Address]]:
    """Add the lens's point at every layer to a resolved SAE set (``STATIC_POINTS=sae+auto``).

    The point is asked of ``block_output_point``, which is what ``/lens/prompt`` itself reads
    through, rather than spelled ``resid_post`` here: that name is wrong on a hyper-connection
    trunk, where the engine refuses it outright and the block output is ``resid_streams``. Asking
    the endpoint's own resolver is what keeps this declaration and that read-out from drifting
    apart -- a set declared under one name and read under the other is a 400 on a pod that
    declared exactly what was wanted.

    Written as reads AND writes, for the reason ``resolve_static_points`` fills the write set in
    behind ``auto``: a read tap alone serves the read-out and then refuses every steer, ablation
    and swap derived from it, which on `/lens/*` is half the endpoint. The SAE writes keep their
    own mapping (``resid_pre[L]`` steers at ``resid_post[L-1]``) and are not touched.

    Deduplicated on the address, so the layers an SAE already covers cost nothing twice -- the
    13-of-26 overlap on `inference-gemma-2-2b-b-static` is the case this is for. Order is the SAE
    sites first, which keeps the log line and any refusal reading as "the SAE set, plus the
    residual set it was missing".
    """
    point = block_output_point(model)
    read_keys = {str(address) for address in reads}
    write_keys = {str(address) for address in writes}
    merged_reads = list(reads)
    merged_writes = list(writes)
    for layer in range(int(num_layers)):
        address = Address(point, layer)
        if str(address) not in read_keys:
            merged_reads.append(address)
        if str(address) not in write_keys:
            merged_writes.append(address)
    return merged_reads, merged_writes


def _vllm_engine_backend(*, generation_only: bool, static_points: Any) -> str:
    """Which of interp-engine's three vLLM backends this pod's two flags ask for.

    The engine takes the mode as ``backend=`` rather than inferring it from a tap set, so the
    mapping happens once, here, instead of being spread over the kwargs a caller happens to pass.

    ``STATIC_POINTS=sae`` and ``sae+auto`` are not cases: their sites are not known until the SAEs
    have loaded, so such a pod is built hooked and promoted by ``configure_static`` before warmup,
    and arrives here with ``static_points=None``.
    """
    if generation_only:
        return "vllm-generate"
    if static_points is not None:
        return "vllm-static"
    return "vllm"


def _vllm_backend_kwargs(
    max_model_len: int,
    *,
    backend: str,
    static_points: Any = None,
) -> dict[str, Any]:
    """Engine construction kwargs for the vLLM backend, beyond what ``load_model`` derives."""
    extra: dict[str, Any] = {
        # Tell vLLM this model is text-only, because on this server it is: no endpoint accepts
        # an image. The flag it reads, `is_mm_prefix_lm`, means "image tokens attend
        # bidirectionally", and vLLM keys it off `model_type` against a list that holds
        # "gemma3" -- so gemma-3-12b (model_type "gemma3") gets it while gemma-3-1b
        # (model_type "gemma3_text") does not. Every attention layer then demands a backend
        # that can express that mask, which rules FlashAttention out and leaves Triton unified
        # attention running even on a pure-text request. It is a no-op on a text-only
        # architecture, where vLLM derives the same False.
        #
        # This is only sound while the "no images" premise holds. Delete it, do not work
        # around it, the day an endpoint takes one.
        "hf_overrides": {"is_mm_prefix_lm": False},
    }
    kwargs: dict[str, Any] = {
        "gpu_memory_utilization": _vllm_gpu_memory_utilization(),
        "max_model_len": max_model_len,
        "extra_vllm_kwargs": extra,
    }
    batched_tokens = _vllm_max_num_batched_tokens()
    if batched_tokens is not None:
        extra["max_num_batched_tokens"] = batched_tokens
    if backend != "vllm":
        # Both graph-replaying backends want inductor. The engine owns enforce_eager for them --
        # it sets False and refuses True -- so this must not pass it, and vllm-generate needs no
        # tap argument because backend= is already the whole declaration.
        extra["compilation_config"] = _graph_compilation_config()
    if backend == "vllm-static":
        kwargs["static_points"] = static_points
    return kwargs


def _resolve_generation_only(args: Any) -> bool:
    """Validate ``GENERATION_ONLY`` against the rest of the startup config, and refuse early.

    The flag selects the engine's ``backend="vllm-generate"``: CUDA graphs kept, no taps declared.
    Graph replay does not run the Python forward that capture and steering hooks live on, so the
    flag does not disable one feature, it disables every hook-dependent one -- which is most of this
    server. Two combinations are configuration errors rather than degraded modes, and both are
    cheaper to reject here than to debug from a pod that boots happily and 400s on every request a
    router sends it:

    - **SAE sets.** An SAE read *is* a capture. A pod with SAEs loaded and no way to capture has
      nothing to do with them but occupy VRAM.
    - **The eager backend.** There is no generate-only variant of it; the flag would silently buy
      nothing while still turning the endpoints off.
    """
    if not getattr(args, "generation_only", False):
        return False
    if args.backend != "vllm":
        raise ValueError(
            f"GENERATION_ONLY=true is only meaningful on the vLLM backend, but this pod resolved to "
            f"{args.backend!r}. It selects backend='vllm-generate', which trades vLLM's capture "
            "hooks for the CUDA graphs they rule out; there is no such tradeoff to make on eager, "
            "which hooks the module tree in-process. Unset GENERATION_ONLY, or force vLLM with "
            "--force-vllm."
        )
    if args.sae_sets:
        raise ValueError(
            f"GENERATION_ONLY=true cannot be combined with SAE_SETS={args.sae_sets!r}: reading an SAE "
            "means capturing an activation, and this mode exists to give up capture (CUDA graph "
            "replay skips the Python forward the hooks are attached to). Start with SAE_SETS='[]' to "
            "serve completions only, or unset GENERATION_ONLY and pass STATIC_POINTS=sae to declare "
            "those SAE sites and read them at graph speed."
        )
    static = getattr(args, "static_points", None)
    if static not in (None, ""):
        raise ValueError(
            "GENERATION_ONLY=true already means backend='vllm-generate' (graphs, no taps). "
            "Do not also pass STATIC_POINTS; omit GENERATION_ONLY for a declared tap set."
        )
    return True


def _is_hf_429_error(exc: BaseException) -> bool:
    """Return True if `exc` (or anything in its cause/context chain) is a
    HuggingFace HTTP 429 (Too Many Requests) response.

    HuggingFace downloads raise `requests.exceptions.HTTPError` /
    `huggingface_hub.utils.HfHubHTTPError`, which carry the originating
    `requests.Response` on `.response`. We walk the exception chain because the
    429 is often re-raised as the cause of a higher-level error.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code == 429:
            return True
        current = current.__cause__ or current.__context__
    return False


def _serving_url() -> str:
    """The address uvicorn was told to bind to, exported by start.py."""
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = os.getenv("SERVER_PORT", "5002")
    # 0.0.0.0 / :: are bind wildcards, not addresses anyone can open.
    display_host = "localhost" if host in ("0.0.0.0", "::", "") else host
    return f"http://{display_host}:{port}"


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


def _log_banner(title: str, lines: list[str]) -> None:
    """One visually obvious block, so the end of a long noisy startup is findable."""
    bar = "=" * 100
    body = "\n".join(f"  {line}" for line in lines)
    logger.info("\n%s\n  %s\n%s\n%s\n%s\n", bar, title, "-" * 100, body, bar)


def _log_ready_banner(elapsed_seconds: float) -> None:
    config = Config.get_instance()
    sae_manager = SAEManager.get_instance()
    configured_saes = sum(len(saes) for saes in sae_manager.sae_set_to_saes.values())
    model_desc = config.custom_hf_model_id or config.override_model_id or config.model_id
    _log_banner(
        f"==== LOADING COMPLETE - SERVING ON {_serving_url()} ====",
        [
            f"model: {model_desc}",
            f"backend: {config.backend} | device: {config.device} | gpus: {config.num_gpus} | "
            f"model dtype: {config.model_dtype} | sae dtype: {config.sae_dtype}",
            f"saes: {len(sae_manager.loaded_saes)} resident of {configured_saes} configured "
            f"({', '.join(sae_manager.valid_sae_sets) or 'none'})",
            f"token limits: prompt={config.token_limit} activation={config.activation_token_limit} "
            f"lens={config.lens_token_limit}",
            f"startup took {_format_duration(elapsed_seconds)}",
        ],
    )


# Initialize logging at module level
initialize_logging()

logger = logging.getLogger(__name__)
logger.info("Server module initialized")

load_dotenv()

global initialized
initialized = False
global initialization_error
initialization_error: str | None = None

# Metadata carried over from the hand-written spec this server used to be generated from, so
# the published clients keep the same title, version and license. Bump `version` when the wire
# format changes; the publish job reads it.
app = FastAPI(
    title="Neuronpedia - Inference Server",
    version="1.9.0",
    contact={"email": "johnny@neuronpedia.org"},
    license_info={"name": "Apache-2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
    generate_unique_id_function=sdk_operation_id,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip compression middleware (only compresses if client sends Accept-Encoding: gzip)
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=6)

args = parse_env_and_args()

if TYPE_CHECKING:
    from sentry_sdk._types import Event, Hint

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


# Module scope on purpose. Production starts this server as `uvicorn ...server:app` (see
# start.py), so any init placed in a `main()` never runs and the server reports nothing. It also
# has to land above the `include_router` calls below, since the FastAPI integration only wraps
# handlers registered after it.
if args.sentry_dsn:
    logger.info("Initializing Sentry")
    sentry_sdk.init(
        dsn=args.sentry_dsn,
        environment=os.getenv("SENTRY_ENVIRONMENT", "development"),
        release=os.getenv("SENTRY_RELEASE"),
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.25")),
        profile_session_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.25")),
        profile_lifecycle="trace",
        before_send=scrub_sentry_event,
        before_send_transaction=scrub_sentry_event,
    )
    # Which model and SAEs a pod serves is a property of that pod, not a deploy target, so it
    # belongs on a tag. Putting it in `environment` (as this used to) makes every new source set
    # look like a new deploy environment in the issue filters.
    sentry_sdk.set_tag("model_id", args.model_id)
    sentry_sdk.set_tag("sae_sets", ",".join(args.sae_sets) or "no-saes")
    sentry_sdk.set_tag("model_dtype", args.model_dtype)
    # Everything else this pod was started with, as a context rather than more tags: tags are a
    # searchable index, and a namespace this size would swamp the ones worth filtering on above.
    # `args` is built entirely from the environment (see args.py), so this is literally the pod's
    # start arguments -- enough to tell two pods of the same model apart by token limit, GPU count
    # or SAE budget. The DSN is dropped rather than echoed back into its own payload.
    sentry_sdk.set_context("start_args", {k: v for k, v in vars(args).items() if k != "sentry_dsn"})
else:
    logger.info("SENTRY_DSN not set, skipping Sentry initialization")


# we have to initialize SAE's AFTER server startup, because some infrastructure providers require
# our server to respond to health checks within a few minutes of starting up
@app.on_event("startup")  # pyright: ignore[reportDeprecated]
async def startup_event():
    # Tests (and any caller that already ran ``initialize()``) must be able to enter
    # ``TestClient`` as a context manager without kicking off a second load -- that
    # would race the already-loaded model / vLLM engine.
    if initialized:
        logger.info("Startup skipped: already initialized")
        return
    logger.info("Starting initialization...")
    # Wait briefly to ensure server is ready
    await asyncio.sleep(3)
    # Start initialization in background
    init_task = asyncio.create_task(initialize(args.custom_hf_model_id))

    def _log_init_task_result(task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except Exception:
            logger.exception("Background initialization task failed")

    init_task.add_done_callback(_log_init_task_result)
    logger.info("Initialization started")


v1_router = APIRouter(prefix="/v1")

v1_router.include_router(capabilities_router)
v1_router.include_router(activation_all_router)
v1_router.include_router(activation_all_batch_router)
v1_router.include_router(steer_completion_chat_router)
v1_router.include_router(steer_completion_router)
v1_router.include_router(activation_single_router)
v1_router.include_router(activation_single_batch_router)
v1_router.include_router(activation_attention_router)
v1_router.include_router(activation_topk_by_token_router)
v1_router.include_router(activation_topk_by_token_batch_router)
v1_router.include_router(sae_topk_by_decoder_cossim_router)
v1_router.include_router(sae_vector_router)
v1_router.include_router(tokenize_router)
v1_router.include_router(chat_template_router)
v1_router.include_router(similarity_matrix_pred_router)
v1_router.include_router(activation_source_router)
v1_router.include_router(activation_raw_router)
v1_router.include_router(lens_prompt_router)
app.include_router(v1_router)


def _openapi_with_secret_key_auth() -> dict[str, Any]:
    """Document the ``X-SECRET-KEY`` header that ``check_secret_key`` below enforces.

    That check is middleware rather than a route dependency, so FastAPI cannot see it and
    would otherwise emit a spec claiming every endpoint is open -- which the clients
    generated from that spec would then believe. ``/health`` is the one exemption, matching
    the middleware.
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
    schema["paths"]["/health"]["get"]["security"] = []
    app.openapi_schema = schema
    return schema


app.openapi = _openapi_with_secret_key_auth


@app.get("/health", responses={200: {"model": HealthResponse}})
async def health_check():
    return {"status": "healthy"}


@app.post("/initialize")
async def initialize(
    custom_hf_model_id: str | None = None,
):
    logger.info("Initializing...")
    global initialization_error
    initialization_error = None
    startup_started_at = time.monotonic()

    # Move the heavy operations to a separate thread pool to prevent blocking
    def load_model_and_sae():
        # Before anything reaches the GPU: args.device is still the *requested* device
        # here (select_backend resolves it below), which is what decides whether a
        # too-old driver is fatal or merely worth a warning.
        check_cuda_driver(args.device, cpu_hint="To serve on CPU instead, pass --device cpu.")

        # The model_id / SAE-set values are not validated against the SAELens directory
        # (get_saelens_neuronpedia_directory_df): sets served here can be local or not yet
        # published, so an unknown name is not an error.
        # iterate through sae_sets and split them by spaces
        args_sae_sets = []
        for sae_set in args.sae_sets:
            args_sae_sets.extend(sae_set.split())
        logger.info("SAE sets: %s", args_sae_sets)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.set_grad_enabled(False)
        checkCudaError("cpu")

        SECRET = os.getenv("SECRET")

        # Auto-select backend (vLLM vs EagerModel) + device + dtype from what
        # the box can do and what the model needs. Explicit DEVICE / MODEL_DTYPE and
        # the backend force (--force-vllm / --force-eager -> FORCE_BACKEND) override.
        # ``--model_id`` is the Hugging Face repo id (override/custom_hf still win when set).
        probe_hf_model_id = custom_hf_model_id or args.override_model_id or args.model_id
        selection = select_backend(
            probe_hf_model_id,
            requested_device=args.device,
            requested_dtype=args.model_dtype,
            force_backend=args.force_backend,
            vllm_available=VLLM_AVAILABLE,
        )
        logger.info("Backend selection for %s: %s", probe_hf_model_id, selection.reason)
        args.device = selection.device
        args.model_dtype = selection.dtype
        args.backend = "vllm" if selection.use_vllm else "eager"
        static_mode = _parse_static_points(getattr(args, "static_points", None))
        if static_mode is not None and args.backend != "vllm":
            raise ValueError(
                f"STATIC_POINTS selects backend='vllm-static', but this pod resolved to "
                f"{args.backend!r}. The eager backend hooks the module tree in-process, so every "
                "site is already reachable and there is nothing to declare. Omit STATIC_POINTS, or "
                "force vLLM with --force-vllm."
            )
        if static_mode in _SAE_RESOLVED_MODES and not args_sae_sets:
            raise ValueError(
                f"STATIC_POINTS={static_mode} needs SAE_SETS so there are hook sites to declare. "
                "Use STATIC_POINTS=auto for the residual set alone."
            )
        extra_points = _parse_extra_static_points(getattr(args, "static_points_extra", None))
        if extra_points and static_mode not in _SAE_RESOLVED_MODES:
            # Refused rather than ignored. Every other value already says everything it is going
            # to: `auto` declares each layer, an explicit list is the caller's own set, and unset
            # is a hooked pod where nothing needs declaring. Merging into one of those would be a
            # no-op that reads, in a deploy config, as though a site had been added.
            raise ValueError(
                f"STATIC_POINTS_EXTRA has nothing to add to STATIC_POINTS={static_mode!r}. It "
                "declares sites beside a set this server resolves after the SAEs load, so it "
                "applies to STATIC_POINTS=sae or sae+auto. Add the sites to the list itself, or "
                "drop STATIC_POINTS_EXTRA."
            )
        num_gpus = max(1, int(getattr(args, "num_gpus", 1) or 1))
        if num_gpus > 1:
            logger.info("Multi-GPU: sharding across %d GPUs (%s)", num_gpus, args.backend)

        config = Config(
            secret=SECRET,
            model_id=args.model_id,
            custom_hf_model_id=custom_hf_model_id,
            sae_sets=args_sae_sets,
            model_dtype=args.model_dtype,
            sae_dtype=args.sae_dtype,
            token_limit=args.token_limit,
            lens_token_limit=args.lens_token_limit,
            device=args.device,
            override_model_id=args.override_model_id,
            include_sae=args.include_sae,
            exclude_sae=args.exclude_sae,
            model_from_pretrained_kwargs=args.model_from_pretrained_kwargs,
            max_loaded_saes=args.max_loaded_saes,
            backend=args.backend,
            num_gpus=num_gpus,
            sae_gpu_budget_gib=args.sae_gpu_budget_gib,
            sae_pinned_host_gib=args.sae_pinned_host_gib,
            generation_only=_resolve_generation_only(args),
        )
        Config._instance = config

        # The engine is keyed by the raw HF repo id.
        model_to_load = config.override_model_id if config.override_model_id else config.model_id
        hf_model_id = config.custom_hf_model_id or model_to_load
        logger.info("Model to load (HF id): %s", hf_model_id)
        # The backend was already resolved above, so pass it explicitly rather than letting
        # load_model re-run the ladder -- Config needs the resolved device/dtype before the
        # model exists, so the app owns the selection call and load_model only constructs.
        # Backend-specific kwargs stay explicit here.
        if args.backend == "vllm":
            # The engine splits vLLM into three backends by what the pod declares up front, so
            # resolve which one these flags mean. config.backend stays the family name, because
            # what the rest of the server asks is "vLLM or eager".
            load_points = None if static_mode in _SAE_RESOLVED_MODES else static_mode
            engine_backend = _vllm_engine_backend(generation_only=config.generation_only, static_points=load_points)
            logger.info("Loading model with engine-owned vLLM backend (%s)...", engine_backend)
            # Lightweight construct (tokenizer + config); the vLLM engine (with native
            # extract_hidden_states on) is created lazily on first async use.
            # num_gpus>1 -> tensor-parallel across that many GPUs on this node.
            backend_kwargs: dict[str, Any] = _vllm_backend_kwargs(
                _engine_context_len(config.token_limit, config.lens_token_limit),
                backend=engine_backend,
                static_points=load_points,
            )
            if config.generation_only:
                logger.warning(
                    "GENERATION_ONLY: loading backend='vllm-generate', which keeps vLLM's CUDA "
                    "graphs. This pod serves completions and tokenization; capture, steering, DFA, "
                    "attention and the lens endpoints are unavailable and report so at "
                    "/capabilities."
                )
            elif static_mode in _SAE_RESOLVED_MODES:
                logger.info(
                    "STATIC_POINTS=%s: will bind static wraps after SAE load, before engine warmup.",
                    static_mode,
                )
        else:
            engine_backend = "eager"
            logger.info("Loading model with interp-engine (raw HF, eager PyTorch)...")
            backend_kwargs = {
                # Force eager attention so the /activation/attention endpoint can read
                # per-head attention probabilities (no-op cost for other endpoints).
                "attn_implementation": "eager",
                "default_prepend_bos": True,
                "model_kwargs": config.model_kwargs,
            }
        model = load_model(
            hf_model_id,
            backend=engine_backend,
            device=args.device,
            dtype=config.model_dtype,
            num_gpus=num_gpus,
            **backend_kwargs,
        )

        Model._instance = model
        num_layers = model.n_layers
        config.set_num_layers(num_layers)

        # Memory-derived serving limits: how many concurrent requests to admit and
        # the per-request token budget. On vLLM we admit up to max_concurrent (vLLM
        # batches); off vLLM (eager) we serve one at a time. See startup_memory.py.
        is_vllm = isinstance(model, VLLMModel)
        if is_vllm:
            attn = model._attn_dims  # type: ignore[attr-defined]
            model_info = ModelMemoryInfo(
                n_layers=num_layers,
                n_kv_heads=attn["n_kv_heads"],
                head_dim=attn["head_dim"],
                dtype=config.model_dtype,
            )
        else:
            model_info = ModelMemoryInfo(
                n_layers=num_layers,
                n_kv_heads=model.n_kv_heads,  # type: ignore[attr-defined]
                head_dim=model.head_dim,  # type: ignore[attr-defined]
                dtype=config.model_dtype,
            )
        serving_limits = compute_serving_limits(device=args.device, is_vllm=is_vllm, model_info=model_info)
        config.set_max_tokens(serving_limits.max_tokens)
        # Bound the prompt caps by the memory-safe sequence budget (never raise them).
        config.token_limit = min(config.token_limit, serving_limits.max_tokens)
        config.lens_token_limit = min(config.lens_token_limit, serving_limits.max_tokens)
        configure_limiter(
            concurrent=is_vllm,
            max_concurrent=serving_limits.max_concurrent_requests,
        )
        logger.info(
            "Serving limits: %s (token_limit=%d, lens_token_limit=%d)",
            serving_limits,
            config.token_limit,
            config.lens_token_limit,
        )

        logger.info(
            f"Loaded {config.custom_hf_model_id if config.custom_hf_model_id else config.override_model_id} on {args.device}"
        )
        checkCudaError()

        logger.info("Loading SAEs...")
        # SAE paging: when a residency budget is configured the SAEs are kept in host RAM
        # and only a bounded slice of them sits on the GPU. Resolved here (rather than in
        # Config) because "auto" depends on the backend that was just selected -- under
        # vLLM the engine's reservation is not on the card yet, so the budget has to be
        # derived from the utilization figure instead of measured.
        sae_gpu_budget_bytes = resolve_sae_gpu_budget_bytes(
            config.sae_gpu_budget_gib,
            device=args.device,
            is_vllm=is_vllm,
            vllm_gpu_utilization=_vllm_gpu_memory_utilization(),
        )
        sae_pinned_host_bytes = (
            measure_pinnable_host_bytes(config.sae_pinned_host_gib) if sae_gpu_budget_bytes > 0 else 0
        )
        SAEManager._instance = SAEManager(
            num_layers,
            args.device,
            sae_gpu_budget_bytes=sae_gpu_budget_bytes,
            sae_pinned_host_bytes=sae_pinned_host_bytes,
        )
        SAEManager._instance.load_saes()

        if is_vllm and static_mode in _SAE_RESOLVED_MODES:
            from neuronpedia_inference.engine_adapter import sae_static_addresses

            reads, writes = sae_static_addresses(SAEManager._instance)
            if static_mode == "sae+auto":
                reads, writes = _with_residual_set(model, reads, writes, num_layers)
            if extra_points:
                reads, writes = _with_extra_points(reads, writes, extra_points)
            logger.info(
                "STATIC_POINTS=%s%s: freezing %d read site(s) and %d write site(s)",
                static_mode,
                f" + {[str(a) for a in extra_points]}" if extra_points else "",
                len(reads),
                len(writes),
            )
            model.configure_static(reads, static_writes=writes)

        # Load the fitted Jacobian lens (best-effort; never fatal). LOGIT_LENS
        # requests work regardless; JACOBIAN_LENS requests error if this fails.
        logger.info("Loading Jacobian lens (if available)...")
        load_jacobian_lens_at_startup(config, args)

        # If a Jacobian lens loaded, run a 1-token pass through the real lens
        # code now so any one-time initialization happens at startup.
        logger.info("Warming up lens code path (if Jacobian lens available)...")
        warmup_lens()

        global initialized
        initialized = True
        logger.info("Initialized: %s", initialized)

    attempt = 0
    while True:
        attempt += 1
        try:
            await asyncio.get_event_loop().run_in_executor(None, load_model_and_sae)
            break
        except Exception as exc:
            # HuggingFace sometimes returns 429 (Too Many Requests) while
            # downloading model/SAE files. Keep retrying indefinitely, waiting a
            # bit between attempts so we don't hammer the API.
            if _is_hf_429_error(exc):
                logger.warning(
                    "HuggingFace returned 429 (Too Many Requests) during "
                    "initialization (attempt %d). Retrying in %d seconds...",
                    attempt,
                    HF_429_RETRY_WAIT_SECONDS,
                )
                await asyncio.sleep(HF_429_RETRY_WAIT_SECONDS)
                continue
            initialization_error = str(exc)
            logger.exception("Initialization failed")
            _log_banner(
                "==== LOADING FAILED - NOT SERVING ====",
                [
                    f"error: {exc}",
                    "see the traceback above for details",
                ],
            )
            raise

    # After the model is loaded: preload the vLLM engine (vLLM backend only), then
    # initialize persona data for the assistant-axis feature. Persona data loads
    # for whichever backend is active (vLLM or EagerModel); it's a no-op/warn
    # when the model has no persona data files on disk.
    model = Model.get_instance()
    if isinstance(model, VLLMModel):
        # warmup() builds the engine, compiles decode kernels, and — when static taps
        # exist — runs a sentinel capture/write. A dead copy_/add_ raises here so we
        # refuse to serve rather than return fluent unsteered text.
        logger.info("Warming up vLLM engine...")
        await model.warmup()
        logger.info("vLLM engine ready")

    config = Config.get_instance()

    # Upload the Jacobian lens now: after the vLLM engine has taken its pool (so the budget
    # is measured against what is really left) and before the transient budget below counts
    # the rest as free. On vLLM it goes into the worker, beside the weights and the residuals
    # it will be applied to; anywhere else, onto this process's device. Both land on the same
    # card, so either way the measurement below sees what the lens took.
    if not await place_jacobian_lens_on_worker(config, args, model):
        place_jacobian_lens_on_device(config, args)

    # ---- size the request working-set budget, LAST ----
    # Deliberately the final step of startup. compute_serving_limits() above runs before the
    # SAE cache and the vLLM reservation exist, so it can only estimate; this measures what is
    # actually left on the card once every persistent allocation is in place. Requests are
    # then admitted against real free memory instead of a flat count (see VramBudget), which
    # is also what makes adding an SAE self-correcting: a bigger cache measures as a smaller
    # budget here, with nothing to reconfigure.
    if config.device is None:
        raise RuntimeError("Config.device must be set before measuring the transient budget")
    budget_bytes = measure_transient_budget(config.device)
    # With paging the cache was warmed before this measurement, so the SAEs it already holds
    # are counted as used. What it may still stage in is not, and it must not be handed to
    # requests as well: hold back the unwarmed remainder of the residency budget.
    unclaimed_sae_bytes = max(0, sae_cache.budget_bytes - sae_cache.resident_bytes)
    if unclaimed_sae_bytes:
        logger.info(
            "[startup_memory] holding back %.2f GiB of unwarmed SAE residency budget",
            unclaimed_sae_bytes / 1024**3,
        )
        budget_bytes = max(0, budget_bytes - unclaimed_sae_bytes)
    configure_budget(budget_bytes)

    # Activation prompts are O(d_sae * tokens) after the streaming top-K rewrite; completion
    # / steer keep the pods.yaml token_limit (the vLLM context is sized from that or the lens
    # cap, whichever is larger -- see _engine_context_len). Shrink the activation
    # cap from the measured budget + widest configured SAE so a newly added wider SAE lowers
    # it automatically rather than OOMing the first all-layers search.
    d_sae, d_in, n_hooks = SAEManager.get_instance().widest_activation_dims()
    config.set_activation_token_limit(
        compute_activation_token_limit(
            budget_bytes=budget_bytes,
            token_limit=config.token_limit,
            d_sae=d_sae,
            d_in=d_in,
            n_hooks=n_hooks,
            sae_dtype=config.sae_dtype,
            model_dtype=config.model_dtype,
        )
    )

    _log_ready_banner(time.monotonic() - startup_started_at)


@app.middleware("http")
async def check_secret_key(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    if request.url.path in ("/health",):
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
async def check_model(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Note, without rejecting, a request that names a model this pod did not load.

    A pod holds exactly one model, so the ``model`` field selects nothing and can only ever
    be a client-side assertion. It used to be enforced here, which meant a caller had to
    spell the id the way the alias expansion happened to produce it -- and pods are started
    for reasons that have nothing to do with the SAE directory those aliases come from.
    """
    if request.method == "POST":
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return await call_next(request)
        if isinstance(body, dict) and body.get("model"):
            Config.get_instance().check_requested_model(body["model"])

    return await call_next(request)


@app.middleware("http")
async def log_and_check_cuda_error(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    if not initialized:
        error_details = f" Initialization error: {initialization_error}" if initialization_error else ""
        return JSONResponse(
            status_code=500,
            content={"error": f"Server not initialized.{error_details}"},
        )
    logger.info("=== Request Info ===")
    logger.info(f"URL: {request.url}")

    response = await call_next(request)

    # Post-request CUDA health probe: if this request poisoned the CUDA context
    # (device-side assert / illegal access / wedged post-OOM), terminate so the
    # supervisor restarts us -- even if the endpoint swallowed the error into a 500.
    probe_cuda_or_die(Config.get_instance().device)
    return response


@app.exception_handler(RequestTooLarge)
async def request_too_large_handler(request: Request, exc: RequestTooLarge):  # noqa: ARG001
    """One request that cannot fit the whole budget: a client problem, not a server one.

    Waiting would never help, so fail immediately with both numbers in the message rather than
    holding the connection for the full lock timeout.
    """
    logger.error("[BUDGET] rejected an over-large request: %s", exc)
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(RecoverableOutOfMemory)
async def recoverable_oom_handler(request: Request, exc: RecoverableOutOfMemory):  # noqa: ARG001
    """503, because the allocator OOM'd but the CUDA context survived -- retrying may work.

    The post-request probe in the middleware above still runs, so if the context turns out to
    be poisoned after all we restart anyway.
    """
    return JSONResponse(status_code=503, content={"error": str(exc)})


@app.exception_handler(TimeoutError)
async def timeout_handler(request: Request, exc: TimeoutError):  # noqa: ARG001
    """Waited too long for a slot or for memory: overloaded, so 503 (retryable) not 500."""
    logger.error("[LIMITER] %s", exc)
    return JSONResponse(status_code=503, content={"error": str(exc)})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):  # noqa: ARG001
    # An unhandled irrecoverable CUDA error means the process is wedged: restart.
    if is_fatal_cuda_error(exc):
        terminate_for_restart(f"unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            # Optionally include traceback in development
            "traceback": traceback.format_exc() if app.debug else None,
        },
    )
