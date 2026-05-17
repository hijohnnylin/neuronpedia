"""
FastAPI server for NLA (Natural Language Autoencoder) inference.

Loads verbalizer model via sgl.Engine in-process (no separate SGLang server needed).
Uses async_generate() to avoid event-loop conflicts with uvicorn's uvloop.

Endpoints:
  POST /describe — activation vector(s) -> natural language descriptions + MSE scores
  POST /score   — text + original vector -> MSE/cosine
  POST /compare — two vectors -> description of the difference vector
  POST /explain — text + position -> extract activation then describe it
  POST /extract — text -> per-token activation vectors from source model

Config via env vars:
  SECRET                          — shared auth secret (X-SECRET-KEY header)
  NLA_VERBALIZER_MODEL            — HF hub ID or local path (e.g. kitft/nla-qwen2.5-7b-actor-step4200)
  NLA_RECONSTRUCTOR_MODEL         — HF hub ID or local path (optional, enables /score)
  NLA_SOURCE_MODEL                — HF hub ID for the base model (default: Qwen/Qwen2.5-7B-Instruct)

Memory / Performance Config Overrides - update these depending on how much VRAM is available.
We have to load the verbalizer, reconstructor, and source model.
Verbalizer = full model
Reconstructor = fraction of the model, based on extraction layer index
Source = full model, because we are doing completion tokens
For example, here's the calculation for how we get to 0.38 for the default Qwen 7b:
Verbalizer = 15 GB (bfloat16)
Reconstructor at layer 20 = 20/28 * 15 GB ~= 11 GB (bfloat16)
Source at layer 20 = 15GB (bfloat16)
Total = 15 GB + 11 GB + 15 GB = 41 GB (bfloat16)
This leaves ~7 GB for KV cache, activations, and PyTorch overhead.
Our NLA_MEM_FRACTION is 0.38 = 18GB for the verbalizer + KV cache (sglang).
This leaves 30GB for reconstructor + source.
Subtracting reconstructor and source models, we get 30GB - 11GB - 15GB = 4GB for KV Cache and activations.
TODO: automatically calculate this based on VRAM, extraction layer index, model size, and dtype.

NLA_MEM_FRACTION values tested to work:
-------------------------------------------------------------
Model                   DType     Layer   GPU+VRAM    NLA_MEM_FRACTION
Qwen2.5-7B-Instruct   bfloat16     20      A40 48GB       0.38
Gemma-3-12b           bfloat16     32      A100 80GB      0.31

  NLA_MEM_FRACTION              — GPU memory fraction for KV cache (default: 0.38)
  NLA_MAX_CONCURRENT            — max concurrent SGLang verbalizer generations
                                  (default: 24). This is now a SERVER-WIDE bound:
                                  the verbalizer fan-outs of /explain and
                                  /describe (across all in-flight requests) all
                                  share one global semaphore sized to this
                                  value, so sglang's KV-pool occupancy stays
                                  capped at MAX_CONCURRENT regardless of how
                                  many requests run in parallel.
  NLA_MAX_CONCURRENT_EXPLAINS   — max concurrent /explain requests (default: 1).
                                  Default of 1 preserves the historical "one
                                  /explain at a time" behavior; raise to allow
                                  parallel /explains. The (N+1)th request gets
                                  HTTP 429 immediately rather than queueing —
                                  same fail-fast contract as before, just with
                                  a higher "N". Verbalizer fan-out across all
                                  parallel /explains is still bounded by
                                  NLA_MAX_CONCURRENT (see above), so raising
                                  this doesn't multiply KV-pool pressure.
  NLA_SOURCE_MAX_CONCURRENT     — max concurrent source-model GPU ops, i.e.
                                  /completion + /extract + /explain prefix
                                  (default: 4). With NLA_MAX_CONCURRENT_EXPLAINS
                                  > 1, raise this to ≥ that value or the
                                  source-extract phase becomes the new
                                  serialization bottleneck across parallel
                                  /explains. Lower if you OOM under load.
  NLA_RECONSTRUCTOR_MAX_CONCURRENT — max concurrent /score reconstructor
                                  forward passes (default: 4). The (N+1)th
                                  request gets HTTP 429 immediately. /score
                                  is otherwise blocking torch on the event
                                  loop; this gate plus an executor-thread
                                  offload (run_in_executor) keeps the loop
                                  responsive under load. The same
                                  reconstructor is also called from
                                  /describe via _ReconstructionBatcher,
                                  which is NOT gated by this semaphore (it
                                  has its own coalescing and is already
                                  bounded by NLA_MAX_CONCURRENT) — bump
                                  this only if you see /score-driven
                                  contention.
  NLA_MAX_DESCRIBE_REQUESTS     — max concurrent in-flight /describe
                                  requests (default: 32). The (N+1)th
                                  request gets HTTP 429 immediately.
                                  Complements NLA_MAX_CONCURRENT (which
                                  bounds verbalizer streams) by capping
                                  the *request* count — without this, a
                                  flash crowd of /describe callers would
                                  pile up unbounded waiters on the
                                  verbalizer semaphore, holding HTTP
                                  connections open until they time out.

Per-request shape limits (bound the SIZE of one request, complementing the
concurrency gates above which bound the COUNT of in-flight requests):
  NLA_MAX_INPUT_CHARS           — max characters in any input `text` field
                                  on /explain, /extract, /tokenize,
                                  /completion (default: 65536, ~16k tokens).
  NLA_MAX_POSITIONS_PER_REQUEST — max positions described in one /explain
                                  (default: 512). Also bounds the implicit
                                  "describe all positions" case on long
                                  inputs.
  NLA_MAX_NEW_TOKENS_LIMIT      — max max_new_tokens accepted on /explain,
                                  /describe, /compare (default: 1024). Each
                                  in-flight verbalizer stream's KV slot
                                  scales with this.
  NLA_MAX_DESCRIBE_BATCH        — max activations per /describe call
                                  (default: 512).
  NLA_MAX_DESCRIPTION_CHARS     — max characters in /score's `description`
                                  argument (default: 8192).
  NLA_RECONSTRUCTION_BATCH_SIZE — how many reconstructor score() calls to
                                  coalesce into one forward pass during
                                  streaming /describe and /explain (default:
                                  4). The batcher fires as soon as this many
                                  streams have produced an explanation —
                                  earlier-finishing streams DO wait briefly
                                  for batch peers, but never for ALL streams
                                  to finish. Partial batches flush
                                  immediately once no more submissions are
                                  pending. Bump for more throughput if your
                                  reconstructor has VRAM headroom; set to 1
                                  to disable batching.
  NLA_TRUNCATE_SOURCE           — if "1"/true (default), drop the source model's
                                  layers past the extraction layer plus its
                                  lm_head and final norm. Saves ~25%+ of source
                                  model VRAM (e.g. ~4 GB for Qwen 7B at layer 20).
                                  /completion is disabled in this mode — use an
                                  external API (e.g. OpenRouter) for completions.
                                  Set to "0" to keep the full model and enable
                                  /completion.
  NLA_FP8_VERBALIZER            — if "1"/true, load the verbalizer with sglang's
                                  runtime FP8 quantization. Quantizes
                                  on-the-fly during load: sglang allocates bf16
                                  weight buffers for each Linear, loads bf16
                                  weights from disk, then converts to fp8 in
                                  process_weights_after_loading. Peak GPU usage
                                  during load is ~bf16 size, NOT fp8 size — so
                                  for a 70B-param model on disk in bf16 you
                                  need ~141 GB free during load. Use TP>1 or
                                  a pre-quantized checkpoint if that doesn't
                                  fit. Auto-falls back to bf16 with a warning
                                  when run on an Ampere or Ada GPU (sglang's
                                  fp8e4nv kernel requires Hopper sm_90+).
                                  Default off.

                                  For pre-quantized checkpoints
                                  (`compressed-tensors`, AWQ, GPTQ, …) leave
                                  this OFF — sglang auto-detects the recipe
                                  from `config.json`'s `quantization_config`
                                  and applies the matching kernel. There is
                                  no longer a separate flag to pin the
                                  quantization mode; if you need that escape
                                  hatch, edit the `quantization=` arg passed
                                  to `sgl.Engine` in `init_models()`. Note
                                  that "torchao" checkpoints are not
                                  loadable by sglang at any version we've
                                  tested; rebuild with
                                  `build_compressed_tensors_*.py`.
  NLA_FP8_SOURCE                — if "1"/true, apply FP8 weight-only
                                  quantization to the source model via torchao
                                  (~50% saving on the kept layers). Requires
                                  the `torchao` package. Default off.
  NLA_FP8_RECONSTRUCTOR         — if "1"/true, apply FP8 weight-only
                                  quantization to the reconstructor backbone
                                  via torchao (~50% saving on the kept layers).
                                  The value_head stays in bf16 to preserve
                                  output-projection numerics. Default off.
                                  Mutually exclusive with NLA_INT4_RECONSTRUCTOR.
  NLA_INT4_RECONSTRUCTOR        — if "1"/true, apply INT4 weight-only
                                  quantization to the reconstructor backbone
                                  via torchao (~75% saving vs bf16, ~50% vs
                                  fp8). Loads via TorchAoConfig+device_map so
                                  no bf16 GPU transient peak. The reconstructor
                                  output is a soft scalar score (MSE/cosine),
                                  so 5-15%% accumulated drift on its predicted
                                  activation is acceptable for relative
                                  ranking — but absolute MSE values will shift
                                  vs an fp8/bf16 baseline. Mutually exclusive
                                  with NLA_FP8_RECONSTRUCTOR. Default off.
 NLA_KV_CACHE_DTYPE            — sglang KV-cache dtype, e.g. "fp8_e5m2" or
                                  "fp8_e4m3". Default: unset (= model dtype,
                                  typically bf16). FP8 KV halves KV-pool
                                  bytes-per-token; on memory-tight setups (e.g.
                                  27B verbalizer at batch 32) this frees
                                  ~1.5–3 GB you can spend on more concurrency
                                  or just headroom. Native compute on
                                  Hopper/Blackwell; output-quality cost is
                                  negligible for a generative verbalizer.
  NLA_CUDA_GRAPH_MAX_BS         — max batch size for sglang's CUDA-graph
                                  capture. Default: unset (sglang picks 8 or
                                  32 depending on version). At high
                                  concurrency (NLA_MAX_CONCURRENT > captured
                                  ceiling), decode falls into eager mode and
                                  loses 5–15%; bump this to match
                                  NLA_MAX_CONCURRENT to keep decode on the
                                  fast path. Slightly more VRAM held by
                                  captured graphs; usually <1 GB.
  NLA_TORCH_COMPILE             — if "1"/true, ask sglang to torch.compile
                                  decode kernels. ~10–20% decode speedup at
                                  the cost of a one-time 1–3 minute warmup at
                                  boot (~2 min for 27B, ~45 s for 7B). Set
                                  TORCHINDUCTOR_CACHE_DIR to a persistent
                                  path to amortize across restarts. Free at
                                  runtime; no steady-state VRAM cost.
                                  Default off.

Multi GPU
  NLA_VERBALIZER_DEVICE         — verbalizer placement (default: cuda:0).
                                  Accepts "cuda", "cuda:N". Passed to sglang
                                  as base_gpu_id.
  NLA_RECONSTRUCTOR_DEVICE      — reconstructor placement (default: cuda:1
                                  on multi-GPU boxes, else cuda:0). Accepts
                                  "cuda", "cuda:N", "cpu".
  NLA_SOURCE_DEVICE             — source-model placement (default: cuda:1
                                  on multi-GPU boxes, else cuda:0). Accepts
                                  "cuda", "cuda:N", "cpu".
  NLA_TP_SIZE                   — tensor parallelism (default: 1). Note: the
                                  HF source/reconstructor are NOT
                                  tensor-parallel; only sglang's verbalizer
                                  honors tp_size. Per-model GPU pinning (the
                                  three vars above) is the recommended
                                  multi-GPU strategy on PCIe-only boxes —
                                  no NCCL, free pipeline parallelism, and
                                  failure-isolated.

Metadata overrides — these are normally loaded from nla_meta.yaml in the HF
checkpoint directory and should NOT be set unless you need to override them:
  NLA_OVERRIDE_D_MODEL                   — model hidden size (e.g. 3584 for Qwen7B)
  NLA_OVERRIDE_INJECTION_SCALE           — L2-norm for injection (e.g. 150 for Qwen7B)
  NLA_OVERRIDE_INJECTION_CHAR            — injection marker char (e.g. ㈎)
  NLA_OVERRIDE_INJECTION_TOKEN_ID        — token ID for injection char
  NLA_OVERRIDE_INJECTION_LEFT_NEIGHBOR   — token ID left of injection position
  NLA_OVERRIDE_INJECTION_RIGHT_NEIGHBOR  — token ID right of injection position
  NLA_OVERRIDE_VERBALIZER_PROMPT_TEMPLATE     — verbalizer prompt template
  NLA_OVERRIDE_RECONSTRUCTOR_PROMPT_TEMPLATE  — reconstructor prompt template
  NLA_OVERRIDE_EXTRACTION_LAYER               — which layer to extract activations from
  NLA_OVERRIDE_MSE_SCALE                      — reconstructor mse_scale (default: sqrt(d_model))
"""

import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from queue import Queue

import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers.generation.streamers import BaseStreamer

# Configure the "nla" logger explicitly with its own handler and disable
# propagation. `logging.basicConfig` is a no-op if any module imported above
# (e.g. `transformers`) has already attached a handler to the root logger,
# which silently drops every `logger.info(...)` we make. Wiring our own
# handler avoids that, and `propagate=False` keeps us independent of
# whatever uvicorn / transformers / sglang do to the root logger later.
logger = logging.getLogger("nla")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
logger.propagate = False

from nla_inference import SourceModel  # noqa: E402
from nla_inference import NLAClient, NLAReconstructor, make_nla_config

load_dotenv()

SECRET = os.environ.get("SECRET")

# ─── NLA config from env ────────────────────────────────────────────────────

VERBALIZER_MODEL = os.environ.get(
    "NLA_VERBALIZER_MODEL", "kitft/nla-qwen2.5-7b-actor-step4200"
)
RECONSTRUCTOR_MODEL = os.environ.get(
    "NLA_RECONSTRUCTOR_MODEL", "kitft/nla-qwen2.5-7b-critic-step4200"
)
SOURCE_MODEL = os.environ.get("NLA_SOURCE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

TP_SIZE = int(os.environ.get("NLA_TP_SIZE", "1"))
MEM_FRACTION = float(os.environ.get("NLA_MEM_FRACTION", "0.38"))
MAX_CONCURRENT = int(os.environ.get("NLA_MAX_CONCURRENT", "24"))
# Max parallel /explain requests. Default 1 = the legacy "lock" behavior:
# request N+1 returns HTTP 429 immediately. Set >1 to permit concurrent
# /explains; the global verbalizer semaphore (sized by MAX_CONCURRENT) still
# bounds total in-flight sglang streams, so raising this doesn't blow up KV
# pool occupancy.
MAX_CONCURRENT_EXPLAINS = int(os.environ.get("NLA_MAX_CONCURRENT_EXPLAINS", "1"))
if MAX_CONCURRENT_EXPLAINS < 1:
    raise ValueError(
        f"NLA_MAX_CONCURRENT_EXPLAINS must be >= 1, got {MAX_CONCURRENT_EXPLAINS}"
    )
SOURCE_MAX_CONCURRENT = int(os.environ.get("NLA_SOURCE_MAX_CONCURRENT", "4"))
# Per-request /score gate. Reconstructor forward passes are blocking torch
# work; we offload them via run_in_executor and cap parallelism to keep
# VRAM bounded and avoid starving the rest of the server. (N+1)th request
# fails fast with 429.
RECONSTRUCTOR_MAX_CONCURRENT = int(
    os.environ.get("NLA_RECONSTRUCTOR_MAX_CONCURRENT", "4")
)
if RECONSTRUCTOR_MAX_CONCURRENT < 1:
    raise ValueError(
        f"NLA_RECONSTRUCTOR_MAX_CONCURRENT must be >= 1, got {RECONSTRUCTOR_MAX_CONCURRENT}"
    )
# Per-request /describe gate. Bounds the number of concurrent /describe
# *requests* (orthogonal to NLA_MAX_CONCURRENT, which bounds verbalizer
# streams across the whole server). (N+1)th request fails fast with 429.
MAX_DESCRIBE_REQUESTS = int(os.environ.get("NLA_MAX_DESCRIBE_REQUESTS", "32"))
if MAX_DESCRIBE_REQUESTS < 1:
    raise ValueError(
        f"NLA_MAX_DESCRIBE_REQUESTS must be >= 1, got {MAX_DESCRIBE_REQUESTS}"
    )
# Reconstructor (HF) batch size — coalesces score() calls from streaming
# fan-outs in /describe and /explain into one forward pass. Larger batches
# amortize the per-call overhead (kernel launches, attention setup) but
# spike VRAM by ~B× the per-prompt activation footprint and add latency
# for early-finishing streams that have to wait for batch peers. Default
# 4 is a safe middle ground for a 7B-class reconstructor on a single GPU;
# bump higher if you have headroom and lots of fan-out.
RECONSTRUCTION_BATCH_SIZE = int(os.environ.get("NLA_RECONSTRUCTION_BATCH_SIZE", "8"))


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"{name}={raw!r} is not a valid boolean")


def _vram_used_gb(device_idx: int = 0) -> float:
    """Query nvidia-smi for current GPU used VRAM in GB.

    Uses nvidia-smi rather than torch.cuda.memory_allocated() because
    sgl.Engine runs the verbalizer in a subprocess — its allocations
    don't show up in this process's torch tracking. Returns 0.0 if
    nvidia-smi is not available (e.g. on CPU/MPS).
    """
    import subprocess

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                f"--id={device_idx}",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return int(out.split("\n")[0]) / 1024
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return 0.0


def _torch_cuda_alloc_gb(device_idx: int = 0) -> float:
    """PyTorch's accounting of allocated VRAM in this process (GB).

    Complements `_vram_used_gb` (which is total system-side via nvidia-smi).
    The gap between them = sglang subprocess + CUDA context overhead +
    fragmentation. Useful for distinguishing "torch allocated this much"
    from "the GPU has this much used."
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(device_idx) / (1024**3)


TRUNCATE_SOURCE = _parse_bool_env("NLA_TRUNCATE_SOURCE", default=True)
FP8_VERBALIZER = _parse_bool_env("NLA_FP8_VERBALIZER", default=False)
FP8_SOURCE = _parse_bool_env("NLA_FP8_SOURCE", default=False)
FP8_RECONSTRUCTOR = _parse_bool_env("NLA_FP8_RECONSTRUCTOR", default=False)
INT4_RECONSTRUCTOR = _parse_bool_env("NLA_INT4_RECONSTRUCTOR", default=False)


# `quantization=` passed to sgl.Engine. Only "fp8" needs to be set
# explicitly (it triggers sglang's runtime bf16->fp8 conversion path).
# Pre-quantized checkpoints (compressed-tensors / AWQ / GPTQ / ...) are
# auto-detected by sglang from `config.json`'s `quantization_config` when
# this is None — same model the source/reconstructor use through HF
# transformers.
VERBALIZER_QUANTIZATION: str | None = "fp8" if FP8_VERBALIZER else None

KV_CACHE_DTYPE: str | None = os.environ.get("NLA_KV_CACHE_DTYPE") or None
_cuda_graph_max_bs_raw = os.environ.get("NLA_CUDA_GRAPH_MAX_BS")
CUDA_GRAPH_MAX_BS: int | None = (
    int(_cuda_graph_max_bs_raw) if _cuda_graph_max_bs_raw else None
)
TORCH_COMPILE = _parse_bool_env("NLA_TORCH_COMPILE", default=False)

if FP8_RECONSTRUCTOR and INT4_RECONSTRUCTOR:
    raise ValueError(
        "NLA_FP8_RECONSTRUCTOR and NLA_INT4_RECONSTRUCTOR are mutually "
        "exclusive — pick at most one for the reconstructor backbone."
    )

# ─── Metadata overrides (normally loaded from nla_meta.yaml in checkpoint) ──

OVERRIDE_D_MODEL = os.environ.get("NLA_OVERRIDE_D_MODEL")
OVERRIDE_INJECTION_SCALE = os.environ.get("NLA_OVERRIDE_INJECTION_SCALE")
OVERRIDE_INJECTION_CHAR = os.environ.get("NLA_OVERRIDE_INJECTION_CHAR")
OVERRIDE_INJECTION_TOKEN_ID = os.environ.get("NLA_OVERRIDE_INJECTION_TOKEN_ID")
OVERRIDE_INJECTION_LEFT_NEIGHBOR = os.environ.get(
    "NLA_OVERRIDE_INJECTION_LEFT_NEIGHBOR"
)
OVERRIDE_INJECTION_RIGHT_NEIGHBOR = os.environ.get(
    "NLA_OVERRIDE_INJECTION_RIGHT_NEIGHBOR"
)
OVERRIDE_VERBALIZER_PROMPT_TEMPLATE = os.environ.get(
    "NLA_OVERRIDE_VERBALIZER_PROMPT_TEMPLATE"
)
OVERRIDE_RECONSTRUCTOR_PROMPT_TEMPLATE = os.environ.get(
    "NLA_OVERRIDE_RECONSTRUCTOR_PROMPT_TEMPLATE"
)
OVERRIDE_EXTRACTION_LAYER = os.environ.get("NLA_OVERRIDE_EXTRACTION_LAYER")
OVERRIDE_MSE_SCALE = os.environ.get("NLA_OVERRIDE_MSE_SCALE")


def _num_cuda_devices() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        return torch.cuda.device_count()
    except Exception:
        return 0


def _detect_device() -> str:
    """Default base device when no per-model override is given.

    Returns "cuda:0" on CUDA boxes (single- or multi-GPU), "mps" on Apple
    silicon, "cpu" otherwise. Per-model overrides (NLA_VERBALIZER_DEVICE,
    NLA_RECONSTRUCTOR_DEVICE, NLA_SOURCE_DEVICE) are layered on top in
    `_compute_default_devices`.
    """
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _normalize_device(spec: str) -> str:
    """Canonicalize a device string for downstream consumers.

    "cuda" -> "cuda:0" so .startswith("cuda") checks and base_gpu_id parsing
    behave consistently. "cpu", "mps", and "cuda:N" pass through unchanged.
    """
    spec = spec.strip()
    if spec == "cuda":
        return "cuda:0"
    return spec


def _compute_default_devices() -> tuple[str, str, str]:
    """Pick (verbalizer, reconstructor, source) defaults given hardware.

    Strategy (per-model GPU pinning, see README "Multi-GPU"):
      - 0 CUDA GPUs: everything goes to mps/cpu — env overrides allowed.
      - 1 CUDA GPU : everything on cuda:0.
      - 2+ CUDA GPUs: verbalizer on cuda:0, reconstructor + source on cuda:1.
        Rationale: the sglang verbalizer is the single biggest VRAM consumer
        (weights + KV pool + sglang overhead), so it gets a GPU to itself.
        The HF source + HF reconstructor are smaller (especially when
        truncated/quantized) and pair well on the second GPU.

    Each value can be overridden via NLA_*_DEVICE env vars, which take
    precedence over these defaults.
    """
    base = _detect_device()
    n_cuda = _num_cuda_devices()
    if n_cuda >= 2:
        verb = "cuda:0"
        recon = "cuda:1"
        source = "cuda:1"
    else:
        verb = recon = source = base
    return verb, recon, source


_DEFAULT_VERB_DEVICE, _DEFAULT_RECON_DEVICE, _DEFAULT_SOURCE_DEVICE = (
    _compute_default_devices()
)
VERBALIZER_DEVICE = _normalize_device(
    os.environ.get("NLA_VERBALIZER_DEVICE", _DEFAULT_VERB_DEVICE)
)
RECONSTRUCTOR_DEVICE = _normalize_device(
    os.environ.get("NLA_RECONSTRUCTOR_DEVICE", _DEFAULT_RECON_DEVICE)
)
SOURCE_DEVICE = _normalize_device(
    os.environ.get("NLA_SOURCE_DEVICE", _DEFAULT_SOURCE_DEVICE)
)


# Globals
nla_client: NLAClient | None = None
nla_reconstructor: NLAReconstructor | None = None
source_model: SourceModel | None = None
# Bounds the number of concurrent /explain requests. Sized by
# NLA_MAX_CONCURRENT_EXPLAINS (default 1 = legacy single-request behavior).
# (N+1)th request gets HTTP 429 — same fail-fast contract as the old lock.
# Initialized in lifespan() because asyncio.Semaphore needs a running loop.
_explain_semaphore: asyncio.Semaphore | None = None
# Server-wide gate on in-flight verbalizer streams (sglang KV-pool occupancy).
# Sized by NLA_MAX_CONCURRENT. Shared across all /explain and /describe
# fan-outs so the pool occupancy bound holds regardless of how many requests
# are running in parallel. Initialized in lifespan().
_verbalizer_semaphore: asyncio.Semaphore | None = None
# Gate concurrent GPU work on the source model (model.generate, extract).
# Initialized in lifespan() because asyncio.Semaphore needs a running loop.
_source_semaphore: asyncio.Semaphore | None = None
# Bounds concurrent /score requests. Each /score does one reconstructor
# forward pass via run_in_executor; sized by NLA_RECONSTRUCTOR_MAX_CONCURRENT.
# (N+1)th request gets HTTP 429. Initialized in lifespan().
_reconstructor_semaphore: asyncio.Semaphore | None = None
# Bounds concurrent /describe requests. Sized by NLA_MAX_DESCRIBE_REQUESTS.
# Distinct from _verbalizer_semaphore (which gates per-vector sglang
# streams across the whole server). (N+1)th request gets HTTP 429.
# Initialized in lifespan().
_describe_semaphore: asyncio.Semaphore | None = None


def _cuda_idx(device: str) -> int | None:
    """Return the GPU index for a "cuda[:N]" string, else None."""
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return None
    return None


def _log_per_device_vram(label: str, indices: list[int]) -> None:
    """Print nvidia-smi + torch allocator VRAM for each unique GPU index."""
    if not indices:
        return
    parts = []
    for idx in indices:
        smi = _vram_used_gb(idx)
        alloc = _torch_cuda_alloc_gb(idx)
        parts.append(f"cuda:{idx} smi={smi:.2f} GB / torch={alloc:.2f} GB")
    print(f"[VRAM] {label}: {' | '.join(parts)}")


def load_models():
    global nla_client, nla_reconstructor, source_model

    n_cuda = _num_cuda_devices()
    cuda_idxs_for_models: list[int] = []
    for d in (VERBALIZER_DEVICE, RECONSTRUCTOR_DEVICE, SOURCE_DEVICE):
        i = _cuda_idx(d)
        if i is not None and i not in cuda_idxs_for_models:
            cuda_idxs_for_models.append(i)
    is_cuda = bool(cuda_idxs_for_models)

    if n_cuda == 0:
        device_summary = (
            f"verbalizer={VERBALIZER_DEVICE} "
            f"reconstructor={RECONSTRUCTOR_DEVICE} "
            f"source={SOURCE_DEVICE}"
        )
    else:
        device_summary = (
            f"verbalizer={VERBALIZER_DEVICE} "
            f"reconstructor={RECONSTRUCTOR_DEVICE} "
            f"source={SOURCE_DEVICE}  (visible CUDA GPUs: {n_cuda})"
        )
    print(f"[NLA] device pinning: {device_summary}")

    if is_cuda:
        _log_per_device_vram("start (before any models)", cuda_idxs_for_models)

    # Build explicit config only when metadata overrides are provided;
    # otherwise NLAClient loads from nla_meta.yaml in the checkpoint.
    _overrides = {}
    if OVERRIDE_D_MODEL is not None:
        _overrides["d_model"] = int(OVERRIDE_D_MODEL)
    if OVERRIDE_INJECTION_SCALE is not None:
        _overrides["injection_scale"] = float(OVERRIDE_INJECTION_SCALE)
    if OVERRIDE_INJECTION_CHAR is not None:
        _overrides["injection_char"] = OVERRIDE_INJECTION_CHAR
    if OVERRIDE_INJECTION_TOKEN_ID is not None:
        _overrides["injection_token_id"] = int(OVERRIDE_INJECTION_TOKEN_ID)
    if OVERRIDE_INJECTION_LEFT_NEIGHBOR is not None:
        _overrides["injection_left_neighbor_id"] = int(OVERRIDE_INJECTION_LEFT_NEIGHBOR)
    if OVERRIDE_INJECTION_RIGHT_NEIGHBOR is not None:
        _overrides["injection_right_neighbor_id"] = int(
            OVERRIDE_INJECTION_RIGHT_NEIGHBOR
        )
    if OVERRIDE_VERBALIZER_PROMPT_TEMPLATE is not None:
        _overrides["verbalizer_prompt_template"] = OVERRIDE_VERBALIZER_PROMPT_TEMPLATE

    cfg = make_nla_config(**_overrides) if _overrides else None

    print(
        f"Loading NLA verbalizer: {VERBALIZER_MODEL} "
        f"(device={VERBALIZER_DEVICE}, "
        f"quantization={VERBALIZER_QUANTIZATION or 'none'}, "
        f"kv_cache_dtype={KV_CACHE_DTYPE or 'default'}, "
        f"cuda_graph_max_bs={CUDA_GRAPH_MAX_BS or 'default'}, "
        f"torch_compile={TORCH_COMPILE})"
    )
    # sglang itself only runs on CUDA; on a no-CUDA box we leave `device`
    # unset so NLAClient skips device validation and sglang fails (or runs
    # whatever fallback it has) on its own. Embedding lookup is tiny
    # (~300 MB for Qwen 7B), so CPU is fine when no GPU is available.
    verb_is_cuda = VERBALIZER_DEVICE.startswith("cuda")
    nla_client = NLAClient(
        VERBALIZER_MODEL,
        nla_config=cfg,
        embed_device=VERBALIZER_DEVICE if verb_is_cuda else "cpu",
        device=VERBALIZER_DEVICE if verb_is_cuda else None,
        tp_size=TP_SIZE,
        mem_fraction_static=MEM_FRACTION,
        quantization=VERBALIZER_QUANTIZATION,
        kv_cache_dtype=KV_CACHE_DTYPE,
        cuda_graph_max_bs=CUDA_GRAPH_MAX_BS,
        enable_torch_compile=TORCH_COMPILE,
    )
    if is_cuda:
        _log_per_device_vram("after verbalizer", cuda_idxs_for_models)

    if RECONSTRUCTOR_MODEL:
        print(
            f"Loading NLA reconstructor: {RECONSTRUCTOR_MODEL} "
            f"(device={RECONSTRUCTOR_DEVICE}, fp8={FP8_RECONSTRUCTOR}, "
            f"int4={INT4_RECONSTRUCTOR})"
        )
        try:
            nla_reconstructor = NLAReconstructor(
                RECONSTRUCTOR_MODEL,
                mse_scale=float(OVERRIDE_MSE_SCALE) if OVERRIDE_MSE_SCALE else None,
                reconstructor_prompt_template=OVERRIDE_RECONSTRUCTOR_PROMPT_TEMPLATE,
                device=RECONSTRUCTOR_DEVICE,
                fp8=FP8_RECONSTRUCTOR,
                int4=INT4_RECONSTRUCTOR,
            )
        except Exception as e:
            print(f"Failed to load reconstructor: {e}")
            print("/score endpoint will be unavailable")

        if is_cuda:
            n_layers_msg = ""
            if nla_reconstructor is not None:
                n = nla_reconstructor.backbone.config.num_hidden_layers
                n_layers_msg = (
                    f" — checkpoint provided {n} layers (NLAReconstructor "
                    f"only zeros lm_head + final norm — it does NOT drop "
                    f"layers, so this is the actual count loaded)"
                )
            _log_per_device_vram(
                f"after reconstructor{n_layers_msg}", cuda_idxs_for_models
            )

    if SOURCE_MODEL:
        extraction_layer = (
            int(OVERRIDE_EXTRACTION_LAYER)
            if OVERRIDE_EXTRACTION_LAYER is not None
            else nla_reconstructor.extraction_layer_index
            if nla_reconstructor is not None
            and nla_reconstructor.extraction_layer_index is not None
            else None
        )
        if extraction_layer is None:
            print(
                "WARNING: extraction layer not found in reconstructor sidecar and "
                "NLA_OVERRIDE_EXTRACTION_LAYER not set — skipping source model"
            )
        else:
            print(
                f"Loading source model: {SOURCE_MODEL} "
                f"(device={SOURCE_DEVICE}, layer {extraction_layer}, "
                f"truncate={TRUNCATE_SOURCE}, fp8={FP8_SOURCE})"
            )
            try:
                source_model = SourceModel(
                    SOURCE_MODEL,
                    layer_index=extraction_layer,
                    device=SOURCE_DEVICE,
                    truncate=TRUNCATE_SOURCE,
                    fp8=FP8_SOURCE,
                )
            except Exception as e:
                print(f"Failed to load source model: {e}")
                print("/extract endpoint will be unavailable")

            if is_cuda:
                _log_per_device_vram("after source model", cuda_idxs_for_models)

    if is_cuda:
        _log_per_device_vram(
            "TOTAL idle (post-load, no requests)", cuda_idxs_for_models
        )
        print(
            "[VRAM] note: gap (nvidia-smi − torch in this process) = sglang's "
            "subprocess (verbalizer weights + KV pool live there) + CUDA "
            "context overhead + fragmentation. The verbalizer's GPU will "
            "show the largest gap."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _source_semaphore, _explain_semaphore, _verbalizer_semaphore
    global _reconstructor_semaphore, _describe_semaphore
    _source_semaphore = asyncio.Semaphore(SOURCE_MAX_CONCURRENT)
    _explain_semaphore = asyncio.Semaphore(MAX_CONCURRENT_EXPLAINS)
    _verbalizer_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    _reconstructor_semaphore = asyncio.Semaphore(RECONSTRUCTOR_MAX_CONCURRENT)
    _describe_semaphore = asyncio.Semaphore(MAX_DESCRIBE_REQUESTS)
    load_models()
    print(
        f"[NLA] source-model concurrency gate: max={SOURCE_MAX_CONCURRENT} "
        f"(NLA_SOURCE_MAX_CONCURRENT)"
    )
    print(
        f"[NLA] /explain concurrency gate: max={MAX_CONCURRENT_EXPLAINS} "
        f"(NLA_MAX_CONCURRENT_EXPLAINS)"
    )
    print(
        f"[NLA] verbalizer fan-out gate (server-wide): max={MAX_CONCURRENT} "
        f"(NLA_MAX_CONCURRENT)"
    )
    print(
        f"[NLA] /score concurrency gate: max={RECONSTRUCTOR_MAX_CONCURRENT} "
        f"(NLA_RECONSTRUCTOR_MAX_CONCURRENT)"
    )
    print(
        f"[NLA] /describe concurrency gate: max={MAX_DESCRIBE_REQUESTS} "
        f"(NLA_MAX_DESCRIBE_REQUESTS)"
    )
    print(
        f"[NLA] reconstruction batch size: {RECONSTRUCTION_BATCH_SIZE} "
        f"(NLA_RECONSTRUCTION_BATCH_SIZE)"
    )
    yield
    global nla_client, nla_reconstructor
    if nla_client is not None:
        nla_client.shutdown()
        nla_client = None
    nla_reconstructor = None
    source_model = None


async def verify_secret(x_secret_key: str | None = Header(default=None)):
    """Verify the secret key header if SECRET is configured."""
    if SECRET is not None:
        if x_secret_key is None:
            raise HTTPException(status_code=401, detail="X-SECRET-KEY header required")
        if x_secret_key != SECRET:
            raise HTTPException(status_code=403, detail="Invalid secret key")


app = FastAPI(
    title="NLA Inference Server",
    description="Natural Language Autoencoder — activation vector <-> text",
    lifespan=lifespan,
    dependencies=[Depends(verify_secret)],
)


# ─── Request-shape limits (bound worst-case memory per request) ─────────────
#
# These complement the global concurrency gates (NLA_MAX_CONCURRENT_EXPLAINS,
# NLA_MAX_CONCURRENT, NLA_SOURCE_MAX_CONCURRENT). The concurrency gates cap
# the number of in-flight units of work; these limits cap the SIZE of each
# unit of work, so a single pathological request can't blow up VRAM.
#
# All overridable via env vars. Defaults are deliberately generous — a
# legitimate /explain over a 16k-token document with 256 positions still
# fits — but reject obvious abuse (multi-million-char text bodies, etc.).
#
# Max characters in any single input `text` field. ~16k tokens at the
# typical 4 chars/token rate. Source-model prefill compute scales with L
# (and attention with L²); a 1M-char body spikes both.
MAX_INPUT_CHARS = int(os.environ.get("NLA_MAX_INPUT_CHARS", "65536"))
# Max number of positions that one /explain may describe. Combined with
# NLA_MAX_CONCURRENT_EXPLAINS this bounds total per-request verbalizer
# fan-out at NLA_MAX_CONCURRENT_EXPLAINS × MAX_POSITIONS_PER_REQUEST,
# though the server-wide _verbalizer_semaphore (NLA_MAX_CONCURRENT) is the
# binding constraint at peak.
MAX_POSITIONS_PER_REQUEST = int(
    os.environ.get("NLA_MAX_POSITIONS_PER_REQUEST", "512")
)
# Max max_new_tokens accepted on /explain, /describe, /compare. Each
# in-flight verbalizer stream's KV-cache slot scales linearly with this;
# a runaway value (e.g. 32k) multiplies sglang's pool occupancy and
# causes runtime OOM rather than a clean rejection.
MAX_NEW_TOKENS_LIMIT = int(os.environ.get("NLA_MAX_NEW_TOKENS_LIMIT", "1024"))
# Max activations per /describe call. Same fan-out bookkeeping as
# MAX_POSITIONS_PER_REQUEST above.
MAX_DESCRIBE_BATCH = int(os.environ.get("NLA_MAX_DESCRIBE_BATCH", "512"))
# Max characters in /score's `description`. Reconstructor scoring is a
# single forward pass, so only matters for extreme inputs; cap is loose.
MAX_DESCRIPTION_CHARS = int(os.environ.get("NLA_MAX_DESCRIPTION_CHARS", "8192"))
# Max completion_tokens for /v1/chat/completions; clamped silently rather
# than rejected (OpenAI-compatible endpoint).
MAX_COMPLETION_TOKENS = 512


# ─── Request/Response models ────────────────────────────────────────────────


class DescribeRequest(BaseModel):
    activations: list[list[float]] = Field(
        ...,
        description="List of activation vectors, each of length d_model",
        min_length=1,
        max_length=MAX_DESCRIBE_BATCH,
    )
    temperature: float = Field(default=0.7, ge=0.0, le=5.0)
    max_new_tokens: int = Field(default=200, gt=0, le=MAX_NEW_TOKENS_LIMIT)
    stream: bool = Field(default=False, description="Stream results as SSE events")


class DescriptionResult(BaseModel):
    description: str
    mse: float | None = Field(
        default=None, description="Reconstruction MSE (if reconstructor available)"
    )
    cosine_similarity: float | None = Field(
        default=None, description="Cosine similarity (if reconstructor available)"
    )


class DescribeResponse(BaseModel):
    results: list[DescriptionResult]


class ScoreRequest(BaseModel):
    description: str = Field(
        ...,
        description="Text description to score",
        max_length=MAX_DESCRIPTION_CHARS,
    )
    activation: list[float] = Field(
        ...,
        description="Original activation vector",
    )


class ScoreResponse(BaseModel):
    mse: float = Field(
        description="Reconstruction MSE, range [0, 4]. ~0.2 good, ~1 mediocre, 2 orthogonal"
    )
    cosine_similarity: float


class CompareRequest(BaseModel):
    activation_a: list[float] = Field(..., description="First activation vector")
    activation_b: list[float] = Field(..., description="Second activation vector")
    temperature: float = Field(default=0.7, ge=0.0, le=5.0)
    max_new_tokens: int = Field(default=200, gt=0, le=MAX_NEW_TOKENS_LIMIT)


class CompareResponse(BaseModel):
    description: str = Field(description="Description of the difference vector (a - b)")
    diff_norm: float = Field(description="L2 norm of the difference vector")
    mse: float | None = Field(default=None)
    cosine_similarity: float | None = Field(default=None)


class TokenizeRequest(BaseModel):
    text: str = Field(
        ...,
        description="Input text to tokenize",
        min_length=1,
        max_length=MAX_INPUT_CHARS,
    )


class CompletionRequest(BaseModel):
    text: str = Field(
        ...,
        description="Prompt text to extend",
        min_length=1,
        max_length=MAX_INPUT_CHARS,
    )
    # completion_tokens is silently clamped server-side to MAX_COMPLETION_TOKENS
    # (see _completion handler) for back-compat; we only enforce a positive
    # lower bound here.
    completion_tokens: int = Field(
        default=16,
        ge=1,
        description=(
            "Number of tokens to generate as continuation "
            f"(clamped server-side to 1-{MAX_COMPLETION_TOKENS})."
        ),
    )
    temperature: float = Field(default=0.7, ge=0.0, le=5.0)
    stream: bool = Field(
        default=False,
        description="If true, stream tokens as SSE events instead of returning the full response.",
    )


# UTF-8 lookahead cap: max bytes per char is 4, and byte-level BPE tokens
# typically map to 1-2 bytes for non-ASCII, so 4 tokens of buffering is
# enough to resolve any legitimate multi-byte glyph. Longer runs are
# treated as genuinely unresolvable bytes and surface as `\ufffd`.
_FRAGMENT_LOOKAHEAD = 4
_REPLACEMENT_CHAR = "\ufffd"


def _decode_with_fragments(tokenizer, ids: list[int]) -> list[tuple[str, int, int]]:
    """Decode token ids to display strings, merging byte-fragment runs.

    Byte-level BPE tokenizers (Qwen, Llama, GPT-style) split multi-byte
    UTF-8 glyphs (emojis, CJK, accents) across token boundaries; decoding
    each id individually then yields `\ufffd` for the partial bytes. To
    keep per-position chips informative without losing position structure,
    this helper finds runs of consecutive ids whose individual decode
    contains `\ufffd` but whose joined decode is clean, and emits the
    full glyph string for *every* id in the run, tagged with
    `(fragment_index, fragment_count)` so the client can render a
    visual marker.

    Returns one tuple per input id; positions / token_ids are unchanged.
    """
    out: list[tuple[str, int, int]] = []
    n = len(ids)
    i = 0
    while i < n:
        single = tokenizer.decode([ids[i]])
        if _REPLACEMENT_CHAR not in single:
            out.append((single, 0, 1))
            i += 1
            continue
        # Byte fragment: extend forward up to _FRAGMENT_LOOKAHEAD additional
        # tokens looking for a clean joined decode.
        joined = single
        run_end = i + 1
        while (
            run_end < n
            and _REPLACEMENT_CHAR in joined
            and (run_end - i) < _FRAGMENT_LOOKAHEAD
        ):
            run_end += 1
            joined = tokenizer.decode(ids[i:run_end])
        if _REPLACEMENT_CHAR in joined:
            # Couldn't resolve: emit just the offending id as-is and
            # advance one. Subsequent ids will be reconsidered on their
            # own merits.
            out.append((single, 0, 1))
            i += 1
            continue
        run_count = run_end - i
        for k in range(run_count):
            out.append((joined, k, run_count))
        i = run_end
    return out


class TokenInfo(BaseModel):
    token: str
    token_id: int
    position: int
    fragment_index: int = Field(
        default=0,
        description=(
            "0-based index within a multi-token byte fragment run. 0 for "
            "tokens that decode cleanly on their own."
        ),
    )
    fragment_count: int = Field(
        default=1,
        description=(
            "Total number of tokens in this byte-fragment run (1 for "
            "tokens that decode cleanly on their own). When >1, the "
            "`token` string is the merged glyph and is identical for "
            "every token in the run."
        ),
    )


class TokenizeResponse(BaseModel):
    tokens: list[TokenInfo]
    prompt_length: int = Field(
        description="Number of tokens from the original input text"
    )
    text: str = Field(description="Full text (original + any generated completion)")


class ExtractRequest(BaseModel):
    text: str = Field(
        ...,
        description="Input text to extract activations from",
        min_length=1,
        max_length=MAX_INPUT_CHARS,
    )


class TokenActivation(BaseModel):
    token: str
    token_id: int
    position: int
    activation: list[float]
    l2_norm: float


class ExtractResponse(BaseModel):
    layer_index: int
    tokens: list[TokenActivation]


class ExplainRequest(BaseModel):
    text: str = Field(
        ...,
        description="Input text to extract activations from",
        min_length=1,
        max_length=MAX_INPUT_CHARS,
    )
    positions: list[int] | None = Field(
        default=None,
        description="Token positions to describe (Python-style indexing, e.g. [-1] for last). "
        "If omitted or empty, all positions are described.",
        max_length=MAX_POSITIONS_PER_REQUEST,
    )
    temperature: float = Field(default=0.7, ge=0.0, le=5.0)
    max_new_tokens: int = Field(
        default=200,
        gt=0,
        le=MAX_NEW_TOKENS_LIMIT,
        description="Max tokens for NLA verbalizer explanation",
    )
    reverse: bool = Field(
        default=True,
        description="Process token positions in reverse order (last to first)",
    )
    stream: bool = Field(default=False, description="Stream results as SSE events")


class ExplainResult(BaseModel):
    token: str
    token_id: int
    position: int
    l2_norm: float
    description: str
    mse: float | None = Field(default=None)
    cosine_similarity: float | None = Field(default=None)
    generated: bool = Field(
        default=False,
        description="True if this token was generated (not from original input)",
    )
    fragment_index: int = Field(default=0)
    fragment_count: int = Field(default=1)


class ExplainResponse(BaseModel):
    layer_index: int
    results: list[ExplainResult]


# ─── Reconstruction batcher ─────────────────────────────────────────────────


_BATCH_SKIP = object()  # sentinel: a registered submitter bailed out


class _ReconstructionBatcher:
    """Coalesce per-stream `reconstructor.score()` calls into batched forward
    passes.

    Lifetime is one batcher per request (one per /describe stream, one per
    /explain stream). Each fan-out task either calls
    ``await batcher.score(description, vector)`` to get its (mse, cos) back,
    or ``batcher.skip()`` to drop out (e.g. on stream error) without
    deadlocking the batch loop.

    Triggering policy — the loop runs a batch as soon as EITHER:
      * the queue has reached `batch_size` items, OR
      * every registered submitter has either submitted or skipped (so
        no more items can possibly arrive).

    A submitter that hasn't yet submitted will block its batch peers, but
    only for the time of its own generation — we never wait for ALL
    submitters at once (that's the policy this class explicitly avoids).
    """

    def __init__(
        self,
        reconstructor: NLAReconstructor,
        total_submitters: int,
        batch_size: int,
    ) -> None:
        self._rec = reconstructor
        self._batch_size = max(1, batch_size)
        self._queue: asyncio.Queue = asyncio.Queue()
        # Submitters registered but not yet submitted/skipped. Drives the
        # "no more items will come" signal that flushes partial batches.
        self._unsubmitted = total_submitters
        self._total = total_submitters
        self._task = asyncio.create_task(self._run())

    async def score(
        self, description: str, vector: np.ndarray
    ) -> tuple[float, float]:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        # Order matters: decrement BEFORE enqueueing so the batcher loop
        # never observes the queued item while still believing more
        # submissions are pending. The two ops together are atomic from
        # the loop's perspective (no awaits between them), so the
        # batcher can't see an inconsistent state.
        self._unsubmitted -= 1
        self._queue.put_nowait((description, vector, fut))
        return await fut

    def skip(self) -> None:
        """Mark a registered submitter as bailed (e.g. generation errored).

        Pushes a sentinel so the batcher loop wakes up if it's blocked
        waiting on `queue.get()` for a submission that will never arrive.
        """
        self._unsubmitted -= 1
        self._queue.put_nowait(_BATCH_SKIP)

    async def aclose(self) -> None:
        """Wait for the batcher loop to finish processing all submitters."""
        try:
            await self._task
        except Exception:
            logger.exception("reconstruction batcher loop failed")

    async def _run(self) -> None:
        processed = 0
        while processed < self._total:
            first = await self._queue.get()
            if first is _BATCH_SKIP:
                processed += 1
                continue
            batch = [first]
            # Drain anything already queued (non-blocking) up to batch size.
            while len(batch) < self._batch_size:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is _BATCH_SKIP:
                    processed += 1
                    continue
                batch.append(item)
            # Wait for more submissions only if the batch isn't full AND
            # more submitters haven't reported in yet. This is the core
            # of the "fire as soon as 4 are ready" policy: we never block
            # for the WHOLE fan-out, only for peers that may arrive soon.
            while len(batch) < self._batch_size and self._unsubmitted > 0:
                item = await self._queue.get()
                if item is _BATCH_SKIP:
                    processed += 1
                    continue
                batch.append(item)

            descriptions = [b[0] for b in batch]
            vectors = [b[1] for b in batch]
            t0 = time.perf_counter()
            try:
                results = await asyncio.to_thread(
                    self._rec.score_batch, descriptions, vectors
                )
            except Exception as e:
                logger.exception(
                    "reconstruction score_batch failed (batch_size=%d)",
                    len(batch),
                )
                for _, _, fut in batch:
                    if not fut.done():
                        fut.set_exception(e)
                processed += len(batch)
                continue
            logger.info(
                f"reconstruction batch: scored {len(batch)} item(s) in "
                f"{time.perf_counter() - t0:.2f}s"
            )
            for (_, _, fut), r in zip(batch, results):
                if not fut.done():
                    fut.set_result(r)
            processed += len(batch)


# ─── Endpoints ──────────────────────────────────────────────────────────────


@app.post("/tokenize", response_model=TokenizeResponse)
async def tokenize(req: TokenizeRequest):
    """Tokenize text using the source model's tokenizer."""
    if source_model is None:
        raise HTTPException(status_code=503, detail="Source model not loaded")

    ids = source_model.tokenizer(req.text, add_special_tokens=True)["input_ids"]
    fragments = _decode_with_fragments(source_model.tokenizer, ids)

    return TokenizeResponse(
        tokens=[
            TokenInfo(
                token=text,
                token_id=tid,
                position=i,
                fragment_index=fidx,
                fragment_count=fcount,
            )
            for i, (tid, (text, fidx, fcount)) in enumerate(zip(ids, fragments))
        ],
        prompt_length=len(ids),
        text=req.text,
    )


class _TokenIdStreamer(BaseStreamer):
    """BaseStreamer subclass that pushes generated token ids onto a thread-safe
    queue. Skips the prompt tokens that `model.generate()` echoes on the
    first `put()` call so consumers only receive newly generated tokens.
    """

    def __init__(self) -> None:
        self.queue: Queue = Queue()
        self._next_tokens_are_prompt = True

    def put(self, value):  # type: ignore[override]
        if self._next_tokens_are_prompt:
            self._next_tokens_are_prompt = False
            return
        # `value` may be a 1D or 2D tensor of token ids.
        ids = value.flatten().tolist()
        for tid in ids:
            self.queue.put(int(tid))

    def end(self):  # type: ignore[override]
        self.queue.put(None)


@app.post("/completion")
async def completion(req: CompletionRequest):
    """Generate a continuation for the prompt and return tokens for the
    full (prompt + completion) text. Set `stream=true` to receive tokens
    incrementally as SSE events.
    """
    if source_model is None:
        raise HTTPException(status_code=503, detail="Source model not loaded")
    if source_model.truncated:
        raise HTTPException(
            status_code=503,
            detail=(
                "Source model is loaded in truncated mode (extraction-only): "
                "lm_head and post-extraction layers were dropped to save VRAM. "
                "Restart with NLA_TRUNCATE_SOURCE=0 (or --no-truncate-source) "
                "to enable /completion, or call an external completion API."
            ),
        )

    completion_tokens = max(1, min(MAX_COMPLETION_TOKENS, req.completion_tokens))

    prompt_ids: list[int] = source_model.tokenizer(req.text, add_special_tokens=True)[
        "input_ids"
    ]
    prompt_length = len(prompt_ids)

    if not req.stream:

        def _do_generate() -> list[int]:
            input_ids = torch.tensor([prompt_ids], device=source_model.device)
            with torch.inference_mode():
                gen_ids = source_model.model.generate(
                    input_ids,
                    max_new_tokens=completion_tokens,
                    do_sample=True,
                    temperature=req.temperature,
                )
            ids_out = gen_ids[0].tolist()
            del input_ids, gen_ids
            if source_model.device.startswith("cuda"):
                torch.cuda.empty_cache()
            return ids_out

        async with _source_semaphore:
            all_ids = await asyncio.to_thread(_do_generate)

        full_text = source_model.tokenizer.decode(all_ids, skip_special_tokens=False)
        fragments = _decode_with_fragments(source_model.tokenizer, all_ids)

        return TokenizeResponse(
            tokens=[
                TokenInfo(
                    token=text,
                    token_id=tid,
                    position=i,
                    fragment_index=fidx,
                    fragment_count=fcount,
                )
                for i, (tid, (text, fidx, fcount)) in enumerate(zip(all_ids, fragments))
            ],
            prompt_length=prompt_length,
            text=full_text,
        )

    # ── Streaming path ────────────────────────────────────────────────────
    # Acquire the source-model gate BEFORE spawning the worker thread, and
    # release in the event_stream's finally so we hold it for the full
    # generation lifetime.
    await _source_semaphore.acquire()
    try:
        input_ids = torch.tensor([prompt_ids], device=source_model.device)
        streamer = _TokenIdStreamer()

        def _run_generate() -> None:
            try:
                with torch.inference_mode():
                    source_model.model.generate(
                        input_ids,
                        max_new_tokens=completion_tokens,
                        do_sample=True,
                        temperature=req.temperature,
                        streamer=streamer,
                    )
            except Exception:
                logger.exception("/completion: generation thread failed")
            finally:
                # Make sure consumers wake up even if generate() errors out.
                streamer.queue.put(None)

        thread = threading.Thread(target=_run_generate, daemon=True)
        thread.start()
    except BaseException:
        _source_semaphore.release()
        raise

    async def event_stream():
        try:
            prompt_fragments = _decode_with_fragments(
                source_model.tokenizer, prompt_ids
            )
            prompt_tokens = [
                TokenInfo(
                    token=text,
                    token_id=tid,
                    position=i,
                    fragment_index=fidx,
                    fragment_count=fcount,
                )
                for i, (tid, (text, fidx, fcount)) in enumerate(
                    zip(prompt_ids, prompt_fragments)
                )
            ]
            prompt_event = {
                "type": "prompt",
                "prompt_length": prompt_length,
                "tokens": [t.model_dump() for t in prompt_tokens],
            }
            yield f"data: {json.dumps(prompt_event)}\n\n"

            loop = asyncio.get_running_loop()
            all_ids = list(prompt_ids)
            position = prompt_length
            # Per-stream byte-fragment buffer. Generated ids may form a
            # multi-token UTF-8 glyph; we hold them until either the joined
            # decode is clean or the run is too long to be a real glyph
            # (>_FRAGMENT_LOOKAHEAD), then emit them as a fragment run with
            # `fragment_index`/`fragment_count` set so the client can mark
            # the chips visually. Each model token still gets its own
            # event (and thus its own position / hidden state for explain),
            # so per-position semantics are preserved.
            pending_ids: list[int] = []
            pending_positions: list[int] = []

            def _drain_pending(force_flush: bool) -> list[TokenInfo]:
                tokenizer = source_model.tokenizer
                emit: list[TokenInfo] = []
                while pending_ids:
                    single = tokenizer.decode([pending_ids[0]])
                    if _REPLACEMENT_CHAR not in single:
                        emit.append(
                            TokenInfo(
                                token=single,
                                token_id=pending_ids[0],
                                position=pending_positions[0],
                            )
                        )
                        pending_ids.pop(0)
                        pending_positions.pop(0)
                        continue
                    max_extend = min(_FRAGMENT_LOOKAHEAD, len(pending_ids))
                    resolved_at = 0
                    joined = single
                    for k in range(2, max_extend + 1):
                        joined = tokenizer.decode(pending_ids[:k])
                        if _REPLACEMENT_CHAR not in joined:
                            resolved_at = k
                            break
                    if resolved_at:
                        for f in range(resolved_at):
                            emit.append(
                                TokenInfo(
                                    token=joined,
                                    token_id=pending_ids[f],
                                    position=pending_positions[f],
                                    fragment_index=f,
                                    fragment_count=resolved_at,
                                )
                            )
                        del pending_ids[:resolved_at]
                        del pending_positions[:resolved_at]
                        continue
                    if not force_flush and len(pending_ids) < _FRAGMENT_LOOKAHEAD:
                        # Buffer might still resolve once the next id arrives.
                        return emit
                    if len(pending_ids) >= _FRAGMENT_LOOKAHEAD:
                        # Lookahead exhausted: give up on the head, emit it
                        # as a solo `\ufffd` and reconsider the tail.
                        emit.append(
                            TokenInfo(
                                token=single,
                                token_id=pending_ids[0],
                                position=pending_positions[0],
                            )
                        )
                        pending_ids.pop(0)
                        pending_positions.pop(0)
                        continue
                    # Stream end with an unresolved short tail: emit the
                    # rest as one fragment run; `joined` may still contain
                    # `\ufffd` but at least we don't drop tokens.
                    joined = tokenizer.decode(pending_ids)
                    k = len(pending_ids)
                    for f in range(k):
                        emit.append(
                            TokenInfo(
                                token=joined,
                                token_id=pending_ids[f],
                                position=pending_positions[f],
                                fragment_index=f,
                                fragment_count=k,
                            )
                        )
                    pending_ids.clear()
                    pending_positions.clear()
                return emit

            while True:
                # Block in a worker thread so we don't stall the event loop.
                tid = await loop.run_in_executor(None, streamer.queue.get)
                if tid is None:
                    break
                all_ids.append(tid)
                pending_ids.append(tid)
                pending_positions.append(position)
                position += 1
                for token_info in _drain_pending(force_flush=False):
                    yield f"data: {json.dumps({'type': 'token', 'token': token_info.model_dump()})}\n\n"

            for token_info in _drain_pending(force_flush=True):
                yield f"data: {json.dumps({'type': 'token', 'token': token_info.model_dump()})}\n\n"

            full_text = source_model.tokenizer.decode(
                all_ids, skip_special_tokens=False
            )
            yield f"data: {json.dumps({'type': 'done', 'text': full_text})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            # Don't block forever if the client disconnects mid-stream.
            await asyncio.get_running_loop().run_in_executor(None, thread.join, 30.0)
            if source_model.device.startswith("cuda"):
                torch.cuda.empty_cache()
            _source_semaphore.release()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "verbalizer_model": VERBALIZER_MODEL,
        "d_model": nla_client.cfg.d_model if nla_client else None,
        "verbalizer_quantization": nla_client.quantization
        if nla_client is not None
        else None,
        "verbalizer_kv_cache_dtype": nla_client.kv_cache_dtype
        if nla_client is not None
        else None,
        "verbalizer_cuda_graph_max_bs": nla_client.cuda_graph_max_bs
        if nla_client is not None
        else None,
        "verbalizer_torch_compile": nla_client.enable_torch_compile
        if nla_client is not None
        else None,
        "verbalizer_device": nla_client.device if nla_client is not None else None,
        "reconstructor_available": nla_reconstructor is not None,
        "reconstructor_fp8": nla_reconstructor.fp8
        if nla_reconstructor is not None
        else None,
        "reconstructor_int4": nla_reconstructor.int4
        if nla_reconstructor is not None
        else None,
        "reconstructor_device": nla_reconstructor.device
        if nla_reconstructor is not None
        else None,
        "source_model": SOURCE_MODEL if source_model is not None else None,
        "extraction_layer": source_model.layer_index
        if source_model is not None
        else None,
        "source_truncated": source_model.truncated
        if source_model is not None
        else None,
        "source_fp8": source_model.fp8 if source_model is not None else None,
        "source_device": source_model.device if source_model is not None else None,
        "completion_available": source_model is not None and not source_model.truncated,
        "num_cuda_devices": _num_cuda_devices(),
        "max_concurrent": MAX_CONCURRENT,
        "max_concurrent_explains": MAX_CONCURRENT_EXPLAINS,
        "source_max_concurrent": SOURCE_MAX_CONCURRENT,
        "reconstructor_max_concurrent": RECONSTRUCTOR_MAX_CONCURRENT,
        "max_describe_requests": MAX_DESCRIBE_REQUESTS,
        "limits": {
            "max_input_chars": MAX_INPUT_CHARS,
            "max_positions_per_request": MAX_POSITIONS_PER_REQUEST,
            "max_new_tokens": MAX_NEW_TOKENS_LIMIT,
            "max_describe_batch": MAX_DESCRIBE_BATCH,
            "max_description_chars": MAX_DESCRIPTION_CHARS,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
        },
    }


@app.post("/describe", response_model=DescribeResponse)
async def describe(req: DescribeRequest):
    """Describe activation vector(s) in natural language, with optional MSE scores.

    Set stream=true to receive partial text updates as SSE events.
    Streaming events:
      - Partial: {"index": i, "text": "partial...", "done": false}
      - Final:   {"index": i, "description": "...", "mse": ..., "cosine_similarity": ...}
      - End:     [DONE]
    """
    if nla_client is None:
        raise HTTPException(status_code=503, detail="NLA client not loaded")

    # Request-level fail-fast — orthogonal to the per-vector verbalizer gate
    # below. Without this, a flash crowd of /describe callers would each
    # consume HTTP connections + per-task buffers while waiting on
    # _verbalizer_semaphore, with no upper bound on the wait queue.
    assert _describe_semaphore is not None  # set in lifespan()
    if _describe_semaphore.locked():
        raise HTTPException(
            status_code=429,
            detail=(
                f"All {MAX_DESCRIBE_REQUESTS} /describe slot(s) in use. "
                f"Try again shortly. Raise NLA_MAX_DESCRIBE_REQUESTS to "
                f"permit more parallel callers."
            ),
        )

    n = len(req.activations)
    logger.info(
        f"/describe: {n} vector(s), shape=({len(req.activations[0])},), "
        f"temp={req.temperature}, max_tokens={req.max_new_tokens}, stream={req.stream}"
    )

    vectors = []
    for i, vec_data in enumerate(req.activations):
        v = np.array(vec_data, dtype=np.float32)
        if v.shape[0] != nla_client.cfg.d_model:
            raise HTTPException(
                status_code=400,
                detail=f"Vector length {v.shape[0]} != d_model {nla_client.cfg.d_model}",
            )
        vectors.append(v)

    # Acquire the request-level slot AFTER cheap validation so 400s don't
    # consume slots, but BEFORE we either spin up streaming tasks or run
    # the non-streaming loop. For streaming, ownership of the slot
    # transfers into event_stream's finally; for non-streaming, the
    # outer try/finally below releases it.
    await _describe_semaphore.acquire()
    sema_owned_by_stream = False

    try:
        if req.stream:

            async def event_stream():
                try:
                    queue: asyncio.Queue = asyncio.Queue()
                    # Server-wide verbalizer gate — shared with /explain and any other
                    # concurrent /describe so total in-flight streams ≤ MAX_CONCURRENT.
                    assert _verbalizer_semaphore is not None  # set in lifespan()
                    verb_sema = _verbalizer_semaphore
                    batcher = (
                        _ReconstructionBatcher(
                            nla_reconstructor, n, RECONSTRUCTION_BATCH_SIZE
                        )
                        if nla_reconstructor is not None
                        else None
                    )

                    async def _stream_one(v: np.ndarray, idx: int):
                        # Whether we still owe the batcher a skip() if we exit
                        # before successfully handing ownership to it.
                        needs_skip = batcher is not None
                        try:
                            async with verb_sema:
                                t0 = time.perf_counter()
                                last_out: dict = {"text": ""}
                                async for out in nla_client.async_generate_stream(
                                    v,
                                    temperature=req.temperature,
                                    max_new_tokens=req.max_new_tokens,
                                ):
                                    last_out = out
                                    await queue.put(
                                        json.dumps(
                                            {
                                                "index": idx,
                                                "text": out["text"],
                                                "done": False,
                                            }
                                        )
                                    )
                                logger.info(
                                    f"/describe: [{idx + 1}/{n}] generated in {time.perf_counter() - t0:.2f}s"
                                )
                                description = nla_client._extract_text(
                                    last_out,
                                    True,
                                    context=f"/describe index={idx}",
                                )
                                mse = None
                                cos = None
                                if batcher is not None:
                                    # Ownership of skip-bookkeeping transfers to
                                    # the batcher the moment we call score(): it
                                    # always either resolves the future or sets
                                    # the exception on it, and decrements the
                                    # unsubmitted counter as part of submission.
                                    needs_skip = False
                                    t0 = time.perf_counter()
                                    mse, cos = await batcher.score(description, v)
                                    logger.info(
                                        f"/describe: [{idx + 1}/{n}] scored mse={mse:.3f} cos={cos:.3f} "
                                        f"in {time.perf_counter() - t0:.2f}s (batched)"
                                    )
                                await queue.put(
                                    DescriptionResult(
                                        description=description,
                                        mse=mse,
                                        cosine_similarity=cos,
                                    ).model_dump_json()
                                )
                        finally:
                            if needs_skip and batcher is not None:
                                batcher.skip()

                    tasks = [
                        asyncio.create_task(_stream_one(v, i))
                        for i, v in enumerate(vectors)
                    ]
                    try:
                        done_count = 0
                        while done_count < n:
                            item = await queue.get()
                            yield f"data: {item}\n\n"
                            parsed = json.loads(item)
                            if "description" in parsed:
                                done_count += 1
                        await asyncio.gather(*tasks)
                        yield "data: [DONE]\n\n"
                    finally:
                        if batcher is not None:
                            await batcher.aclose()
                finally:
                    # Always release the request-level slot, regardless of
                    # whether the stream completed cleanly, errored, or the
                    # client disconnected mid-stream (FastAPI calls
                    # generator.aclose(), which fires this finally).
                    _describe_semaphore.release()

            sema_owned_by_stream = True
            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # Non-streaming path
        results = []
        for i, v in enumerate(vectors):
            t0 = time.perf_counter()
            description = await nla_client.async_generate(
                v,
                temperature=req.temperature,
                max_new_tokens=req.max_new_tokens,
                context=f"/describe index={i}",
            )
            logger.info(
                f"/describe: [{i + 1}/{n}] generated in {time.perf_counter() - t0:.2f}s"
            )

            mse = None
            cos = None
            if nla_reconstructor is not None:
                t0 = time.perf_counter()
                mse, cos = nla_reconstructor.score(description, v)
                logger.info(
                    f"/describe: [{i + 1}/{n}] scored mse={mse:.3f} cos={cos:.3f} "
                    f"in {time.perf_counter() - t0:.2f}s"
                )

            results.append(
                DescriptionResult(
                    description=description,
                    mse=mse,
                    cosine_similarity=cos,
                )
            )

        return DescribeResponse(results=results)
    finally:
        # Streaming path transferred ownership into event_stream's finally.
        # All other exits (validation error, non-streaming success, raised
        # exception in the non-streaming loop) release here.
        if not sema_owned_by_stream:
            _describe_semaphore.release()


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    """Score an existing text description against the original activation vector.

    The reconstructor forward pass is blocking torch work, so we run it on
    a thread-pool executor (via run_in_executor) to keep the FastAPI event
    loop responsive — otherwise concurrent /describe streams, /health,
    etc. would all stall while a /score is in flight. Concurrency is
    capped by NLA_RECONSTRUCTOR_MAX_CONCURRENT to bound VRAM and keep the
    executor pool from drowning.
    """
    if nla_reconstructor is None:
        raise HTTPException(status_code=503, detail="NLA reconstructor not loaded")

    assert _reconstructor_semaphore is not None  # set in lifespan()
    if _reconstructor_semaphore.locked():
        raise HTTPException(
            status_code=429,
            detail=(
                f"All {RECONSTRUCTOR_MAX_CONCURRENT} /score slot(s) in use. "
                f"Try again shortly. Raise NLA_RECONSTRUCTOR_MAX_CONCURRENT "
                f"to permit more parallel callers."
            ),
        )

    logger.info(
        f"/score: description={req.description[:80]!r}... "
        f"activation_shape=({len(req.activation)},)"
    )

    async with _reconstructor_semaphore:
        t0 = time.perf_counter()
        v = np.array(req.activation, dtype=np.float32)
        loop = asyncio.get_running_loop()
        # `NLAReconstructor.reconstruct()` (called inside .score()) is NOT
        # decorated with @torch.no_grad / @torch.inference_mode, so under a
        # default-grad context it would build an autograd graph per call.
        # We wrap the executor body in inference_mode to (a) keep VRAM
        # bounded — no graph retained — and (b) sidestep any autograd-engine
        # thread-safety quirks now that we're invoking from a thread pool
        # rather than the event loop. Concurrency is bounded by
        # _reconstructor_semaphore above; PyTorch's default CUDA stream
        # serializes the actual GPU work, so executor parallelism mostly
        # buys us a responsive event loop, not raw throughput.
        # Bind locally so the closure captures a guaranteed-non-None
        # reference (the module-level `nla_reconstructor` could in
        # principle be cleared during shutdown).
        reconstructor = nla_reconstructor
        description = req.description

        def _blocking_score():
            with torch.inference_mode():
                return reconstructor.score(description, v)

        mse, cos = await loop.run_in_executor(None, _blocking_score)
        logger.info(
            f"/score: mse={mse:.3f} cos={cos:.3f} in {time.perf_counter() - t0:.2f}s"
        )
        return ScoreResponse(mse=mse, cosine_similarity=cos)


@app.post("/compare", response_model=CompareResponse)
async def compare(req: CompareRequest):
    """Describe the difference between two activation vectors (a - b)."""
    if nla_client is None:
        raise HTTPException(status_code=503, detail="NLA client not loaded")

    logger.info(
        f"/compare: activation_a=({len(req.activation_a)},) "
        f"activation_b=({len(req.activation_b)},) "
        f"temp={req.temperature}, max_tokens={req.max_new_tokens}"
    )

    a = np.array(req.activation_a, dtype=np.float32)
    b = np.array(req.activation_b, dtype=np.float32)

    if a.shape[0] != nla_client.cfg.d_model:
        raise HTTPException(
            status_code=400,
            detail=f"activation_a length {a.shape[0]} != d_model {nla_client.cfg.d_model}",
        )
    if b.shape[0] != nla_client.cfg.d_model:
        raise HTTPException(
            status_code=400,
            detail=f"activation_b length {b.shape[0]} != d_model {nla_client.cfg.d_model}",
        )

    diff = a - b
    diff_norm = float(np.linalg.norm(diff))
    logger.info(f"/compare: diff_norm={diff_norm:.2f}")

    t0 = time.perf_counter()
    description = await nla_client.async_generate(
        diff,
        temperature=req.temperature,
        max_new_tokens=req.max_new_tokens,
        context="/compare diff=a-b",
    )
    logger.info(f"/compare: generated in {time.perf_counter() - t0:.2f}s")

    mse = None
    cos = None
    if nla_reconstructor is not None:
        t0 = time.perf_counter()
        mse, cos = nla_reconstructor.score(description, diff)
        logger.info(
            f"/compare: scored mse={mse:.3f} cos={cos:.3f} "
            f"in {time.perf_counter() - t0:.2f}s"
        )

    return CompareResponse(
        description=description,
        diff_norm=diff_norm,
        mse=mse,
        cosine_similarity=cos,
    )


def _resolve_positions(
    positions: list[int] | None, n_tokens: int, *, reverse: bool = False
) -> list[int]:
    """Resolve position indices, supporting Python-style negative indexing."""
    if not positions:
        indices = list(range(n_tokens))
    else:
        indices = []
        for p in positions:
            idx = p if p >= 0 else n_tokens + p
            if idx < 0 or idx >= n_tokens:
                raise HTTPException(
                    status_code=400,
                    detail=f"Position {p} out of range (0..{n_tokens - 1})",
                )
            indices.append(idx)
    if reverse:
        indices.reverse()
    return indices


async def _generate_explain_result(
    tok: dict,
    idx: int,
    i: int,
    n: int,
    req: ExplainRequest,
    *,
    is_generated: bool = False,
    batcher: _ReconstructionBatcher | None = None,
) -> ExplainResult:
    """Generate a single ExplainResult for a token position.

    If `batcher` is provided, defer reconstructor scoring to it so multiple
    sibling tasks can share a single batched forward pass through the
    reconstructor. Caller is responsible for skip-bookkeeping if this
    coroutine raises before reaching the batcher.score() call.
    """
    v = np.array(tok["activation"], dtype=np.float32)

    needs_skip = batcher is not None
    try:
        t0 = time.perf_counter()
        description = await nla_client.async_generate(
            v,
            temperature=req.temperature,
            max_new_tokens=req.max_new_tokens,
            context=f"/explain token={tok['token']!r} pos={idx}",
        )
        logger.info(
            f"/explain: [{i + 1}/{n}] token={tok['token']!r} pos={idx} "
            f"generated in {time.perf_counter() - t0:.2f}s"
        )

        mse = None
        cos = None
        if batcher is not None:
            needs_skip = False
            t0 = time.perf_counter()
            mse, cos = await batcher.score(description, v)
            logger.info(
                f"/explain: [{i + 1}/{n}] scored mse={mse:.3f} cos={cos:.3f} "
                f"in {time.perf_counter() - t0:.2f}s (batched)"
            )
        elif nla_reconstructor is not None:
            t0 = time.perf_counter()
            mse, cos = nla_reconstructor.score(description, v)
            logger.info(
                f"/explain: [{i + 1}/{n}] scored mse={mse:.3f} cos={cos:.3f} "
                f"in {time.perf_counter() - t0:.2f}s"
            )

        return ExplainResult(
            token=tok["token"],
            token_id=tok["token_id"],
            position=tok["position"],
            l2_norm=tok["l2_norm"],
            description=description,
            mse=mse,
            cosine_similarity=cos,
            generated=is_generated,
            fragment_index=tok.get("fragment_index", 0),
            fragment_count=tok.get("fragment_count", 1),
        )
    finally:
        if needs_skip and batcher is not None:
            batcher.skip()


@app.post("/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest):
    """Extract activations and describe them in natural language.

    If positions is omitted, all token positions are described.
    Positions use Python-style indexing (negative values count from end).
    Set stream=true to receive results as SSE events.

    Up to NLA_MAX_CONCURRENT_EXPLAINS requests run in parallel; the
    (N+1)th request gets HTTP 429 (fail-fast, no queueing). Verbalizer
    fan-out across all parallel /explains is bounded by NLA_MAX_CONCURRENT
    via a server-wide semaphore, so KV-pool occupancy stays capped.
    """
    if source_model is None:
        raise HTTPException(status_code=503, detail="Source model not loaded")
    if nla_client is None:
        raise HTTPException(status_code=503, detail="NLA client not loaded")
    assert _explain_semaphore is not None  # set in lifespan()

    # asyncio.Semaphore.locked() returns True iff the internal counter is 0,
    # i.e. no slots free. Same fail-fast contract as the legacy lock — we
    # never queue requests, just reject when the gate is full.
    if _explain_semaphore.locked():
        raise HTTPException(
            status_code=429,
            detail=(
                f"All {MAX_CONCURRENT_EXPLAINS} /explain slot(s) in use. "
                "Please retry."
            ),
        )

    if req.stream:
        # Streaming path: acquire before returning the StreamingResponse and
        # release inside the generator's finally so the slot stays held for
        # the lifetime of the SSE stream. If _explain_inner raises before
        # constructing the StreamingResponse (e.g. extract returns no tokens),
        # the generator's finally never runs, so we release here. Once the
        # StreamingResponse is returned, ownership passes to the generator.
        await _explain_semaphore.acquire()
        try:
            return await _explain_inner(req)
        except BaseException:
            _explain_semaphore.release()
            raise
    else:
        async with _explain_semaphore:
            return await _explain_inner(req)


async def _explain_inner(req: ExplainRequest):
    logger.info(
        f"/explain start: text_len={len(req.text)} text={req.text[:80]!r}... "
        f"positions={req.positions} reverse={req.reverse} "
        f"stream={req.stream} temp={req.temperature} "
        f"max_new_tokens={req.max_new_tokens}"
    )
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    async with _source_semaphore:
        tokens = await asyncio.to_thread(source_model.extract, req.text)
    if not tokens:
        raise HTTPException(status_code=400, detail="No tokens extracted")
    n_tokens = len(tokens)
    prompt_length = n_tokens  # no generation in explain; all tokens are from input
    # Overwrite the per-token display strings with fragment-aware decoding so
    # ExplainResult.token doesn't surface `\ufffd` for byte-split glyphs.
    # `position`/`token_id` (and the activation vector) are untouched —
    # explanations remain per-position, just with human-readable labels.
    fragments = _decode_with_fragments(
        source_model.tokenizer, [tok["token_id"] for tok in tokens]
    )
    for tok, (text, fidx, fcount) in zip(tokens, fragments):
        tok["token"] = text
        tok["fragment_index"] = fidx
        tok["fragment_count"] = fcount
    logger.info(
        f"/explain: extracted {n_tokens} tokens in {time.perf_counter() - t0:.2f}s"
    )

    indices = _resolve_positions(req.positions, n_tokens, reverse=req.reverse)
    n = len(indices)
    # Pydantic max_length on req.positions covers explicit lists; this catches
    # the implicit "describe all positions" case for long inputs (e.g. a
    # 10k-token text with positions=None would otherwise fan out 10k streams).
    # The /explain wrapper releases the explain-gate slot on any exception
    # raised from here for both streaming and non-streaming paths.
    if n > MAX_POSITIONS_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=(
                f"resolved {n} positions from text of {n_tokens} tokens, "
                f"but max is {MAX_POSITIONS_PER_REQUEST} per request. "
                "Pass an explicit `positions` list to scope it down."
            ),
        )
    logger.info(f"/explain: describing {n} position(s)")

    if req.stream:

        async def event_stream():
            batcher: _ReconstructionBatcher | None = None
            try:
                # First event: metadata
                yield f"data: {json.dumps({'layer_index': source_model.layer_index, 'total': n, 'prompt_length': prompt_length})}\n\n"

                # Multiplex streaming from all positions concurrently via a queue
                queue: asyncio.Queue = asyncio.Queue()
                # Server-wide verbalizer gate — shared across all parallel
                # /explain and /describe fan-outs so sglang's KV-pool
                # occupancy stays capped at NLA_MAX_CONCURRENT in total.
                assert _verbalizer_semaphore is not None  # set in lifespan()
                verb_sema = _verbalizer_semaphore
                batcher = (
                    _ReconstructionBatcher(
                        nla_reconstructor, n, RECONSTRUCTION_BATCH_SIZE
                    )
                    if nla_reconstructor is not None
                    else None
                )

                async def _stream_one(tok: dict, idx: int, pos_i: int):
                    """Stream text for one position, then score and send final result."""
                    needs_skip = batcher is not None
                    try:
                        async with verb_sema:
                            v = np.array(tok["activation"], dtype=np.float32)
                            t0 = time.perf_counter()
                            last_out: dict = {"text": ""}
                            async for out in nla_client.async_generate_stream(
                                v,
                                temperature=req.temperature,
                                max_new_tokens=req.max_new_tokens,
                            ):
                                last_out = out
                                await queue.put(
                                    json.dumps(
                                        {
                                            "position": tok["position"],
                                            "text": out["text"],
                                            "done": False,
                                        }
                                    )
                                )
                            logger.info(
                                f"/explain: [{pos_i + 1}/{n}] token={tok['token']!r} pos={idx} "
                                f"generated in {time.perf_counter() - t0:.2f}s"
                            )
                            description = nla_client._extract_text(
                                last_out,
                                True,
                                context=f"/explain token={tok['token']!r} pos={idx}",
                            )
                            mse = None
                            cos = None
                            if batcher is not None:
                                needs_skip = False
                                t0 = time.perf_counter()
                                mse, cos = await batcher.score(description, v)
                                logger.info(
                                    f"/explain: [{pos_i + 1}/{n}] scored mse={mse:.3f} cos={cos:.3f} "
                                    f"in {time.perf_counter() - t0:.2f}s (batched)"
                                )
                            await queue.put(
                                ExplainResult(
                                    token=tok["token"],
                                    token_id=tok["token_id"],
                                    position=tok["position"],
                                    l2_norm=tok["l2_norm"],
                                    description=description,
                                    mse=mse,
                                    cosine_similarity=cos,
                                    generated=idx >= prompt_length,
                                    fragment_index=tok.get("fragment_index", 0),
                                    fragment_count=tok.get("fragment_count", 1),
                                ).model_dump_json()
                            )
                    finally:
                        if needs_skip and batcher is not None:
                            batcher.skip()

                tasks = [
                    asyncio.create_task(_stream_one(tokens[idx], idx, i))
                    for i, idx in enumerate(indices)
                ]
                done_count = 0
                while done_count < n:
                    item = await queue.get()
                    yield f"data: {item}\n\n"
                    # Check if this was a final result (has "description" key)
                    parsed = json.loads(item)
                    if "description" in parsed:
                        done_count += 1
                # Ensure all tasks are finished
                await asyncio.gather(*tasks)
                logger.info(
                    f"/explain done (stream): {n} position(s) in "
                    f"{time.perf_counter() - t_total:.2f}s"
                )
                yield "data: [DONE]\n\n"
            finally:
                if batcher is not None:
                    await batcher.aclose()
                # Mirrors the await _explain_semaphore.acquire() that ran
                # before this StreamingResponse was returned. Release here
                # (rather than in the outer endpoint) because the slot must
                # stay held for the entire SSE lifetime.
                assert _explain_semaphore is not None
                _explain_semaphore.release()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming: run all positions in parallel under the server-wide
    # verbalizer gate (shared with concurrent /explain and /describe so total
    # in-flight sglang streams ≤ NLA_MAX_CONCURRENT).
    assert _verbalizer_semaphore is not None  # set in lifespan()
    verb_sema = _verbalizer_semaphore
    batcher = (
        _ReconstructionBatcher(nla_reconstructor, n, RECONSTRUCTION_BATCH_SIZE)
        if nla_reconstructor is not None
        else None
    )

    async def _limited(coro):
        async with verb_sema:
            return await coro

    tasks = [
        _limited(
            _generate_explain_result(
                tokens[idx],
                idx,
                i,
                n,
                req,
                is_generated=idx >= prompt_length,
                batcher=batcher,
            )
        )
        for i, idx in enumerate(indices)
    ]
    try:
        results = await asyncio.gather(*tasks)
    finally:
        if batcher is not None:
            await batcher.aclose()

    logger.info(
        f"/explain done: {n} position(s) in "
        f"{time.perf_counter() - t_total:.2f}s"
    )

    return ExplainResponse(
        layer_index=source_model.layer_index,
        results=list(results),
    )


@app.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest):
    """Extract per-token activation vectors from the source model."""
    if source_model is None:
        raise HTTPException(status_code=503, detail="Source model not loaded")

    logger.info(f"/extract: text={req.text[:80]!r}...")

    t0 = time.perf_counter()
    async with _source_semaphore:
        results = await asyncio.to_thread(source_model.extract, req.text)
    logger.info(f"/extract: {len(results)} tokens in {time.perf_counter() - t0:.2f}s")

    return ExtractResponse(
        layer_index=source_model.layer_index,
        tokens=[TokenActivation(**r) for r in results],
    )


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="NLA Inference Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5009)
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Max concurrent SGLang generations (default: 24, or NLA_MAX_CONCURRENT env)",
    )
    parser.add_argument(
        "--mem-fraction",
        type=float,
        default=None,
        help="GPU memory fraction for KV cache (default: 0.38, or NLA_MEM_FRACTION env)",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=None,
        help="Tensor parallelism size (default: 1, or NLA_TP_SIZE env)",
    )
    parser.add_argument(
        "--truncate-source",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Truncate source model to layers 0..extraction_layer (saves VRAM, "
            "disables /completion). Use --no-truncate-source to load the full "
            "model. Default: truncate (or NLA_TRUNCATE_SOURCE env)."
        ),
    )
    parser.add_argument(
        "--fp8-verbalizer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use sglang's runtime FP8 quantization for the verbalizer "
            "(~50%% VRAM saving plus FP8 tensor-core speedup on Hopper+). "
            "Loads bf16 weights from disk first then converts on-GPU, so "
            "peak GPU usage during load is ~bf16 size — for very large "
            "models on disk in bf16, prefer a pre-quantized "
            "compressed-tensors checkpoint built with "
            "`build_compressed_tensors_verbalizer.py`. Such pre-quantized "
            "checkpoints are auto-detected by sglang from "
            "`config.json`'s `quantization_config`; no flag needed here. "
            "Auto-falls back to bf16 with a warning if the GPU is "
            "Ampere/Ada (sglang's fp8e4nv kernel needs Hopper sm_90+). "
            "Default: off (or NLA_FP8_VERBALIZER env)."
        ),
    )
    parser.add_argument(
        "--fp8-source",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use FP8 weight-only quantization for the source model via "
            "torchao (~50%% VRAM saving on kept layers). Requires the "
            "`torchao` package. Default: off (or NLA_FP8_SOURCE env)."
        ),
    )
    parser.add_argument(
        "--fp8-reconstructor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use FP8 weight-only quantization for the reconstructor backbone "
            "via torchao (~50%% VRAM saving on kept layers). value_head stays "
            "in bf16. Mutually exclusive with --int4-reconstructor. "
            "Requires the `torchao` package. Default: off "
            "(or NLA_FP8_RECONSTRUCTOR env)."
        ),
    )
    parser.add_argument(
        "--verbalizer-device",
        default=None,
        help=(
            "GPU placement for the sglang verbalizer (e.g. 'cuda', 'cuda:0', "
            "'cuda:1'). Passed to sglang as base_gpu_id. Default: cuda:0 (or "
            "NLA_VERBALIZER_DEVICE env)."
        ),
    )
    parser.add_argument(
        "--reconstructor-device",
        default=None,
        help=(
            "Device for the HF reconstructor (e.g. 'cuda', 'cuda:0', "
            "'cuda:1', 'cpu'). Default: cuda:1 on multi-GPU boxes, else "
            "cuda:0 (or NLA_RECONSTRUCTOR_DEVICE env)."
        ),
    )
    parser.add_argument(
        "--source-device",
        default=None,
        help=(
            "Device for the HF source model (e.g. 'cuda', 'cuda:0', "
            "'cuda:1', 'cpu'). Default: cuda:1 on multi-GPU boxes, else "
            "cuda:0 (or NLA_SOURCE_DEVICE env)."
        ),
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default=None,
        help=(
            "sglang KV-cache dtype, e.g. 'fp8_e5m2' or 'fp8_e4m3'. Halves "
            "KV-pool bytes-per-token vs bf16 — useful when memory-tight (e.g. "
            "27B verbalizer at high concurrency). Default: unset = model dtype "
            "(or NLA_KV_CACHE_DTYPE env)."
        ),
    )
    parser.add_argument(
        "--cuda-graph-max-bs",
        type=int,
        default=None,
        help=(
            "Max batch size for sglang CUDA-graph capture. Bump to match "
            "--max-concurrent so peak fan-out stays on the captured fast path "
            "(decode 5–15%% faster). Default: unset = sglang default (or "
            "NLA_CUDA_GRAPH_MAX_BS env)."
        ),
    )
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Ask sglang to torch.compile decode kernels. ~10–20%% decode "
            "speedup at the cost of a one-time 1–3 minute warmup at boot "
            "(~2 min for 27B, ~45 s for 7B). Set TORCHINDUCTOR_CACHE_DIR to "
            "a persistent path to amortize across restarts. "
            "Default: off (or NLA_TORCH_COMPILE env)."
        ),
    )
    parser.add_argument(
        "--int4-reconstructor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use INT4 weight-only quantization for the reconstructor backbone "
            "via torchao (~75%% VRAM saving vs bf16, ~50%% vs fp8). Loads "
            "via TorchAoConfig+device_map (streaming, no bf16 GPU peak). "
            "Adds ~5-15%% drift to predicted activation — acceptable for "
            "soft scoring (relative ranking) but absolute MSE values shift "
            "vs an fp8/bf16 baseline. Mutually exclusive with "
            "--fp8-reconstructor. Default: off (or NLA_INT4_RECONSTRUCTOR env)."
        ),
    )
    args = parser.parse_args()

    if args.max_concurrent is not None:
        MAX_CONCURRENT = args.max_concurrent
    if args.mem_fraction is not None:
        MEM_FRACTION = args.mem_fraction
    if args.tp_size is not None:
        TP_SIZE = args.tp_size
    if args.truncate_source is not None:
        TRUNCATE_SOURCE = args.truncate_source
    if args.fp8_verbalizer is not None:
        FP8_VERBALIZER = args.fp8_verbalizer
        VERBALIZER_QUANTIZATION = "fp8" if FP8_VERBALIZER else None
    if args.fp8_source is not None:
        FP8_SOURCE = args.fp8_source
    if args.fp8_reconstructor is not None:
        FP8_RECONSTRUCTOR = args.fp8_reconstructor
    if args.int4_reconstructor is not None:
        INT4_RECONSTRUCTOR = args.int4_reconstructor
    if args.kv_cache_dtype is not None:
        KV_CACHE_DTYPE = args.kv_cache_dtype
    if args.cuda_graph_max_bs is not None:
        CUDA_GRAPH_MAX_BS = args.cuda_graph_max_bs
    if args.torch_compile is not None:
        TORCH_COMPILE = args.torch_compile
    if args.verbalizer_device is not None:
        VERBALIZER_DEVICE = _normalize_device(args.verbalizer_device)
    if args.reconstructor_device is not None:
        RECONSTRUCTOR_DEVICE = _normalize_device(args.reconstructor_device)
    if args.source_device is not None:
        SOURCE_DEVICE = _normalize_device(args.source_device)

    if FP8_RECONSTRUCTOR and INT4_RECONSTRUCTOR:
        raise ValueError(
            "--fp8-reconstructor and --int4-reconstructor are mutually "
            "exclusive — pick at most one."
        )

    uvicorn.run(app, host=args.host, port=args.port)
