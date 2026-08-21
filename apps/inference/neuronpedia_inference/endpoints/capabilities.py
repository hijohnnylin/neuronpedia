"""``/capabilities`` -- advertise what this instance can serve.

A capability-aware router (webapp / pods) maps ``(endpoint, model)`` to an instance
that supports it, and the webapp hides tabs a model/backend can't serve. Endpoints
themselves still return a clean 4xx (``BackendUnsupported`` -> 400) when asked for
something unsupported, so this is advisory, not the enforcement.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from interp_engine import HOOK_CAPTURE_POINTS, EagerModel, VLLMModel

from neuronpedia_inference.config import Config
from neuronpedia_inference.endpoints.activation.all import MAX_NUM_RESULTS
from neuronpedia_inference.endpoints.lens.lens_loader import JacobianLensStore
from neuronpedia_inference.engine_adapter import (
    native_resid_available,
    vllm_attention_unsupported_reason,
    vllm_served_capture_points,
)
from neuronpedia_inference.sae_cache import sae_cache
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.shared import Model, budget, limiter

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/capabilities")
async def capabilities():
    """Report the loaded model, backend, concurrency/token limits, and feature support."""
    config = Config.get_instance()
    model = Model.get_instance()
    is_vllm = isinstance(model, VLLMModel)
    is_eager = isinstance(model, EagerModel)

    # Capture points available for SAE reads, from the engine's point table rather than restated
    # here -- a hand-written copy of this list is what let the webapp be told `attn_out` was
    # unavailable for a year after the engine started serving it. Eager serves every hookable point
    # (and more besides, but the rest are not SAE sites); vLLM serves the same set less whatever its
    # GPU sharding splits across ranks. DFA/attention need eager attn or the vLLM recompute (both
    # wired, the latter only unsharded).
    #
    # A GENERATION_ONLY pod serves none of them: it keeps vLLM's CUDA graphs, which never call the
    # Python forward the hooks are attached to. That is the whole reason this flag is advertised
    # rather than merely enforced -- a router that reads /capabilities can route capture traffic
    # elsewhere, where one that only sees 400s can only retry.
    hooks = model.hooks_available
    declared = tuple(getattr(model, "static_points", ()) or ())
    writes = tuple(getattr(model, "static_writes", ()) or ())
    graph_replay = bool(getattr(model, "graph_replay", False))
    can_capture = bool(hooks or declared)
    declared_names = {getattr(a, "name", str(a).split(".", 1)[0]) for a in declared}
    can_residual = bool(hooks or "resid_post" in declared_names or native_resid_available(model))
    if not can_capture:
        capture_points = []
    elif is_eager:
        capture_points = sorted(HOOK_CAPTURE_POINTS)
    else:
        capture_points = sorted(vllm_served_capture_points(model))

    attention = (hooks or "attn" in declared_names) and (is_eager or vllm_attention_unsupported_reason(model) is None)

    lens_jacobian = can_residual and JacobianLensStore.get() is not None

    # Gradient support as a fact rather than as an inference from the backend name: eager serves
    # gradients only when loaded with requires_grad=True (serving does not), and vLLM cannot serve
    # them through the forward at all. Cheap and side-effect-free on both backends.
    grad_support = model.grad_support.describe()

    return {
        "model": config.custom_hf_model_id or config.override_model_id or config.model_id,
        "backend": "vllm" if is_vllm else "eager",
        "device": config.device,
        "max_concurrent_requests": limiter.max_concurrent,
        "max_tokens": config.max_tokens,
        "token_limit": config.token_limit,
        "lens_token_limit": config.lens_token_limit,
        # May be lower than token_limit: derived at startup from the measured VRAM budget
        # and the widest configured SAE. Completion/steer keep token_limit.
        "activation_token_limit": config.activation_token_limit,
        # Working-set budget shared by all in-flight requests, measured after warmup. A
        # request is admitted only when its estimated cost fits in what is free, so
        # max_concurrent_requests is a ceiling rather than a promise. 0 means unrationed.
        "vram_budget_bytes": budget.total_bytes,
        "vram_budget_available_bytes": budget.available_bytes,
        # SAE paging: when enabled, SAE masters live in host RAM and only `budget_bytes` of
        # them are GPU-resident at a time. A rising miss/hit ratio here means the residency
        # budget is too small for the traffic and requests are paying stage-in latency.
        "sae_cache": sae_cache.stats(),
        "max_num_results": MAX_NUM_RESULTS,
        "capture_points": capture_points,
        "grad_support": grad_support,
        # False only on a GENERATION_ONLY pod. Reported next to the endpoint map rather than in place
        # of it, so a client sees both which endpoints are off and the one reason they are.
        "hooks_available": hooks,
        "graph_replay": graph_replay,
        "static_points": [str(a) for a in declared],
        "static_writes": [str(a) for a in writes],
        # Their pre-1.3 names. Nothing in this repo reads them, but they shipped to origin/main
        # before the rename, and a router that keys off them would read a missing list as "this pod
        # captures nothing" and stop sending it traffic -- a silent routing change, not an error.
        # Delete both once no deployed caller reads them.
        "frozen_points": [str(a) for a in declared],
        "writes_available": [str(a) for a in writes],
        "generation_only": config.generation_only,
        # Layers /activation/raw will return when the request does not name any.
        "num_layers": config.num_layers,
        # Empty on a pod started with no SAE sets, where only the model-only endpoints below
        # are servable.
        "sae_sets": SAEManager.get_instance().get_valid_sae_sets(),
        # Everything that reads an activation is gated on `hooks`. The two steer endpoints stay True
        # either way: they serve the unsteered completion types on any pod, and refuse only the
        # STEERED type when hooks are gone (a 400 naming the flag). Reporting them False would hide
        # the completions a generation-only pod exists to serve.
        "endpoints": {
            "tokenize": True,
            "activation_single": can_capture,
            "activation_all": can_capture,
            "activation_source": can_capture,
            "activation_raw": can_residual,
            "activation_topk_by_token": can_capture,
            "activation_attention": attention,  # eager output_attentions / vLLM off-kernel recompute
            "dfa": attention,  # eager value+attn_probs / vLLM recompute; only on -att- sources
            "steer_completion": True,
            "steer_completion_chat": True,
            "lens_logit": can_residual,
            "lens_jacobian": lens_jacobian,
            "neurons": False,  # mlp.hook_post not served by the engine backends
        },
    }
