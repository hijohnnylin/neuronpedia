"""``/capabilities`` -- advertise what this instance can serve.

A capability-aware router (webapp / pods) maps ``(endpoint, model)`` to an instance
that supports it, and the webapp hides tabs a model/backend can't serve. Endpoints
themselves still return a clean 4xx (``BackendUnsupported`` -> 400) when asked for
something unsupported, so this is advisory, not the enforcement.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from neuronpedia_inference.config import Config
from neuronpedia_inference.endpoints.activation.all import MAX_NUM_RESULTS
from neuronpedia_inference.endpoints.lens.lens_loader import JacobianLensStore, JppLensStore
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

    # What the pod can serve is the model's answer, for this backend, build and checkpoint at once.
    # A GENERATION_ONLY pod serves no capture point: it keeps vLLM's CUDA graphs, which never call
    # the Python forward the hooks are attached to. A router that reads this can send capture
    # traffic elsewhere; one that only sees 400s can only retry.
    described = model.describe()
    hooks = described.hooks_available
    can_capture = hooks or bool(described.static_points)
    can_residual = described.residual_readable
    attention = described.attention

    lens_jacobian = can_residual and JacobianLensStore.get() is not None
    lens_jpp = can_residual and JppLensStore.get() is not None

    # Gradient support as a fact rather than as an inference from the backend name: eager serves
    # gradients only when loaded with requires_grad=True (serving does not), and vLLM cannot serve
    # them through the forward at all. Cheap and side-effect-free on both backends.
    grad_support = model.grad_support.describe()

    # What the checkpoint's generation_config.json states, as the engine read it. A steer request
    # that leaves a knob unset runs with this, so a client that shows its sliders at their defaults
    # has this to show. Null when the checkpoint states nothing (Qwen3.5 ships no file).
    stated = model.recommended_sampling
    recommended_sampling = (
        None
        if stated.is_empty
        else {
            "temperature": stated.temperature,
            "top_k": stated.top_k,
            "top_p": stated.top_p,
            "do_sample": stated.do_sample,
        }
    )

    return {
        "model": config.custom_hf_model_id or config.override_model_id or config.model_id,
        "backend": described.backend,
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
        "capture_points": described.capture_points,
        "grad_support": grad_support,
        "recommended_sampling": recommended_sampling,
        # False only on a GENERATION_ONLY pod. Reported next to the endpoint map rather than in place
        # of it, so a client sees both which endpoints are off and the one reason they are.
        "hooks_available": hooks,
        "graph_replay": described.graph_replay,
        "static_points": [str(a) for a in described.static_points],
        "static_writes": [str(a) for a in described.static_writes],
        # Their pre-1.3 names. Nothing in this repo reads them, but they shipped to origin/main
        # before the rename, and a router that keys off them would read a missing list as "this pod
        # captures nothing" and stop sending it traffic -- a silent routing change, not an error.
        # Delete both once no deployed caller reads them.
        "frozen_points": [str(a) for a in described.static_points],
        "writes_available": [str(a) for a in described.static_writes],
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
            "lens_jpp": lens_jpp,
            "neurons": False,  # mlp.hook_post not served by the engine backends
        },
    }
