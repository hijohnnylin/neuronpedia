import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import nullcontext
from typing import Any

import torch
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from interp_engine import (
    Address,
    AddSpec,
    InterpModel,
    OrthogonalDecompSpec,
    ProjectionCapSpec,
    SamplingSettings,
    SteeringOp,
    SteeringSpec,
)
from interp_engine import steer as engine_steer

from neuronpedia_inference.config import Config
from neuronpedia_inference.engine_adapter import (
    BackendUnsupported,
    assert_steer_layers_declared,
    assert_steering_available,
    declares_static_taps,
    tlens_hook_to_point,
)
from neuronpedia_inference.inference_utils.sampling import (
    resolve_request_sampling,
    sampling_report,
    state_settings_once,
)
from neuronpedia_inference.inference_utils.steering import (
    SteeringSettings,
    format_sse_message,
    process_features_vectorized,
    remove_sse_formatting,
    stop_when_client_leaves,
    stream_lock,
)
from neuronpedia_inference.inference_utils.token_limit import reject_if_over_token_limit
from neuronpedia_inference.memory_cost import steer_cost
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.schemas import (
    NPLogprob,
    NPSteerCompletionOutput,
    NPSteerFeature,
    NPSteerMethod,
    NPSteerType,
    NPSteerVector,
    SteerCompletionRequest,
    SteerCompletionResponse,
)
from neuronpedia_inference.shared import Model, with_request_lock

logger = logging.getLogger(__name__)

router = APIRouter()


def get_layer_num_from_sae_id(sae_id: str) -> int:
    return int(sae_id.split("-")[0]) if not sae_id.isdigit() else int(sae_id)


def resolve_max_new_tokens(prompt_len: int, requested: int) -> tuple[int, JSONResponse | None]:
    """Clamp a generation length to the sequence budget, or explain why it can't be.

    ``clamp_completion_tokens`` returns 0 when the prompt already fills ``max_tokens``,
    which generation treats as "produce nothing" -- an empty completion the caller has
    no way to distinguish from the model choosing to stop. The prompt-length check that
    runs before this only bounds the prompt against ``token_limit``; nothing bounds
    prompt + generation together, so this is where that runs out.
    """
    config = Config.get_instance()
    clamped = config.clamp_completion_tokens(prompt_len, requested)
    if clamped > 0:
        return clamped, None
    logger.error(
        "No room to generate: %s prompt tokens against a %s-token budget",
        prompt_len,
        config.max_tokens,
    )
    return 0, JSONResponse(
        content={
            "error": (
                f"No room to generate: the prompt is {prompt_len} tokens and the "
                f"per-request budget (prompt + completion) is {config.max_tokens}. "
                "Shorten the prompt or start a new conversation."
            )
        },
        status_code=400,
    )


@router.post("/steer/completion", responses={200: {"model": SteerCompletionResponse}})
@with_request_lock(exclusive=False, cost=steer_cost)
async def completion(request: SteerCompletionRequest, http_request: Request):
    config = Config.get_instance()
    model = Model.get_instance()
    steer_method = request.steer_method
    normalize_steering = request.normalize_steering

    # See the equivalent guard in completion_chat.py: without the worker's write-hooks a STEERED
    # request would come back fluent, unsteered, and labelled as steered.
    if NPSteerType.STEERED in request.types:
        try:
            assert_steering_available(model, "Steered generation")
        except BackendUnsupported as e:
            return JSONResponse(content={"error": str(e)}, status_code=400)

    # Ensure exactly one of features or vector is provided
    if (request.features is not None) == (request.vectors is not None):
        logger.error("Invalid request data: exactly one of features or vectors must be provided")
        return JSONResponse(
            content={"error": "Invalid request data: exactly one of features or vectors must be provided"},
            status_code=400,
        )

    prompt = request.prompt

    # if the prompt doesn't start with the bos, prepend it (models like Qwen have no BOS)
    bos_token = model.tokenizer.bos_token
    if bos_token and not prompt.startswith(bos_token):
        prompt = bos_token + prompt

    # The prompt carries its BOS as text now, so the tokenizer must not add a second one. Asking it
    # to prepend as well doubled the BOS on every tokenizer that adds its own (Gemma, Llama).
    tokens = model.to_tokens(prompt, prepend_bos=False, truncate=False)[0]

    too_long = reject_if_over_token_limit(len(tokens), config.token_limit)
    if too_long is not None:
        return too_long

    if request.features is not None:
        features = process_features_vectorized(request.features)
    elif request.vectors is not None:
        features = request.vectors

    else:
        return JSONResponse(
            content={"error": "No features or vectors provided"},
            status_code=400,
        )

    # Asked here because the spec that writes is built inside the generator, where a refusal is a
    # 500 mid-stream rather than a status this can return.
    if NPSteerType.STEERED in request.types and declares_static_taps(model):
        try:
            for point, layers in steer_write_targets(features).items():
                assert_steer_layers_declared(model, layers, point=point)
        except BackendUnsupported as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=400)

    max_new_tokens, no_room = resolve_max_new_tokens(len(tokens), int(request.n_completion_tokens))
    if no_room is not None:
        return no_room

    seed = int(request.seed)
    sampling = resolve_request_sampling(model, request)
    report = sampling_report(sampling, seed)
    generator = run_batched_generate(
        prompt=prompt,
        settings=SteeringSettings(
            features=features,
            strength_multiplier=float(request.strength_multiplier),
            steer_method=steer_method,
            normalize_steering=normalize_steering,
        ),
        steer_types=request.types,
        seed=seed,
        sampling=sampling,
        max_new_tokens=max_new_tokens,
        use_stream_lock=request.stream if request.stream is not None else False,
    )

    if request.stream:
        logger.info("Streaming response")
        generator = state_settings_once(generator, report, SteerCompletionResponse)
        return StreamingResponse(
            stop_when_client_leaves(generator, http_request, "STEER"),
            media_type="text/event-stream",
        )

    logger.info("Non-streaming response")
    # Each frame carries the whole completion so far, so the last one is the answer. The
    # generators emit at least one frame per steer type, empty completion included.
    last_frame = None
    async for frame in generator:
        last_frame = frame
    if last_frame is None:
        raise ValueError("Steer generator emitted no frame for any steer type")

    response = SteerCompletionResponse.model_validate_json(remove_sse_formatting(last_frame))
    # The stream stated its settings on the first frame; the one response states them here.
    response.sampling = report
    # Drop unset fields rather than serializing them as null: callers predating
    # `logprobs` expect the key to be absent when there is nothing to report.
    return JSONResponse(content=response.model_dump(exclude_none=True))


async def run_batched_generate(
    prompt: str,
    settings: SteeringSettings,
    steer_types: list[NPSteerType],
    seed: int | None = None,
    use_stream_lock: bool = False,
    **kwargs: Any,
):
    async with await stream_lock(use_stream_lock):
        model = Model.get_instance()

        if seed is not None:
            torch.manual_seed(seed)

        # One path for every backend: `steer()` records the spec and `generate_stream` honors it,
        # on eager through hooks and elsewhere per request. A point a backend cannot steer is the
        # engine's refusal, with its reason, not a branch here.
        async for msg in _run_batched_generate(
            model=model,
            prompt=prompt,
            settings=settings,
            steer_types=steer_types,
            seed=seed,
            **kwargs,
        ):
            yield msg


def steer_target(hook_name: str) -> Address:
    """The engine point a steer at a TransformerLens ``hook_name`` writes.

    ``resid_pre[X]`` is the output of decoder layer ``X-1`` (``resid_post[X-1]``), or the embedding
    output for ``X == 0``; ``resid_post[X]`` maps directly; ``hook_z`` steers the concatenated
    per-head attention output that attention-output SAEs live in. One mapping for every backend,
    so a pod passes a check for one target and reaches the engine at the same one.
    """
    address = tlens_hook_to_point(hook_name)
    name, layer = address.name, address.layer
    if layer is None:
        raise ValueError(f"Engine steering needs a per-layer hook, but {hook_name!r} maps to the global point {name!r}")
    if name == "resid_pre":
        return Address("embeddings") if layer == 0 else Address("resid_post", layer - 1)
    if name in ("resid_post", "z"):
        return Address(name, layer)
    raise ValueError(f"Engine steering supports resid_pre/resid_post/attn.hook_z hooks only, got {hook_name!r}")


def _hook_of(feature: NPSteerFeature | NPSteerVector) -> str:
    sae_manager = SAEManager.get_instance()
    return sae_manager.get_sae_hook(feature.source) if isinstance(feature, NPSteerFeature) else feature.hook


def steer_write_targets(features: Sequence[NPSteerFeature | NPSteerVector]) -> dict[str, list[int]]:
    """The layers a steer over ``features`` will write, per engine point, sorted and deduplicated.

    Answerable in the request handler, before a StreamingResponse has taken the reply away, where
    :func:`features_to_steering_specs` runs inside the generator and cannot return a status.
    Hooks it cannot map are left to that function, whose ValueError is already handled.
    """
    targets: dict[str, set[int]] = {}
    for feature in features:
        try:
            spec = SteeringSpec.at(steer_target(_hook_of(feature)))
        except ValueError:
            continue
        targets.setdefault(spec.point, set()).update(spec.layers)
    return {point: sorted(layers) for point, layers in targets.items()}


def _steer_op(method: NPSteerMethod, vector: list[float], coeff: float, *, normalize: bool) -> SteeringOp:
    """The engine op for one feature. The other methods use only the direction, so they ignore ``normalize``."""
    if method == NPSteerMethod.SIMPLE_ADDITIVE:
        return AddSpec(vector=vector, scale=coeff, normalize=normalize)
    if method == NPSteerMethod.ORTHOGONAL_DECOMP:
        return OrthogonalDecompSpec(vector=vector, coeff=coeff)
    return ProjectionCapSpec(vector=vector, max=coeff)


def features_to_steering_specs(settings: SteeringSettings) -> list[SteeringSpec]:
    """The engine ``SteeringSpec``s for ``settings``, one per feature, in request order.

    Shared by ``/steer/completion`` and ``/steer/completion-chat``. The list is one ``steer()``
    block. Each strength is scaled by ``strength_multiplier``; ``normalize_steering`` makes each
    vector unit length first.
    """
    if not settings.features:
        raise ValueError("A steered generation needs at least one feature or vector to steer with")
    specs = []
    for f in settings.features:
        if f.steering_vector is None:
            raise ValueError("A feature has no steering vector")
        coeff = settings.strength_multiplier * f.strength
        op = _steer_op(settings.steer_method, f.steering_vector, coeff, normalize=settings.normalize_steering)
        specs.append(SteeringSpec.at(steer_target(_hook_of(f)), op))
    return specs


def _completion_frame(steer_types: list[NPSteerType], output_by_type: dict[NPSteerType, str]) -> str:
    """Build one streaming SSE frame from the running per-type outputs.

    ``make_steer_completion_response`` emits only the entries named in ``steer_types``; a
    steer type not yet started reads as empty.
    """
    return format_sse_message(
        make_steer_completion_response(
            steer_types,
            output_by_type.get(NPSteerType.STEERED, ""),
            output_by_type.get(NPSteerType.DEFAULT, ""),
        ).to_wire_json()
    )


async def _generate_text(
    model: InterpModel,
    tokens: Sequence[int],
    specs: list[SteeringSpec] | None,
    *,
    max_new_tokens: int,
    sampling: SamplingSettings,
    seed: int | None,
    position_mask: Any = None,
) -> AsyncIterator[str]:
    """Stream decoded text deltas from the engine, under ``specs`` when given.

    ``specs`` is one ``steer()`` block, which stays open until the stream is exhausted: that is
    what keeps a served backend's per-request steer attached to the request. A backend that
    cannot carry ``position_mask`` refuses it there.
    """
    block = engine_steer(model, specs, prompt_token_ids=tokens, position_mask=position_mask) if specs else nullcontext()
    with block:
        async for delta in model.generate_stream(
            tokens,
            max_tokens=max_new_tokens,
            temperature=sampling.temperature,
            top_k=sampling.top_k,
            top_p=sampling.top_p,
            presence_penalty=sampling.presence_penalty,
            seed=seed,
        ):
            yield delta


async def _run_batched_generate(
    model: InterpModel,
    prompt: str,
    settings: SteeringSettings,
    steer_types: list[NPSteerType],
    seed: int | None,
    **kwargs: Any,
):
    """SSE generator for the STEERED/DEFAULT flow, on whichever backend is loaded."""
    # BOS is in the prompt text already; see the handler.
    tokens = [int(t) for t in model.to_tokens(prompt, prepend_bos=False, truncate=False)[0]]
    max_new_tokens = int(kwargs.get("max_new_tokens") or 0)
    sampling: SamplingSettings = kwargs["sampling"]

    # Built only for a run that steers, so a DEFAULT-only request with no features is not a 500.
    specs = features_to_steering_specs(settings) if NPSteerType.STEERED in steer_types else None

    output_by_type: dict[NPSteerType, str] = {}
    for flag in steer_types:
        active_specs = specs if flag == NPSteerType.STEERED else None
        text = ""
        async for delta in _generate_text(
            model,
            tokens,
            active_specs,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            seed=seed,
        ):
            text += delta
            output_by_type[flag] = text
            yield _completion_frame(steer_types, output_by_type)
        output_by_type[flag] = text
        if not text:
            # No delta arrives when the model samples EOS first, or emits only special tokens.
            # One frame per type, even for an empty completion: no frame is a 500 downstream.
            yield _completion_frame(steer_types, output_by_type)


def make_steer_completion_response(
    steer_types: list[NPSteerType],
    steered_result: str,
    default_result: str,
    steered_logprobs: list[NPLogprob] | None = None,
    default_logprobs: list[NPLogprob] | None = None,
) -> SteerCompletionResponse:
    """Assemble the response, emitting one entry per requested steer type.

    Nothing populates the logprobs arguments today -- neither generation backend hands
    back per-token scores -- but ``logprobs`` is part of the published response schema,
    so the parameters stay as the seam a backend would fill in.
    """
    output_by_type = {
        NPSteerType.STEERED: steered_result,
        NPSteerType.DEFAULT: default_result,
    }
    logprobs_by_type = {
        NPSteerType.STEERED: steered_logprobs,
        NPSteerType.DEFAULT: default_logprobs,
    }
    return SteerCompletionResponse(
        outputs=[
            NPSteerCompletionOutput(
                type=steer_type,
                output=output_by_type[steer_type],
                logprobs=logprobs_by_type[steer_type],
            )
            for steer_type in steer_types
        ]
    )
