import logging
from collections.abc import AsyncGenerator, Sequence
from typing import Any, cast

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from interp_engine import (
    AddSpec,
    EagerModel,
    LayerSteeringSpec,
    OrthogonalDecompSpec,
    ProjectionCapSpec,
    SteeringOp,
    SteeringSpec,
    SteerSpec,
    VLLMModel,
)
from interp_engine import generate_stream as engine_generate_stream
from interp_engine import steer as engine_steer

from neuronpedia_inference.config import Config
from neuronpedia_inference.engine_adapter import (
    BackendUnsupported,
    assert_steer_layers_declared,
    assert_steering_available,
    declares_static_taps,
    tlens_hook_to_point,
)
from neuronpedia_inference.inference_utils.steering import (
    SteeringSettings,
    format_sse_message,
    process_features_vectorized,
    remove_sse_formatting,
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
from neuronpedia_inference.vllm_optional import SamplingParams

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
async def completion(request: SteerCompletionRequest):
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

    tokens = []
    if isinstance(model, EagerModel):
        tokens = model.to_tokens(prompt, prepend_bos=model.tok.tokenizer_prepends_bos, truncate=False)[0]
    elif isinstance(model, VLLMModel):
        # prompt already has BOS prepended above; model.generate tokenizes it with
        # add_special_tokens=False, so tokenize the same way just for the length check.
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
            assert_steer_layers_declared(model, steer_write_layers(features))
        except BackendUnsupported as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=400)

    max_new_tokens, no_room = resolve_max_new_tokens(len(tokens), int(request.n_completion_tokens))
    if no_room is not None:
        return no_room

    generator = run_batched_generate(
        prompt=prompt,
        settings=SteeringSettings(
            features=features,
            strength_multiplier=float(request.strength_multiplier),
            steer_method=steer_method,
            normalize_steering=normalize_steering,
        ),
        steer_types=request.types,
        seed=int(request.seed),
        temperature=float(request.temperature),
        freq_penalty=float(request.freq_penalty),
        max_new_tokens=max_new_tokens,
        use_stream_lock=request.stream if request.stream is not None else False,
    )

    if request.stream:
        logger.info("Streaming response")
        return StreamingResponse(generator, media_type="text/event-stream")

    logger.info("Non-streaming response")
    # Each frame carries the whole completion so far, so the last one is the answer.
    last_frame = None
    async for frame in generator:
        last_frame = frame
    if last_frame is None:
        raise ValueError("Generator yielded no items")

    response = SteerCompletionResponse.model_validate_json(remove_sse_formatting(last_frame))
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

        # Two backends: EagerModel (forward write-hooks) and the engine-owned
        # vLLM backend (worker steering write-hooks + streaming generation).
        if isinstance(model, VLLMModel):
            async for msg in _vllm_run_batched_generate(
                model=model,
                prompt=prompt,
                settings=settings,
                steer_types=steer_types,
                seed=seed,
                **kwargs,
            ):
                yield msg
            return
        if not isinstance(model, EagerModel):
            raise ValueError("The /steer/completion endpoint only supports the interp-engine and vLLM backends")
        # NOTE: this is an async generator, so `yield from` is illegal here.
        for msg in _engine_run_batched_generate(  # noqa: UP028
            model=model,
            prompt=prompt,
            settings=settings,
            steer_types=steer_types,
            seed=seed,
            **kwargs,
        ):
            yield msg


def _feature_to_steerspec(
    feature: NPSteerFeature | NPSteerVector,
    settings: SteeringSettings,
) -> SteerSpec:
    """Map an SAE feature / raw vector (with its TLens hook) to an engine ``SteerSpec``.

    A ``resid_pre[X]`` steering point equals the output of decoder layer ``X-1``
    (``resid_post[X-1]``), or the embedding output for ``X == 0``; ``resid_post[X]`` maps
    directly. This mirrors adding the vector at the same residual position TransformerLens
    steers at.
    """
    sae_manager = SAEManager.get_instance()
    hook_name = sae_manager.get_sae_hook(feature.source) if isinstance(feature, NPSteerFeature) else feature.hook
    address = tlens_hook_to_point(hook_name)
    name = address.name
    if address.layer is None:
        raise ValueError(f"Engine steering needs a per-layer hook, but {hook_name!r} maps to the global point {name!r}")
    layer = address.layer
    vector = torch.tensor(feature.steering_vector, dtype=torch.float32)
    coeff = settings.strength_multiplier * feature.strength
    method = "orthogonal" if settings.steer_method == NPSteerMethod.ORTHOGONAL_DECOMP else "additive"

    if name == "resid_post":
        point, spec_layer = "resid_post", layer
    elif name == "resid_pre":
        if layer == 0:
            point, spec_layer = "embeddings", 0
        else:
            point, spec_layer = "resid_post", layer - 1
    elif name == "z":
        # Attention-output SAEs steer in hook_z space: add the vector to the concatenated
        # per-head attention output (the attention output projection's input).
        point, spec_layer = "z", layer
    else:
        raise ValueError(f"Engine steering supports resid_pre/resid_post/attn.hook_z hooks only, got {hook_name!r}")
    return SteerSpec(
        vector=vector,
        layer=spec_layer,
        coeff=coeff,
        method=method,
        point=point,
        normalize=settings.normalize_steering,
    )


def steer_layer_for_hook(hook_name: str) -> int:
    """Which layer a steer at ``hook_name`` writes: ``resid_pre[X]`` lands at ``X-1``.

    Split out so the pre-flight write check and the spec that does the writing read the layer the
    same way. Computing it twice is how a pod passes a check for one layer and fails the engine at
    another.
    """
    if "resid_post" in hook_name:
        return int(hook_name.split(".")[1])
    if "resid_pre" in hook_name:
        return int(hook_name.split(".")[1]) - 1
    raise ValueError(f"Unsupported hook name for vLLM steering: {hook_name}")


def steer_write_layers(features: Sequence[NPSteerFeature | NPSteerVector]) -> list[int]:
    """The layers a steer over ``features`` will write to, sorted and deduplicated.

    Answerable in the request handler, before a StreamingResponse has taken the reply away, where
    :func:`features_to_vllm_steering_spec` runs inside the generator and cannot return a status.
    Hooks it cannot map are left to that function, whose ValueError is already handled.
    """
    sae_manager = SAEManager.get_instance()
    layers: set[int] = set()
    for feature in features:
        hook_name = sae_manager.get_sae_hook(feature.source) if isinstance(feature, NPSteerFeature) else feature.hook
        try:
            layers.add(steer_layer_for_hook(hook_name))
        except ValueError:
            continue
    return sorted(layers)


def features_to_vllm_steering_spec(settings: SteeringSettings) -> SteeringSpec:
    """Build an engine ``SteeringSpec`` (per-layer Add/ProjectionCap ops) for the vLLM backend.

    Shared by ``/steer/completion`` and ``/steer/completion-chat`` so both backends stay in
    sync. Supports ``resid_pre``/``resid_post`` hooks (``resid_pre[X]`` -> layer ``X-1``).
    """
    sae_manager = SAEManager.get_instance()
    layer_features: dict[int, list[tuple[NPSteerFeature | NPSteerVector, torch.Tensor]]] = {}
    for feature in settings.features:
        hook_name = sae_manager.get_sae_hook(feature.source) if isinstance(feature, NPSteerFeature) else feature.hook
        layer = steer_layer_for_hook(hook_name)

        steering_vector = torch.tensor(feature.steering_vector, dtype=torch.float32)
        if not torch.isfinite(steering_vector).all():
            raise ValueError("Steering vector contains inf or nan values")
        if settings.normalize_steering:
            norm = torch.norm(steering_vector)
            if norm == 0:
                raise ValueError("Zero norm steering vector")
            steering_vector = steering_vector / norm
        layer_features.setdefault(layer, []).append((feature, steering_vector))

    steering_spec_layers: dict[int, LayerSteeringSpec] = {}
    for layer, layer_feature_list in layer_features.items():
        operations: list[SteeringOp] = []
        if settings.steer_method == NPSteerMethod.SIMPLE_ADDITIVE:
            for feature, steering_vector in layer_feature_list:
                coeff = settings.strength_multiplier * feature.strength
                norm = torch.norm(steering_vector)
                if norm > 0:
                    operations.append(AddSpec(vector=steering_vector / norm, scale=norm.item() * coeff))
        elif settings.steer_method == NPSteerMethod.ORTHOGONAL_DECOMP:
            # h -> (I-P)h + coeff*P h; the worker uses only the vector's direction, so pass the
            # raw (un-normalized) vector. Matches the eager OrthogonalProjector numerics.
            for feature, steering_vector in layer_feature_list:
                coeff = settings.strength_multiplier * feature.strength
                operations.append(OrthogonalDecompSpec(vector=steering_vector, coeff=coeff))
        elif settings.steer_method == NPSteerMethod.PROJECTION_CAP:
            for feature, steering_vector in layer_feature_list:
                coeff = settings.strength_multiplier * feature.strength
                operations.append(ProjectionCapSpec(vector=steering_vector, min=None, max=coeff))
        if operations:
            steering_spec_layers[layer] = LayerSteeringSpec(operations=operations)

    if not steering_spec_layers:
        raise ValueError(
            "No valid steering layers found. All features may have zero-norm vectors or invalid configurations."
        )
    return SteeringSpec(layers=steering_spec_layers)


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


async def _vllm_run_batched_generate(
    model: VLLMModel,
    prompt: str,
    settings: SteeringSettings,
    steer_types: list[NPSteerType],
    seed: int | None,
    **kwargs: Any,
):
    """SSE generator for the vLLM backend, mirroring the STEERED/DEFAULT engine flow."""
    max_new_tokens = int(kwargs.get("max_new_tokens") or 0)
    temperature = float(kwargs.get("temperature", 1.0))
    if kwargs.get("freq_penalty"):
        logger.warning("freq_penalty is not supported on the vLLM backend; ignoring")

    # See the note in completion_chat.py's _vllm_chat_generate: build the spec only for a
    # run that steers, so a DEFAULT-only request with no features is not a 500.
    spec = features_to_vllm_steering_spec(settings) if NPSteerType.STEERED in steer_types else None

    def _sampling_params():
        return SamplingParams(temperature=temperature, max_tokens=max_new_tokens, seed=seed)

    async def _stream_type(
        active_spec: SteeringSpec | None,
    ) -> AsyncGenerator[str, None]:
        # With stream=True the backend returns an async generator of text deltas; it only
        # returns the full string when stream=False.
        return cast(
            AsyncGenerator[str, None],
            await model.generate(prompt, _sampling_params(), steering_spec=active_spec, stream=True),
        )

    output_by_type: dict[NPSteerType, str] = {}
    for flag in steer_types:
        active_spec = spec if flag == NPSteerType.STEERED else None
        text = ""
        async for delta in await _stream_type(active_spec):
            text += delta
            output_by_type[flag] = text
            yield _completion_frame(steer_types, output_by_type)
        output_by_type[flag] = text


def _engine_generate_text(
    model: EagerModel,
    tokens: torch.Tensor,
    specs: list[SteerSpec] | None,
    *,
    max_new_tokens: int,
    temperature: float,
    seed: int | None,
    position_mask: Any = None,
):
    """Stream decoded text deltas from the engine, optionally under steering.

    ``position_mask`` (a ``interp_engine.SteerMask`` preset or ``list[int]``) excludes
    prompt positions from steering (e.g. special tokens); resolved against ``tokens``.
    """
    from contextlib import nullcontext

    ctx = engine_steer(model, specs, prompt_token_ids=tokens, position_mask=position_mask) if specs else nullcontext()
    with ctx:
        for step in engine_generate_stream(
            model,
            tokens,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop_at_eos=True,
            seed=seed,
        ):
            yield step.token_str


def _engine_run_batched_generate(
    model: EagerModel,
    prompt: str,  # noqa: ARG001 - tokens are derived from the model tokenizer below
    settings: SteeringSettings,
    steer_types: list[NPSteerType],
    seed: int | None,
    **kwargs: Any,
):
    """SSE generator for the engine backend, mirroring the TLens STEERED/DEFAULT flow."""
    tokens = model.to_tokens(prompt, prepend_bos=model.tok.tokenizer_prepends_bos, truncate=False)[0]
    max_new_tokens = int(kwargs.get("max_new_tokens") or 0)
    temperature = float(kwargs.get("temperature", 1.0))

    specs = [_feature_to_steerspec(f, settings) for f in settings.features]

    output_by_type: dict[NPSteerType, str] = {}
    for flag in steer_types:
        active_specs = specs if flag == NPSteerType.STEERED else None
        text = ""
        for delta in _engine_generate_text(
            model,
            tokens,
            active_specs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
        ):
            text += delta
            output_by_type[flag] = text
            yield _completion_frame(steer_types, output_by_type)
        output_by_type[flag] = text


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
