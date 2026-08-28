import logging
import os
import time
from collections.abc import AsyncGenerator
from typing import Any, cast

import numpy as np
import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from interp_engine import (
    Address,
    EagerModel,
    SteerMask,
    VLLMModel,
    compose_assistant_turns,
    strip_wire_reasoning,
)

from neuronpedia_inference.config import Config
from neuronpedia_inference.endpoints.steer.completion import (
    _engine_generate_text,
    _feature_to_steerspec,
    features_to_vllm_steering_spec,
    resolve_max_new_tokens,
)
from neuronpedia_inference.engine_adapter import BackendUnsupported, assert_steering_available, get_tokenize
from neuronpedia_inference.inference_utils.persona import (
    AxisAsset,
    AxisRequestError,
    RenderConditions,
    project_axis_with_percentile,
    resolve_request_axes,
    truncate_content,
)
from neuronpedia_inference.inference_utils.persona.capture_engine import (
    capture_turn_means_engine,
    capture_turn_means_vllm,
    turn_means_from_generation_capture,
)
from neuronpedia_inference.inference_utils.steering import (
    SteeringSettings,
    format_sse_message,
    process_features_vectorized,
    remove_sse_formatting,
    stream_lock,
)
from neuronpedia_inference.inference_utils.token_limit import reject_if_over_token_limit
from neuronpedia_inference.inference_utils.vllm_monitor import get_monitor
from neuronpedia_inference.memory_cost import steer_cost
from neuronpedia_inference.schemas import (
    NPLogprob,
    NPSteerChatMessage,
    NPSteerChatResult,
    NPSteerType,
    SteerAxisReadout,
    SteerAxisTurn,
    SteerCompletionChatRequest,
    SteerCompletionChatResponse,
)
from neuronpedia_inference.shared import Model, with_request_lock
from neuronpedia_inference.vllm_optional import VLLM_AVAILABLE, SamplingParams

logger = logging.getLogger(__name__)

# A base model has no chat template, and this endpoint used to paper over that with a
# generic ChatML render. That produced a 200 carrying the prompt parroted back with
# `<|im_start|>` markers the model has never seen — a failure indistinguishable from
# success unless you read the output. Refusing sends the caller to the route that fits
# the model, which is what the UI already picks for a non-instruct model.
#
# The verdict comes from the engine rather than from `tokenizer.chat_template`, so a model
# that defines its chat format in code (DeepSeek-V4) is served rather than refused.
NO_CHAT_TEMPLATE_ERROR = (
    "This model has no chat template, so it cannot accept chat messages. "
    "Use /v1/steer/completion with a raw `prompt` instead."
)

# Enable background health monitoring if env var is set
ENABLE_BACKGROUND_MONITOR = os.environ.get("ENABLE_VLLM_MONITOR", "0") == "1"
MONITOR_INTERVAL = float(os.environ.get("VLLM_MONITOR_INTERVAL", "30"))


router = APIRouter()


@router.get("/steer/health")
async def health_check():
    """
    Get health stats for the vLLM engine.

    Returns GPU memory usage, system RAM, active requests, threads, etc.
    Useful for debugging hanging requests.
    """
    model = Model.get_instance()
    monitor = get_monitor()

    # Set the model if it's a VLLMSteerModel
    if VLLM_AVAILABLE and isinstance(model, VLLMModel):
        monitor.set_model(model)

    stats = await monitor.get_stats()
    return JSONResponse(
        content={
            "stats": stats.to_dict(),
            "summary": stats.summary(),
        }
    )


def messages_for_render(promptChat: list[NPSteerChatMessage], *, blank_system_prompt: bool) -> list[dict[str, str]]:
    """The request messages as the chat template should see them.

    Distinct from ``promptChat``, which is echoed back to the client verbatim: this is only
    what the model reads, so the returned transcript and the stored row stay faithful to what
    was generated while the prompt drops what shouldn't be re-rendered.

    Prior-turn reasoning is what gets dropped. Composition folds it into the message as
    ``<think>...</think>`` so the client can render it, but harmony's convention is to discard
    earlier analysis and ``<think>`` is not one of its delimiters — re-rendering it would put
    literal tag text inside a ``final``-channel block. Reasoning-tag families would merely pay
    the context window for it.
    """
    rendered: list[dict[str, str]] = []
    for index, message in enumerate(promptChat):
        content = "" if (index == 0 and blank_system_prompt) else message.content
        if message.role == "assistant":
            content = strip_wire_reasoning(content)
            if not content:
                # A turn that was nothing but reasoning has nothing left to render. Dropping it
                # beats rendering an empty assistant turn the model would try to continue.
                continue
        rendered.append({"role": message.role, "content": content})
    return rendered


@router.post("/steer/completion-chat", responses={200: {"model": SteerCompletionChatResponse}})
@with_request_lock(exclusive=False, cost=steer_cost)
async def completion_chat(request: SteerCompletionChatRequest):
    request_start = time.time()
    model = Model.get_instance()
    config = Config.get_instance()
    steer_method = request.steer_method
    normalize_steering = request.normalize_steering
    steer_special_tokens = request.steer_special_tokens

    # Start background monitoring if enabled (once) for VLLMSteerModel
    if ENABLE_BACKGROUND_MONITOR and VLLM_AVAILABLE and isinstance(model, VLLMModel):
        monitor = get_monitor()
        monitor.set_model(model)
        if monitor._background_task is None:
            monitor.start_background_logging(interval=MONITOR_INTERVAL)

    # Every readout axis this request wants comes with it: this server ships none, so there is
    # nothing to resolve a name against. The backend check comes first because it is what makes
    # the model's width and depth readable, and because an artifact should not be fetched for a
    # backend that could not have read it.
    axes: list[AxisAsset] = []
    if request.custom_axes:
        vllm_backend = VLLM_AVAILABLE and isinstance(model, VLLMModel)
        if not (vllm_backend or isinstance(model, EagerModel)):
            return JSONResponse(
                content={"error": "Axis readouts require the vLLM or interp-engine (EagerModel) backend"},
                status_code=400,
            )
        try:
            axes = await resolve_request_axes(
                request.custom_axes,
                hidden_size=int(model.d_model),
                n_layers=int(model.n_layers),
            )
        except AxisRequestError as exc:
            logger.warning("Axis request validation failed", exc_info=True)
            return JSONResponse(content={"error": "Invalid axis request"}, status_code=exc.status_code)

    # Every requested axis has to agree about how the conversation is rendered, because those
    # conditions are applied before generation and so change the text itself. There is no way to
    # render one conversation two ways in a single generation, and honouring one axis's
    # conditions while reporting another's numbers would quietly project onto a direction fitted
    # off-distribution.
    render, render_conflict = _agreed_render_conditions(axes)
    if render_conflict is not None:
        return JSONResponse(content={"error": render_conflict}, status_code=400)

    # A steered generation needs the worker's write-hooks, and a GENERATION_ONLY pod has none: on
    # that pod this would otherwise generate happily and return UNSTEERED text under a "STEERED"
    # label, since a hook that never fires reports nothing. Conditional on the request, because an
    # unsteered completion through this endpoint is exactly what such a pod is for.
    #
    # Deliberately not conditional on the axes: a readout needs capture hooks, not write hooks, so
    # an unsteered readout is exactly what a generation-only pod can serve.
    if NPSteerType.STEERED in request.types:
        try:
            assert_steering_available(model, "Steered generation")
        except BackendUnsupported as e:
            return JSONResponse(content={"error": str(e)}, status_code=400)

    # Features or vectors are what steering steers with, so they are required only when something
    # will be steered. A readout-only request (types=[DEFAULT], axes=[...]) legitimately carries
    # neither, and demanding a placeholder would make the caller fake a steer to measure one.
    wants_steering = NPSteerType.STEERED in request.types
    if wants_steering and (request.features is not None) == (request.vectors is not None):
        logger.error("Invalid request data: exactly one of features or vectors must be provided")
        return JSONResponse(
            content={"error": "Invalid request data: exactly one of features or vectors must be provided"},
            status_code=400,
        )
    if not wants_steering and request.features is not None and request.vectors is not None:
        return JSONResponse(
            content={"error": "Invalid request data: provide at most one of features or vectors"},
            status_code=400,
        )

    promptChat = request.prompt

    # Blank a caller-supplied system prompt only when the requested axes were fitted that way —
    # their directions are meaningless against activations from a conversation rendered
    # differently. Gating on the assets (rather than on "a readout was requested") keeps us from
    # silently discarding the system prompt of a model whose axes have no such requirement.
    blank_system_prompt = bool(promptChat) and promptChat[0].role == "system" and render.blank_system_prompt

    promptChatFormatted = messages_for_render(promptChat, blank_system_prompt=blank_system_prompt)

    if model.tokenizer is None:
        raise ValueError("Tokenizer is not initialized")

    tok = get_tokenize(model)
    if not tok.has_chat_template():
        return JSONResponse(content={"error": NO_CHAT_TEMPLATE_ERROR}, status_code=400)

    # Render the prompt to a string, then tokenize to a flat list of ids. We render
    # first (rather than apply_chat_template(tokenize=True)) because transformers 5
    # returns a BatchEncoding from the tokenizing path; rendering + a plain tokenizer
    # call is deterministic and backend-uniform. The rendered string already carries
    # the model's special tokens, so add_special_tokens=False (no double BOS). Both
    # backends' generation consumes the token ids directly.
    #
    # Rendered through the engine's `Tokenize`, not the tokenizer: that is the layer holding
    # the code formatter for a family whose format is not a Jinja template.
    # `render.template_kwargs` is whatever the fit pinned about the template itself. Llama 3.1
    # injects the current date into the system block, so an axis fitted on it pins `date_string`
    # and would otherwise drift off distribution as the calendar moves.
    #
    # Typed `dict[str, Any]` because the pinned names are the asset's, not ours: as `dict[str, str]`
    # a pin that collides with a named parameter (`continue_final_message`) reads as a str-for-bool
    # type error at every call site rather than at the one asset that would do it.
    template_kwargs: dict[str, Any] = dict(render.template_kwargs)
    rendered_prompt = tok.apply_chat_template(
        promptChatFormatted,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    promptTokenized = model.tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"]
    # Normalize any nested [[...]] shape to 1D.
    if promptTokenized and isinstance(promptTokenized[0], list):
        promptTokenized = promptTokenized[0]
    promptTokenized = torch.tensor(promptTokenized)

    # logger.info("promptTokenized: %s", promptTokenized)
    too_long = reject_if_over_token_limit(len(promptTokenized), config.token_limit)
    if too_long is not None:
        return too_long

    if request.features is not None:
        features = process_features_vectorized(request.features)
    elif request.vectors is not None:
        features = request.vectors
    elif not wants_steering:
        # Nothing will be steered, so there is nothing to steer with.
        features = []
    else:
        return JSONResponse(
            content={"error": "No features or vectors provided"},
            status_code=400,
        )

    # Convert promptChatFormatted to NPSteerChatMessage for the readouts, so they analyze the
    # same conversation (including the system message, blanked or not) that generation saw.
    inputPromptForAxes = [NPSteerChatMessage(role=msg["role"], content=msg["content"]) for msg in promptChatFormatted]

    max_new_tokens, no_room = resolve_max_new_tokens(len(promptTokenized), int(request.n_completion_tokens))
    if no_room is not None:
        return no_room

    generation_start = time.time()

    generator = run_batched_generate(
        promptTokenized=promptTokenized,
        inputPrompt=inputPromptForAxes if axes else promptChat,
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
        steer_special_tokens=steer_special_tokens,
        use_stream_lock=request.stream if request.stream is not None else False,
        axes=axes,
    )

    if request.stream:
        # For streaming, wrap the generator to add timing logs
        async def timed_generator():
            chunk_count = 0
            try:
                async for item in generator:
                    chunk_count += 1
                    yield item
                generation_time = time.time() - generation_start
                total_time = time.time() - request_start
                logger.info(
                    f"[REQUEST COMPLETE] total={total_time:.2f}s, generation={generation_time:.2f}s, "
                    f"~chunks={chunk_count}"
                )
            except Exception:
                logger.exception(f"[REQUEST ERROR] Error during generation after {time.time() - request_start:.2f}s")
                raise

        return StreamingResponse(timed_generator(), media_type="text/event-stream")

    # for non-streaming request, get last item from generator
    last_item = None
    chunk_count = 0
    async for item in generator:
        chunk_count += 1
        last_item = item

    generation_time = time.time() - generation_start
    total_time = time.time() - request_start
    logger.info(f"[REQUEST COMPLETE] total={total_time:.2f}s, generation={generation_time:.2f}s, ~chunks={chunk_count}")

    if last_item is None:
        raise ValueError("No response generated")
    results = remove_sse_formatting(last_item)
    response = SteerCompletionChatResponse.model_validate_json(results)
    # set exclude_none to True to omit the logprobs field when n_logprobs isn't set in the request, for backwards compatibility
    return JSONResponse(content=response.model_dump(exclude_none=True))


def _agreed_render_conditions(axes: list[AxisAsset]) -> tuple[RenderConditions, str | None]:
    """The rendering conditions shared by every requested axis, or why there are none.

    Returns the agreed conditions and ``None`` when the axes agree, or when none were requested
    and so nothing about the prompt changes. Otherwise the second element names the
    disagreement, which the endpoint turns into a 400: the conversation is rendered once and
    generated from once, so honouring one axis's conditions while reporting another's numbers
    would project onto a direction fitted off-distribution.
    """
    if not axes:
        return RenderConditions(), None

    by_conditions: dict[tuple, list[AxisAsset]] = {}
    for axis in axes:
        by_conditions.setdefault(axis.render.key(), []).append(axis)
    if len(by_conditions) == 1:
        return axes[0].render, None

    groups = "; ".join(
        f"[{', '.join(axis.id for axis in group)}] need ({group[0].render.describe()})"
        for group in by_conditions.values()
    )
    return axes[0].render, (
        "The requested readout axes were fitted under different rendering conditions, so they "
        f"cannot be measured in one generation: {groups}. Request them separately."
    )


async def _capture_means(
    model: Any,
    conversation: list[NPSteerChatMessage],
    layers: list[int],
    steering_spec: Any,
    template_kwargs: dict[str, str],
) -> dict[int, torch.Tensor]:
    """One capture pass covering every layer in ``layers``, steered when a spec is given."""
    if not layers:
        return {}
    if isinstance(model, EagerModel):
        # steering_spec is a list[SteerSpec] here (engine specs).
        return capture_turn_means_engine(
            model, conversation, layers, specs=steering_spec, template_kwargs=template_kwargs
        )
    if isinstance(model, VLLMModel):
        # steering_spec is an engine SteeringSpec here (built from Add/ProjectionCap specs).
        return await capture_turn_means_vllm(
            model, conversation, layers, steering_spec=steering_spec, template_kwargs=template_kwargs
        )
    raise ValueError(f"axis readouts unsupported for backend {type(model).__name__}")


async def capture_axis_means(
    model: Any,
    conversation: list[NPSteerChatMessage],
    layers: list[int],
    steering_spec: Any = None,
    pre_cap_means: dict[int, torch.Tensor] | None = None,
    post_cap_means: dict[int, torch.Tensor] | None = None,
    template_kwargs: dict[str, str] | None = None,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor] | None]:
    """Per-message mean activations at every layer the requested axes read from.

    At most one forward per condition, whatever the axis count: the pre-cap read captures every
    outstanding layer at once, and so does the post-cap read. Six axes across five layers
    therefore cost what one axis costs, which is what makes a multi-axis panel affordable.

    Args:
        model: the loaded model (VLLMModel or EagerModel)
        conversation: the full conversation, including the generated assistant turn
        layers: the distinct layers the requested axes were fitted at
        steering_spec: steering for the post-cap read (engine ``list[SteerSpec]`` for
            EagerModel, ``SteeringSpec`` for vLLM). None means no post-cap read is wanted.
        pre_cap_means / post_cap_means: means already captured during generation, keyed by
            layer. Every layer supplied is a layer not captured again here. An unsteered
            generation supplies pre-cap; a steered one supplies post-cap and still needs the
            pre-cap read, because the cap covers the layers being projected at.
        template_kwargs: what the endpoint rendered the generation prompt with. A re-capture
            renders the conversation again, so anything the requested axes pinned about the
            template has to be pinned the same way here or the two renderings diverge.

    Returns:
        ``(pre_cap, post_cap)``, each keyed by layer. ``post_cap`` is None when there is none.
    """
    capture_start = time.time()
    wanted = sorted(set(layers))
    kwargs = template_kwargs or {}

    pre_cap = dict(pre_cap_means or {})
    pre_cap.update(await _capture_means(model, conversation, [x for x in wanted if x not in pre_cap], None, kwargs))

    post_cap = dict(post_cap_means or {})
    if steering_spec is not None:
        post_cap.update(
            await _capture_means(model, conversation, [x for x in wanted if x not in post_cap], steering_spec, kwargs)
        )

    logger.debug(
        f"[AXIS] captured layers {sorted(pre_cap)} pre-cap / {sorted(post_cap)} post-cap "
        f"in {time.time() - capture_start:.3f}s"
    )
    return pre_cap, post_cap or None


def build_axis_readouts(
    conversation: list[NPSteerChatMessage],
    steer_type: NPSteerType,
    axes: list[AxisAsset],
    pre_cap: dict[int, torch.Tensor],
    post_cap: dict[int, torch.Tensor] | None,
) -> list[SteerAxisReadout]:
    """Project the captured means onto each axis, one readout per axis.

    Assistant turns are located by role rather than by position, so a conversation carrying a
    system message (or one whose turns do not alternate) still lines its values up with its
    turns. An axis whose layer failed to capture is dropped rather than reported empty.
    """
    assistant_indices = [i for i, msg in enumerate(conversation) if msg.role == "assistant"]
    snippets = [truncate_content(conversation[i].content) for i in assistant_indices]

    readouts: list[SteerAxisReadout] = []
    for axis in axes:
        pre_means = pre_cap.get(axis.layer)
        if pre_means is None or pre_means.shape[0] == 0:
            logger.warning(f"[AXIS] no activations at layer {axis.layer}, skipping axis '{axis.id}'")
            continue
        values, percentiles = project_axis_with_percentile(pre_means, axis)

        post_means = (post_cap or {}).get(axis.layer)
        values_post_cap: np.ndarray | None = None
        percentiles_post_cap: np.ndarray | None = None
        if post_means is not None and post_means.shape[0] > 0:
            values_post_cap, percentiles_post_cap = project_axis_with_percentile(post_means, axis)

        turns: list[SteerAxisTurn] = []
        for position, index in enumerate(assistant_indices):
            if index >= len(values):
                break
            has_post = values_post_cap is not None and index < len(values_post_cap)
            turns.append(
                SteerAxisTurn(
                    value=float(values[index]),
                    value_post_cap=float(values_post_cap[index]) if has_post else None,  # type: ignore[index]
                    # None rather than 0 for an axis with no tables: absent says "this axis
                    # cannot report a percentile", where 0 would say "dead centre".
                    percentile=float(percentiles[index]) if percentiles is not None else None,
                    percentile_post_cap=(
                        float(percentiles_post_cap[index]) if has_post and percentiles_post_cap is not None else None
                    ),
                    snippet=snippets[position],
                )
            )

        readouts.append(
            SteerAxisReadout(
                id=axis.id,
                author=axis.author,
                title=axis.title,
                type=steer_type,
                layer=axis.layer,
                caveat=axis.caveat,
                pole_positive=axis.pole_positive,
                pole_negative=axis.pole_negative,
                pole_positive_description=axis.pole_positive_description,
                pole_negative_description=axis.pole_negative_description,
                source_revision=axis.source_revision,
                turns=turns,
            )
        )
    return readouts


async def run_batched_generate(
    promptTokenized: torch.Tensor,
    inputPrompt: list[NPSteerChatMessage],
    settings: SteeringSettings,
    steer_types: list[NPSteerType],
    seed: int | None = None,
    steer_special_tokens: bool = False,
    use_stream_lock: bool = False,
    axes: list[AxisAsset] | None = None,
    **kwargs: Any,
):
    async with await stream_lock(use_stream_lock):
        model = Model.get_instance()

        if seed is not None:
            torch.manual_seed(seed)

        # steer_special_tokens=False -> exclude the model's special tokens (BOS/EOS + chat
        # markers) from steering; the engine resolves the exact positions per model family
        # (see SteerMask.SPECIAL_TOKENS), replacing the old Gemma-only masking. This applies
        # to both the eager (EagerModel) and vLLM backends.
        steer_position_mask = None if steer_special_tokens else SteerMask.SPECIAL_TOKENS

        # Both backends stream STEERED/DEFAULT over the chat-templated prompt (eager via
        # forward write-hooks, vLLM via worker steering) and project the requested axes after
        # generation; they share the frame + capture helpers below.
        if isinstance(model, EagerModel):
            async for msg in _engine_chat_generate(
                model=model,
                promptTokenized=promptTokenized,
                inputPrompt=inputPrompt,
                settings=settings,
                steer_types=steer_types,
                seed=seed,
                temperature=float(kwargs.get("temperature", 1.0)),
                max_new_tokens=int(kwargs.get("max_new_tokens") or 0),
                axes=axes or [],
                position_mask=steer_position_mask,
            ):
                yield msg
            return

        if not (VLLM_AVAILABLE and isinstance(model, VLLMModel)):
            raise ValueError("The /steer/completion-chat endpoint only supports the interp-engine and vLLM backends")
        if kwargs.get("freq_penalty"):
            logger.warning("freq_penalty is not supported on the vLLM backend; ignoring")
        async for msg in _vllm_chat_generate(
            model=model,
            promptTokenized=promptTokenized,
            inputPrompt=inputPrompt,
            settings=settings,
            steer_types=steer_types,
            seed=seed,
            temperature=float(kwargs.get("temperature", 1.0)),
            max_new_tokens=int(kwargs.get("max_new_tokens") or 0),
            axes=axes or [],
            position_mask=steer_position_mask,
        ):
            yield msg


def _chat_stream_frame(
    steer_types: list[NPSteerType],
    output_by_type: dict[NPSteerType, str],
    prompt_string: str,
    model: "VLLMModel | EagerModel",
    promptTokenized: torch.Tensor,
    inputPrompt: list[NPSteerChatMessage],
) -> str:
    """Build one streaming SSE frame from the running per-type outputs.

    A type that hasn't started yet reads as empty. ``make_steer_completion_chat_response``
    only emits the entries named in ``steer_types``.
    """
    return format_sse_message(
        make_steer_completion_chat_response(
            steer_types,
            output_by_type.get(NPSteerType.STEERED, ""),
            output_by_type.get(NPSteerType.DEFAULT, ""),
            prompt_string,
            model,
            promptTokenized,
            inputPrompt,
        ).to_wire_json()
    )


def _axis_capture_points(axes: list[AxisAsset]) -> dict[int, Address]:
    """The addresses a generation should capture so the requested axes need no extra forward.

    Keyed by layer, deduplicated: axes sharing a layer share one capture. Empty when nothing
    was requested, in which case generation captures nothing at all.
    """
    return {layer: Address("resid_post", layer) for layer in sorted({axis.layer for axis in axes})}


def _pool_generation_capture(
    model: "VLLMModel | EagerModel",
    inputPrompt: list[NPSteerChatMessage],
    prompt_token_ids: list[int],
    captures: dict[Address, torch.Tensor],
    points: dict[int, Address],
    template_kwargs: dict[str, str],
) -> dict[int, torch.Tensor]:
    """Per-message means per layer from a generation-time capture, skipping anything unusable.

    A capture that came back short or misaligned is not a degraded result here -- the projection
    is indexed per message -- so that layer is left out and ``capture_axis_means`` re-captures
    it rather than pooling over misaligned positions.
    """
    pooled: dict[int, torch.Tensor] = {}
    if not points:
        return pooled
    tok = get_tokenize(model)
    for layer, point in points.items():
        acts = captures.get(point)
        if acts is None or acts.shape[0] == 0:
            continue
        try:
            means = turn_means_from_generation_capture(tok, list(inputPrompt), prompt_token_ids, acts, template_kwargs)
        except Exception:
            logger.exception(f"[AXIS] pooling the generation capture at layer {layer} failed; will re-capture")
            continue
        if means is not None:
            pooled[layer] = means
    return pooled


async def _chat_axis_frame(
    *,
    model: "VLLMModel | EagerModel",
    inputPrompt: list[NPSteerChatMessage],
    output_by_type: dict[NPSteerType, str],
    steer_types: list[NPSteerType],
    prompt_string: str,
    promptTokenized: torch.Tensor,
    axes: list[AxisAsset],
    steered_spec: Any,
    gen_means_by_type: dict[NPSteerType, dict[int, torch.Tensor]] | None = None,
) -> str:
    """Project every requested axis for every generated type, and build the final frame.

    ``steered_spec`` (engine ``list[SteerSpec]`` for EagerModel, ``SteeringSpec`` for vLLM) is
    passed only for the STEERED type so post-cap activations are captured under steering.

    ``gen_means_by_type`` holds per-layer means captured during each type's own generation. A
    STEERED generation steers, so its means are post-cap; a DEFAULT one doesn't, so its means
    are pre-cap and no further forward is needed at all.
    """
    layers = sorted({axis.layer for axis in axes})
    # Validated to agree back in the endpoint, so any axis's conditions are all of theirs.
    render, _conflict = _agreed_render_conditions(axes)
    readouts: list[SteerAxisReadout] = []
    for steer_type, output_text in output_by_type.items():
        full_conversation = list(inputPrompt) + [NPSteerChatMessage(role="assistant", content=output_text)]
        is_steered = steer_type == NPSteerType.STEERED
        gen_means = (gen_means_by_type or {}).get(steer_type) or {}
        pre_cap, post_cap = await capture_axis_means(
            model,
            full_conversation,
            layers,
            steering_spec=steered_spec if is_steered else None,
            pre_cap_means=None if is_steered else gen_means,
            post_cap_means=gen_means if is_steered else None,
            template_kwargs=render.template_kwargs,
        )
        readouts.extend(build_axis_readouts(full_conversation, steer_type, axes, pre_cap, post_cap))

    to_return = make_steer_completion_chat_response(
        steer_types,
        output_by_type.get(NPSteerType.STEERED, ""),
        output_by_type.get(NPSteerType.DEFAULT, output_by_type.get(NPSteerType.STEERED, "")),
        prompt_string,
        model,
        promptTokenized,
        inputPrompt,
        axis_readouts=readouts or None,
    )
    return format_sse_message(to_return.to_wire_json())


async def _vllm_chat_generate(
    *,
    model: VLLMModel,
    promptTokenized: torch.Tensor,
    inputPrompt: list[NPSteerChatMessage],
    settings: SteeringSettings,
    steer_types: list[NPSteerType],
    seed: int | None,
    temperature: float,
    max_new_tokens: int,
    axes: list[AxisAsset] | None = None,
    position_mask: Any = None,
):
    """SSE generator for the vLLM backend over a chat-templated prompt.

    Mirrors :func:`_engine_chat_generate` but streams via the async vLLM backend. The
    steering spec is built once (shared with ``/steer/completion`` via
    ``features_to_vllm_steering_spec``) and reused for STEERED generation and for the
    post-cap axis read. ``position_mask`` excludes prompt positions (e.g. special tokens)
    from steering.

    When axes are requested, each generation also captures their layers on its own request, so
    the activations the readouts need come out of the forwards that produced the text: a DEFAULT
    turn needs no further pass, and a STEERED one is left with only the unsteered pre-cap read
    (the cap covers those layers, so its own activations there are clipped).
    """
    axes = axes or []
    prompt_string = model.tokenizer.decode(promptTokenized)
    # Only build the spec if a pass will actually use it. A DEFAULT-only request is
    # legitimate — the webapp collapses to it when the feature list is empty, and a
    # readout-only request carries no features at all — and building the spec eagerly turns
    # that into a 500, since an empty feature list has no steering layers. The engine path
    # never had this problem: it steers under `if specs`.
    steering_spec = features_to_vllm_steering_spec(settings) if NPSteerType.STEERED in steer_types else None

    axis_points = _axis_capture_points(axes)
    # Validated to agree back in the endpoint, so any axis's conditions are all of theirs. The
    # pooling below re-renders the prompt to find its message spans, and has to render it the
    # same way the prompt it is pooling over was rendered.
    axis_render, _conflict = _agreed_render_conditions(axes)
    prompt_token_ids = [int(t) for t in promptTokenized.tolist()]

    output_by_type: dict[NPSteerType, str] = {}
    gen_means_by_type: dict[NPSteerType, dict[int, torch.Tensor]] = {}
    for flag in steer_types:
        if seed is not None:
            torch.manual_seed(seed)
        active_spec = steering_spec if flag == NPSteerType.STEERED else None
        captures: dict[Address, torch.Tensor] = {}
        # With stream=True the backend returns an async generator of text deltas; it only
        # returns the full string when stream=False. Generating from the endpoint's own
        # token ids (rather than a string the backend would re-tokenize) keeps one
        # tokenization in play, which is what lets the pooling below trust that captured
        # row i is prompt token i.
        stream_generator = cast(
            AsyncGenerator[str, None],
            await model.generate_steered(
                prompt_token_ids,
                SamplingParams(
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    seed=seed,
                    # A chat response's structure is carried by special tokens (harmony's
                    # <|channel|>/<|message|>, turn-end markers). vLLM's detokenizer drops
                    # them by default, which left the assistant turn unrecoverable on this
                    # backend; composition strips whatever the client shouldn't see.
                    skip_special_tokens=False,
                ),
                steering_spec=active_spec,
                position_mask=position_mask if active_spec is not None else None,
                stream=True,
                capture_points=list(axis_points.values()) if axis_points else None,
                capture_out=captures if axis_points else None,
            ),
        )
        text = ""
        async for delta in stream_generator:
            text += delta
            output_by_type[flag] = text
            yield _chat_stream_frame(
                steer_types,
                output_by_type,
                prompt_string,
                model,
                promptTokenized,
                inputPrompt,
            )
        output_by_type[flag] = text
        if axis_points:
            gen_means_by_type[flag] = _pool_generation_capture(
                model, inputPrompt, prompt_token_ids, captures, axis_points, axis_render.template_kwargs
            )

    if axes:
        yield await _chat_axis_frame(
            model=model,
            inputPrompt=inputPrompt,
            output_by_type=output_by_type,
            steer_types=steer_types,
            prompt_string=prompt_string,
            promptTokenized=promptTokenized,
            axes=axes,
            steered_spec=steering_spec,
            gen_means_by_type=gen_means_by_type,
        )


async def _engine_chat_generate(
    *,
    model: EagerModel,
    promptTokenized: torch.Tensor,
    inputPrompt: list[NPSteerChatMessage],
    settings: SteeringSettings,
    steer_types: list[NPSteerType],
    seed: int | None,
    temperature: float,
    max_new_tokens: int,
    axes: list[AxisAsset] | None = None,
    position_mask: Any = None,
):
    """SSE generator for the eager engine backend over a chat-templated prompt.

    Mirrors `steer/completion.py`'s STEERED/DEFAULT engine flow but emits
    `SteerCompletionChatResponse` frames. The prompt tokens come from the
    endpoint's `apply_chat_template` output (`promptTokenized`); generated text
    is prefixed with the decoded prompt to match the vLLM path's `raw` output.
    When axes are requested, projects them on the generated conversation after
    streaming (pre-cap always; post-cap under steering).
    """
    axes = axes or []
    tokens = promptTokenized.to(model.device)
    prompt_string = model.tokenizer.decode(promptTokenized)

    specs = [_feature_to_steerspec(f, settings) for f in settings.features]

    # Track each steer type's generated text so the readouts can analyze the
    # full conversation (prompt + assistant turn) afterwards.
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
            position_mask=position_mask,
        ):
            text += delta
            output_by_type[flag] = text
            yield _chat_stream_frame(
                steer_types,
                output_by_type,
                prompt_string,
                model,
                promptTokenized,
                inputPrompt,
            )
        output_by_type[flag] = text

    if axes:
        yield await _chat_axis_frame(
            model=model,
            inputPrompt=inputPrompt,
            output_by_type=output_by_type,
            steer_types=steer_types,
            prompt_string=prompt_string,
            promptTokenized=promptTokenized,
            axes=axes,
            steered_spec=specs,
        )


def make_steer_completion_chat_response(
    steer_types: list[NPSteerType],
    steered_output: str,
    default_output: str,
    prompt_string: str,
    model: "VLLMModel | EagerModel",
    promptTokenized: torch.Tensor,
    promptChat: list[NPSteerChatMessage],
    steered_logprobs: list[NPLogprob] | None = None,
    default_logprobs: list[NPLogprob] | None = None,
    axis_readouts: list[SteerAxisReadout] | None = None,
) -> SteerCompletionChatResponse:
    """Build the response from the prompt messages plus the text generated for each type.

    ``*_output`` is generation only (no prompt). The returned ``chat_template`` is composed:
    the prompt messages we rendered from, plus the assistant turns implied by the generation.
    Nothing re-parses the prompt scaffold, so this stays cheap to call per streaming frame.
    """
    output_by_type = {
        NPSteerType.STEERED: steered_output,
        NPSteerType.DEFAULT: default_output,
    }
    logprobs_by_type = {
        NPSteerType.STEERED: steered_logprobs,
        NPSteerType.DEFAULT: default_logprobs,
    }
    steerChatResults = [
        NPSteerChatResult(
            raw=prompt_string + output_by_type[steer_type],  # type: ignore
            chat_template=list(promptChat)
            + [
                NPSteerChatMessage(role=turn.role, content=turn.content)
                for turn in compose_assistant_turns(
                    output_by_type[steer_type],
                    model.tokenizer,
                    prompt=prompt_string,
                )
            ],
            type=steer_type,
            logprobs=logprobs_by_type[steer_type],
        )
        for steer_type in steer_types
    ]

    # Handle token to string conversion for both model types (vLLM + EagerModel).
    prompt_raw = model.tokenizer.decode(promptTokenized) if model.tokenizer is not None else ""

    return SteerCompletionChatResponse(
        axes=axis_readouts,
        outputs=steerChatResults,
        input=NPSteerChatResult(
            raw=prompt_raw,  # type: ignore
            chat_template=promptChat,
        ),
    )
