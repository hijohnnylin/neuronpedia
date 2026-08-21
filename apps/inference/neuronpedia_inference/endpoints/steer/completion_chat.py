import logging
import os
import time
from collections.abc import AsyncGenerator
from typing import Any, cast

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
    PersonaData,
    _truncate_content,
    pc_projection,
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
    SteerAssistantAxis,
    SteerAssistantAxisTurn,
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

    # if is_assistant_axis is true, then we also send the persona monitor results, and add a system prompt for short responses
    is_assistant_axis = request.is_assistant_axis if request.is_assistant_axis is not None else False

    if is_assistant_axis:
        vllm_backend = VLLM_AVAILABLE and isinstance(model, VLLMModel)
        if not (vllm_backend or isinstance(model, EagerModel)):
            return JSONResponse(
                content={"error": "Assistant axis requires the vLLM or interp-engine (EagerModel) backend"},
                status_code=400,
            )

    # A steered generation needs the worker's write-hooks, and a GENERATION_ONLY pod has none: on
    # that pod this would otherwise generate happily and return UNSTEERED text under a "STEERED"
    # label, since a hook that never fires reports nothing. Conditional on the request, because an
    # unsteered completion through this endpoint is exactly what such a pod is for.
    if NPSteerType.STEERED in request.types or is_assistant_axis:
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

    promptChat = request.prompt

    # Blank a caller-supplied system prompt only when the loaded persona asset was
    # fitted that way — its PCs are meaningless against activations from a conversation
    # rendered differently. Gating on the asset (rather than on `is_assistant_axis`)
    # keeps us from silently discarding the system prompt of a model whose asset has no
    # such requirement, or of a request that never reaches persona monitoring.
    blank_system_prompt = (
        is_assistant_axis
        and bool(promptChat)
        and promptChat[0].role == "system"
        and PersonaData.get_instance().fit.blank_system_prompt
    )

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
    rendered_prompt = tok.apply_chat_template(promptChatFormatted, tokenize=False, add_generation_prompt=True)
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
    else:
        return JSONResponse(
            content={"error": "No features or vectors provided"},
            status_code=400,
        )

    # Convert promptChatFormatted to NPSteerChatMessage for persona monitor
    # This ensures persona monitor analyzes the same conversation (including system message) as generation
    inputPromptForPersona = [
        NPSteerChatMessage(role=msg["role"], content=msg["content"]) for msg in promptChatFormatted
    ]

    max_new_tokens, no_room = resolve_max_new_tokens(len(promptTokenized), int(request.n_completion_tokens))
    if no_room is not None:
        return no_room

    generation_start = time.time()

    generator = run_batched_generate(
        promptTokenized=promptTokenized,
        inputPrompt=inputPromptForPersona if is_assistant_axis else promptChat,
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
        is_assistant_axis=is_assistant_axis,
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


async def run_persona_monitor(
    model: Any,
    conversation: list[NPSteerChatMessage],
    steer_type: NPSteerType,
    layer: int | None = None,
    steering_spec: Any = None,
    pre_cap_means: torch.Tensor | None = None,
    post_cap_means: torch.Tensor | None = None,
) -> SteerAssistantAxis | None:
    """
    Run persona monitoring on the conversation and return assistant_axis data.

    This extracts activations and projects them onto pre-computed principal components
    that capture persona-related variation in the model's representations.

    Args:
        model: The loaded model (VLLMSteerModel or EagerModel)
        conversation: List of chat messages (user/assistant turns)
        steer_type: The steer type this analysis corresponds to
        layer: Layer to extract activations from. Defaults to the layer the loaded
            persona asset was fitted at, which is the only layer its PCs are valid for.
        steering_spec: Optional steering to apply during capture (vLLM
            SteeringSpec for vLLM, list[SteerSpec] for the engine). If provided,
            both pre-cap (base model) and post-cap (with steering) activations are captured.
        pre_cap_means / post_cap_means: Per-message means ``[len(conversation), hidden]``
            already captured during generation. Each one supplied skips the corresponding
            capture here. An unsteered generation supplies pre-cap; a steered one supplies
            post-cap, and still needs the pre-cap capture below because the cap covers the
            layer this projects at.

    Returns:
        AssistantAxis response data, or None if persona data not available
    """
    persona_start = time.time()

    # Get pre-loaded PCA data
    logger.debug("[PERSONA] Getting PersonaData instance...")
    persona_data = PersonaData.get_instance()
    if not persona_data.is_initialized():
        logger.warning("Persona data not initialized, skipping persona monitor")
        return None

    if layer is None:
        layer = persona_data.primary_layer
    if layer is None:
        logger.warning("No persona PCA layer loaded, skipping persona monitor")
        return None
    logger.debug(
        f"[PERSONA] run_persona_monitor called for steer_type={steer_type}, layer={layer}, has_steering_spec={steering_spec is not None}"
    )

    pca_results = persona_data.get_pca_data(layer)
    if pca_results is None:
        logger.warning(f"PCA data not available for layer {layer}")
        return None
    logger.debug(f"[PERSONA] PCA data loaded in {time.time() - persona_start:.3f}s")

    # Extract mean activations per message (pre-cap / base model), plus post-cap
    # under steering when a spec is provided. Backend-specific capture only; the
    # projection below is backend-agnostic.
    extract_start = time.time()
    needs_post_cap = steering_spec is not None and post_cap_means is None
    if pre_cap_means is not None and not needs_post_cap:
        # Everything this projection needs came out of the generation's own forwards.
        mean_acts_per_turn = pre_cap_means
        mean_acts_per_turn_post_cap = post_cap_means
    elif isinstance(model, EagerModel):
        # Eager backend: capture resid_post per message directly. steering_spec is
        # a list[SteerSpec] here (engine specs), not a vendored steerllm SteeringSpec.
        mean_acts_per_turn = (
            pre_cap_means if pre_cap_means is not None else capture_turn_means_engine(model, conversation, layer)
        )
        mean_acts_per_turn_post_cap = (
            capture_turn_means_engine(model, conversation, layer, specs=steering_spec)
            if needs_post_cap
            else post_cap_means
        )
    elif isinstance(model, VLLMModel):
        # vLLM backend: native resid capture + per-message pooling. steering_spec is
        # an engine SteeringSpec here (built from Add/ProjectionCap specs).
        mean_acts_per_turn = (
            pre_cap_means if pre_cap_means is not None else await capture_turn_means_vllm(model, conversation, layer)
        )
        mean_acts_per_turn_post_cap = (
            await capture_turn_means_vllm(model, conversation, layer, steering_spec=steering_spec)
            if needs_post_cap
            else post_cap_means
        )
    else:
        raise ValueError(f"persona monitor unsupported for backend {type(model).__name__}")
    logger.debug(
        f"[PERSONA] Activations extracted in {time.time() - extract_start:.3f}s, shape={mean_acts_per_turn.shape}"
    )

    # Handle empty activations
    if mean_acts_per_turn.shape[0] == 0:
        logger.warning("No activations extracted, skipping persona monitor")
        return None

    # How many PCs to project onto is the same fact as how many are labelled, so it comes
    # from the loaded asset's manifest rather than being fixed here. An asset that fits
    # more components than it labels surfaces only the labelled ones.
    pc_titles = persona_data.pc_titles
    n_pcs = len(pc_titles)

    # Compute projections (pre-cap)
    role_projs = pc_projection(mean_acts_per_turn, pca_results, n_pcs=n_pcs)

    # Compute projections (post-cap) if available
    role_projs_post_cap = None
    if mean_acts_per_turn_post_cap is not None and mean_acts_per_turn_post_cap.shape[0] > 0:
        role_projs_post_cap = pc_projection(mean_acts_per_turn_post_cap, pca_results, n_pcs=n_pcs)

    # Find indices of assistant turns in the conversation (by actual role, not position assumption)
    # This handles conversations with system messages where indices don't alternate user/assistant
    assistant_indices = [i for i, msg in enumerate(conversation) if msg.role == "assistant"]

    # Select projections for assistant turns only
    assistant_role_projs = role_projs[assistant_indices] if assistant_indices else role_projs[0:0]
    assistant_role_projs_post_cap = None
    if role_projs_post_cap is not None:
        assistant_role_projs_post_cap = (
            role_projs_post_cap[assistant_indices] if assistant_indices else role_projs_post_cap[0:0]
        )

    # Get assistant turns for snippets
    assistant_turns = [msg for msg in conversation if msg.role == "assistant"]

    turns_data = []
    for i in range(len(assistant_role_projs)):
        pc_values = {pc_titles[j]: float(assistant_role_projs[i][j]) for j in range(len(pc_titles))}

        # Add post-cap values if available
        pc_values_post_cap = None
        if assistant_role_projs_post_cap is not None and i < len(assistant_role_projs_post_cap):
            pc_values_post_cap = {
                pc_titles[j]: float(assistant_role_projs_post_cap[i][j]) for j in range(len(pc_titles))
            }

        snippet = ""
        if i < len(assistant_turns):
            snippet = _truncate_content(assistant_turns[i].content)

        turns_data.append(
            SteerAssistantAxisTurn(
                pc_values=pc_values,
                pc_values_post_cap=pc_values_post_cap,
                snippet=snippet,
            )
        )

    logger.debug(f"[PERSONA] Complete in {time.time() - persona_start:.3f}s, {len(turns_data)} assistant turns")
    return SteerAssistantAxis(type=steer_type, pc_titles=pc_titles, turns=turns_data)


async def run_batched_generate(
    promptTokenized: torch.Tensor,
    inputPrompt: list[NPSteerChatMessage],
    settings: SteeringSettings,
    steer_types: list[NPSteerType],
    seed: int | None = None,
    steer_special_tokens: bool = False,
    use_stream_lock: bool = False,
    is_assistant_axis: bool = False,
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
        # forward write-hooks, vLLM via worker steering) and run persona monitoring after
        # generation when is_assistant_axis; they share the frame + persona helpers below.
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
                is_assistant_axis=is_assistant_axis,
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
            is_assistant_axis=is_assistant_axis,
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


def _persona_capture_point() -> Address | None:
    """The address a generation should capture for persona monitoring.

    None when no persona asset is loaded, in which case generation captures nothing and
    ``run_persona_monitor`` bails out for the same reason.
    """
    persona_data = PersonaData.get_instance()
    if not persona_data.is_initialized() or persona_data.primary_layer is None:
        return None
    return Address("resid_post", int(persona_data.primary_layer))


def _pool_generation_capture(
    model: "VLLMModel | EagerModel",
    inputPrompt: list[NPSteerChatMessage],
    prompt_token_ids: list[int],
    acts: torch.Tensor | None,
) -> torch.Tensor | None:
    """Per-message means from a generation-time capture, or None to fall back.

    A capture that came back short or misaligned is not a degraded result here -- the
    projection is indexed per message -- so anything unexpected returns None and
    ``run_persona_monitor`` re-captures instead.
    """
    if acts is None or acts.shape[0] == 0:
        return None
    try:
        return turn_means_from_generation_capture(get_tokenize(model), list(inputPrompt), prompt_token_ids, acts)
    except Exception:
        logger.exception("[PERSONA] pooling the generation capture failed; falling back to re-capture")
        return None


async def _chat_persona_axis_frame(
    *,
    model: "VLLMModel | EagerModel",
    inputPrompt: list[NPSteerChatMessage],
    output_by_type: dict[NPSteerType, str],
    steer_types: list[NPSteerType],
    prompt_string: str,
    promptTokenized: torch.Tensor,
    steered_spec: Any,
    gen_means_by_type: dict[NPSteerType, torch.Tensor] | None = None,
) -> str:
    """Run persona monitoring per steer type and build the final assistant-axis frame.

    ``steered_spec`` (engine ``list[SteerSpec]`` for EagerModel, ``SteeringSpec`` for vLLM) is
    passed only for the STEERED type so post-cap activations are captured under steering.

    ``gen_means_by_type`` holds per-message means captured during each type's own
    generation. A STEERED generation steers, so its means are post-cap; a DEFAULT one
    doesn't, so its means are pre-cap and no further forward is needed at all.
    """
    assistant_axis_data_list: list[SteerAssistantAxis] = []
    for steer_type, output_text in output_by_type.items():
        full_conversation = list(inputPrompt) + [NPSteerChatMessage(role="assistant", content=output_text)]
        is_steered = steer_type == NPSteerType.STEERED
        gen_means = (gen_means_by_type or {}).get(steer_type)
        axis_data = await run_persona_monitor(
            model,
            full_conversation,
            steer_type,
            steering_spec=steered_spec if is_steered else None,
            pre_cap_means=None if is_steered else gen_means,
            post_cap_means=gen_means if is_steered else None,
        )
        if axis_data is not None:
            assistant_axis_data_list.append(axis_data)

    to_return = make_steer_completion_chat_response(
        steer_types,
        output_by_type.get(NPSteerType.STEERED, ""),
        output_by_type.get(NPSteerType.DEFAULT, output_by_type.get(NPSteerType.STEERED, "")),
        prompt_string,
        model,
        promptTokenized,
        inputPrompt,
        assistant_axis_data=assistant_axis_data_list if assistant_axis_data_list else None,
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
    is_assistant_axis: bool = False,
    position_mask: Any = None,
):
    """SSE generator for the vLLM backend over a chat-templated prompt.

    Mirrors :func:`_engine_chat_generate` but streams via the async vLLM backend. The
    steering spec is built once (shared with ``/steer/completion`` via
    ``features_to_vllm_steering_spec``) and reused for STEERED generation and, when
    ``is_assistant_axis`` is set, for post-cap persona monitoring. ``position_mask`` excludes
    prompt positions (e.g. special tokens) from steering.

    Under ``is_assistant_axis`` each generation also captures the persona layer on its own
    request, so the activations persona monitoring needs come out of the forwards that
    produced the text: a DEFAULT turn needs no further pass, and a STEERED one is left
    with only the unsteered pre-cap read (the cap covers the persona layer, so its own
    activations there are clipped).
    """
    prompt_string = model.tokenizer.decode(promptTokenized)
    # Only build the spec if a pass will actually use it. A DEFAULT-only request is
    # legitimate — the webapp collapses to it when the feature list is empty — and
    # building the spec eagerly turns that into a 500, since an empty feature list has
    # no steering layers. The engine path never had this problem: it steers under
    # `if specs`.
    steering_spec = (
        features_to_vllm_steering_spec(settings) if NPSteerType.STEERED in steer_types or is_assistant_axis else None
    )

    persona_point = _persona_capture_point() if is_assistant_axis else None
    prompt_token_ids = [int(t) for t in promptTokenized.tolist()]

    output_by_type: dict[NPSteerType, str] = {}
    gen_means_by_type: dict[NPSteerType, torch.Tensor] = {}
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
                capture_points=[persona_point] if persona_point else None,
                capture_out=captures if persona_point else None,
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
        if persona_point is not None:
            means = _pool_generation_capture(model, inputPrompt, prompt_token_ids, captures.get(persona_point))
            if means is not None:
                gen_means_by_type[flag] = means

    if is_assistant_axis:
        yield await _chat_persona_axis_frame(
            model=model,
            inputPrompt=inputPrompt,
            output_by_type=output_by_type,
            steer_types=steer_types,
            prompt_string=prompt_string,
            promptTokenized=promptTokenized,
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
    is_assistant_axis: bool = False,
    position_mask: Any = None,
):
    """SSE generator for the eager engine backend over a chat-templated prompt.

    Mirrors `steer/completion.py`'s STEERED/DEFAULT engine flow but emits
    `SteerCompletionChatResponse` frames. The prompt tokens come from the
    endpoint's `apply_chat_template` output (`promptTokenized`); generated text
    is prefixed with the decoded prompt to match the vLLM path's `raw` output.
    When ``is_assistant_axis`` is set, runs persona monitoring on the generated
    conversation after streaming (pre-cap always; post-cap under steering).
    """
    tokens = promptTokenized.to(model.device)
    prompt_string = model.tokenizer.decode(promptTokenized)

    specs = [_feature_to_steerspec(f, settings) for f in settings.features]

    # Track each steer type's generated text so persona monitoring can analyze the
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

    if is_assistant_axis:
        yield await _chat_persona_axis_frame(
            model=model,
            inputPrompt=inputPrompt,
            output_by_type=output_by_type,
            steer_types=steer_types,
            prompt_string=prompt_string,
            promptTokenized=promptTokenized,
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
    assistant_axis_data: list[SteerAssistantAxis] | None = None,
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
        assistant_axis=assistant_axis_data,
        outputs=steerChatResults,
        input=NPSteerChatResult(
            raw=prompt_raw,  # type: ignore
            chat_template=promptChat,
        ),
    )
