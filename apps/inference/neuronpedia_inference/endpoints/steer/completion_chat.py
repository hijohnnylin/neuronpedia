import logging
import os
import time
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from interp_engine import (
    InterpModel,
    SamplingSettings,
    SteeringSpec,
    SteerMask,
    compose_assistant_turns,
    strip_wire_reasoning,
)

from neuronpedia_inference.config import Config
from neuronpedia_inference.endpoints.steer.completion import (
    _generate_text,
    features_to_steering_specs,
    resolve_max_new_tokens,
    steer_write_targets,
)
from neuronpedia_inference.engine_adapter import (
    BackendUnsupported,
    assert_capture_layers_declared,
    assert_steer_layers_declared,
    assert_steering_available,
    declares_static_taps,
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
from neuronpedia_inference.inference_utils.vectors import (
    CaptureKey,
    RenderConditions,
    TokenSelection,
    VectorAsset,
    VectorRequestError,
    project_vector_with_percentile,
    resolve_request_reads,
    truncate_content,
)
from neuronpedia_inference.inference_utils.vectors.capture_engine import capture_turn_means
from neuronpedia_inference.inference_utils.vllm_monitor import get_monitor
from neuronpedia_inference.memory_cost import steer_cost
from neuronpedia_inference.schemas import (
    NPLogprob,
    NPSteerChatMessage,
    NPSteerChatResult,
    NPSteerType,
    SteerCompletionChatRequest,
    SteerCompletionChatResponse,
    SteerReadoutTurn,
    SteerVectorReadout,
)
from neuronpedia_inference.shared import Model, with_request_lock

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
    Get health stats for the loaded engine.

    Returns GPU memory usage, system RAM, active requests, threads, etc.
    Useful for debugging hanging requests.
    """
    monitor = get_monitor()
    monitor.set_model(Model.get_instance())

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
async def completion_chat(request: SteerCompletionChatRequest, http_request: Request):
    request_start = time.time()
    model = Model.get_instance()
    config = Config.get_instance()
    steer_method = request.steer_method
    normalize_steering = request.normalize_steering
    steer_special_tokens = request.steer_special_tokens

    # Start background monitoring if enabled (once).
    if ENABLE_BACKGROUND_MONITOR:
        monitor = get_monitor()
        monitor.set_model(model)
        if monitor._background_task is None:
            monitor.start_background_logging(interval=MONITOR_INTERVAL)

    # Every vector this request wants read comes with it: this server ships none, so there is
    # nothing to resolve a name against.
    reads: list[VectorAsset] = []
    if request.reads:
        try:
            reads = await resolve_request_reads(
                request.reads,
                hidden_size=int(model.d_model),
                n_layers=int(model.n_layers),
            )
        except VectorRequestError as exc:
            # Returned verbatim, and safe to: every message this can carry is a literal written in
            # `vector_request`, naming the vector and the field that was wrong. The upstream text that
            # used to be interpolated into the Hub failures -- and with it this pod's cache paths --
            # is logged there instead. Blanking the message would leave a caller who sent eight
            # reads unable to tell which one this rejected, which is the whole point of the type.
            logger.warning(f"Vector read rejected: {exc}", exc_info=True)
            return JSONResponse(content={"error": str(exc)}, status_code=exc.status_code)

        # Asked before anything is rendered or generated. A static pod's tap set is fixed when its
        # graphs are recorded, and a vector names its layer at request time, so this is the one
        # mismatch that no startup check could have caught.
        try:
            assert_capture_layers_declared(
                model,
                [vector.layer for vector in reads],
                f"Vector read {[vector.id for vector in reads]}",
            )
        except BackendUnsupported as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=400)

    # Every requested vector has to agree about how the conversation is rendered, because those
    # conditions are applied before generation and so change the text itself. There is no way to
    # render one conversation two ways in a single generation, and honouring one vector's
    # conditions while reporting another's numbers would quietly project onto a direction fitted
    # off-distribution.
    render, render_conflict = _agreed_render_conditions(reads)
    if render_conflict is not None:
        return JSONResponse(content={"error": render_conflict}, status_code=400)

    # A steered generation needs the worker's write-hooks, and a GENERATION_ONLY pod has none: on
    # that pod this would otherwise generate happily and return UNSTEERED text under a "STEERED"
    # label, since a hook that never fires reports nothing. Conditional on the request, because an
    # unsteered completion through this endpoint is exactly what such a pod is for.
    #
    # Deliberately not conditional on the reads: a readout needs capture hooks, not write hooks, so
    # an unsteered readout is exactly what a generation-only pod can serve.
    if NPSteerType.STEERED in request.types:
        try:
            assert_steering_available(model, "Steered generation")
        except BackendUnsupported as e:
            return JSONResponse(content={"error": str(e)}, status_code=400)

    # Features or vectors are what steering steers with, so they are required only when something
    # will be steered. A readout-only request (types=[DEFAULT], reads=[...]) legitimately carries
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

    # Blank a caller-supplied system prompt only when the requested reads were fitted that way —
    # their directions are meaningless against activations from a conversation rendered
    # differently. Gating on the assets (rather than on "a readout was requested") keeps us from
    # silently discarding the system prompt of a model whose reads have no such requirement.
    blank_system_prompt = bool(promptChat) and promptChat[0].role == "system" and render.blank_system_prompt

    promptChatFormatted = messages_for_render(promptChat, blank_system_prompt=blank_system_prompt)

    if model.tokenizer is None:
        raise ValueError("Tokenizer is not initialized")

    tok = model.tok
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
    # injects the current date into the system block, so a vector fitted on it pins `date_string`
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

    # The write-side twin of the vector-layer check above, and asked for the same reason: the spec
    # that writes is built inside the SSE generator, so a refusal there is a 500 mid-stream. A
    # projection cap writes wherever the vector it caps was fitted, which on this endpoint is not
    # the layer the vector reads -- the 70B pod declared 40 for the readout and was asked to write 32.
    if wants_steering and declares_static_taps(model):
        try:
            for point, layers in steer_write_targets(features).items():
                assert_steer_layers_declared(model, layers, point=point)
        except BackendUnsupported as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=400)

    # Convert promptChatFormatted to NPSteerChatMessage for the readouts, so they analyze the
    # same conversation (including the system message, blanked or not) that generation saw.
    inputPromptForReads = [NPSteerChatMessage(role=msg["role"], content=msg["content"]) for msg in promptChatFormatted]

    max_new_tokens, no_room = resolve_max_new_tokens(len(promptTokenized), int(request.n_completion_tokens))
    if no_room is not None:
        return no_room

    generation_start = time.time()

    seed = int(request.seed)
    sampling = resolve_request_sampling(model, request)
    report = sampling_report(sampling, seed)
    generator = run_batched_generate(
        promptTokenized=promptTokenized,
        inputPrompt=inputPromptForReads if reads else promptChat,
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
        steer_special_tokens=steer_special_tokens,
        use_stream_lock=request.stream if request.stream is not None else False,
        reads=reads,
    )

    if request.stream:
        stated = state_settings_once(generator, report, SteerCompletionChatResponse)

        # For streaming, wrap the generator to add timing logs
        async def timed_generator():
            chunk_count = 0
            try:
                async for item in stated:
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

        return StreamingResponse(
            stop_when_client_leaves(timed_generator(), http_request, "STEER-CHAT"),
            media_type="text/event-stream",
        )

    # For a non-streaming request, the last frame is the answer. The generators emit at least
    # one frame per steer type, empty completion included.
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
    # The stream states its settings on the first frame; the one response states them here.
    response.sampling = report
    # set exclude_none to True to omit the logprobs field when n_logprobs isn't set in the request, for backwards compatibility
    return JSONResponse(content=response.model_dump(exclude_none=True))


def _agreed_render_conditions(reads: list[VectorAsset]) -> tuple[RenderConditions, str | None]:
    """The rendering conditions shared by every requested vector, or why there are none.

    Returns the agreed conditions and ``None`` when the reads agree, or when none were requested
    and so nothing about the prompt changes. Otherwise the second element names the
    disagreement, which the endpoint turns into a 400: the conversation is rendered once and
    generated from once, so honouring one vector's conditions while reporting another's numbers
    would project onto a direction fitted off-distribution.
    """
    if not reads:
        return RenderConditions(), None

    by_conditions: dict[tuple, list[VectorAsset]] = {}
    for vector in reads:
        by_conditions.setdefault(vector.render.key(), []).append(vector)
    if len(by_conditions) == 1:
        return reads[0].render, None

    groups = "; ".join(
        f"[{', '.join(vector.id for vector in group)}] need ({group[0].render.describe()})"
        for group in by_conditions.values()
    )
    return reads[0].render, (
        "The requested vectors were fitted under different rendering conditions, so they "
        f"cannot be measured in one generation: {groups}. Request them separately."
    )


async def _capture_means(
    model: InterpModel,
    conversation: list[NPSteerChatMessage],
    keys: list[CaptureKey],
    specs: list[SteeringSpec] | None,
    template_kwargs: dict[str, str],
) -> dict[CaptureKey, torch.Tensor]:
    """One capture pass covering every key in ``keys``, steered when specs are given."""
    if not keys:
        return {}
    return await capture_turn_means(model, conversation, keys, specs=specs, template_kwargs=template_kwargs)


async def capture_read_means(
    model: InterpModel,
    conversation: list[NPSteerChatMessage],
    keys: list[CaptureKey],
    steering_specs: list[SteeringSpec] | None = None,
    template_kwargs: dict[str, str] | None = None,
) -> tuple[dict[CaptureKey, torch.Tensor], dict[CaptureKey, torch.Tensor] | None]:
    """Per-message pooled activations for every capture the requested reads need.

    One forward per condition, whatever the number of reads: the pre-cap read captures every key
    at once, and so does the post-cap read. Six vectors across five layers therefore cost what one
    costs, which is what makes a multi-vector panel affordable. Two keys at one layer are two
    poolings of one captured tensor, not two forwards.

    Args:
        model: the loaded model, on any backend
        conversation: the full conversation, including the generated assistant turn
        keys: the distinct captures the requested reads need
        steering_specs: steering for the post-cap read, one spec per point written. None means
            no post-cap read is wanted.
        template_kwargs: what the endpoint rendered the generation prompt with. A re-capture
            renders the conversation again, so anything the requested reads pinned about the
            template has to be pinned the same way here or the two renderings diverge.

    Returns:
        ``(pre_cap, post_cap)``, each keyed by capture. ``post_cap`` is None when there is none.
    """
    capture_start = time.time()
    wanted = sorted(set(keys))
    kwargs = template_kwargs or {}

    pre_cap = await _capture_means(model, conversation, wanted, None, kwargs)
    post_cap = await _capture_means(model, conversation, wanted, steering_specs, kwargs) if steering_specs else {}

    logger.debug(
        f"[READ] captured {_describe_keys(pre_cap)} pre-cap / {_describe_keys(post_cap)} post-cap "
        f"in {time.time() - capture_start:.3f}s"
    )
    return pre_cap, post_cap or None


def _describe_key(key: CaptureKey) -> str:
    """One capture as ``resid_post:12/mean``, for a log line that says what was actually read."""
    return f"{key.point}:{key.layer}/{key.pool}"


def _describe_keys(captures: dict[CaptureKey, torch.Tensor]) -> str:
    return ", ".join(sorted(_describe_key(key) for key in captures)) or "nothing"


def _selected_indices(conversation: list[NPSteerChatMessage], tokens: TokenSelection) -> list[int]:
    """Which of a conversation's messages get a reading.

    Assistant turns are located by role rather than by position, so a conversation carrying a
    system message (or one whose turns do not alternate) still lines its values up with its turns.

    Exhaustive over ``TokenSelection`` on purpose: a member added to that alias without a branch
    here raises rather than falling back to the assistant turns, which would report a reading of
    something other than what was asked for.
    """
    if tokens == "assistant_turns":
        return [i for i, msg in enumerate(conversation) if msg.role == "assistant"]
    if tokens == "all_turns":
        return list(range(len(conversation)))
    raise ValueError(f"token selection {tokens!r} is not implemented")


def build_readouts(
    conversation: list[NPSteerChatMessage],
    steer_type: NPSteerType,
    reads: list[VectorAsset],
    pre_cap: dict[CaptureKey, torch.Tensor],
    post_cap: dict[CaptureKey, torch.Tensor] | None,
) -> list[SteerVectorReadout]:
    """Project the captured means onto each vector, one readout per vector.

    Each vector reads the capture its own spec names and reports the messages its own selection
    names, so two reads in one response may differ in both. A vector whose capture failed is dropped
    rather than reported empty.
    """
    readouts: list[SteerVectorReadout] = []
    for vector in reads:
        pre_means = pre_cap.get(vector.capture_key)
        if pre_means is None or pre_means.shape[0] == 0:
            logger.warning(
                f"[READ] no activations for {_describe_key(vector.capture_key)}, skipping vector '{vector.id}'"
            )
            continue
        values, percentiles = project_vector_with_percentile(pre_means, vector)

        post_means = (post_cap or {}).get(vector.capture_key)
        values_post_cap: np.ndarray | None = None
        percentiles_post_cap: np.ndarray | None = None
        if post_means is not None and post_means.shape[0] > 0:
            values_post_cap, percentiles_post_cap = project_vector_with_percentile(post_means, vector)

        selected = _selected_indices(conversation, vector.read.tokens)
        snippets = [truncate_content(conversation[i].content) for i in selected]

        turns: list[SteerReadoutTurn] = []
        for position, index in enumerate(selected):
            if index >= len(values):
                break
            has_post = values_post_cap is not None and index < len(values_post_cap)
            turns.append(
                SteerReadoutTurn(
                    value=float(values[index]),
                    value_post_cap=float(values_post_cap[index]) if has_post else None,  # type: ignore[index]
                    # None rather than 0 for a vector with no tables: absent says "this vector
                    # cannot report a percentile", where 0 would say "dead centre".
                    percentile=float(percentiles[index]) if percentiles is not None else None,
                    percentile_post_cap=(
                        float(percentiles_post_cap[index]) if has_post and percentiles_post_cap is not None else None
                    ),
                    snippet=snippets[position],
                )
            )

        readouts.append(
            SteerVectorReadout(
                id=vector.id,
                author=vector.author,
                title=vector.title,
                type=steer_type,
                layer=vector.layer,
                caveat=vector.caveat,
                pole_positive=vector.pole_positive,
                pole_negative=vector.pole_negative,
                pole_positive_description=vector.pole_positive_description,
                pole_negative_description=vector.pole_negative_description,
                source_revision=vector.source_revision,
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
    reads: list[VectorAsset] | None = None,
    **kwargs: Any,
):
    async with await stream_lock(use_stream_lock):
        model = Model.get_instance()

        # steer_special_tokens=False -> exclude the model's special tokens (BOS/EOS + chat
        # markers) from steering; the engine resolves the exact positions per model family
        # (see SteerMask.SPECIAL_TOKENS). A backend that cannot carry a mask refuses it.
        steer_position_mask = None if steer_special_tokens else SteerMask.SPECIAL_TOKENS

        async for msg in _run_chat_generate(
            model=model,
            promptTokenized=promptTokenized,
            inputPrompt=inputPrompt,
            settings=settings,
            steer_types=steer_types,
            seed=seed,
            sampling=kwargs["sampling"],
            max_new_tokens=int(kwargs.get("max_new_tokens") or 0),
            reads=reads or [],
            position_mask=steer_position_mask,
        ):
            yield msg


def _chat_stream_frame(
    steer_types: list[NPSteerType],
    output_by_type: dict[NPSteerType, str],
    prompt_string: str,
    model: InterpModel,
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


async def _chat_readout_frame(
    *,
    model: InterpModel,
    inputPrompt: list[NPSteerChatMessage],
    output_by_type: dict[NPSteerType, str],
    steer_types: list[NPSteerType],
    prompt_string: str,
    promptTokenized: torch.Tensor,
    reads: list[VectorAsset],
    steered_specs: list[SteeringSpec] | None,
) -> str:
    """Project every requested vector for every generated type, and build the final frame.

    ``steered_specs`` is passed only for the STEERED type, so post-cap activations are captured
    under the same steering the text was generated under.
    """
    keys = sorted({vector.capture_key for vector in reads})
    # Validated to agree back in the endpoint, so any vector's conditions are all of theirs.
    render, _conflict = _agreed_render_conditions(reads)
    readouts: list[SteerVectorReadout] = []
    for steer_type, output_text in output_by_type.items():
        full_conversation = list(inputPrompt) + [NPSteerChatMessage(role="assistant", content=output_text)]
        is_steered = steer_type == NPSteerType.STEERED
        pre_cap, post_cap = await capture_read_means(
            model,
            full_conversation,
            keys,
            steering_specs=steered_specs if is_steered else None,
            template_kwargs=render.template_kwargs,
        )
        readouts.extend(build_readouts(full_conversation, steer_type, reads, pre_cap, post_cap))

    to_return = make_steer_completion_chat_response(
        steer_types,
        output_by_type.get(NPSteerType.STEERED, ""),
        output_by_type.get(NPSteerType.DEFAULT, output_by_type.get(NPSteerType.STEERED, "")),
        prompt_string,
        model,
        promptTokenized,
        inputPrompt,
        readouts=readouts or None,
    )
    return format_sse_message(to_return.to_wire_json())


async def _run_chat_generate(
    *,
    model: InterpModel,
    promptTokenized: torch.Tensor,
    inputPrompt: list[NPSteerChatMessage],
    settings: SteeringSettings,
    steer_types: list[NPSteerType],
    seed: int | None,
    sampling: SamplingSettings,
    max_new_tokens: int,
    reads: list[VectorAsset] | None = None,
    position_mask: Any = None,
):
    """SSE generator over a chat-templated prompt, on whichever backend is loaded.

    Mirrors `steer/completion.py`'s STEERED/DEFAULT flow but emits `SteerCompletionChatResponse`
    frames. The prompt tokens come from the endpoint's `apply_chat_template` output
    (`promptTokenized`) and go to the backend as ids, so one tokenization is in play. The
    steering specs are built once (shared with ``/steer/completion``) and reused for STEERED
    generation and for the post-cap vector read. ``position_mask`` excludes prompt positions
    (e.g. special tokens) from steering.

    When reads are requested, projects them on the generated conversation after streaming
    (pre-cap always; post-cap under steering).
    """
    reads = reads or []
    prompt_token_ids = [int(t) for t in promptTokenized.tolist()]
    prompt_string = model.tokenizer.decode(promptTokenized)
    # Only build the specs if a pass will actually use them. A DEFAULT-only request is
    # legitimate -- the webapp collapses to it when the feature list is empty, and a
    # readout-only request carries no features at all -- and building the specs eagerly turns
    # that into a 500, since an empty feature list has no steering layers.
    specs = features_to_steering_specs(settings) if NPSteerType.STEERED in steer_types else None

    # Track each steer type's generated text so the readouts can analyze the
    # full conversation (prompt + assistant turn) afterwards.
    output_by_type: dict[NPSteerType, str] = {}
    for flag in steer_types:
        active_specs = specs if flag == NPSteerType.STEERED else None
        text = ""
        async for delta in _generate_text(
            model,
            prompt_token_ids,
            active_specs,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
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
        if not text:
            # No delta arrives when the model samples EOS first. One frame per type, even for an
            # empty completion: a stream with no frame is a 500 downstream.
            yield _chat_stream_frame(
                steer_types,
                output_by_type,
                prompt_string,
                model,
                promptTokenized,
                inputPrompt,
            )

    if reads:
        yield await _chat_readout_frame(
            model=model,
            inputPrompt=inputPrompt,
            output_by_type=output_by_type,
            steer_types=steer_types,
            prompt_string=prompt_string,
            promptTokenized=promptTokenized,
            reads=reads,
            steered_specs=specs,
        )


def make_steer_completion_chat_response(
    steer_types: list[NPSteerType],
    steered_output: str,
    default_output: str,
    prompt_string: str,
    model: InterpModel,
    promptTokenized: torch.Tensor,
    promptChat: list[NPSteerChatMessage],
    steered_logprobs: list[NPLogprob] | None = None,
    default_logprobs: list[NPLogprob] | None = None,
    readouts: list[SteerVectorReadout] | None = None,
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

    prompt_raw = model.tokenizer.decode(promptTokenized) if model.tokenizer is not None else ""

    return SteerCompletionChatResponse(
        readouts=readouts,
        outputs=steerChatResults,
        input=NPSteerChatResult(
            raw=prompt_raw,  # type: ignore
            chat_template=promptChat,
        ),
    )
