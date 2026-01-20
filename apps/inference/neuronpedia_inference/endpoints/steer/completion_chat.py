import asyncio
import logging
import os
import threading
import time
from typing import Any

import torch
from transformers import TextIteratorStreamer
from chatspace.generation import VLLMSteerModel
from chatspace.generation.vllm_steer_model import (
    AddSpec,
    LayerSteeringSpec,
    ProjectionCapSpec,
    SteeringOp,
    SteeringSpec,
)
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from neuronpedia_inference.config import Config
from neuronpedia_inference.inference_utils.steering import (
    OrthogonalProjector,
    apply_generic_chat_template,
    convert_to_chat_array,
    format_sse_message,
    process_features_vectorized,
    remove_sse_formatting,
    stream_lock,
)
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.shared import Model, with_request_lock, is_nnterp_model
from neuronpedia_inference.utils import make_logprob_from_logits
from neuronpedia_inference_client.models.np_logprob import NPLogprob
from neuronpedia_inference_client.models.np_steer_chat_message import NPSteerChatMessage
from neuronpedia_inference_client.models.np_steer_chat_result import NPSteerChatResult
from neuronpedia_inference_client.models.np_steer_feature import NPSteerFeature
from neuronpedia_inference_client.models.np_steer_method import NPSteerMethod
from neuronpedia_inference_client.models.np_steer_type import NPSteerType
from neuronpedia_inference_client.models.np_steer_vector import NPSteerVector
from neuronpedia_inference_client.models.steer_completion_chat_post200_response import (
    SteerCompletionChatPost200Response,
)
from neuronpedia_inference_client.models.steer_completion_chat_post200_response_assistant_axis_inner import (
    SteerCompletionChatPost200ResponseAssistantAxisInner,
)
from neuronpedia_inference_client.models.steer_completion_chat_post200_response_assistant_axis_inner_turns_inner import (
    SteerCompletionChatPost200ResponseAssistantAxisInnerTurnsInner,
)
from neuronpedia_inference_client.models.steer_completion_chat_post_request import (
    SteerCompletionChatPostRequest,
)
from nnterp import StandardizedTransformer
from transformer_lens import HookedTransformer
from vllm import SamplingParams

from neuronpedia_inference.config import Config
from neuronpedia_inference.inference_utils.steering import (
    OrthogonalProjector,
    apply_generic_chat_template,
    convert_to_chat_array,
    format_sse_message,
    process_features_vectorized,
    remove_sse_formatting,
    stream_lock,
    _strip_llama3_system_preamble,
)
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.shared import Model, with_request_lock, is_nnterp_model
from neuronpedia_inference.utils import make_logprob_from_logits
from neuronpedia_inference.endpoints.persona.monitor import pc_projection, _truncate_content
from neuronpedia_inference.endpoints.persona.utils import (
    ConversationEncoder,
    ProbingModelChatSpace,
    SpanMapperChatSpace,
    PersonaData,
    DEFAULT_LAYER,
    ROLE_PC_TITLES,
)
from neuronpedia_inference.inference_utils.vllm_monitor import (
    VLLMMonitor,
    get_monitor,
    get_health_stats,
)

logger = logging.getLogger(__name__)

# Enable background health monitoring if env var is set
ENABLE_BACKGROUND_MONITOR = os.environ.get("ENABLE_VLLM_MONITOR", "0") == "1"
MONITOR_INTERVAL = float(os.environ.get("VLLM_MONITOR_INTERVAL", "30"))


async def stream_nnsight_generation(
    model: Any,
    prompt_string: str,
    features: list,
    sae_manager: Any,
    steer_type: NPSteerType,
    strength_multiplier: float,
    normalize_steering: bool,
    steer_method: NPSteerMethod,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
    do_sample: bool = True,
):
    """
    Stream tokens from nnsight model using TextIteratorStreamer.
    
    This runs generation in a separate thread and yields tokens as they're generated.
    Interventions (steering) are applied inside the generation context.
    """
    # Create a TextIteratorStreamer for streaming tokens
    # Note: skip_special_tokens=False to preserve special tokens like <|channel|>, <|message|>, etc.
    streamer = TextIteratorStreamer(
        model.tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
    )
    
    generation_complete = threading.Event()
    generation_error: list[Exception] = []
    
    def run_generation():
        """Run the generation with interventions in a separate thread."""
        try:
            with model.generate(
                prompt_string,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                streamer=streamer,
            ) as tracer:
                # Apply interventions if steering is requested
                if steer_type == NPSteerType.STEERED and features:
                    with tracer.all():
                        # Sort features by layer number for nnsight (must be accessed in order)
                        sorted_features = sorted(
                            features,
                            key=lambda f: _get_feature_layer_for_nnsight(f, sae_manager),
                        )
                        for feature in sorted_features:
                            # get layer number
                            hook_name = (
                                sae_manager.get_sae_hook(feature.source)
                                if isinstance(feature, NPSteerFeature)
                                else feature.hook
                            )
                            if "resid_post" in hook_name:
                                layer = int(hook_name.split(".")[1])
                            elif "resid_pre" in hook_name:
                                layer = int(hook_name.split(".")[1]) - 1
                            else:
                                raise ValueError(
                                    f"Unsupported hook name for nnsight: {hook_name}"
                                )

                            steering_vector = torch.tensor(feature.steering_vector).to(
                                model.device
                            )

                            if not torch.isfinite(steering_vector).all():
                                raise ValueError(
                                    "Steering vector contains inf or nan values"
                                )

                            if normalize_steering:
                                norm = torch.norm(steering_vector)
                                if norm == 0:
                                    raise ValueError("Zero norm steering vector")
                                steering_vector = steering_vector / norm

                            coeff = strength_multiplier * feature.strength

                            if steer_method == NPSteerMethod.SIMPLE_ADDITIVE:
                                model.layers_output[layer - 1] += (
                                    coeff
                                    * steering_vector.to(model.layers_output[layer - 1].device)
                                )
                            elif steer_method == NPSteerMethod.ORTHOGONAL_DECOMP:
                                projector = OrthogonalProjector(
                                    steering_vector.to(model.layers_output[layer - 1].device)
                                )
                                model.layers_output[layer - 1] = projector.project(
                                    model.layers_output[layer - 1], coeff
                                )
        except Exception as e:
            generation_error.append(e)
            logger.exception("Error in nnsight generation thread")
        finally:
            generation_complete.set()
    
    # Start generation in background thread
    thread = threading.Thread(target=run_generation, daemon=True)
    thread.start()
    
    # Yield tokens as they're generated
    try:
        for text in streamer:
            if text:
                yield text
            # Allow other async tasks to run
            await asyncio.sleep(0)
    except Exception as e:
        logger.exception("Error iterating over streamer")
        raise
    
    # Wait for generation to complete
    thread.join(timeout=300)  # 5 minute timeout
    
    if generation_error:
        raise generation_error[0]

ASSISTANT_AXIS_ALLOWED_MODELS = ["meta-llama/Meta-Llama-3.3-70B-Instruct"]

def _get_feature_layer_for_nnsight(
    feature: NPSteerFeature | NPSteerVector, sae_manager: SAEManager
) -> int:
    """Get the layer number for sorting features in nnsight (must be accessed in order)."""
    hook_name = (
        sae_manager.get_sae_hook(feature.source)
        if isinstance(feature, NPSteerFeature)
        else feature.hook
    )
    # Extract layer from hook_name like "blocks.0.hook_resid_post"
    return int(hook_name.split(".")[1])


router = APIRouter()

TOKENS_PER_YIELD = 1


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
    if isinstance(model, VLLMSteerModel):
        monitor.set_model(model)
    
    stats = await monitor.get_stats()
    return JSONResponse(content={
        "stats": stats.to_dict(),
        "summary": stats.summary(),
    })

def _get_feature_layer_for_nnsight(
    feature: NPSteerFeature | NPSteerVector, sae_manager: SAEManager
) -> int:
    """Get the layer number for sorting features in nnsight (must be accessed in order)."""
    hook_name = (
        sae_manager.get_sae_hook(feature.source)
        if isinstance(feature, NPSteerFeature)
        else feature.hook
    )
    # Extract layer from hook_name like "blocks.0.hook_resid_post"
    return int(hook_name.split(".")[1])


def _get_feature_layer_for_nnsight(
    feature: NPSteerFeature | NPSteerVector, sae_manager: SAEManager
) -> int:
    """Get the layer number for sorting features in nnsight (must be accessed in order)."""
    hook_name = (
        sae_manager.get_sae_hook(feature.source)
        if isinstance(feature, NPSteerFeature)
        else feature.hook
    )
    # Extract layer from hook_name like "blocks.0.hook_resid_post"
    return int(hook_name.split(".")[1])


@router.post("/steer/completion-chat")
@with_request_lock()
async def completion_chat(request: SteerCompletionChatPostRequest):
    request_start = time.time()
    model = Model.get_instance()
    config = Config.get_instance()
    steer_method = request.steer_method
    normalize_steering = request.normalize_steering
    steer_special_tokens = request.steer_special_tokens
    custom_hf_model_id = config.custom_hf_model_id

    # Start background monitoring if enabled (once) for VLLMSteerModel
    if ENABLE_BACKGROUND_MONITOR and isinstance(model, VLLMSteerModel):
        monitor = get_monitor()
        monitor.set_model(model)
        if monitor._background_task is None:
            monitor.start_background_logging(interval=MONITOR_INTERVAL)

    # if is_assistant_axis is true, then we also send the persona monitor results, and add a system prompt for short responses
    is_assistant_axis = request.is_assistant_axis if request.is_assistant_axis is not None else False

    if is_assistant_axis:
        if not isinstance(model, VLLMSteerModel):
            return JSONResponse(
                content={
                    "error": "Assistant axis is only supported for Chatspace/VLLMSteer model"
                },
                status_code=400,
            )
    

    # Ensure exactly one of features or vector is provided
    if (request.features is not None) == (request.vectors is not None):
        logger.error(
            "Invalid request data: exactly one of features or vectors must be provided"
        )
        return JSONResponse(
            content={
                "error": "Invalid request data: exactly one of features or vectors must be provided"
            },
            status_code=400,
        )

    promptChat = request.prompt
    promptChatFormatted = []

    if is_assistant_axis:
        # Check if first message is already a system message
        if promptChat and promptChat[0].role == "system":
            # Strip the default Llama system prompt if present
            system_content = promptChat[0].content
            system_content = _strip_llama3_system_preamble(system_content)
            promptChatFormatted.append({"role": "system", "content": system_content})
            # Add remaining messages (skip the first system message we already processed)
            for message in promptChat[1:]:
                promptChatFormatted.append({"role": message.role, "content": message.content})
        else:
            # No existing system message, just add all messages as-is
            for message in promptChat:
                promptChatFormatted.append({"role": message.role, "content": message.content})
    else:
        for message in promptChat:
                promptChatFormatted.append({"role": message.role, "content": message.content})
    
    if model.tokenizer is None:
        raise ValueError("Tokenizer is not initialized")

    # If the tokenizer does not support chat templates, we need to apply a generic chat template
    if (
        not hasattr(model.tokenizer, "chat_template")
        or model.tokenizer.chat_template is None
    ):
        logger.warning(
            "Model's tokenizer does not support chat templates. Using generic chat template."
        )
        template_applied_prompt = apply_generic_chat_template(
            promptChatFormatted, add_generation_prompt=True
        )
        if isinstance(model, HookedTransformer):
            promptTokenized = model.to_tokens(
                template_applied_prompt, prepend_bos=True
            )[0]
        elif is_nnterp_model(model) or isinstance(
            model, VLLMSteerModel
        ):
            promptTokenized = model.tokenizer(
                template_applied_prompt, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]
            if (request.n_logprobs is not None) and (request.n_logprobs > 0):
                request.n_logprobs = 0
    else:
        # tokenize = True adds a BOS
        promptTokenized = model.tokenizer.apply_chat_template(
            promptChatFormatted, tokenize=True, add_generation_prompt=True
        )
        if is_nnterp_model(model) or isinstance(
            model, VLLMSteerModel
        ):
            if promptTokenized[0] == model.tokenizer.bos_token_id:
                promptTokenized = promptTokenized[1:]
    promptTokenized = torch.tensor(promptTokenized)

    # logger.info("promptTokenized: %s", promptTokenized)
    if len(promptTokenized) > config.token_limit:
        logger.error(
            "Text too long: %s tokens, max is %s",
            len(promptTokenized),
            config.token_limit,
        )
        return JSONResponse(
            content={
                "error": f"Text too long: {len(promptTokenized)} tokens, max is {config.token_limit}"
            },
            status_code=400,
        )

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
        NPSteerChatMessage(role=msg["role"], content=msg["content"])
        for msg in promptChatFormatted
    ]
    
    generation_start = time.time()
    
    generator = run_batched_generate(
        promptTokenized=promptTokenized,
        inputPrompt=inputPromptForPersona if is_assistant_axis else promptChat,
        features=features,
        steer_types=request.types,
        strength_multiplier=float(request.strength_multiplier),
        seed=int(request.seed),
        temperature=float(request.temperature),
        freq_penalty=float(request.freq_penalty),
        max_new_tokens=int(request.n_completion_tokens),
        steer_special_tokens=steer_special_tokens,
        steer_method=steer_method,
        normalize_steering=normalize_steering,
        use_stream_lock=request.stream if request.stream is not None else False,
        custom_hf_model_id=custom_hf_model_id,
        n_logprobs=(request.n_logprobs or 0),
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
            except Exception as e:
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
    logger.info(
        f"[REQUEST COMPLETE] total={total_time:.2f}s, generation={generation_time:.2f}s, "
        f"~chunks={chunk_count}"
    )
    
    if last_item is None:
        raise ValueError("No response generated")
    results = remove_sse_formatting(last_item)
    response = SteerCompletionChatPost200Response.from_json(results)
    if response is None:
        raise ValueError("Failed to parse response")
    # set exclude_none to True to omit the logprobs field when n_logprobs isn't set in the request, for backwards compatibility
    return JSONResponse(content=response.model_dump(exclude_none=True))


async def run_persona_monitor(
    model: Any,
    conversation: list[NPSteerChatMessage],
    steer_type: NPSteerType,
    layer: int = DEFAULT_LAYER,
) -> SteerCompletionChatPost200ResponseAssistantAxisInner | None:
    """
    Run persona monitoring on the conversation and return assistant_axis data.
    
    This extracts activations and projects them onto pre-computed principal components
    that capture persona-related variation in the model's representations.
    
    Args:
        model: The VLLMSteerModel instance
        conversation: List of chat messages (user/assistant turns)
        steer_type: The steer type this analysis corresponds to
        layer: Layer to extract activations from
        
    Returns:
        AssistantAxis response data, or None if persona data not available
    """
    logger.debug(f"[PERSONA] run_persona_monitor called for steer_type={steer_type}, layer={layer}")
    persona_start = time.time()
    
    config = Config.get_instance()
    model_id_for_data = config.override_model_id or config.model_id
    
    # Get pre-loaded PCA data
    logger.debug(f"[PERSONA] Getting PersonaData instance...")
    persona_data = PersonaData.get_instance()
    if not persona_data.is_initialized():
        logger.warning("Persona data not initialized, skipping persona monitor")
        return None
    
    pca_results = persona_data.get_pca_data(layer)
    if pca_results is None:
        logger.warning(f"PCA data not available for layer {layer}")
        return None
    logger.debug(f"[PERSONA] PCA data loaded in {time.time() - persona_start:.3f}s")
    
    # Wrap model with ProbingModelChatSpace
    logger.debug(f"[PERSONA] Creating ProbingModelChatSpace wrapper...")
    probing_model = ProbingModelChatSpace.from_existing(
        model,
        tokenizer=None,
        model_name=model_id_for_data
    )
    
    tokenizer = probing_model.tokenizer
    encoder = ConversationEncoder(tokenizer, model_id_for_data)
    mapper = SpanMapperChatSpace(tokenizer)
    
    # Convert NPSteerChatMessage to the format expected by mapper
    conversation_turns = [
        {"role": msg.role, "content": msg.content}
        for msg in conversation
    ]
    
    # Extract mean activations per turn
    logger.debug(f"[PERSONA] Extracting activations for {len(conversation_turns)} turns...")
    extract_start = time.time()
    mean_acts_per_turn = await mapper.mean_all_turn_activations_async(
        probing_model, encoder, conversation_turns, layer=layer
    )
    logger.debug(f"[PERSONA] Activations extracted in {time.time() - extract_start:.3f}s, shape={mean_acts_per_turn.shape}")
    
    # Handle empty activations
    if mean_acts_per_turn.shape[0] == 0:
        logger.warning("No activations extracted, skipping persona monitor")
        return None
    
    # Compute projections
    role_projs = pc_projection(mean_acts_per_turn, pca_results, n_pcs=1)
    
    # Find indices of assistant turns in the conversation (by actual role, not position assumption)
    # This handles conversations with system messages where indices don't alternate user/assistant
    assistant_indices = [i for i, msg in enumerate(conversation) if msg.role == "assistant"]
    
    # Select projections for assistant turns only
    assistant_role_projs = role_projs[assistant_indices] if assistant_indices else role_projs[0:0]
    
    # Get assistant turns for snippets
    assistant_turns = [
        msg for msg in conversation if msg.role == "assistant"
    ]
    
    turns_data = []
    for i in range(len(assistant_role_projs)):
        pc_values = {
            ROLE_PC_TITLES[j]: float(assistant_role_projs[i][j])
            for j in range(len(ROLE_PC_TITLES))
        }
        
        snippet = ""
        if i < len(assistant_turns):
            snippet = _truncate_content(assistant_turns[i].content)
        
        turns_data.append(SteerCompletionChatPost200ResponseAssistantAxisInnerTurnsInner(
            pc_values=pc_values,
            snippet=snippet
        ))
    
    logger.debug(f"[PERSONA] Complete in {time.time() - persona_start:.3f}s, {len(turns_data)} assistant turns")
    return SteerCompletionChatPost200ResponseAssistantAxisInner(
        type=steer_type,
        pc_titles=list(ROLE_PC_TITLES),
        turns=turns_data
    )


async def run_batched_generate(
    promptTokenized: torch.Tensor,
    inputPrompt: list[NPSteerChatMessage],
    features: list[NPSteerFeature] | list[NPSteerVector],
    steer_types: list[NPSteerType],
    strength_multiplier: float,
    seed: int | None = None,
    steer_method: NPSteerMethod = NPSteerMethod.SIMPLE_ADDITIVE,
    normalize_steering: bool = False,
    steer_special_tokens: bool = False,
    use_stream_lock: bool = False,
    custom_hf_model_id: str | None = None,
    n_logprobs: int = 0,
    is_assistant_axis: bool = False,
    **kwargs: Any,
):
    async with await stream_lock(use_stream_lock):
        model = Model.get_instance()
        sae_manager = SAEManager.get_instance()

        # Add device logging
        # logger.info(f"Model device: {model.cfg.device}")
        # logger.info(f"Input tensor device: {promptTokenized.device}")

        if seed is not None:
            torch.manual_seed(seed)

        def steering_hook(activations: torch.Tensor, hook: Any) -> torch.Tensor:  # noqa: ARG001
            # log activation device
            # logger.info(f"Activations device: {activations.device}")

            for i, flag in enumerate(steer_types):
                if flag == NPSteerType.STEERED:
                    if model.tokenizer is None:
                        raise ValueError("Tokenizer is not initialized")

                    # If we want to steer special tokens, then just pass it through without masking
                    if steer_special_tokens:
                        mask = torch.ones(
                            activations.shape[1], device=activations.device
                        )
                    else:
                        # TODO: Need to generalize beyond the gemma tokenizer

                        # Get the current tokens for this batch
                        current_tokens = promptTokenized.to(activations.device)

                        mask = torch.ones(
                            activations.shape[1], device=activations.device
                        )

                        # Find indices of special tokens

                        bos_indices = (
                            current_tokens == model.tokenizer.bos_token_id
                        ).nonzero(as_tuple=True)[0]  # type: ignore
                        start_of_turn_indices = (
                            current_tokens
                            == model.tokenizer.encode("<start_of_turn>")[0]
                        ).nonzero(as_tuple=True)[0]
                        end_of_turn_indices = (
                            current_tokens == model.tokenizer.encode("<end_of_turn>")[0]
                        ).nonzero(as_tuple=True)[0]

                        # Apply masking rules
                        # 1. Don't steer <bos>
                        mask[bos_indices] = 0

                        # 2. Don't steer <start_of_turn> and the next two tokens
                        for idx in start_of_turn_indices:
                            mask[idx : idx + 3] = 0

                        # 3. Don't steer <end_of_turn> and the next token
                        for idx in end_of_turn_indices:
                            mask[idx : idx + 2] = 0
                    # Apply steering with the mask
                    for feature in features:
                        steering_vector = torch.tensor(feature.steering_vector).to(
                            activations.device
                        )

                        if not torch.isfinite(steering_vector).all():
                            raise ValueError(
                                "Steering vector contains inf or nan values"
                            )

                        if normalize_steering:
                            norm = torch.norm(steering_vector)
                            if norm == 0:
                                raise ValueError("Zero norm steering vector")
                            steering_vector = steering_vector / norm

                        # If it's attention hook, reshape it to (n_heads, head_dim)
                        if isinstance(
                            feature, NPSteerFeature
                        ) and "attn.hook_z" in sae_manager.get_sae_hook(feature.source):
                            n_heads = model.cfg.n_heads
                            d_head = model.cfg.d_head
                            steering_vector = steering_vector.view(n_heads, d_head)

                        coeff = strength_multiplier * feature.strength

                        if steer_method == NPSteerMethod.SIMPLE_ADDITIVE:
                            activations[i] += (
                                coeff * steering_vector * mask.unsqueeze(-1)
                            )

                        elif steer_method == NPSteerMethod.ORTHOGONAL_DECOMP:
                            projector = OrthogonalProjector(steering_vector)
                            projected = projector.project(activations[i], coeff)
                            activations[i] = activations[i] * (
                                1 - mask.unsqueeze(-1)
                            ) + projected * mask.unsqueeze(-1)

            return activations

        # Check if we need to generate both STEERED and DEFAULT
        generate_both = (
            NPSteerType.STEERED in steer_types and NPSteerType.DEFAULT in steer_types
        )

        if generate_both:
            steered_partial_result = ""
            default_partial_result = ""
            steered_logprobs = None
            default_logprobs = None

            steered_partial_result_array: list[str] = []
            default_partial_result_array: list[str] = []
            steered_logprobs = None
            default_logprobs = None

            # Generate STEERED and DEFAULT separately
            for flag in [NPSteerType.STEERED, NPSteerType.DEFAULT]:
                if seed is not None:
                    torch.manual_seed(seed)  # Reset seed for each generation

                if isinstance(model, HookedTransformer):
                    model.reset_hooks()
                    if flag == NPSteerType.STEERED:
                        logger.info("Running Steered")
                        editing_hooks = [
                            (
                                (
                                    sae_manager.get_sae_hook(feature.source)
                                    if isinstance(feature, NPSteerFeature)
                                    else feature.hook
                                ),
                                steering_hook,
                            )
                            for feature in features
                        ]
                    else:
                        logger.info("Running Default")
                        editing_hooks = []

                    logprobs = []

                    with model.hooks(fwd_hooks=editing_hooks):  # type: ignore
                        for i, (result, logits) in enumerate(
                            model.generate_stream(
                                max_tokens_per_yield=TOKENS_PER_YIELD,
                                stop_at_eos=(model.cfg.device != "mps"),
                                input=promptTokenized.unsqueeze(0),
                                do_sample=True,
                                return_logits=True,
                                **kwargs,
                            )
                        ):
                            to_append = ""
                            if i == 0:
                                to_append = model.to_string(result[0][1:])  # type: ignore
                            else:
                                to_append = model.to_string(result[0])  # type: ignore

                            if n_logprobs > 0:
                                current_logprobs = make_logprob_from_logits(
                                    result,  # type: ignore
                                    logits,  # type: ignore
                                    model,
                                    n_logprobs,
                                )
                                logprobs.append(current_logprobs)

                            if flag == NPSteerType.STEERED:
                                steered_partial_result += to_append  # type: ignore
                                steered_logprobs = logprobs.copy() or None
                            else:
                                default_partial_result += to_append  # type: ignore
                                default_logprobs = logprobs.copy() or None

                            to_return = make_steer_completion_chat_response(
                                steer_types,
                                steered_partial_result,
                                default_partial_result,
                                model,
                                promptTokenized,
                                inputPrompt,
                                custom_hf_model_id,
                                steered_logprobs,
                                default_logprobs,
                            )  # type: ignore
                            yield format_sse_message(to_return.to_json())

                elif is_nnterp_model(model):
                    logger.info("nnsight streaming")
                    if kwargs.get("freq_penalty"):
                        logger.warning(
                            "freq_penalty is not supported for StandardizedTransformer models, it will be ignored"
                        )

                    # Convert promptTokenized to string for nnsight
                    prompt_string = model.tokenizer.decode(promptTokenized)

                    # Use streaming generation with TextIteratorStreamer
                    output_total = ""
                    async for token_text in stream_nnsight_generation(
                        model=model,
                        prompt_string=prompt_string,
                        features=features if flag == NPSteerType.STEERED else [],
                        sae_manager=sae_manager,
                        steer_type=flag,
                        strength_multiplier=strength_multiplier,
                        normalize_steering=normalize_steering,
                        steer_method=steer_method,
                        temperature=kwargs.get("temperature"),
                        max_new_tokens=kwargs.get("max_new_tokens"),
                        do_sample=kwargs.get("do_sample", True),
                    ):
                        output_total += token_text
                        
                        if flag == NPSteerType.STEERED:
                            to_return = make_steer_completion_chat_response(
                                steer_types,
                                prompt_string + output_total,
                                prompt_string + "".join(default_partial_result_array),
                                model,
                                promptTokenized,
                                inputPrompt,
                                custom_hf_model_id,
                                steered_logprobs,
                                default_logprobs,
                            )
                        else:
                            to_return = make_steer_completion_chat_response(
                                steer_types,
                                prompt_string + "".join(steered_partial_result_array),
                                prompt_string + output_total,
                                model,
                                promptTokenized,
                                inputPrompt,
                                custom_hf_model_id,
                                steered_logprobs,
                                default_logprobs,
                            )
                        yield format_sse_message(to_return.to_json())
                    
                    # Update result array after streaming completes
                    if flag == NPSteerType.STEERED:
                        steered_partial_result_array = [output_total]
                    else:
                        default_partial_result_array = [output_total]

                elif isinstance(model, VLLMSteerModel):
                    if kwargs.get("freq_penalty"):
                        logger.warning(
                            "freq_penalty is not supported for VLLMSteerModel models, it will be ignored"
                        )

                    # Convert promptTokenized to string for VLLMSteerModel
                    prompt_string = model.tokenizer.decode(promptTokenized)

                    sampling_params = SamplingParams(
                        temperature=kwargs.get("temperature"),
                        max_tokens=kwargs.get("max_new_tokens"),
                        seed=seed,
                    )

                    if flag == NPSteerType.STEERED:
                        # Build steering spec from all features
                        steering_spec_layers = {}

                        # Group features by layer
                        layer_features: dict[
                            int,
                            list[tuple[NPSteerFeature | NPSteerVector, torch.Tensor]],
                        ] = {}

                        for feature in features:
                            hook_name = (
                                sae_manager.get_sae_hook(feature.source)
                                if isinstance(feature, NPSteerFeature)
                                else feature.hook
                            )
                            if "resid_post" in hook_name:
                                layer = int(
                                    hook_name.split(".")[1]
                                )  # blocks.0.hook_resid_post -> 0
                            elif "resid_pre" in hook_name:
                                layer = (
                                    int(hook_name.split(".")[1]) - 1
                                )  # blocks.1.hook_resid_pre -> 0
                            else:
                                raise ValueError(
                                    f"Unsupported hook name for chatspace: {hook_name}"
                                )

                            steering_vector = torch.tensor(feature.steering_vector)

                            if not torch.isfinite(steering_vector).all():
                                raise ValueError(
                                    "Steering vector contains inf or nan values"
                                )

                            if normalize_steering:
                                norm = torch.norm(steering_vector)
                                if norm == 0:
                                    raise ValueError("Zero norm steering vector")
                                steering_vector = steering_vector / norm

                            if layer not in layer_features:
                                layer_features[layer] = []
                            layer_features[layer].append((feature, steering_vector))

                        # Build LayerSteeringSpec for each layer
                        for layer, layer_feature_list in layer_features.items():
                            operations: list[SteeringOp] = []

                            if steer_method == NPSteerMethod.SIMPLE_ADDITIVE:
                                # Add each feature as a separate AddSpec operation
                                for feature, steering_vector in layer_feature_list:
                                    coeff = strength_multiplier * feature.strength
                                    norm = torch.norm(steering_vector)
                                    if norm > 0:
                                        normalized_vector = steering_vector / norm
                                        operations.append(
                                            AddSpec(
                                                vector=normalized_vector,
                                                scale=norm.item() * coeff,
                                            )
                                        )

                            elif steer_method == NPSteerMethod.ORTHOGONAL_DECOMP:
                                raise ValueError(
                                    "Orthogonal decomposition is not supported for chatspace."
                                )

                            elif steer_method == NPSteerMethod.PROJECTION_CAP:
                                # logger.info("projection cap")
                                # Add each feature as a separate ProjectionCapSpec operation
                                for feature, steering_vector in layer_feature_list:
                                    coeff = strength_multiplier * feature.strength
                                    operations.append(
                                        ProjectionCapSpec(
                                            vector=steering_vector, min=None, max=coeff
                                        )
                                    )

                            if operations:
                                steering_spec_layers[layer] = LayerSteeringSpec(
                                    operations=operations
                                )

                        # Validate that we have at least one layer to steer
                        if not steering_spec_layers:
                            raise ValueError(
                                "No valid steering layers found. All features may have zero-norm vectors or invalid configurations."
                            )

                        # Use streaming generation
                        stream_generator = await model.generate(
                            prompt_string,
                            sampling_params,
                            steering_spec=SteeringSpec(layers=steering_spec_layers),
                            stream=True,
                        )
                        output_total = ""
                        async for delta in stream_generator:
                            output_total += delta
                            to_return = make_steer_completion_chat_response(
                                steer_types,
                                prompt_string + output_total,
                                prompt_string + "".join(default_partial_result_array),
                                model,
                                promptTokenized,
                                inputPrompt,
                                custom_hf_model_id,
                                steered_logprobs,
                                default_logprobs,
                            )  # type: ignore
                            yield format_sse_message(to_return.to_json())
                        # Update array after streaming completes
                        steered_partial_result_array = [output_total]  # type: ignore
                    else:
                        # Use streaming generation for DEFAULT
                        stream_generator = await model.generate(
                            prompt_string, sampling_params, stream=True
                        )
                        output_total = ""
                        async for delta in stream_generator:
                            output_total += delta
                            to_return = make_steer_completion_chat_response(
                                steer_types,
                                prompt_string + "".join(steered_partial_result_array),
                                prompt_string + output_total,
                                model,
                                promptTokenized,
                                inputPrompt,
                                custom_hf_model_id,
                                steered_logprobs,
                                default_logprobs,
                            )  # type: ignore
                            yield format_sse_message(to_return.to_json())
                        # Update array after streaming completes
                        default_partial_result_array = [output_total]  # type: ignore

            # After both STEERED and DEFAULT streaming completes for VLLMSteerModel,
            # run persona monitor if is_assistant_axis
            if isinstance(model, VLLMSteerModel) and is_assistant_axis:
                steered_output = "".join(steered_partial_result_array)
                default_output = "".join(default_partial_result_array)
                
                # Run persona monitor for each steer type
                assistant_axis_data_list: list[SteerCompletionChatPost200ResponseAssistantAxisInner] = []
                
                for steer_type_for_monitor in steer_types:
                    if steer_type_for_monitor == NPSteerType.STEERED:
                        output_for_monitor = steered_output
                    else:
                        output_for_monitor = default_output
                    
                    # Build full conversation including the generated assistant response
                    full_conversation = list(inputPrompt) + [
                        NPSteerChatMessage(role="assistant", content=output_for_monitor)
                    ]
                    
                    axis_data = await run_persona_monitor(
                        model, full_conversation, steer_type_for_monitor, DEFAULT_LAYER
                    )
                    if axis_data is not None:
                        assistant_axis_data_list.append(axis_data)
                
                # Yield final message with persona monitor results
                to_return = make_steer_completion_chat_response(
                    steer_types,
                    prompt_string + steered_output,
                    prompt_string + default_output,
                    model,
                    promptTokenized,
                    inputPrompt,
                    custom_hf_model_id,
                    steered_logprobs,
                    default_logprobs,
                    assistant_axis_data_list if assistant_axis_data_list else None,
                )  # type: ignore
                yield format_sse_message(to_return.to_json())

            # Send final complete message for nnsight after both streams complete
            if is_nnterp_model(model):
                to_return = make_steer_completion_chat_response(
                    steer_types,
                    prompt_string + "".join(steered_partial_result_array),
                    prompt_string + "".join(default_partial_result_array),
                    model,
                    promptTokenized,
                    inputPrompt,
                    custom_hf_model_id,
                    steered_logprobs,
                    default_logprobs,
                )  # type: ignore
                yield format_sse_message(to_return.to_json())
        else:
            steer_type = steer_types[0]
            if seed is not None:
                torch.manual_seed(seed)

            partial_result_array: list[str] = []

            if isinstance(model, HookedTransformer):
                model.reset_hooks()
                editing_hooks = [
                    (
                        (
                            sae_manager.get_sae_hook(feature.source)
                            if isinstance(feature, NPSteerFeature)
                            else feature.hook
                        ),
                        steering_hook,
                    )
                    for feature in features
                ]
                logger.info("steer_type: %s", steer_type)

                with model.hooks(fwd_hooks=editing_hooks):  # type: ignore
                    partial_result = ""
                    logprobs = []

                    for i, (result, logits) in enumerate(
                        model.generate_stream(
                            max_tokens_per_yield=TOKENS_PER_YIELD,
                            stop_at_eos=(model.cfg.device != "mps"),
                            input=promptTokenized.unsqueeze(0),
                            do_sample=True,
                            return_logits=True,
                            **kwargs,
                        )
                    ):
                        if i == 0:
                            partial_result = model.to_string(result[0][1:])  # type: ignore
                        else:
                            partial_result += model.to_string(result[0])  # type: ignore

                        if n_logprobs > 0:
                            current_logprobs = make_logprob_from_logits(
                                result,  # type: ignore
                                logits,  # type: ignore
                                model,
                                n_logprobs,
                            )
                            logprobs.append(current_logprobs)

                        to_return = make_steer_completion_chat_response(
                            [steer_type],
                            partial_result,  # type: ignore
                            partial_result,  # type: ignore
                            model,
                            promptTokenized,
                            inputPrompt,
                            custom_hf_model_id,
                            logprobs or None,
                            logprobs or None,
                        )
                        yield format_sse_message(to_return.to_json())

            elif is_nnterp_model(model):
                logger.info("nnsight streaming")
                if kwargs.get("freq_penalty"):
                    logger.warning(
                        "freq_penalty is not supported for StandardizedTransformer models, it will be ignored"
                    )

                # Convert promptTokenized to string for nnsight
                prompt_string = model.tokenizer.decode(promptTokenized)

                # Use streaming generation with TextIteratorStreamer
                output_total = ""
                async for token_text in stream_nnsight_generation(
                    model=model,
                    prompt_string=prompt_string,
                    features=features if steer_type == NPSteerType.STEERED else [],
                    sae_manager=sae_manager,
                    steer_type=steer_type,
                    strength_multiplier=strength_multiplier,
                    normalize_steering=normalize_steering,
                    steer_method=steer_method,
                    temperature=kwargs.get("temperature"),
                    max_new_tokens=kwargs.get("max_new_tokens"),
                    do_sample=kwargs.get("do_sample", True),
                ):
                    output_total += token_text
                    to_return = make_steer_completion_chat_response(
                        [steer_type],
                        prompt_string + output_total,
                        prompt_string + output_total,
                        model,
                        promptTokenized,
                        inputPrompt,
                        custom_hf_model_id,
                        None,
                        None,
                    )
                    yield format_sse_message(to_return.to_json())
            elif isinstance(model, VLLMSteerModel):
                if kwargs.get("freq_penalty"):
                    logger.warning(
                        "freq_penalty is not supported for VLLMSteerModel models, it will be ignored"
                    )

                # Convert promptTokenized to string for chatspace
                prompt_string = model.tokenizer.decode(promptTokenized)

                sampling_params = SamplingParams(
                    temperature=kwargs.get("temperature"),
                    max_tokens=kwargs.get("max_new_tokens"),
                    seed=seed,
                )

                if steer_type == NPSteerType.STEERED:
                    # Build steering spec from all features
                    steering_spec_layers = {}

                    # Group features by layer
                    layer_features: dict[
                        int, list[tuple[NPSteerFeature | NPSteerVector, torch.Tensor]]
                    ] = {}

                    for feature in features:
                        hook_name = (
                            sae_manager.get_sae_hook(feature.source)
                            if isinstance(feature, NPSteerFeature)
                            else feature.hook
                        )
                        if "resid_post" in hook_name:
                            layer = int(
                                hook_name.split(".")[1]
                            )  # blocks.0.hook_resid_post -> 0
                        elif "resid_pre" in hook_name:
                            layer = (
                                int(hook_name.split(".")[1]) - 1
                            )  # blocks.1.hook_resid_pre -> 0
                        else:
                            raise ValueError(
                                f"Unsupported hook name for chatspace: {hook_name}"
                            )

                        steering_vector = torch.tensor(feature.steering_vector)

                        if not torch.isfinite(steering_vector).all():
                            raise ValueError(
                                "Steering vector contains inf or nan values"
                            )

                        if normalize_steering:
                            norm = torch.norm(steering_vector)
                            if norm == 0:
                                raise ValueError("Zero norm steering vector")
                            steering_vector = steering_vector / norm

                        if layer not in layer_features:
                            layer_features[layer] = []
                        layer_features[layer].append((feature, steering_vector))

                    # Build LayerSteeringSpec for each layer
                    for layer, layer_feature_list in layer_features.items():
                        layer_operations: list[SteeringOp] = []

                        if steer_method == NPSteerMethod.SIMPLE_ADDITIVE:
                            # Add each feature as a separate AddSpec operation
                            for feature, steering_vector in layer_feature_list:
                                coeff = strength_multiplier * feature.strength
                                norm = torch.norm(steering_vector)
                                if norm > 0:
                                    normalized_vector = steering_vector / norm
                                    layer_operations.append(
                                        AddSpec(
                                            vector=normalized_vector,
                                            scale=norm.item() * coeff,
                                        )
                                    )

                        elif steer_method == NPSteerMethod.ORTHOGONAL_DECOMP:
                            raise ValueError(
                                "Orthogonal decomposition is not supported for chatspace."
                            )

                        elif steer_method == NPSteerMethod.PROJECTION_CAP:
                            # logger.info("projection cap")
                            # Add each feature as a separate ProjectionCapSpec operation
                            for feature, steering_vector in layer_feature_list:
                                coeff = strength_multiplier * feature.strength
                                layer_operations.append(
                                    ProjectionCapSpec(
                                        vector=steering_vector, min=None, max=coeff
                                    )
                                )

                        if layer_operations:
                            steering_spec_layers[layer] = LayerSteeringSpec(
                                operations=layer_operations
                            )

                    # Validate that we have at least one layer to steer
                    if not steering_spec_layers:
                        raise ValueError(
                            "No valid steering layers found. All features may have zero-norm vectors or invalid configurations."
                        )

                    # Use streaming generation
                    stream_generator = await model.generate(
                        prompt_string,
                        sampling_params,
                        steering_spec=SteeringSpec(layers=steering_spec_layers),
                        stream=True,
                    )
                    output_total = ""
                    async for delta in stream_generator:
                        output_total += delta
                        to_return = make_steer_completion_chat_response(
                            [steer_type],
                            prompt_string + output_total,
                            prompt_string + output_total,
                            model,
                            promptTokenized,
                            inputPrompt,
                            custom_hf_model_id,
                            None,
                            None,
                        )  # type: ignore
                        yield format_sse_message(to_return.to_json())
                else:
                    # Use streaming generation for DEFAULT
                    stream_generator = await model.generate(
                        prompt_string, sampling_params, stream=True
                    )
                    output_total = ""
                    async for delta in stream_generator:
                        output_total += delta
                        to_return = make_steer_completion_chat_response(
                            [steer_type],
                            prompt_string + output_total,
                            prompt_string + output_total,
                            model,
                            promptTokenized,
                            inputPrompt,
                            custom_hf_model_id,
                            None,
                            None,
                        )  # type: ignore
                        yield format_sse_message(to_return.to_json())

                # After streaming completes, run persona monitor if is_assistant_axis
                if is_assistant_axis:
                    # Build full conversation including the generated assistant response
                    full_conversation = list(inputPrompt) + [
                        NPSteerChatMessage(role="assistant", content=output_total)
                    ]
                    assistant_axis_data = await run_persona_monitor(
                        model, full_conversation, steer_type, DEFAULT_LAYER
                    )
                    # Yield final message with persona monitor results
                    to_return = make_steer_completion_chat_response(
                        [steer_type],
                        prompt_string + output_total,
                        prompt_string + output_total,
                        model,
                        promptTokenized,
                        inputPrompt,
                        custom_hf_model_id,
                        None,
                        None,
                        [assistant_axis_data] if assistant_axis_data else None,
                    )  # type: ignore
                    yield format_sse_message(to_return.to_json())


def make_steer_completion_chat_response(
    steer_types: list[NPSteerType],
    steered_result: str,
    default_result: str,
    model: HookedTransformer | StandardizedTransformer | VLLMSteerModel,
    promptTokenized: torch.Tensor,
    promptChat: list[NPSteerChatMessage],
    custom_hf_model_id: str | None = None,
    steered_logprobs: list[NPLogprob] | None = None,
    default_logprobs: list[NPLogprob] | None = None,
    assistant_axis_data: list[SteerCompletionChatPost200ResponseAssistantAxisInner] | None = None,
) -> SteerCompletionChatPost200Response:
    steerChatResults = []
    for steer_type in steer_types:
        if steer_type == NPSteerType.STEERED:
            steerChatResults.append(
                NPSteerChatResult(
                    raw=steered_result,  # type: ignore
                    chat_template=convert_to_chat_array(
                        steered_result,
                        model.tokenizer,
                        custom_hf_model_id,  # type: ignore
                    ),
                    type=steer_type,
                    logprobs=steered_logprobs,
                )
            )
        else:
            steerChatResults.append(
                NPSteerChatResult(
                    raw=default_result,  # type: ignore
                    chat_template=convert_to_chat_array(
                        default_result,
                        model.tokenizer,
                        custom_hf_model_id,  # type: ignore
                    ),
                    type=steer_type,
                    logprobs=default_logprobs,
                )
            )

    # Handle token to string conversion for both model types
    if isinstance(model, HookedTransformer):
        prompt_raw = model.to_string(promptTokenized)  # type: ignore
    elif is_nnterp_model(model) or isinstance(
        model, VLLMSteerModel
    ):
        prompt_raw = model.tokenizer.decode(promptTokenized)
    else:
        prompt_raw = ""

    return SteerCompletionChatPost200Response(
        assistant_axis=assistant_axis_data,
        outputs=steerChatResults,
        input=NPSteerChatResult(
            raw=prompt_raw,  # type: ignore
            chat_template=promptChat,
        ),
    )
