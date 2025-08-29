import logging
from typing import Any

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
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
from neuronpedia_inference_client.models.steer_completion_chat_post_request import (
    SteerCompletionChatPostRequest,
)
from transformer_lens.model_bridge.bridge import TransformerBridge

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
from neuronpedia_inference.shared import Model, with_request_lock
from neuronpedia_inference.utils import make_logprob_from_logits

logger = logging.getLogger(__name__)


router = APIRouter()

TOKENS_PER_YIELD = 1


@router.post("/steer/completion-chat")
@with_request_lock()
async def completion_chat(request: SteerCompletionChatPostRequest):
    model = Model.get_instance()
    config = Config.get_instance()
    steer_method = request.steer_method
    normalize_steering = request.normalize_steering
    steer_special_tokens = request.steer_special_tokens
    custom_hf_model_id = config.custom_hf_model_id

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

    # assert that steered comes before default
    # TODO: unsure why this is needed? some artifact of a refactoring done last summer
    if NPSteerType.STEERED in request.types and NPSteerType.DEFAULT in request.types:
        index_steer = request.types.index(NPSteerType.STEERED)
        index_default = request.types.index(NPSteerType.DEFAULT)
        # assert index_steer < index_default, "STEERED must come before DEFAULT, we have a bug otherwise"
        if index_steer > index_default:
            logger.error("STEERED must come before DEFAULT. We have a bug otherwise.")
            return JSONResponse(
                content={
                    "error": "STEERED must come before DEFAULT. We have a bug otherwise."
                },
                status_code=400,
            )

    promptChat = request.prompt
    promptChatFormatted = []
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
        promptTokenized = model.to_tokens(template_applied_prompt, prepend_bos=True)[0]
    else:
        # tokenize = True adds a BOS
        promptTokenized = model.tokenizer.apply_chat_template(
            promptChatFormatted, tokenize=True, add_generation_prompt=True
        )
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

    generator = run_batched_generate(
        promptTokenized=promptTokenized,
        inputPrompt=promptChat,
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
    )

    if request.stream:
        return StreamingResponse(generator, media_type="text/event-stream")
    # for non-streaming request, get last item from generator
    last_item = None
    async for item in generator:
        last_item = item
    if last_item is None:
        raise ValueError("No response generated")
    results = remove_sse_formatting(last_item)
    response = SteerCompletionChatPost200Response.from_json(results)
    if response is None:
        raise ValueError("Failed to parse response")
    # set exclude_none to True to omit the logprobs field when n_logprobs isn't set in the request, for backwards compatibility
    return JSONResponse(content=response.model_dump(exclude_none=True))


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

            # Generate STEERED and DEFAULT separately
            for flag in [NPSteerType.STEERED, NPSteerType.DEFAULT]:
                if seed is not None:
                    torch.manual_seed(seed)  # Reset seed for each generation

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
        else:
            steer_type = steer_types[0]
            if seed is not None:
                torch.manual_seed(seed)

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


def make_steer_completion_chat_response(
    steer_types: list[NPSteerType],
    steered_result: str,
    default_result: str,
    model: TransformerBridge,
    promptTokenized: torch.Tensor,
    promptChat: list[NPSteerChatMessage],
    custom_hf_model_id: str | None = None,
    steered_logprobs: list[NPLogprob] | None = None,
    default_logprobs: list[NPLogprob] | None = None,
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

    return SteerCompletionChatPost200Response(
        outputs=steerChatResults,
        input=NPSteerChatResult(
            raw=model.to_string(promptTokenized),  # type: ignore
            chat_template=promptChat,
        ),
    )
