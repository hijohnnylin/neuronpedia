import logging
from typing import Any

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
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
from transformer_lens import HookedTransformer

from neuronpedia_inference.config import Config
from neuronpedia_inference.inference_utils.steering import (
    OrthogonalProjector,
    convert_to_chat_array,
    format_sse_message,
    process_features_vectorized,
    remove_sse_formatting,
    stream_lock,
)
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.shared import (
    Model,
    generate_stream,
    with_request_lock,
)

logger = logging.getLogger(__name__)


router = APIRouter()

TOKENS_PER_YIELD = 2


@router.post("/steer/completion-chat")
@with_request_lock()
async def completion_chat(request: SteerCompletionChatPostRequest):
    model = Model.get_instance()
    config = Config.get_instance()
    steer_method = request.steer_method
    normalize_steering = request.normalize_steering
    steer_special_tokens = request.steer_special_tokens
    custom_hf_model_id = config.CUSTOM_HF_MODEL_ID

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

    # tokenize = True adds a BOS
    if model.tokenizer is None:
        raise ValueError("Tokenizer is not initialized")
    promptTokenized = model.tokenizer.apply_chat_template(
        promptChatFormatted, tokenize=True, add_generation_prompt=True
    )
    promptTokenized = torch.tensor(promptTokenized)

    # logger.info("promptTokenized: %s", promptTokenized)
    if len(promptTokenized) > config.TOKEN_LIMIT:
        logger.error(
            "Text too long: %s tokens, max is %s",
            len(promptTokenized),
            config.TOKEN_LIMIT,
        )
        return JSONResponse(
            content={
                "error": f"Text too long: {len(promptTokenized)} tokens, max is {config.TOKEN_LIMIT}"
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
    )

    if request.stream:
        return StreamingResponse(generator, media_type="text/event-stream")
    last_item = None
    async for item in generator:
        last_item = item
    if last_item is None:
        raise ValueError("No response generated")
    results = remove_sse_formatting(last_item)
    return SteerCompletionChatPost200Response.from_json(results)


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

                with model.hooks(fwd_hooks=editing_hooks):
                    for result in generate_stream(
                        model,
                        max_tokens_per_yield=TOKENS_PER_YIELD,
                        stop_at_eos=(model.cfg.device != "mps"),
                        input=promptTokenized.unsqueeze(0),
                        do_sample=True,
                        **kwargs,
                    ):
                        if flag == NPSteerType.STEERED:
                            steered_partial_result += model.to_string(result[0])  # type: ignore
                        else:
                            default_partial_result += model.to_string(result[0])  # type: ignore
                        to_return = make_steer_completion_chat_response(
                            steer_types,
                            steered_partial_result,
                            default_partial_result,
                            model,
                            promptTokenized,
                            inputPrompt,
                            custom_hf_model_id,
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
                for result in generate_stream(
                    model,
                    max_tokens_per_yield=TOKENS_PER_YIELD,
                    stop_at_eos=(model.cfg.device != "mps"),
                    input=promptTokenized.unsqueeze(0),
                    do_sample=True,
                    **kwargs,
                ):
                    partial_result += model.to_string(result[0])  # type: ignore
                    to_return = make_steer_completion_chat_response(
                        [steer_type],
                        partial_result,
                        partial_result,
                        model,
                        promptTokenized,
                        inputPrompt,
                        custom_hf_model_id,
                    )  # type: ignore
                    logger.info("to_return: %s", to_return)
                    yield format_sse_message(to_return.to_json())


def make_steer_completion_chat_response(
    steer_types: list[NPSteerType],
    steered_result: str,
    default_result: str,
    model: HookedTransformer,
    promptTokenized: torch.Tensor,
    promptChat: list[NPSteerChatMessage],
    custom_hf_model_id: str | None = None,
) -> SteerCompletionChatPost200Response:
    # Add tensor device logging
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
                )
            )

    return SteerCompletionChatPost200Response(
        outputs=steerChatResults,
        input=NPSteerChatResult(
            raw=model.to_string(promptTokenized),  # type: ignore
            chat_template=promptChat,
        ),
    )
