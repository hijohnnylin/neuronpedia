import logging
from typing import Any, AsyncGenerator

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from neuronpedia_inference_client.models.np_steer_completion_response_inner import (
    NPSteerCompletionResponseInner,
)
from neuronpedia_inference_client.models.np_steer_feature import NPSteerFeature
from neuronpedia_inference_client.models.np_steer_method import NPSteerMethod
from neuronpedia_inference_client.models.np_steer_type import NPSteerType
from neuronpedia_inference_client.models.np_steer_vector import NPSteerVector
from neuronpedia_inference_client.models.steer_completion_post200_response import (
    SteerCompletionPost200Response,
)
from neuronpedia_inference_client.models.steer_completion_request import (
    SteerCompletionRequest,
)

from neuronpedia_inference.config import Config
from neuronpedia_inference.inference_utils.steering import (
    OrthogonalProjector,
    format_sse_message,
    process_features_vectorized,
    remove_sse_formatting,
    stream_lock,
)
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.shared import Model, with_request_lock

logger = logging.getLogger(__name__)

router = APIRouter()

TOKENS_PER_YIELD = 1


@router.post("/steer/completion")
@with_request_lock()
async def completion(request: SteerCompletionRequest):
    config = Config.get_instance()
    model = Model.get_instance()
    steer_method = request.steer_method
    normalize_steering = request.normalize_steering

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

    prompt = request.prompt

    tokens = model.to_tokens(
        prompt, prepend_bos=model.cfg.tokenizer_prepends_bos, truncate=False
    )[0]

    if len(tokens) > config.TOKEN_LIMIT:
        logger.error(
            "Text too long: %s tokens, max is %s",
            len(tokens),
            config.TOKEN_LIMIT,
        )
        return JSONResponse(
            content={
                "error": f"Text too long: {len(tokens)} tokens, max is {config.TOKEN_LIMIT}"
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
        prompt=prompt,
        features=features,
        steer_types=request.types,
        strength_multiplier=float(request.strength_multiplier),
        seed=int(request.seed),
        temperature=float(request.temperature),
        freq_penalty=float(request.freq_penalty),
        max_new_tokens=int(request.n_completion_tokens),
        steer_method=steer_method,
        normalize_steering=normalize_steering,
        use_stream_lock=request.stream if request.stream is not None else False,
    )

    if request.stream:
        logger.info("Streaming response")
        return StreamingResponse(generator, media_type="text/event-stream")
    logger.info("Non-streaming response")
    # Get the last item from the generator
    item = None
    async for item in generator:
        pass
    if item is None:
        raise ValueError("Generator yielded no items")
    results = remove_sse_formatting(item)
    return SteerCompletionPost200Response.from_json(results)


async def run_batched_generate(
    prompt: str,
    features: list[NPSteerFeature] | list[NPSteerVector],
    steer_types: list[NPSteerType],
    strength_multiplier: float,
    seed: int | None = None,
    steer_method: NPSteerMethod = NPSteerMethod.SIMPLE_ADDITIVE,
    normalize_steering: bool = False,
    use_stream_lock: bool = False,
    **kwargs: Any,
):
    async with await stream_lock(use_stream_lock):
        model = Model.get_instance()
        sae_manager = SAEManager.get_instance()

        # Add device logging
        logger.info(f"Model device: {model.cfg.device}")

        # Check if we need to generate both STEERED and DEFAULT
        generate_both = (
            NPSteerType.STEERED in steer_types and NPSteerType.DEFAULT in steer_types
        )

        tokenized = model.to_tokens(prompt)[0]
        logger.info(f"Tokenized input device: {tokenized.device}")

        if generate_both:
            # Try batch generation with different steering for each batch item
            logger.info("Attempting batch generation for steered and default")

            # Pre-process features and create steering vectors
            processed_steering_vectors = []
            for feature in features:
                steering_vector = torch.tensor(feature.steering_vector).to(
                    model.cfg.device
                )
                if normalize_steering:
                    norm = torch.norm(steering_vector)
                    if norm > 0:
                        steering_vector = steering_vector / norm
                processed_steering_vectors.append(
                    (feature, steering_vector, strength_multiplier * feature.strength)
                )

            # Create batched input (2 copies of the same prompt)
            batched_input = tokenized.unsqueeze(0).repeat(2, 1)  # Shape: [2, seq_len]

            def batched_steering_hook(
                activations: torch.Tensor, hook: Any
            ) -> torch.Tensor:  # noqa: ARG001
                # Apply steering only to the first item in batch (index 0)
                for _, steering_vector, coeff in processed_steering_vectors:
                    if steer_method == NPSteerMethod.SIMPLE_ADDITIVE:
                        activations[0] += coeff * steering_vector
                    elif steer_method == NPSteerMethod.ORTHOGONAL_DECOMP:
                        projector = OrthogonalProjector(steering_vector)
                        activations[0] = projector.project(activations[0], coeff)
                # Leave activations[1] unmodified for DEFAULT
                return activations

            # Set up hooks
            model.reset_hooks()
            editing_hooks = [
                (
                    (
                        sae_manager.get_sae_hook(feature.source)
                        if isinstance(feature, NPSteerFeature)
                        else feature.hook
                    ),
                    batched_steering_hook,
                )
                for feature in features
            ]

            # Try batched generation
            try:
                steered_result = ""
                default_result = ""

                with model.hooks(fwd_hooks=editing_hooks):  # type: ignore
                    for i, result in enumerate(
                        model.generate_stream(
                            stop_at_eos=(model.cfg.device != "mps"),
                            input=batched_input,
                            do_sample=True,
                            max_tokens_per_yield=TOKENS_PER_YIELD,
                            **kwargs,
                        )
                    ):
                        # Extract results for both batch items
                        if i == 0:
                            steered_append = model.to_string(result[0][1:])  # type: ignore
                            default_append = model.to_string(result[1][1:])  # type: ignore
                        else:
                            steered_append = model.to_string(result[0])  # type: ignore
                            default_append = model.to_string(result[1])  # type: ignore

                        steered_result += str(steered_append)  # type: ignore
                        default_result += str(default_append)  # type: ignore

                        to_return = make_steer_completion_response(
                            steer_types, steered_result, default_result
                        )
                        yield format_sse_message(to_return.to_json())

            except Exception as e:
                logger.warning(
                    f"Batch generation failed, falling back to sequential: {e}"
                )
                # Fall back to sequential generation
                async for item in sequential_generate(
                    prompt,
                    features,
                    steer_types,
                    strength_multiplier,
                    seed,
                    steer_method,
                    normalize_steering,
                    tokenized,
                    **kwargs,
                ):
                    yield item

        else:
            # Single generation case
            steer_type = steer_types[0]
            async for partial_result in generate_single_completion(
                prompt=prompt,
                features=features,
                steer_type=steer_type,
                strength_multiplier=strength_multiplier,
                seed=seed,
                steer_method=steer_method,
                normalize_steering=normalize_steering,
                tokenized=tokenized,
                **kwargs,
            ):
                to_return = make_steer_completion_response(
                    [steer_type],
                    partial_result,
                    partial_result,
                )
                yield format_sse_message(to_return.to_json())


async def sequential_generate(
    prompt: str,
    features: list[NPSteerFeature] | list[NPSteerVector],
    steer_types: list[NPSteerType],
    strength_multiplier: float,
    seed: int | None,
    steer_method: NPSteerMethod,
    normalize_steering: bool,
    tokenized: torch.Tensor,
    **kwargs: Any,
):
    """Fallback to sequential generation if batch generation fails."""
    steered_partial_result = ""
    default_partial_result = ""

    # Generate STEERED and DEFAULT separately
    for flag in [NPSteerType.STEERED, NPSteerType.DEFAULT]:
        async for partial_result in generate_single_completion(
            prompt=prompt,
            features=features,
            steer_type=flag,
            strength_multiplier=strength_multiplier,
            seed=seed,
            steer_method=steer_method,
            normalize_steering=normalize_steering,
            tokenized=tokenized,
            **kwargs,
        ):
            if flag == NPSteerType.STEERED:
                steered_partial_result = partial_result
            else:
                default_partial_result = partial_result

            to_return = make_steer_completion_response(
                steer_types, steered_partial_result, default_partial_result
            )
            yield format_sse_message(to_return.to_json())


async def generate_single_completion(
    prompt: str,
    features: list[NPSteerFeature] | list[NPSteerVector],
    steer_type: NPSteerType,
    strength_multiplier: float,
    seed: int | None,
    steer_method: NPSteerMethod,
    normalize_steering: bool,
    tokenized: torch.Tensor,
    **kwargs: Any,
) -> AsyncGenerator[str, None]:
    """Generate a single completion (steered or default)."""
    model = Model.get_instance()
    sae_manager = SAEManager.get_instance()

    if seed is not None:
        torch.manual_seed(seed)

    def steering_hook(activations: torch.Tensor, hook: Any) -> torch.Tensor:  # noqa: ARG001
        if steer_type == NPSteerType.STEERED:
            for feature in features:
                steering_vector = torch.tensor(feature.steering_vector).to(
                    activations.device
                )

                if not torch.isfinite(steering_vector).all():
                    raise ValueError("Steering vector contains inf or nan values")

                if normalize_steering:
                    norm = torch.norm(steering_vector)
                    if norm == 0:
                        raise ValueError("Zero norm steering vector")
                    steering_vector = steering_vector / norm

                coeff = strength_multiplier * feature.strength

                if steer_method == NPSteerMethod.SIMPLE_ADDITIVE:
                    activations[0] += coeff * steering_vector
                elif steer_method == NPSteerMethod.ORTHOGONAL_DECOMP:
                    projector = OrthogonalProjector(steering_vector)
                    activations[0] = projector.project(activations[0], coeff)
        return activations

    model.reset_hooks()
    editing_hooks = []

    if steer_type == NPSteerType.STEERED:
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

    partial_result = ""
    with model.hooks(fwd_hooks=editing_hooks):  # type: ignore
        for i, result in enumerate(
            model.generate_stream(
                stop_at_eos=(model.cfg.device != "mps"),
                input=tokenized.unsqueeze(0),
                do_sample=True,
                max_tokens_per_yield=TOKENS_PER_YIELD,
                **kwargs,
            )
        ):
            if i == 0:
                to_append = model.to_string(result[0][1:])  # type: ignore
            else:
                to_append = model.to_string(result[0])  # type: ignore
            partial_result += to_append  # type: ignore
            yield partial_result


def make_steer_completion_response(
    steer_types: list[NPSteerType],
    steered_result: str,
    default_result: str,
) -> SteerCompletionPost200Response:
    steerResults = []
    for steer_type in steer_types:
        if steer_type == NPSteerType.STEERED:
            steerResults.append(
                NPSteerCompletionResponseInner(type=steer_type, output=steered_result)
            )
        else:
            steerResults.append(
                NPSteerCompletionResponseInner(type=steer_type, output=default_result)
            )
    return SteerCompletionPost200Response(outputs=steerResults)
