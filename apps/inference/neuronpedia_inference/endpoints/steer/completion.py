import logging
from typing import Any

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from neuronpedia_inference_client.models.np_logprob import NPLogprob
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
from nnterp import StandardizedTransformer
from transformer_lens import HookedTransformer

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
from neuronpedia_inference.utils import make_logprob_from_logits

logger = logging.getLogger(__name__)

router = APIRouter()

TOKENS_PER_YIELD = 1


def get_layer_num_from_sae_id(sae_id: str) -> int:
    return int(sae_id.split("-")[0]) if not sae_id.isdigit() else int(sae_id)


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

    # if the prompt doesn't start with the bos, prepend it
    bos_token = model.tokenizer.bos_token
    if not prompt.startswith(bos_token):
        prompt = bos_token + prompt

    tokens = []
    if isinstance(model, HookedTransformer):
        tokens = model.to_tokens(
            prompt, prepend_bos=model.cfg.tokenizer_prepends_bos, truncate=False
        )[0]
    elif isinstance(model, StandardizedTransformer):
        tokens = model.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ][0]
        if (request.n_logprobs is not None) and (request.n_logprobs > 0):
            request.n_logprobs = 0

    if len(tokens) > config.token_limit:
        logger.error(
            "Text too long: %s tokens, max is %s",
            len(tokens),
            config.token_limit,
        )
        return JSONResponse(
            content={
                "error": f"Text too long: {len(tokens)} tokens, max is {config.token_limit}"
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
        n_logprobs=(request.n_logprobs or 0),
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
    response = SteerCompletionPost200Response.from_json(results)
    if response is None:
        raise ValueError("Failed to parse response")
    # set exclude_none to True to omit the logprobs field when n_logprobs isn't set in the request, for backwards compatibility
    return JSONResponse(content=response.model_dump(exclude_none=True))


async def run_batched_generate(
    prompt: str,
    features: list[NPSteerFeature] | list[NPSteerVector],
    steer_types: list[NPSteerType],
    strength_multiplier: float,
    seed: int | None = None,
    steer_method: NPSteerMethod = NPSteerMethod.SIMPLE_ADDITIVE,
    normalize_steering: bool = False,
    use_stream_lock: bool = False,
    n_logprobs: int = 0,
    **kwargs: Any,
):
    async with await stream_lock(use_stream_lock):
        model = Model.get_instance()
        sae_manager = SAEManager.get_instance()

        if seed is not None:
            torch.manual_seed(seed)

        # for TransformerLens, use steering_hook
        def steering_hook(
            activations: torch.Tensor, hook: Any
        ) -> torch.Tensor:  # noqa: ARG001
            # Log activation device
            logger.info(f"Activations device: {activations.device}")

            for i, flag in enumerate(steer_types):
                if flag == NPSteerType.STEERED:
                    for feature in features:
                        steering_vector = torch.tensor(feature.steering_vector).to(
                            activations.device
                        )
                        logger.info(f"Steering vector device: {steering_vector.device}")

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
                        # get n_heads from the transformerlens model
                        if isinstance(
                            feature, NPSteerFeature
                        ) and "attn.hook_z" in sae_manager.get_sae_hook(feature.source):
                            n_heads = model.cfg.n_heads
                            d_head = model.cfg.d_head
                            steering_vector = steering_vector.view(n_heads, d_head)

                        coeff = strength_multiplier * feature.strength

                        if steer_method == NPSteerMethod.SIMPLE_ADDITIVE:
                            activations[i] += coeff * steering_vector

                        elif steer_method == NPSteerMethod.ORTHOGONAL_DECOMP:
                            projector = OrthogonalProjector(steering_vector)
                            activations[i] = projector.project(activations[i], coeff)
            return activations

        # Check if we need to generate both STEERED and DEFAULT
        generate_both = (
            NPSteerType.STEERED in steer_types and NPSteerType.DEFAULT in steer_types
        )

        tokenized = []
        if isinstance(model, HookedTransformer):
            tokenized = model.to_tokens(
                prompt, prepend_bos=model.cfg.tokenizer_prepends_bos, truncate=False
            )[0]
        elif isinstance(model, StandardizedTransformer):
            tokenized = model.tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]
        logger.info(f"Tokenized input device: {tokenized.device}")

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
                        editing_hooks = []

                    logprobs = []

                    with model.hooks(fwd_hooks=editing_hooks):  # type: ignore
                        for i, (result, logits) in enumerate(
                            model.generate_stream(
                                stop_at_eos=(model.cfg.device != "mps"),
                                input=tokenized.unsqueeze(0),
                                do_sample=True,
                                max_tokens_per_yield=TOKENS_PER_YIELD,
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

                            to_return = make_steer_completion_response(
                                steer_types,
                                steered_partial_result,
                                default_partial_result,
                                steered_logprobs,
                                default_logprobs,
                            )  # type: ignore
                            yield format_sse_message(to_return.to_json())

                elif isinstance(model, StandardizedTransformer):
                    logger.info("nnsight")
                    if kwargs.get("freq_penalty"):
                        logger.warning(
                            "freq_penalty is not supported for StandardizedTransformer models, it will be ignored"
                        )

                    # for nnsight we don't yield one token at a time (it hangs for some reason)
                    # so we just send one message at the end
                    with model.generate(
                        prompt,
                        temperature=kwargs.get("temperature"),
                        max_new_tokens=kwargs.get("max_new_tokens"),
                        do_sample=kwargs.get("do_sample", True),
                    ) as tracer:
                        with tracer.all():
                            token = model.generator.streamer.output
                            token_str = model.tokenizer.decode(token[-1])

                            to_append = token_str  # type: ignore

                            # if n_logprobs > 0:
                            #     current_logprobs = make_logprob_from_logits(
                            #         result,  # type: ignore
                            #         logits,  # type: ignore
                            #         model,
                            #         n_logprobs,
                            #     )
                            #     logprobs.append(current_logprobs)

                            if flag == NPSteerType.STEERED:
                                steered_partial_result_array.append(to_append)  # type: ignore
                                # steered_logprobs = logprobs.copy() or None

                                for feature in features:
                                    # get layer number
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
                                            f"Unsupported hook name for nnsight: {hook_name}"
                                        )

                                    # only supporting resid_pre and post in nnsight for now
                                    steering_vector = torch.tensor(
                                        feature.steering_vector
                                    ).to(model.device)

                                    if not torch.isfinite(steering_vector).all():
                                        raise ValueError(
                                            "Steering vector contains inf or nan values"
                                        )

                                    if normalize_steering:
                                        norm = torch.norm(steering_vector)
                                        if norm == 0:
                                            raise ValueError(
                                                "Zero norm steering vector"
                                            )
                                        steering_vector = steering_vector / norm

                                    coeff = strength_multiplier * feature.strength

                                    if steer_method == NPSteerMethod.SIMPLE_ADDITIVE:
                                        model.layers_output[
                                            layer - 1
                                        ] += coeff * steering_vector.to(
                                            model.layers_output[layer - 1].device
                                        )
                                    elif (
                                        steer_method == NPSteerMethod.ORTHOGONAL_DECOMP
                                    ):
                                        projector = OrthogonalProjector(
                                            steering_vector.to(
                                                model.layers_output[layer - 1].device
                                            )
                                        )
                                        model.layers_output[layer - 1] = (
                                            projector.project(
                                                model.layers_output[layer - 1], coeff
                                            )
                                        )
                            else:
                                default_partial_result_array.append(to_append)  # type: ignore
                                # default_logprobs = logprobs.copy() or None

                            # to_return = make_steer_completion_response(
                            #     steer_types,
                            #     "".join(steered_partial_result_array),
                            #     "".join(default_partial_result_array),
                            #     steered_logprobs,
                            #     default_logprobs,
                            # )  # type: ignore

            # for nnsight we don't yield one token at a time (it hangs for some reason)
            # so we just send one message at the end
            if isinstance(model, StandardizedTransformer):
                to_return = make_steer_completion_response(
                    steer_types,
                    "".join(steered_partial_result_array),
                    "".join(default_partial_result_array),
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

                with model.hooks(fwd_hooks=editing_hooks):  # type: ignore
                    partial_result = ""
                    logprobs = []

                    for i, (result, logits) in enumerate(
                        model.generate_stream(
                            stop_at_eos=(model.cfg.device != "mps"),
                            input=tokenized.unsqueeze(0),
                            do_sample=True,
                            max_tokens_per_yield=TOKENS_PER_YIELD,
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

                        to_return = make_steer_completion_response(
                            [steer_type],
                            partial_result,  # type: ignore
                            partial_result,  # type: ignore
                            logprobs or None,
                            logprobs or None,
                        )
                        yield format_sse_message(to_return.to_json())

            elif isinstance(model, StandardizedTransformer):
                logger.info("nnsight")
                if kwargs.get("freq_penalty"):
                    logger.warning(
                        "freq_penalty is not supported for StandardizedTransformer models, it will be ignored"
                    )

                with model.generate(
                    prompt,
                    temperature=kwargs.get("temperature"),
                    max_new_tokens=kwargs.get("max_new_tokens"),
                    do_sample=kwargs.get("do_sample", True),
                ) as tracer:
                    with tracer.all():
                        token = model.generator.streamer.output
                        partial_result_array.append(model.tokenizer.decode(token[-1]))  # type: ignore

                        if steer_type == NPSteerType.STEERED:
                            # steered_logprobs = logprobs.copy() or None

                            for feature in features:
                                # get layer number
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
                                        f"Unsupported hook name for nnsight: {hook_name}"
                                    )

                                # only supporting resid_pre and post in nnsight for now
                                steering_vector = torch.tensor(
                                    feature.steering_vector
                                ).to(model.device)

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
                                    model.layers_output[
                                        layer - 1
                                    ] += coeff * steering_vector.to(
                                        model.layers_output[layer - 1].device
                                    )
                                elif steer_method == NPSteerMethod.ORTHOGONAL_DECOMP:
                                    projector = OrthogonalProjector(
                                        steering_vector.to(
                                            model.layers_output[layer - 1].device
                                        )
                                    )
                                    model.layers_output[layer - 1] = projector.project(
                                        model.layers_output[layer - 1], coeff
                                    )
                        else:
                            pass  # not steering
                            # default_logprobs = logprobs.copy() or None
                to_return = make_steer_completion_response(
                    [steer_type],
                    "".join(partial_result_array),
                    "".join(partial_result_array),
                    None,
                    None,
                )  # type: ignore
                yield format_sse_message(to_return.to_json())


def make_steer_completion_response(
    steer_types: list[NPSteerType],
    steered_result: str,
    default_result: str,
    steered_logprobs: list[NPLogprob] | None = None,
    default_logprobs: list[NPLogprob] | None = None,
) -> SteerCompletionPost200Response:
    steerResults = []
    for steer_type in steer_types:
        if steer_type == NPSteerType.STEERED:
            steerResults.append(
                NPSteerCompletionResponseInner(
                    type=steer_type, output=steered_result, logprobs=steered_logprobs
                )
            )
        else:
            steerResults.append(
                NPSteerCompletionResponseInner(
                    type=steer_type, output=default_result, logprobs=default_logprobs
                )
            )
    return SteerCompletionPost200Response(outputs=steerResults)
