import logging
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass

import torch
from fastapi import Request

from neuronpedia_inference.config import Config
from neuronpedia_inference.sae_manager import SAE_TYPE, SAEManager
from neuronpedia_inference.schemas import NPSteerFeature, NPSteerMethod, NPSteerVector
from neuronpedia_inference.shared import limiter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SteeringSettings:
    """What to steer with, and how to apply it.

    These four decide the steering vectors and nothing else, so they are the complete
    input to building a backend steering spec. They also travel together unchanged from
    the endpoint all the way down to whichever backend runs the generation, which is
    why they move as one value instead of as four parallel parameters repeated across
    every function in that chain.
    """

    features: list[NPSteerFeature] | list[NPSteerVector]
    strength_multiplier: float
    steer_method: NPSteerMethod = NPSteerMethod.SIMPLE_ADDITIVE
    normalize_steering: bool = False


async def stream_lock(is_stream: bool):
    # Streaming generation runs AFTER the handler returns its StreamingResponse (so
    # the with_request_lock decorator has already released), and must hold a slot for
    # the stream's lifetime. With the per-request demux, steering is per-request-safe,
    # so this takes a NON-exclusive slot (concurrent streams on vLLM; still serialized
    # by the single mutex off vLLM). The non-stream path already holds a slot via the
    # decorator, so it gets a no-op.
    if is_stream:
        return limiter.slot(exclusive=False)

    class DummyLock:
        async def __aenter__(self):
            pass

        async def __aexit__(self, *args):  # type: ignore
            pass

    return DummyLock()


async def stop_when_client_leaves(
    frames: AsyncIterator[str],
    http_request: Request,
    label: str,
) -> AsyncIterator[str]:
    """Pass frames through, and stop generating once nobody is listening.

    The webapp now abandons a pod that has not answered in time and tries another, so without
    this a slow pod would finish the whole completion for a client that left -- holding a
    request slot the retry then queues behind. Checked per frame, which is per token.

    ``is_disconnected()`` only answers truthfully while every middleware is pure ASGI; see the
    middleware section of ``server.py``.
    """
    async for frame in frames:
        if await http_request.is_disconnected():
            logger.info("[%s] Client disconnected; aborting generation.", label)
            return
        yield frame


def format_sse_message(data: str) -> str:
    return f"data: {data}\n\n"


def remove_sse_formatting(data: str) -> str:
    if data.startswith("data: "):
        data = data[6:]  # Remove "data: " prefix
    return data.rstrip("\n")


class NeuronSteerError(ValueError):
    """A neuron steer this pod cannot run. The handlers answer it with a 400."""


def neuron_steering_vector(index: int, d_mlp: int | None) -> list[float]:
    """One-hot over the MLP neurons, so an additive steer adds its strength to one neuron's activation."""
    if not d_mlp:
        raise NeuronSteerError("This model does not state its MLP width, so its neurons cannot be steered")
    if not 0 <= index < d_mlp:
        raise NeuronSteerError(f"Neuron index {index} is out of range: this model has {d_mlp} neurons per layer")
    vector = [0.0] * d_mlp
    vector[index] = 1.0
    return vector


def refuse_split_neuron_steer(features: list[NPSteerFeature], steer_method: NPSteerMethod) -> None:
    """Raise when a neuron steer needs the whole MLP activation and vLLM splits it across GPUs.

    Under tensor parallelism each GPU holds a block of the neurons. An additive write is cut to
    each block, but the other methods project onto the whole vector, so the engine refuses them.
    Asked here so the caller gets a 400, not an error in the middle of a stream.
    """
    if steer_method == NPSteerMethod.SIMPLE_ADDITIVE:
        return
    config = Config.get_instance()
    if config.num_gpus <= 1 or not str(config.backend).startswith("vllm"):
        return
    sae_manager = SAEManager.get_instance()
    if any(sae_manager.get_sae_type(f.source) == SAE_TYPE.NEURONS for f in features):
        raise NeuronSteerError(
            f"Neurons on this model can only be steered with {NPSteerMethod.SIMPLE_ADDITIVE.value}: "
            f"it runs on {config.num_gpus} GPUs, and each holds only part of every MLP layer"
        )


def process_features_vectorized(features: list[NPSteerFeature]):
    sae_manager = SAEManager.get_instance()
    # Group features by source
    source_groups: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, feature in enumerate(features):
        source_groups[feature.source].append((i, int(feature.index)))

    # Process by each source
    for source, indices in source_groups.items():
        if sae_manager.get_sae_type(source) == SAE_TYPE.NEURONS:
            if not sae_manager.mlp_width_uniform:
                raise NeuronSteerError("This model's MLP width differs by layer, so its neurons cannot be steered")
            for feature_idx, index in indices:
                features[feature_idx].steering_vector = neuron_steering_vector(index, sae_manager.mlp_width)
            continue
        sae = sae_manager.get_sae(source)
        feature_indices = torch.tensor([idx for _, idx in indices], device=sae.W_dec.device)
        steering_vectors = sae.W_dec[feature_indices]

        # Assign steering vectors back to features
        for (feature_idx, _), steer_vector in zip(indices, steering_vectors):
            features[feature_idx].steering_vector = steer_vector

    return features
