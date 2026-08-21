from collections import defaultdict
from dataclasses import dataclass

import torch

from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.schemas import NPSteerFeature, NPSteerMethod, NPSteerVector
from neuronpedia_inference.shared import limiter


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


def format_sse_message(data: str) -> str:
    return f"data: {data}\n\n"


def remove_sse_formatting(data: str) -> str:
    if data.startswith("data: "):
        data = data[6:]  # Remove "data: " prefix
    return data.rstrip("\n")


def process_features_vectorized(features: list[NPSteerFeature]):
    # Group features by source
    source_groups: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, feature in enumerate(features):
        source_groups[feature.source].append((i, int(feature.index)))

    # Process by each source
    for source, indices in source_groups.items():
        sae = SAEManager.get_instance().get_sae(source)
        feature_indices = torch.tensor([idx for _, idx in indices], device=sae.W_dec.device)
        steering_vectors = sae.W_dec[feature_indices]

        # Assign steering vectors back to features
        for (feature_idx, _), steer_vector in zip(indices, steering_vectors):
            features[feature_idx].steering_vector = steer_vector

    return features
