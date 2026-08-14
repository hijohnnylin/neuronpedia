from dataclasses import dataclass
import os
from typing import List

from neuronpedia.requests.activation_request import Activation


@dataclass
class NPVector:
    """A Vector returned by the Neuronpedia API."""

    label: str
    model_id: str
    source: str
    index: str
    values: List[float]
    hook_name: str
    default_steer_strength: float | None
    url: str | None = None

    def __post_init__(self):
        if not self.url:
            USE_LOCALHOST = os.getenv("USE_LOCALHOST", False)
            BASE_URL = "https://neuronpedia.org/api" if not USE_LOCALHOST else "http://localhost:3000/api"
            self.url = f"{BASE_URL}/{self.model_id}/{self.source}/{self.index}"

    def __eq__(self, other: "NPVector") -> bool:
        return (
            self.model_id == other.model_id
            and self.source == other.source
            and self.index == other.index
            and self.label == other.label
            and self.hook_name == other.hook_name
            and self.values == other.values
            and self.default_steer_strength == other.default_steer_strength
        )

    def delete(self):
        # import here to avoid circular import
        from neuronpedia.requests.vector_request import VectorRequest

        return VectorRequest().delete(self)

    def steer_chat(self, steered_chat_messages: list[dict[str, str]]):
        # import here to avoid circular import
        from neuronpedia.requests.steer_request import SteerChatRequest

        return SteerChatRequest().steer(
            model_id=self.model_id, vectors=[self], steered_chat_messages=steered_chat_messages
        )

    def steer_completion(self, prompt: str):
        # import here to avoid circular import
        from neuronpedia.requests.steer_request import SteerCompletionRequest

        return SteerCompletionRequest().steer(model_id=self.model_id, vectors=[self], prompt=prompt)

    def compute_activation_for_text(self, text: str) -> Activation:
        # import here to avoid circular import
        from neuronpedia.requests.activation_request import ActivationRequest

        return ActivationRequest().compute_activation_for_text(self.model_id, self.source, self.index, text)

    def upload_activations(self, activations: List[Activation]):
        from neuronpedia.requests.activation_request import ActivationRequest

        return ActivationRequest().upload_batch(self.model_id, self.source, self.index, activations)

    @classmethod
    def get(cls, model_id: str, source: str, index: str) -> "NPVector":
        # import here to avoid circular import
        from neuronpedia.requests.vector_request import VectorRequest

        return VectorRequest().get(model_id, source, index)

    @classmethod
    def get_owned(cls) -> List["NPVector"]:
        # import here to avoid circular import
        from neuronpedia.requests.vector_request import VectorRequest

        return VectorRequest().get_owned()

    @classmethod
    def new(
        cls,
        label: str,
        model_id: str,
        layer_num: int,
        hook_type: str,
        vector: list[float],
        default_steer_strength: float | None = 10,
    ) -> "NPVector":
        # import here to avoid circular import
        from neuronpedia.requests.vector_request import VectorRequest

        np_vector = VectorRequest().new(
            label=label,
            model_id=model_id,
            layer_num=layer_num,
            hook_type=hook_type,
            vector=vector,
            default_steer_strength=default_steer_strength,
        )

        return np_vector
