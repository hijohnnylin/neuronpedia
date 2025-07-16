from typing import Optional

from neuronpedia.np_vector import NPVector
from neuronpedia.requests.base_request import NPRequest
from requests import Response

ChatMessage = dict[str, str]


class SteerChatRequest(NPRequest):
    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        super().__init__("steer-chat", api_key=api_key)

    def steer(
        self,
        model_id: str,
        vectors: list[NPVector],
        default_chat_messages: list[ChatMessage] = [
            {"role": "user", "content": "Write a one sentence story."}
        ],
        steered_chat_messages: list[ChatMessage] = [
            {"role": "user", "content": "Write a one sentence story."}
        ],
        temperature: float = 0.5,
        n_tokens: int = 32,
        freq_penalty: float = 2,
        seed: int = 16,
        strength_multiplier: float = 4,
        steer_special_tokens: bool = True,
    ) -> Response:
        # convert the vectors to the feature format
        features = [
            {
                "modelId": vector.model_id,
                "layer": vector.source,
                "index": vector.index,
                "strength": vector.default_steer_strength,
            }
            for vector in vectors
        ]
        payload = {
            "modelId": model_id,
            "features": features,
            "defaultChatMessages": default_chat_messages,
            "steeredChatMessages": steered_chat_messages,
            "temperature": temperature,
            "n_tokens": n_tokens,
            "freq_penalty": freq_penalty,
            "seed": seed,
            "strength_multiplier": strength_multiplier,
            "steer_special_tokens": steer_special_tokens,
        }
        return self.send_request(method="POST", json=payload)


class SteerCompletionRequest(NPRequest):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("steer", api_key=api_key)

    def steer(
        self,
        model_id: str,
        vectors: list[NPVector],
        prompt: str,
        temperature: float = 0.5,
        n_tokens: int = 32,
        freq_penalty: float = 2,
        seed: int = 42,
        strength_multiplier: float = 4,
    ) -> Response:
        # convert the vectors to the feature format
        features = [
            {
                "modelId": vector.model_id,
                "layer": vector.source,
                "index": vector.index,
                "strength": vector.default_steer_strength,
            }
            for vector in vectors
        ]
        payload = {
            "modelId": model_id,
            "features": features,
            "prompt": prompt,
            "temperature": temperature,
            "n_tokens": n_tokens,
            "freq_penalty": freq_penalty,
            "seed": seed,
            "strength_multiplier": strength_multiplier,
        }
        return self.send_request(method="POST", json=payload)
