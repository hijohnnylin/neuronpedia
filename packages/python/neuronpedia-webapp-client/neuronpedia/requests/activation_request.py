from typing import List, Optional, TypedDict

from neuronpedia.requests.base_request import NPRequest
from requests import Response


class Activation(TypedDict):
    tokens: List[str]
    values: List[float]

    def __init__(self, tokens: List[str], values: List[float]):
        if len(tokens) != len(values):
            raise ValueError("tokens and activation values must have the same length")
        self["tokens"] = tokens
        self["values"] = values


class ActivationRequest(NPRequest):
    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        super().__init__("activation", api_key=api_key)

    def compute_activation_for_text(
        self, model_id: str, source: str, index: str, text: str
    ) -> Activation:
        payload = {
            "feature": {
                "modelId": model_id,
                "layer": source,
                "index": index,
            },
            "customText": text,
        }

        result = self.send_request(method="POST", json=payload, uri="new")

        return Activation(tokens=result["tokens"], values=result["values"])

    def upload_batch(
        self,
        model_id: str,
        source: str,
        index: str,
        activations: List[Activation],
    ) -> Response:
        payload = {
            "modelId": model_id,
            "source": source,
            "index": index,
            "activations": activations,
        }
        return self.send_request(method="POST", json=payload, uri="upload-batch")
