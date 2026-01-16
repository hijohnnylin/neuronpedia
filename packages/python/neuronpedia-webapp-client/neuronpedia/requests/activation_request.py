from typing import List, Optional

from neuronpedia.np_activation import Activation
from neuronpedia.requests.base_request import NPRequest
from requests import Response


class ActivationRequest(NPRequest):
    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        super().__init__("activation", api_key=api_key)

    def compute_all_source_activations_for_text(
        self, model_id: str, source: str, text: str | list[str]
    ):
        payload = {
            "customText": text,
            "modelId": model_id,
            "source": source,
        }
        data = self.send_request(method="POST", json=payload, uri="source")

        # allow the consumer to parse the results as they see fit
        return data["results"]

    def compute_activation_for_text(
        self, model_id: str, source: str, index: str, text: str
    ) -> Activation:
        if isinstance(index, int):
            index = str(index)
        payload = {
            "feature": {
                "modelId": model_id,
                "layer": source,
                "index": index,
            },
            "customText": text,
        }

        result = self.send_request(method="POST", json=payload, uri="new")

        return Activation(
            modelId=model_id,
            source=source,
            index=index,
            tokens=result["tokens"],
            values=result["values"],
        )

    def compute_activation_for_texts(
        self, model_id: str, source: str, index: str, texts: List[str]
    ) -> List[Activation]:
        if isinstance(index, int):
            index = str(index)
        payload = {
            "feature": {
                "modelId": model_id,
                "layer": source,
                "index": index,
            },
            "customText": texts,
        }

        results = self.send_request(method="POST", json=payload, uri="new")
        return [
            Activation(
                modelId=model_id,
                source=source,
                index=index,
                tokens=result["tokens"],
                values=result["values"],
            )
            for result in results
        ]

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
