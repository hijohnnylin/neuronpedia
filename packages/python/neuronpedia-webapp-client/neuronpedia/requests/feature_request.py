from typing import List, Optional

from neuronpedia.np_activation import Activation
from neuronpedia.np_explanation import Explanation
from neuronpedia.np_feature import Feature
from neuronpedia.requests.base_request import NPRequest

MAX_FEATURES_PER_BATCH = 128


class FeatureRequest(NPRequest):
    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        super().__init__("feature", api_key=api_key)

    def get(self, model_id: str, source: str, index: int) -> Feature:
        result = self.send_request(method="GET", uri=f"{model_id}/{source}/{index}")

        explanations = [
            Explanation.from_np_db_json(explanation)
            for explanation in result["explanations"]
        ]

        activations = [
            Activation.from_np_db_json(activation)
            for activation in result["activations"]
        ]

        return Feature(
            modelId=result["modelId"],
            source=result["layer"],
            index=result["index"],
            density=result["frac_nonzero"],
            explanations=explanations,
            activations=activations,
        )

    def upload_batch(self, model_id: str, source: str, features: List[Feature]):
        if len(features) > MAX_FEATURES_PER_BATCH:
            raise ValueError(
                f"Cannot upload more than {MAX_FEATURES_PER_BATCH} features at a time"
            )
        if len(features) == 0:
            raise ValueError("Cannot upload 0 features")

        # make the features payload
        features_payload = []
        for feature in features:
            to_append = {
                "index": feature.index,
                "density": feature.density,
                "activations": [],
                "explanations": [],
                "topLogits": [
                    {"token": logit.token, "value": logit.value}
                    for logit in feature.top_logits
                ]
                if feature.top_logits is not None
                else None,
                "bottomLogits": [
                    {"token": logit.token, "value": logit.value}
                    for logit in feature.bottom_logits
                ]
                if feature.bottom_logits is not None
                else None,
            }
            if feature.explanations is not None:
                for explanation in feature.explanations:
                    to_append["explanations"].append(
                        {
                            "text": explanation.text,
                            "methodName": explanation.method,
                            "modelName": explanation.explainer_model,
                        }
                    )
            if feature.activations is not None:
                for activation in feature.activations:
                    to_append["activations"].append(
                        {
                            "tokens": activation.tokens,
                            "values": activation.values,
                            "quantileMax": activation.quantileMax,
                            "quantileMin": activation.quantileMin,
                            "quantileFraction": activation.quantileFraction,
                        }
                    )
            features_payload.append(to_append)
        payload = {
            "modelId": model_id,
            "source": source,
            "features": features_payload,
        }
        return self.send_request(method="POST", uri="upload-batch", json=payload)
