from typing import List, Optional
from dataclasses import dataclass

from neuronpedia.requests.base_request import NPRequest
from neuronpedia.types.common.feature import NPFeature


@dataclass
class FeatureWithTopActivation:
    feature: NPFeature
    activationValue: float


@dataclass
class TopKByTokenResult:
    position: int
    token: str
    topFeatures: List[FeatureWithTopActivation]


@dataclass
class TopKByTokenRequest(NPRequest):
    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        super().__init__("search-topk-by-token", api_key=api_key)

    def compute_top_features_by_token_for_text(
        self,
        model_id: str,
        source: str,
        text: str,
        num_results: int = 5,
        ignore_bos: bool = True,
        density_threshold: float = 0.0,
    ) -> List[TopKByTokenResult]:
        payload = {
            "modelId": model_id,
            "source": source,
            "text": text,
            "numResults": num_results,
            "ignoreBos": ignore_bos,
            "densityThreshold": density_threshold,
        }

        data = self.send_request(method="POST", json=payload, uri="")

        token_results = [
            TopKByTokenResult(
                position=item["position"],
                token=item["token"],
                topFeatures=[
                    FeatureWithTopActivation(
                        feature=NPFeature(model=model_id, source=source, index=feature["featureIndex"]),
                        activationValue=feature["activationValue"],
                    )
                    for feature in item["topFeatures"]
                ],
            )
            for item in data["results"]
        ]

        return token_results

    def compute_top_features_by_token_for_texts(
        self,
        model_id: str,
        source: str,
        texts: List[str],
        num_results: int = 10,
        ignore_bos: bool = True,
        density_threshold: float = 0.0,
    ) -> List[List[TopKByTokenResult]]:
        payload = {
            "modelId": model_id,
            "source": source,
            "text": texts,
            "numResults": num_results,
            "ignoreBos": ignore_bos,
            "densityThreshold": density_threshold,
        }

        data = self.send_request(method="POST", json=payload, uri="")

        results = data["results"]

        return [
            [
                TopKByTokenResult(
                    position=item["position"],
                    token=item["token"],
                    topFeatures=[
                        FeatureWithTopActivation(
                            feature=NPFeature(model=model_id, source=source, index=feature["featureIndex"]),
                            activationValue=feature["activationValue"],
                        )
                        for feature in item["topFeatures"]
                    ],
                )
                for item in result["results"]
            ]
            for result in results
        ]
