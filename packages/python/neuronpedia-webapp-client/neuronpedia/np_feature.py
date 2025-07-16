from dataclasses import dataclass
from typing import List

from neuronpedia.np_activation import Activation
from neuronpedia.np_explanation import Explanation


@dataclass
class Logit:
    token: str
    value: float


@dataclass
class Feature:
    modelId: str
    source: str
    index: int
    density: float | None = None
    top_logits: List[Logit] | None = None
    bottom_logits: List[Logit] | None = None
    explanations: List[Explanation] | None = None
    activations: List[Activation] | None = None

    @classmethod
    def get(cls, model_id: str, source: str, index: int) -> "Feature":
        from neuronpedia.requests.feature_request import FeatureRequest

        request = FeatureRequest()
        return request.get(model_id, source, index)

    def open_in_browser(self):
        import webbrowser

        webbrowser.open(
            f"https://neuronpedia.org/{self.modelId}/{self.source}/{self.index}"
        )
