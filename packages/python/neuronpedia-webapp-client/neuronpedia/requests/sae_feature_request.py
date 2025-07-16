import json
from typing import Optional

from neuronpedia.np_sae_feature import SAEFeature
from neuronpedia.requests.base_request import NPRequest


class SAEFeatureRequest(NPRequest):
    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        super().__init__("feature", api_key=api_key)

    def get(self, model_id: str, source: str, index: str) -> SAEFeature:
        result = self.send_request(method="GET", uri=f"{model_id}/{source}/{index}")

        return SAEFeature(
            modelId=result["modelId"],
            source=result["layer"],
            index=result["index"],
            jsonData=json.dumps(result),
        )
