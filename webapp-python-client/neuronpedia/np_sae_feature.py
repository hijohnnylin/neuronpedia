from dataclasses import dataclass


@dataclass
class SAEFeature:
    modelId: str
    source: str
    index: str
    jsonData: str

    @classmethod
    def get(cls, model_id: str, source: str, index: str) -> "SAEFeature":

        from neuronpedia.requests.sae_feature_request import SAEFeatureRequest

        request = SAEFeatureRequest()
        return request.get(model_id, source, index)
