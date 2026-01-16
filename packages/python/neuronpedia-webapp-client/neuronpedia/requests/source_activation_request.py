from dataclasses import dataclass
from typing import List, Optional

from neuronpedia.np_activation import Activation
from neuronpedia.requests.base_request import NPRequest


@dataclass
class SourceActivationResult:
    modelId: str
    layer: str
    index: str
    values: List[float]


@dataclass
class SourceActivationRequest(NPRequest):
    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        super().__init__("search-all", api_key=api_key)

    def _get_source_set_from_source(self, source: str | list[str]) -> str:
        # if source is a list, return the source set name for the first source, ensuring that all sources have the same source set name
        if isinstance(source, list):
            source_set_name = source[0].split("-", 1)[1]
            for s in source:
                if s.split("-", 1)[1] != source_set_name:
                    raise ValueError(
                        f"All sources must have the same source set name, got {source_set_name} and {s.split('-', 1)[1]}"
                    )
            return source_set_name
        # if source is a string, return the source set name for the source
        return source.split("-", 1)[1]

    def compute_top_features_for_text(
        self,
        model_id: str,
        source: str | list[str],
        text: str,
        ignore_bos: bool = True,
        sort_results_by_activations_at_token_indexes: List[int] = [],
        density_threshold: float = -1,
        num_results: int = 20,
    ) -> list[Activation]:
        payload = {
            "text": text,
            "modelId": model_id,
            "sourceSet": self._get_source_set_from_source(source),
            "selectedLayers": [source] if isinstance(source, str) else source,
            "ignoreBos": ignore_bos,
            "sortIndexes": sort_results_by_activations_at_token_indexes,
            "densityThreshold": density_threshold,
            "numResults": num_results,
        }

        data = self.send_request(method="POST", json=payload, uri="")

        feature_results = [
            SourceActivationResult(
                modelId=item["modelId"],
                layer=item["layer"],
                index=item["index"],
                values=item["values"],
            )
            for item in data["result"]
        ]

        tokens = data["tokens"]

        return [
            Activation(
                modelId=result.modelId,
                source=result.layer,
                index=result.index,
                tokens=tokens,
                values=result.values,
            )
            for result in feature_results
        ]

    def compute_top_features_for_texts(
        self,
        model_id: str,
        source: str | list[str],
        texts: List[str],
        ignore_bos: bool = True,
        sort_results_by_activations_at_token_indexes: List[int] = [],
        density_threshold: float = -1,
        num_results: int = 20,
    ) -> list[list[Activation]]:
        payload = {
            "text": texts,
            "modelId": model_id,
            "sourceSet": self._get_source_set_from_source(source),
            "selectedLayers": [source] if isinstance(source, str) else source,
            "ignoreBos": ignore_bos,
            "sortIndexes": sort_results_by_activations_at_token_indexes,
            "densityThreshold": density_threshold,
            "numResults": num_results,
        }

        data = self.send_request(method="POST", json=payload, uri="")

        # response is a list of results under "results"
        results = data["results"]

        # iterate through each result to create array of SourceActivationResult for each result
        # each result has its own result array
        batch_feature_results = []
        for result in results:
            batch_feature_result = {"tokens": result["tokens"], "results": []}
            for item in result["result"]:
                batch_feature_result["results"].append(
                    SourceActivationResult(
                        modelId=item["modelId"],
                        layer=item["layer"],
                        index=item["index"],
                        values=item["values"],
                    )
                )
            batch_feature_results.append(batch_feature_result)

        # now return the array of array of activations, one for each prompt, in the same order as the input prompts
        batch_activations = []
        for batch_feature_result in batch_feature_results:
            batch_activations.append(
                [
                    Activation(
                        modelId=result.modelId,
                        source=result.layer,
                        index=result.index,
                        tokens=batch_feature_result["tokens"],
                        values=result.values,
                    )
                    for result in batch_feature_result["results"]
                ]
            )

        return batch_activations
