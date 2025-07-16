from typing import Optional

from neuronpedia.np_source_set import SourceSet
from neuronpedia.requests.base_request import NPRequest


class SourceSetRequest(NPRequest):
    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        super().__init__("source-set", api_key=api_key)

    def new(
        self,
        name: str,
        model_id: str,
        description: str | None = None,
        url: str | None = None,
    ) -> SourceSet:
        """
        Create a new source set on Neuronpedia.
        Before uploading features, you must create a source set, which is a collection of sources.
        Each *source* is a set of features for a layer of a model. For example, one sparse autoencoder, which can contain thousands of features, is one source.
        Sources are named as [layer]-[source_set_name]. For example, "20-gemmascope-res-16k" is the residual stream of the 20th layer, with 16k features. Its source set name is "gemmascope-res-16k".

        When you create a source set, it automatically creates a source for each layer of the model.
        If you create a source set called "puppyscope-res-16k" for a 3 layer "pup-model-3" model, it will create 3 sources: "0-puppyscope-res-16k", "1-puppyscope-res-16k", and "2-puppyscope-res-16k".

        After creating the source set, you can upload features (and activations + explanations) to it, by referencing its source. For the example above, you upload the to layer 1 by calling the Feature.upload_batch() method with modelId: "pup-model-3", source: "1-puppyscope-res-16k", and the feature details for each feature index.
        Only the creator of the source set can upload features to it.

        Args:
            name: Name of the source set (eg "gemmascope-res-16k" - no strict rules, but usually include the author name or abbreviation, plus abbreviation of the hook used)
            model_id: ID of the model this source set belongs to (eg "gemma-2-2b")
            description: Optional description of the source set (eg "Residual Stream - 16k")
            url: Optional URL associated with the source set (eg "https://huggingface.co/google/gemma-scope-2b-pt-res/tree/main")

        Returns:
            SourceSet: Response containing the created source set information
        """
        payload = {
            "name": name,
            "modelId": model_id,
            "description": description,
            "url": url,
        }
        response = self.send_request(
            method="POST",
            json=payload,
            uri="new",
        )
        return SourceSet.from_np_db_json(response)

    def get(self, model_id: str, name: str) -> SourceSet:
        response = self.send_request(
            method="GET",
            uri=f"{model_id}/{name}",
        )
        return SourceSet.from_np_db_json(response)
