from typing import Optional

from neuronpedia.np_model import Model
from neuronpedia.requests.base_request import NPRequest


class ModelRequest(NPRequest):
    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        super().__init__("model", api_key=api_key)

    def new(
        self,
        id: str,
        layers: int,
        display_name: str | None = None,
        url: str | None = None,
    ) -> Model:
        """
        Create a new model on Neuronpedia. It will be UNLISTED, meaning that anyone who knows its ID will be
        able to view it, but it won't be listed publicly in the home page and dropdowns.
        Args:
            id: Unique identifier for the model (eg "gemma-2-2b")
            layers: Number of layers in the model (eg 26)
            display_name: Human-readable name for the model (optional, eg "Gemma 2 2B")
            url: URL associated with the model or its creator (optional, eg "https://deepmind.google/, or
                Huggingface URL")

        Returns:
            Model: Response containing the created model information
        """
        payload = {
            "id": id,
            "layers": layers,
            "displayName": display_name,
            "url": url,
        }
        response = self.send_request(
            method="POST",
            json=payload,
            uri="new",
        )
        return Model(
            id=response["id"],
            layers=response["layers"],
            display_name=response["displayName"],
            url=response["website"],
        )
