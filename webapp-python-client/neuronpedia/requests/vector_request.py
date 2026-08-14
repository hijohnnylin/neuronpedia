from typing import Optional

from neuronpedia.np_vector import NPVector
from neuronpedia.requests.base_request import NPRequest
from requests import Response

VALID_HOOK_TYPES = ["hook_resid_pre", "hook_resid_post"]


class VectorRequest(NPRequest):
    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        super().__init__("vector", api_key=api_key)

    def new(
        self,
        label: str,
        model_id: str,
        layer_num: int,
        hook_type: str,
        vector: list[float],
        default_steer_strength: float | None = 10,
    ) -> NPVector:
        if hook_type not in VALID_HOOK_TYPES:
            raise ValueError(
                f"Invalid hook name: {hook_type}. Valid hook names are {VALID_HOOK_TYPES}."
            )
        payload = {
            "modelId": model_id,
            "layerNumber": layer_num,
            "hookType": hook_type,
            "vector": vector,
            "vectorDefaultSteerStrength": default_steer_strength,
            "vectorLabel": label,
        }
        response = self.send_request(
            method="POST",
            json=payload,
            uri="new",
        )

        return NPVector(
            model_id=response["vector"]["modelId"],
            source=response["vector"]["source"],
            index=response["vector"]["index"],
            label=response["vector"]["label"],
            hook_name=response["vector"]["hookName"],
            values=response["vector"]["values"],
            default_steer_strength=response["vector"]["defaultSteerStrength"],
            url=response["url"],
        )

    def delete(self, vector: NPVector) -> Response:
        return self._delete(vector.model_id, vector.source, vector.index)

    def _delete(self, modelId: str, source: str, index: str) -> Response:
        payload = {
            "modelId": modelId,
            "source": source,
            "index": index,
        }
        return self.send_request(
            method="POST",
            uri="delete",
            json=payload,
        )

    def get(self, model_id: str, source: str, index: str) -> NPVector:
        response = self.send_request(
            method="POST",
            uri="get",
            json={"modelId": model_id, "source": source, "index": index},
        )
        return NPVector(
            model_id=response["vector"]["modelId"],
            source=response["vector"]["layer"],
            index=response["vector"]["index"],
            label=response["vector"]["vectorLabel"],
            hook_name=response["vector"].get("hookName", ""),
            values=response["vector"]["vector"],
            default_steer_strength=response["vector"]["vectorDefaultSteerStrength"],
        )

    def get_owned(self) -> list[NPVector]:
        response = self.send_request(
            method="POST",
            uri="list-owned",
        )
        return [
            NPVector(
                model_id=vector["modelId"],
                source=vector["layer"],
                index=vector["index"],
                label=vector["vectorLabel"],
                hook_name=vector["hookName"],
                values=vector["vector"],
                default_steer_strength=vector["vectorDefaultSteerStrength"],
            )
            for vector in response["vectors"]
        ]
