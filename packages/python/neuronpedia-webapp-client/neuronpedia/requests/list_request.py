from typing import Optional

from neuronpedia.np_list import NPList, NPListItem
from neuronpedia.requests.base_request import NPRequest


class ListRequest(NPRequest):
    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        super().__init__("list", api_key=api_key)

    def get_owned(self) -> list[NPList]:
        response = self.send_request(method="POST", uri="list")

        return [
            NPList(
                id=list["id"],
                name=list["name"],
                description=list["description"],
                items=[],
            )
            for list in response
        ]

    def new(self, name: str, description: str = "") -> NPList:
        payload = {"name": name, "description": description}
        response = self.send_request(method="POST", json=payload, uri="new")
        return NPList(
            id=response["id"],
            name=response["name"],
            description=response["description"],
            items=[],
        )

    def add_items(self, nplist: NPList, items: list[NPListItem]) -> NPList:
        formatted_items = [
            {
                "modelId": item.model_id,
                "layer": item.source,
                "index": item.index,
                "description": item.description,
            }
            for item in items
        ]
        payload = {"listId": nplist.id, "featuresToAdd": formatted_items}
        response = self.send_request(method="POST", json=payload, uri="add-features")
        for item in response:
            nplist.items.append(
                NPListItem(
                    model_id=item["modelId"],
                    source=item["layer"],
                    index=item["index"],
                )
            )
        return nplist

    def get(self, list_id: str) -> NPList:
        payload = {"listId": list_id}
        response = self.send_request(method="POST", json=payload, uri="get")

        return NPList(
            id=response["id"],
            name=response["name"],
            description=response["description"],
            items=[
                NPListItem(
                    model_id=item["modelId"],
                    source=item["layer"],
                    index=item["index"],
                )
                for item in response["neurons"]
            ],
        )
