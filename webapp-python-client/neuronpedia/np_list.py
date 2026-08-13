from dataclasses import dataclass
from typing import List


@dataclass
class NPListItem:
    model_id: str
    source: str
    index: str
    description: str = ""


@dataclass
class NPList:
    id: str
    name: str
    description: str
    items: List[NPListItem]

    @classmethod
    def new(cls, name: str, description: str = "") -> "NPList":
        # import here to avoid circular import
        from neuronpedia.requests.list_request import ListRequest

        return ListRequest().new(name, description)

    @classmethod
    def get_owned(cls) -> list["NPList"]:
        # import here to avoid circular import
        from neuronpedia.requests.list_request import ListRequest

        return ListRequest().get_owned()

    def add_items(self, items: list[NPListItem]) -> "NPList":
        # import here to avoid circular import
        from neuronpedia.requests.list_request import ListRequest

        return ListRequest().add_items(self, items)

    @classmethod
    def get(cls, list_id: str) -> "NPList":
        # import here to avoid circular import
        from neuronpedia.requests.list_request import ListRequest

        return ListRequest().get(list_id)
