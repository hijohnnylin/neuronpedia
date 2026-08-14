from dataclasses import dataclass, field

from neuronpedia.np_source import Source


@dataclass
class SourceSet:
    name: str
    model_id: str
    description: str | None = None
    url: str | None = None
    sources: list[Source] = field(default_factory=list)

    @classmethod
    def new(
        cls,
        name: str,
        model_id: str,
        description: str | None = None,
        url: str | None = None,
    ) -> "SourceSet":
        from neuronpedia.requests.source_set_request import SourceSetRequest

        return SourceSetRequest().new(name, model_id, description, url)

    @classmethod
    def get(cls, model_id: str, name: str) -> "SourceSet":
        from neuronpedia.requests.source_set_request import SourceSetRequest

        return SourceSetRequest().get(model_id, name)

    def get_source_for_layer_number(self, layer_number: int) -> Source:
        for source in self.sources:
            if source.id == str(layer_number) + "-" + self.name:
                return source
        raise ValueError(
            f"Source for layer {layer_number} not found. Does this model have that many layers?"
        )

    @classmethod
    def from_np_db_json(cls, json_data: dict) -> "SourceSet":
        return cls(
            name=json_data["name"],
            model_id=json_data["modelId"],
            description=json_data["description"],
            url=json_data["urls"][0] if json_data["urls"] else None,
            sources=[
                Source(
                    id=source["id"],
                    model_id=source["modelId"],
                    set_name=source["setName"],
                )
                for source in json_data["sources"]
            ],
        )

    def open_in_browser(self):
        import webbrowser

        webbrowser.open(f"https://neuronpedia.org/{self.model_id}/{self.name}")
