from dataclasses import dataclass


@dataclass
class Model:
    id: str
    layers: int
    display_name: str | None = None
    url: str | None = None

    def open_in_browser(self):
        import webbrowser

        webbrowser.open(f"https://neuronpedia.org/{self.id}")

    @classmethod
    def new(
        cls,
        id: str,
        layers: int,
        display_name: str | None = None,
        url: str | None = None,
    ) -> "Model":
        from neuronpedia.requests.model_request import ModelRequest

        return ModelRequest().new(id, layers, display_name, url)
