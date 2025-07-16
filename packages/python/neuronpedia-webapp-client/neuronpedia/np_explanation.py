from dataclasses import dataclass


@dataclass
class Explanation:
    modelId: str
    source: str
    index: int
    text: str
    method: str | None = None
    explainer_model: str | None = None

    # converts from the Neuronpedia database type
    @classmethod
    def from_np_db_json(cls, json_data: dict) -> "Explanation":
        return cls(
            modelId=json_data["modelId"],
            source=json_data["source"],
            index=int(json_data["index"]),
            text=json_data["description"],
            method=json_data.get("typeName"),
            explainer_model=json_data.get("explanationModelName"),
        )
