from dataclasses import dataclass
from typing import List


@dataclass
class Activation:
    modelId: str
    source: str
    index: int
    tokens: List[str]
    values: List[float]
    quantileMax: float | None = None
    quantileMin: float | None = None
    quantileFraction: float | None = None

    def __post_init__(self):
        if self.quantileFraction is not None:
            if not (0 <= self.quantileFraction <= 1):
                raise ValueError(
                    f"quantileFraction must be between 0 and 1 inclusive, got {self.quantileFraction}"
                )

    # converts from the Neuronpedia database type
    @classmethod
    def from_np_db_json(cls, json_data: dict) -> "Activation":
        # Ensure tokens and values have the same length
        if len(json_data["tokens"]) != len(json_data["values"]):
            raise ValueError(
                f"Length mismatch: tokens has {len(json_data['tokens'])} elements, values has {len(json_data['values'])} elements"
            )
        return cls(
            modelId=json_data["modelId"],
            source=json_data["source"],
            index=int(json_data["index"]),
            tokens=json_data["tokens"],
            values=json_data["values"],
            quantileMax=json_data["binMax"],
            quantileMin=json_data["binMin"],
            quantileFraction=json_data["binContains"],
        )
