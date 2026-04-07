from dataclasses import dataclass

from torch.nn import Module


@dataclass(frozen=True)
class ResidualStreamPattern:
    name: str
    model_id_contains: tuple[str, ...]

    def matches(self, model_id: str) -> bool:
        lowered = model_id.lower()
        return all(part in lowered for part in self.model_id_contains)

    def get_layer_modules(self, model: Module) -> list[Module]:
        if self.name == "llama-3.1-8b-instruct":
            # LlamaForCausalLM -> .model.layers gives decoder blocks whose outputs are
            # the per-layer residual stream states (pre-final-norm).
            if not hasattr(model, "model") or not hasattr(model.model, "layers"):  # type: ignore[attr-defined]
                raise ValueError(
                    "Unsupported model object for llama residual stream mapping."
                )
            return list(model.model.layers)  # type: ignore[attr-defined]

        raise ValueError(f"Unknown residual stream pattern: {self.name}")


RESIDUAL_STREAM_PATTERNS: list[ResidualStreamPattern] = [
    ResidualStreamPattern(
        name="llama-3.1-8b-instruct",
        model_id_contains=("llama-3.1", "8b", "instruct"),
    ),
]


def get_residual_stream_layers_for_model(model_id: str, model: Module) -> list[Module]:
    for pattern in RESIDUAL_STREAM_PATTERNS:
        if pattern.matches(model_id):
            return pattern.get_layer_modules(model)
    raise ValueError(
        f"No residual stream mapping found for model '{model_id}'. "
        "Add a new pattern in model_patterns.py."
    )
