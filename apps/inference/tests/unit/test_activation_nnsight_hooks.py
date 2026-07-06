from neuronpedia_inference.endpoints.activation.nnsight_hooks import save_nnsight_hook_outputs


class FakeTraceOutput:
    def __init__(self, name: str):
        self.name = name

    def save(self) -> str:
        return self.name


class FakeModel:
    def __init__(self):
        self.layers_output = [FakeTraceOutput("layer_0"), FakeTraceOutput("layer_1")]
        self.mlps_input = [FakeTraceOutput("mlp_0"), FakeTraceOutput("mlp_1")]
        self.embeddings_output = FakeTraceOutput("embeddings")


def test_save_nnsight_hook_outputs_maps_ln2_normalized_to_mlp_input():
    model = FakeModel()

    outputs = save_nnsight_hook_outputs(model, "blocks.0.ln2.hook_normalized", 0)

    assert outputs == "mlp_0"


def test_save_nnsight_hook_outputs_maps_resid_pre_layer_zero_to_embeddings():
    model = FakeModel()

    outputs = save_nnsight_hook_outputs(model, "blocks.0.hook_resid_pre", 0)

    assert outputs == "embeddings"