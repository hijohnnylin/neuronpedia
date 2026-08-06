import nnsight

import neuronpedia_inference.endpoints.activation.nnsight_hooks as nnsight_hooks
from neuronpedia_inference.endpoints.activation.nnsight_hooks import (
    save_nnsight_hook_outputs,
)


class FakeTraceOutput:
    def __init__(self, name: str):
        self.name = name


class FakeModel:
    def __init__(self):
        self.layers_output = [FakeTraceOutput("layer_0"), FakeTraceOutput("layer_1")]
        self.mlps_input = [FakeTraceOutput("mlp_0"), FakeTraceOutput("mlp_1")]
        self.embeddings_output = FakeTraceOutput("embeddings")


def _identity_save(monkeypatch):
    # save_nnsight_hook_outputs wraps the selected node with nnsight.save(...) (the
    # nnsight>=0.6 module-level API); its unit contract is the node *selection*, so pin
    # save to identity rather than depending on version-specific proxy semantics.
    monkeypatch.setattr(nnsight, "save", lambda value: value)
    monkeypatch.setattr(nnsight_hooks.nnsight, "save", lambda value: value)


def test_save_nnsight_hook_outputs_maps_ln2_normalized_to_mlp_input(monkeypatch):
    _identity_save(monkeypatch)
    model = FakeModel()

    outputs = save_nnsight_hook_outputs(model, "blocks.0.ln2.hook_normalized", 0)

    assert outputs is model.mlps_input[0]


def test_save_nnsight_hook_outputs_maps_resid_pre_layer_zero_to_embeddings(monkeypatch):
    _identity_save(monkeypatch)
    model = FakeModel()

    outputs = save_nnsight_hook_outputs(model, "blocks.0.hook_resid_pre", 0)

    assert outputs is model.embeddings_output


def test_save_nnsight_hook_outputs_maps_resid_pre_later_layer_to_previous_layer_output(
    monkeypatch,
):
    _identity_save(monkeypatch)
    model = FakeModel()

    outputs = save_nnsight_hook_outputs(model, "blocks.1.hook_resid_pre", 1)

    assert outputs is model.layers_output[0]


def test_save_nnsight_hook_outputs_maps_resid_post_to_layer_output(monkeypatch):
    _identity_save(monkeypatch)
    model = FakeModel()

    outputs = save_nnsight_hook_outputs(model, "blocks.1.hook_resid_post", 1)

    assert outputs is model.layers_output[1]
