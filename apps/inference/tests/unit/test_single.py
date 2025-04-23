import time

import pytest
import torch
from sae_lens import SAE, SAEConfig
from transformer_lens import HookedTransformer

from neuronpedia_inference.endpoints.activation.single import calculate_dfa


@pytest.fixture
def model() -> HookedTransformer:
    model = HookedTransformer.from_pretrained("tiny-stories-1M", device="cpu")
    model.eval()
    return model


@pytest.fixture
def sae() -> SAE:
    cfg = SAEConfig(
        architecture="standard",
        d_in=64,
        d_sae=128,
        apply_b_dec_to_input=False,
        context_size=128,
        model_name="TEST",
        hook_name="test",
        hook_layer=0,
        prepend_bos=True,
        dataset_path="test/test",
        dtype="float32",
        activation_fn_str="relu",
        finetuning_scaling_factor=False,
        hook_head_index=None,
        normalize_activations="none",
        device="cpu",
        sae_lens_training_version=None,
        dataset_trust_remote_code=True,
    )

    sae = SAE(cfg)
    # set weights and biases to hardcoded values so tests are consistent
    seed1 = torch.tensor([0.1, -0.2, 0.3, -0.4] * 16)  # 64
    seed2 = torch.tensor([0.2, -0.1, 0.4, -0.2] * 32)  # 64 x 2
    seed3 = torch.tensor([0.3, -0.3, 0.6, -0.6] * 16)  # 64
    seed4 = torch.tensor([-0.4, 0.4, 0.8, -0.8] * 32)  # 64 x 2
    W_enc_base = torch.cat([torch.eye(64), torch.eye(64)], dim=-1)
    W_dec_base = torch.cat([torch.eye(64), torch.eye(64)], dim=0)
    sae.load_state_dict(
        {
            "W_enc": W_enc_base + torch.outer(seed1, seed2),
            "W_dec": W_dec_base + torch.outer(seed4, seed3),
            "b_enc": torch.zeros_like(sae.b_enc) + 0.5,
            "b_dec": torch.zeros_like(sae.b_dec) + 0.3,
        }
    )
    return sae


@pytest.fixture
def tokens(model: HookedTransformer) -> torch.Tensor:
    return model.to_tokens(
        [
            "But what about second breakfast?" * 3,
            "Nothing is cheesier than cheese." * 3,
        ]
    )


def test_calculate_dfa_shape(model: HookedTransformer, sae: SAE, tokens: torch.Tensor):
    layer_num = 0
    index = 0
    max_value_index = 5

    result = calculate_dfa(
        model,
        sae=sae,
        layer_num=layer_num,
        index=index,
        max_value_index=max_value_index,
        tokens=tokens,
    )

    assert "dfa_values" in result
    assert "dfa_target_index" in result
    assert "dfa_max_value" in result

    assert isinstance(result["dfa_values"], list)
    assert isinstance(result["dfa_target_index"], int)
    assert isinstance(result["dfa_max_value"], float)
    assert len(result["dfa_values"]) == tokens.shape[1]  # Should match sequence length
    assert result["dfa_target_index"] == max_value_index


def test_calculate_dfa_values(model: HookedTransformer, sae: SAE, tokens: torch.Tensor):
    layer_num = 0
    index = 0
    max_value_index = 3

    result = calculate_dfa(
        model,
        sae=sae,
        layer_num=layer_num,
        index=index,
        max_value_index=max_value_index,
        tokens=tokens,
    )

    dfa_values = result["dfa_values"]
    assert isinstance(dfa_values, list)
    assert len(dfa_values) == tokens.shape[1]
    assert not all(v == 0 for v in dfa_values)
    assert result["dfa_max_value"] == max(dfa_values)


def test_calculate_dfa_different_layers(
    model: HookedTransformer, sae: SAE, tokens: torch.Tensor
):
    index = 0
    max_value_index = 2

    result_layer0 = calculate_dfa(
        model,
        sae=sae,
        layer_num=0,
        index=index,
        max_value_index=max_value_index,
        tokens=tokens,
    )
    result_layer1 = calculate_dfa(
        model,
        sae=sae,
        layer_num=1,
        index=index,
        max_value_index=max_value_index,
        tokens=tokens,
    )

    # Results should be different for different layers
    assert result_layer0["dfa_values"] != result_layer1["dfa_values"]


def test_calculate_dfa_target_index(
    model: HookedTransformer, sae: SAE, tokens: torch.Tensor
):
    layer_num = 0
    index = 0

    # Test with max_value_index at sequence boundaries
    result_first = calculate_dfa(
        model,
        sae=sae,
        layer_num=layer_num,
        index=index,
        max_value_index=0,
        tokens=tokens,
    )
    result_last = calculate_dfa(
        model,
        sae=sae,
        layer_num=layer_num,
        index=index,
        max_value_index=tokens.shape[1] - 1,
        tokens=tokens,
    )
    assert result_first["dfa_target_index"] == 0
    assert result_last["dfa_target_index"] == tokens.shape[1] - 1


def test_calculate_dfa_performance_comparison(model: HookedTransformer, sae: SAE):
    # Create larger tokens for performance testing
    large_tokens = model.to_tokens(["This is a longer test sequence. " * 20] * 4)

    layer_num = 0
    index = 0
    max_value_index = large_tokens.shape[1] // 2

    start_time = time.time()
    result = calculate_dfa(
        model,
        sae=sae,
        layer_num=layer_num,
        index=index,
        max_value_index=max_value_index,
        tokens=large_tokens,
    )
    end_time = time.time()

    # Should complete in reasonable time
    assert end_time - start_time < 10, "Function took too long"
    dfa_values = result["dfa_values"]
    assert isinstance(dfa_values, list)
    assert len(dfa_values) == large_tokens.shape[1]


def test_calculate_dfa_different_shapes(model: HookedTransformer, sae: SAE):
    layer_num = 0
    index = 0

    # Test with different sequence lengths
    shapes = [
        model.to_tokens(["Short text."]),
        model.to_tokens(["Medium length text sequence here."]),
        model.to_tokens(
            [
                "Much longer text sequence that spans multiple tokens and tests the function with larger inputs."
            ]
        ),
    ]

    for tokens in shapes:
        max_value_index = tokens.shape[1] // 2
        result = calculate_dfa(
            model,
            sae=sae,
            layer_num=layer_num,
            index=index,
            max_value_index=max_value_index,
            tokens=tokens,
        )

        dfa_values = result["dfa_values"]
        assert isinstance(dfa_values, list)
        assert len(dfa_values) == tokens.shape[1]
        assert result["dfa_target_index"] == max_value_index
        assert isinstance(result["dfa_max_value"], float)
