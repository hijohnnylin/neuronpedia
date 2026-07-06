from neuronpedia_inference.shared import (
    replace_tlens_model_id_with_hf_model_id,
    resolve_model_id_for_load,
)


def test_replace_tlens_model_id_with_hf_model_id_maps_gemma3_aliases():
    assert replace_tlens_model_id_with_hf_model_id("gemma-3-1b-it") == "google/gemma-3-1b-it"
    assert replace_tlens_model_id_with_hf_model_id("google/gemma-3-1b-it") == "google/gemma-3-1b-it"


def test_resolve_model_id_for_load_prefers_custom_hf_model_id():
    assert (
        resolve_model_id_for_load(
            model_id="gemma-3-1b-it",
            override_model_id="gemma-3-1b-it",
            custom_hf_model_id="google/gemma-3-1b-it",
        )
        == "google/gemma-3-1b-it"
    )


def test_resolve_model_id_for_load_uses_alias_mapping_without_custom_hf_model_id():
    assert (
        resolve_model_id_for_load(
            model_id="gemma-3-4b-it",
            override_model_id="gemma-3-4b-it",
        )
        == "google/gemma-3-4b-it"
    )