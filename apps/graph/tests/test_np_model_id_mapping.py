"""Guard the HuggingFace -> Neuronpedia model id mapping a graph is labelled with.

The dict this replaced was hand-kept and carried `meta-llama/Llama-3.2-1B -> llama3.1-8b`, a
different model. Nothing compared it against `np_model_to_hf.json` or the webapp's own copy, so the
wrong id reached the `scan` field of every graph such a pod produced and no test failed.
"""

import json
from pathlib import Path

import pytest

from neuronpedia_graph.model_ids import (
    FALLBACK_HF_MODEL_ID_TO_NP_MODEL_ID,
    _load_np_model_to_hf,
    hf_model_id_to_np_model_id,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MAPPING_PATH = REPO_ROOT / "np_model_to_hf.json"


def test_mapping_file_is_present_and_readable():
    """Fail rather than skip. A silently absent file means the fallback ships as the real table."""
    assert MAPPING_PATH.exists(), f"{MAPPING_PATH} is missing"
    assert _load_np_model_to_hf(), "np_model_to_hf.json read as empty"


def test_mapping_file_is_injective():
    """One repo, one Neuronpedia model. `Model.hfRepoId` is unique in the webapp schema, so a
    duplicate here would reverse into a table that silently drops a model."""
    with open(MAPPING_PATH, encoding="utf-8") as f:
        mapping = json.load(f)
    repos = list(mapping.values())
    duplicates = {repo for repo in repos if repos.count(repo) > 1}
    assert not duplicates, f"repo ids mapped to more than one model: {sorted(duplicates)}"


def test_reverse_lookup_matches_the_file():
    forward = _load_np_model_to_hf()
    reverse = hf_model_id_to_np_model_id()
    assert len(reverse) == len(forward)
    for np_id, hf_id in forward.items():
        assert reverse[hf_id] == np_id


@pytest.mark.parametrize(
    ("hf_model_id", "expected_np_model_id"),
    [
        ("google/gemma-2-2b", "gemma-2-2b"),
        ("google/gemma-3-4b-it", "gemma-3-4b-it"),
        ("Qwen/Qwen3-1.7B", "qwen3-1.7b"),
        ("Qwen/Qwen3-4B", "qwen3-4b"),
        # Not a stripped namespace: the Neuronpedia id is a different string entirely.
        ("openai-community/gpt2", "gpt2-small"),
    ],
)
def test_known_models_resolve(hf_model_id: str, expected_np_model_id: str):
    assert hf_model_id_to_np_model_id()[hf_model_id] == expected_np_model_id


def test_llama_3_2_1b_is_not_mapped_to_llama_3_1_8b():
    """The specific regression. Llama-3.2-1B is not a model Neuronpedia carries; if it is ever
    added, it gets its own id rather than borrowing an 8B model's."""
    assert hf_model_id_to_np_model_id().get("meta-llama/Llama-3.2-1B") != "llama3.1-8b"


def test_fallback_entries_all_exist_in_the_mapping_file():
    """The fallback is a copy for availability, not a second source. An entry that drifts out of
    the file would serve a stale id on any pod whose image lacks the repo root."""
    reverse = hf_model_id_to_np_model_id()
    for hf_model_id, np_model_id in FALLBACK_HF_MODEL_ID_TO_NP_MODEL_ID.items():
        assert reverse.get(hf_model_id) == np_model_id, f"{hf_model_id} disagrees with np_model_to_hf.json"
