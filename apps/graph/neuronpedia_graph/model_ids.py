"""Translation between HuggingFace repo ids and Neuronpedia model ids.

Its own module rather than part of ``server.py`` so that importing it costs nothing: the server
refuses to import without a SECRET and an attribution backend, and a table of strings should not
need either. That is also what lets the test guarding it run everywhere, rather than skipping on a
machine that has no attribution extra installed.
"""

import json
from functools import cache
from pathlib import Path


def _load_np_model_to_hf() -> dict[str, str]:
    """Neuronpedia short id -> Hugging Face repo id, read from the repo root.

    A second copy of the loader in ``apps/inference/neuronpedia_inference/config.py``. The apps are
    separate projects with no shared package, so the duplicated function is deliberate; what
    matters is that both read the one file rather than each keeping its own table.
    """
    # apps/graph/neuronpedia_graph/model_ids.py -> the repo root is three parents up.
    path = Path(__file__).resolve().parents[3] / "np_model_to_hf.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to read {path}: {exc}")
        return {}


# Used only where np_model_to_hf.json is not on disk, i.e. an image holding apps/graph without the
# repo root. Every entry here must also be in that file: this is a copy for availability, not a
# place to add a model, and a test asserts they agree.
FALLBACK_HF_MODEL_ID_TO_NP_MODEL_ID = {
    "google/gemma-2-2b": "gemma-2-2b",
    "google/gemma-3-4b-it": "gemma-3-4b-it",
    "Qwen/Qwen3-1.7B": "qwen3-1.7b",
    "Qwen/Qwen3-4B": "qwen3-4b",
}


@cache
def hf_model_id_to_np_model_id() -> dict[str, str]:
    """Hugging Face repo id -> the Neuronpedia model id a graph is labelled with.

    Reversed from np_model_to_hf.json rather than written out here. The hand-kept dict this
    replaces carried ``"meta-llama/Llama-3.2-1B": "llama3.1-8b"`` -- a different model, whose id
    went into the `scan` field of every graph such a pod produced, and which nothing compared
    against the mapping file or the webapp's own copy.

    Not interchangeable with NP_MODEL_ID. That labels graphs on the lm-saes-crm path only, and
    always holds a value because start.py defaults it, so preferring it here would relabel every
    circuit-tracer graph as whatever that default happened to be.
    """
    mapping = _load_np_model_to_hf()
    if not mapping:
        return dict(FALLBACK_HF_MODEL_ID_TO_NP_MODEL_ID)
    return {hf_id: np_id for np_id, hf_id in mapping.items()}
