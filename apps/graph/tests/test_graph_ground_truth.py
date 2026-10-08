"""A graph built here agrees with the one production Neuronpedia built from the same request.

The fixtures under ``fixtures/graphs/`` are graphs ``POST https://www.neuronpedia.org/api/graph/generate``
returned for the prompt ``123``, one per transcoder set this server offers, pruned as hard as that
API allows so they stay small. Each records the model, the transcoder set and every generation and
pruning parameter in its ``metadata``, and the test builds a graph from exactly those through
:func:`neuronpedia_graph.circuit_tracer_graph.build_circuit_tracer_graph`, the function the endpoint
calls. So what is compared is the whole circuit-tracer path on this machine against the same path on
production's CUDA box -- a different device, and a different checkout of circuit-tracer.

Agreement is measured rather than asserted exact. Two bfloat16 attributions do not reproduce bit for
bit across devices, and pruning turns a small difference in influence at the threshold into a node
present on one side and absent on the other. The tolerances below leave room over what an MPS run
showed against production on the three sets: feature-node Jaccard 0.92 to 1.0, activations within
0.7% at the median and 11% at worst, influence within 0.045, edge weights correlated past 0.997.

Heavy: a model and a transcoder set per case, and the CLT set alone is 170 GB. Opt in with
``RUN_GRAPH_GROUND_TRUTH=1`` and pick cases with ``-k``. The self-agreement test at the bottom
runs everywhere and keeps the fixtures and the comparison honest without weights.
"""

import gc
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

import pytest
import torch

from neuronpedia_graph.model_ids import hf_model_id_to_np_model_id
from neuronpedia_graph.runtime_env import get_device

FIXTURES = Path(__file__).parent / "fixtures" / "graphs"

#: Fixture -> the HF checkpoint it was generated on. The transcoder set is in the fixture's metadata.
CASES = {
    "123-gemma-2-2b-gemmascope-transcoder-16k.json": "google/gemma-2-2b",
    "123-gemma-2-2b-clt-hp.json": "google/gemma-2-2b",
    "123-qwen3-4b-transcoder-hp.json": "Qwen/Qwen3-4B",
}

PROMPT = "123"

#: Tolerances. See the module docstring for where they come from. Influence is in [0, 1] and is
#: compared by difference: a correlation says little on the dozen nodes a hard-pruned graph keeps.
FEATURE_JACCARD_MIN = 0.85
ACTIVATION_REL_MEDIAN_MAX = 0.02
ACTIVATION_REL_MAX = 0.2
INFLUENCE_ABS_MEDIAN_MAX = 0.02
INFLUENCE_ABS_MAX = 0.1
EDGE_PEARSON_MIN = 0.99
SHARED_LINKS_MIN_FRACTION = 0.9
LOGIT_PROB_ATOL = 0.03


def _load(name: str) -> dict[str, Any]:
    with (FIXTURES / name).open(encoding="utf-8") as f:
        return json.load(f)


def _pearson(x: list[float], y: list[float]) -> float:
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / (sx * sy)


def _features(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Feature nodes by ``node_id`` (``layer_feature_ctx``), the key that is stable across generators."""
    return {n["node_id"]: n for n in graph["nodes"] if n["feature_type"] == "cross layer transcoder"}


def _logits(graph: dict[str, Any]) -> dict[str, float]:
    """Token -> probability for the logit nodes, the token read out of the node's label."""
    return {n["clerp"].split('"')[1]: n["token_prob"] for n in graph["nodes"] if n["feature_type"] == "logit"}


def _links(graph: dict[str, Any]) -> dict[tuple[str, str], float]:
    return {(link["source"], link["target"]): link["weight"] for link in graph["links"]}


def assert_graphs_agree(reference: dict[str, Any], got: dict[str, Any]) -> None:
    """The comparison, on the wire format both sides produce."""
    assert got["metadata"]["prompt_tokens"] == reference["metadata"]["prompt_tokens"]

    ref_features, got_features = _features(reference), _features(got)
    shared = set(ref_features) & set(got_features)
    jaccard = len(shared) / len(set(ref_features) | set(got_features))
    assert jaccard >= FEATURE_JACCARD_MIN, (
        f"feature nodes: {len(ref_features)} on production, {len(got_features)} here, {len(shared)} shared "
        f"(jaccard {jaccard:.3f})"
    )

    relative = [
        abs(ref_features[k]["activation"] - got_features[k]["activation"]) / abs(ref_features[k]["activation"])
        for k in shared
        if ref_features[k]["activation"]
    ]
    assert statistics.median(relative) <= ACTIVATION_REL_MEDIAN_MAX, (
        f"activation median rel {statistics.median(relative):.4f}"
    )
    assert max(relative) <= ACTIVATION_REL_MAX, f"activation max rel {max(relative):.4f}"

    influence = [abs(ref_features[k]["influence"] - got_features[k]["influence"]) for k in shared]
    assert statistics.median(influence) <= INFLUENCE_ABS_MEDIAN_MAX, (
        f"influence median diff {statistics.median(influence):.4f}"
    )
    assert max(influence) <= INFLUENCE_ABS_MAX, f"influence max diff {max(influence):.4f}"

    ref_logits, got_logits = _logits(reference), _logits(got)
    assert max(ref_logits, key=lambda t: ref_logits[t]) == max(got_logits, key=lambda t: got_logits[t]), (
        f"top logit: {ref_logits} vs {got_logits}"
    )
    for token, prob in ref_logits.items():
        assert abs(prob - got_logits.get(token, 0.0)) <= LOGIT_PROB_ATOL, (
            f"logit {token!r}: {prob} vs {got_logits.get(token)}"
        )

    ref_links, got_links = _links(reference), _links(got)
    shared_links = set(ref_links) & set(got_links)
    assert len(shared_links) >= SHARED_LINKS_MIN_FRACTION * len(ref_links), (
        f"links: {len(ref_links)} on production, {len(got_links)} here, {len(shared_links)} shared"
    )
    edges = _pearson([ref_links[k] for k in shared_links], [got_links[k] for k in shared_links])
    assert edges >= EDGE_PEARSON_MIN, f"edge weight pearson {edges:.4f}"


@pytest.mark.parametrize("fixture", list(CASES))
def test_each_fixture_agrees_with_itself(fixture: str) -> None:
    """Weight-free: the fixture parses, records what the heavy test reads, and passes its own comparison."""
    graph = _load(fixture)
    metadata = graph["metadata"]
    assert metadata["prompt_tokens"][1:] == list(PROMPT)
    assert metadata["info"]["transcoder_set"]
    assert set(metadata["generation_settings"]) >= {
        "max_n_logits",
        "desired_logit_prob",
        "batch_size",
        "max_feature_nodes",
    }
    assert set(metadata["pruning_settings"]) == {"node_threshold", "edge_threshold"}
    assert_graphs_agree(graph, graph)


@pytest.mark.skipif(
    os.environ.get("RUN_GRAPH_GROUND_TRUTH") != "1",
    reason="Set RUN_GRAPH_GROUND_TRUTH=1 to run (downloads a model and a transcoder set per case, needs a GPU or MPS)",
)
@pytest.mark.parametrize("fixture", list(CASES))
def test_a_graph_built_here_agrees_with_production(fixture: str) -> None:
    from circuit_tracer.replacement_model import ReplacementModel

    from neuronpedia_graph.circuit_tracer_graph import build_circuit_tracer_graph

    device = get_device()
    if device.type == "cpu":
        pytest.skip("attribution on cpu takes too long to be a test")

    reference = _load(fixture)
    metadata = reference["metadata"]
    hf_model_id = CASES[fixture]
    generation, pruning = metadata["generation_settings"], metadata["pruning_settings"]

    # As the server loads it: bfloat16, decoders read lazily, interp-engine running the model.
    model = ReplacementModel.from_pretrained(
        hf_model_id,
        metadata["info"]["transcoder_set"],
        device=device,
        dtype=torch.bfloat16,
        lazy_encoder=False,
        lazy_decoder=True,
        backend="interp_engine",
    )
    try:
        output_model, _ = build_circuit_tracer_graph(
            PROMPT,
            model,
            np_model_id=hf_model_id_to_np_model_id()[hf_model_id],
            slug=metadata["slug"],
            max_n_logits=generation["max_n_logits"],
            desired_logit_prob=generation["desired_logit_prob"],
            batch_size=generation["batch_size"],
            max_feature_nodes=generation["max_feature_nodes"],
            node_threshold=pruning["node_threshold"],
            edge_threshold=pruning["edge_threshold"],
            device=device,
        )
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    assert_graphs_agree(reference, output_model.model_dump())
