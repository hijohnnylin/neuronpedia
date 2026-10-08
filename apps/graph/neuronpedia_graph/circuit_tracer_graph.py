"""The circuit-tracer half of ``/generate-graph``, from prompt to output model.

One function, so the endpoint and a test build a graph through the same steps: attribute, prune,
create the nodes and edges, and assemble the wire model. The endpoint adds the lock, the token
limit, the upload and the metadata block around this; a test compares the output model against a
graph production generated from the same prompt and parameters (``tests/test_graph_ground_truth.py``).
"""

import time
from typing import Any

import torch
from circuit_tracer import attribute
from circuit_tracer.graph import prune_graph
from circuit_tracer.utils.create_graph_files import (
    build_model,
    create_nodes,
    create_used_nodes_and_edges,
)
from transformers import AutoTokenizer


def build_circuit_tracer_graph(
    prompt: str,
    model: Any,
    *,
    np_model_id: str,
    slug: str,
    max_n_logits: int,
    desired_logit_prob: float,
    batch_size: int,
    max_feature_nodes: int,
    node_threshold: float,
    edge_threshold: float,
    device: torch.device,
    offload: str | None = None,
    update_interval: int = 1000,
) -> tuple[Any, float]:
    """Attribute ``prompt`` on ``model`` and return the output model and the attribution time in ms.

    ``prompt`` is what circuit-tracer tokenizes, BOS included where the model needs one. The graph
    is pruned on ``device`` and moved to the CPU before the node and edge files are built from it.
    """
    started = time.time()
    graph = attribute(
        prompt,
        model,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        offload=offload,
        update_interval=update_interval,
    )
    attribution_time_ms = (time.time() - started) * 1000

    graph.to(device)
    node_mask, edge_mask, cumulative_scores = (el.cpu() for el in prune_graph(graph, node_threshold, edge_threshold))
    graph.to("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model.cfg.tokenizer_name)
    nodes = create_nodes(graph, node_mask, tokenizer, cumulative_scores)
    used_nodes, used_edges = create_used_nodes_and_edges(graph, nodes, edge_mask)
    output_model = build_model(graph, used_nodes, used_edges, slug, np_model_id, node_threshold, tokenizer)
    return output_model, attribution_time_ms
