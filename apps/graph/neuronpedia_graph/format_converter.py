"""Converts lm-saes AttributionResult + pruned graph into Neuronpedia's graph-schema.json format."""

from __future__ import annotations

import re
from importlib.metadata import version as pkg_version
from typing import Any

import torch
from llamascopium.circuits.attribution import AttributionResult
from llamascopium.circuits.indexed_tensor import NodeIndexedMatrix
from llamascopium.models.lorsa import LorsaConfig
from llamascopium.models.sparse_dictionary import SparseDictionary


def cantor_pair(layer: int, index: int) -> int:
    return ((layer + index) * (layer + index + 1)) // 2 + index


def _compute_node_influence(
    node_entries: list[dict[str, Any]],
    links: list[dict[str, Any]],
    logit_probs: torch.Tensor,
    max_iter: int = 1000,
) -> list[float]:
    """Compute per-node cumulative influence scores matching circuit-tracer's format.

    The returned scores are cumulative fractions (0 to 1): the most influential node
    gets a low score (small fraction of total), the least influential gets ~1.0.
    The frontend slider filters nodes where influence <= threshold.
    """
    n = len(node_entries)
    if n == 0:
        return []

    node_id_to_idx = {entry["node_id"]: i for i, entry in enumerate(node_entries)}

    adjacency = torch.zeros(n, n)
    for link in links:
        src_idx = node_id_to_idx.get(link["source"])
        tgt_idx = node_id_to_idx.get(link["target"])
        if src_idx is not None and tgt_idx is not None:
            adjacency[tgt_idx, src_idx] = link["weight"]

    normalized = adjacency.abs()
    row_sums = normalized.sum(dim=1, keepdim=True).clamp(min=1e-10)
    normalized = normalized / row_sums

    logit_weights = torch.zeros(n)
    for i, entry in enumerate(node_entries):
        if entry["feature_type"] == "logit" and entry.get("token_prob") is not None:
            logit_weights[i] = entry["token_prob"]

    current = logit_weights @ normalized
    raw_influence = current.clone()
    for _ in range(max_iter):
        if not current.any():
            break
        current = current @ normalized
        raw_influence += current

    # Convert raw influence to cumulative fraction (matching circuit-tracer's format)
    sorted_scores, sorted_indices = torch.sort(raw_influence, descending=True)
    total = sorted_scores.sum()
    if total > 0:
        cumulative = torch.cumsum(sorted_scores, dim=0) / total
    else:
        cumulative = torch.zeros_like(sorted_scores)
    final_scores = torch.zeros_like(raw_influence)
    final_scores[sorted_indices] = cumulative

    return final_scores.tolist()


def _parse_hook_layer(hook_key: str) -> int:
    match = re.search(r"blocks\.(\d+)\.", hook_key)
    return int(match.group(1)) if match else 0


def _build_sae_metadata(
    replacement_modules: list[SparseDictionary],
) -> dict[str, dict[str, Any]]:
    """Build a hook_point_out -> metadata mapping from the loaded SAE modules."""
    sae_metadata: dict[str, dict[str, Any]] = {}
    for sae in replacement_modules:
        cfg = sae.cfg
        hook_point_out = cfg.hook_point_out
        layer_match = re.search(r"blocks\.(\d+)\.", hook_point_out)
        layer_idx = int(layer_match.group(1)) if layer_match else 0
        sae_metadata[hook_point_out] = {
            "is_lorsa": isinstance(cfg, LorsaConfig),
            "layer_idx": layer_idx,
        }
    return sae_metadata


def _attach_qk_tracing_results(
    ar: AttributionResult,
    nodes: Any,
    node_ids: list[str],
    node_entries: list[dict[str, Any]],
) -> None:
    """Populate a ``qk_tracing_results`` object on each target lorsa node entry.

    The shape mirrors OpenMOSS' reference server logic:
        {
            "pair_wise_contributors": [(q_node_id, k_node_id, attribution), ...],
            "top_q_marginal_contributors": [(q_node_id, attribution), ...],
            "top_k_marginal_contributors": [(k_node_id, attribution), ...],
        }

    Contributors whose upstream node was pruned out of the graph are dropped so
    every referenced id is guaranteed to exist in ``node_entries``.
    """
    qk = getattr(ar, "qk_trace_results", None)
    if qk is None:
        return

    def _node_offsets(dim: Any) -> list[int]:
        offsets = nodes.nodes_to_offsets(dim)
        if hasattr(offsets, "cpu"):
            offsets = offsets.cpu()
        return offsets.tolist()

    def _ensure_entry(target_offset: int) -> dict[str, Any]:
        entry = node_entries[target_offset].setdefault(
            "qk_tracing_results",
            {
                "pair_wise_contributors": [],
                "top_q_marginal_contributors": [],
                "top_k_marginal_contributors": [],
            },
        )
        return entry

    target_offsets = _node_offsets(qk.pairs.dimensions[0])

    for target_offset, (pairs_ni, _target_info) in zip(target_offsets, qk.pairs):
        if target_offset < 0:
            continue
        q_offsets = _node_offsets(pairs_ni.dimensions[0])
        k_offsets = _node_offsets(pairs_ni.dimensions[1])
        values = pairs_ni.value.detach().cpu().tolist()
        contributors: list[tuple[str, str, float]] = []
        for q_off, k_off, value in zip(q_offsets, k_offsets, values):
            if q_off < 0 or k_off < 0:
                continue
            contributors.append((node_ids[q_off], node_ids[k_off], float(value)))
        _ensure_entry(target_offset)["pair_wise_contributors"] = contributors

    for marginal, out_key in (
        (qk.q_marginal, "top_q_marginal_contributors"),
        (qk.k_marginal, "top_k_marginal_contributors"),
    ):
        marginal_target_offsets = _node_offsets(marginal.dimensions[0])
        for target_offset, (marg_ni, _target_info) in zip(marginal_target_offsets, marginal):
            if target_offset < 0:
                continue
            src_offsets = _node_offsets(marg_ni.dimensions[0])
            values = marg_ni.value.detach().cpu().tolist()
            contributors_1d: list[tuple[str, float]] = []
            for src_off, value in zip(src_offsets, values):
                if src_off < 0:
                    continue
                contributors_1d.append((node_ids[src_off], float(value)))
            _ensure_entry(target_offset)[out_key] = contributors_1d


def convert_to_neuronpedia_graph(
    ar: AttributionResult,
    pruned_attribution: NodeIndexedMatrix,
    sae_metadata: dict[str, dict[str, Any]],
    *,
    slug: str,
    np_model_id: str,
    prompt: str,
    node_threshold: float,
    edge_threshold: float,
    np_transcoder_source_set: str | None = None,
    np_lorsa_source_set: str | None = None,
    generation_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert an lm-saes AttributionResult + pruned attribution into Neuronpedia graph JSON."""

    logit_token_map = {
        tid: tok for tid, tok in zip(ar.logit_token_ids, ar.logit_tokens)
    }
    logit_prob_map = {
        tid: float(p.item()) for tid, p in zip(ar.logit_token_ids, ar.probs)
    }
    n_layers = max((int(m["layer_idx"]) for m in sae_metadata.values()), default=-1) + 1

    edge_weights, (targets, sources) = pruned_attribution.nonzero()
    nodes = (sources + targets).unique()
    node_activations = ar.activations[nodes].data

    def make_node(
        node_key: str, indices_tuple: tuple[int, ...], activation: float
    ) -> dict[str, Any]:
        indices = list(indices_tuple)

        if node_key == "hook_embed":
            pos = indices[0]
            token_id = (
                int(ar.prompt_token_ids[pos]) if pos < len(ar.prompt_token_ids) else -1
            )
            token = ar.prompt_tokens[pos] if pos < len(ar.prompt_tokens) else ""
            node_id = f"E_{token_id}_{pos}"
            return {
                "node_id": node_id,
                "jsNodeId": node_id,
                "feature": None,
                "feature_type": "embedding",
                "layer": "E",
                "ctx_idx": pos,
                "clerp": f'Emb: "{token}"',
                "token": token,
                "activation": None,
            }

        if node_key == "logits":
            vocab_idx = indices[0]
            ctx_idx = max(len(ar.prompt_token_ids) - 1, 0)
            layer = n_layers
            token = logit_token_map.get(vocab_idx, "?")
            prob = logit_prob_map.get(vocab_idx, 0.0)
            node_id = f"{layer}_{vocab_idx}_{ctx_idx}"
            return {
                "node_id": node_id,
                "jsNodeId": node_id,
                "feature": vocab_idx,
                "feature_type": "logit",
                "layer": layer,
                "ctx_idx": ctx_idx,
                "clerp": f'Logit "{token}" (p={prob:.3f})',
                "token_prob": prob,
                "token": token,
                "activation": None,
            }

        if node_key.endswith(".error"):
            hook_point_out = node_key.removesuffix(".error")
            metadata = sae_metadata.get(hook_point_out)
            is_lorsa = bool(metadata["is_lorsa"]) if metadata else False
            layer_base = (
                int(metadata["layer_idx"])
                if metadata
                else _parse_hook_layer(hook_point_out)
            )
            layer = layer_base
            pos = indices[0]
            feature_type = "lorsa error" if is_lorsa else "mlp reconstruction error"
            token = ar.prompt_tokens[pos] if pos < len(ar.prompt_tokens) else ""
            prefix = "attn" if is_lorsa else "mlp"
            node_id = f"{layer}_error_{prefix}_{pos}"
            return {
                "node_id": node_id,
                "jsNodeId": node_id,
                "feature": None,
                "feature_type": feature_type,
                "layer": layer,
                "ctx_idx": pos,
                "clerp": f'Err: {prefix} "{token}"',
                "activation": None,
            }

        if node_key.endswith(".sae.hook_feature_acts"):
            hook_point_out = node_key.removesuffix(".sae.hook_feature_acts")
            metadata = sae_metadata.get(hook_point_out)
            is_lorsa = bool(metadata["is_lorsa"]) if metadata else False
            layer_base = (
                int(metadata["layer_idx"])
                if metadata
                else _parse_hook_layer(hook_point_out)
            )
            layer = layer_base
            pos = indices[0] if len(indices) > 0 else 0
            feature_idx = indices[1] if len(indices) > 1 else 0
            feature_type = "lorsa" if is_lorsa else "cross layer transcoder"
            cantor_value = cantor_pair(layer, feature_idx)
            prefix = "" if is_lorsa else ""
            node_id = f"{layer}_{feature_type[:3]}_{feature_idx}_{pos}"
            return {
                "node_id": node_id,
                "jsNodeId": node_id,
                "feature": cantor_value,
                "feature_type": feature_type,
                "layer": layer,
                "ctx_idx": pos,
                "clerp": f"{prefix}F{cantor_value}",
                "activation": activation,
            }

        # Fallback (bias nodes, etc.)
        fallback_id = f"{node_key}:{'_'.join(str(v) for v in indices)}"
        return {
            "node_id": fallback_id,
            "jsNodeId": fallback_id,
            "feature": None,
            "feature_type": "bias",
            "layer": 0,
            "ctx_idx": indices[0] if len(indices) > 0 else 0,
            "clerp": "bias",
            "activation": None,
        }

    node_entries = [
        make_node(str(ni.key), tuple(ni.indices[0].tolist()), float(act))
        for ni, act in zip(nodes, node_activations)
    ]
    node_ids = [n["node_id"] for n in node_entries]

    target_node_offsets = nodes.nodes_to_offsets(targets).tolist()
    source_node_offsets = nodes.nodes_to_offsets(sources).tolist()
    edge_weight_values = edge_weights.tolist()
    links = [
        {
            "source": node_ids[src],
            "target": node_ids[tgt],
            "weight": float(w),
        }
        for w, src, tgt in zip(
            edge_weight_values, source_node_offsets, target_node_offsets
        )
    ]

    influence_scores = _compute_node_influence(node_entries, links, ar.probs)
    for entry, score in zip(node_entries, influence_scores):
        entry["influence"] = score

    _attach_qk_tracing_results(ar, nodes, node_ids, node_entries)

    feature_details: dict[str, str] = {}
    if np_transcoder_source_set:
        feature_details["neuronpedia_source_set"] = np_transcoder_source_set
    if np_lorsa_source_set:
        feature_details["neuronpedia_lorsa_source_set"] = np_lorsa_source_set

    metadata: dict[str, Any] = {
        "slug": slug,
        "scan": np_model_id,
        "prompt": prompt,
        "prompt_tokens": ar.prompt_tokens,
        "schema_version": 1,
        "node_threshold": node_threshold,
        "info": {
            "generator": {
                "name": "llamascopium (CRM) by OpenMOSS",
                "version": pkg_version("llamascopium"),
                "url": "https://github.com/OpenMOSS/Language-Model-SAEs",
            },
        },
        "pruning_settings": {
            "node_threshold": node_threshold,
            "edge_threshold": edge_threshold,
        },
    }
    if feature_details:
        metadata["feature_details"] = feature_details
    if generation_settings:
        metadata["generation_settings"] = generation_settings

    return {
        "metadata": metadata,
        "qParams": {},
        "nodes": node_entries,
        "links": links,
    }
