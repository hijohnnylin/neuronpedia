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
    make_node: Any,
    qk_only_nodes: dict[str, dict[str, Any]],
) -> None:
    """Populate a ``qk_tracing_results`` object on each target lorsa node entry.

    The shape mirrors OpenMOSS' reference server logic:
        {
            "pair_wise_contributors": [(q_node_id, k_node_id, attribution), ...],
            "top_q_marginal_contributors": [(q_node_id, attribution), ...],
            "top_k_marginal_contributors": [(k_node_id, attribution), ...],
        }

    QK contributors whose upstream node was pruned out of the main graph are
    still referenced by their ``node_id``. Their descriptor is written to the
    side-channel ``qk_only_nodes`` dict (keyed by ``node_id``) so the UI can
    resolve the id to a label/layer/feature for display in the node
    connections panel without polluting the primary ``nodes`` / ``links``
    arrays.
    """
    qk = getattr(ar, "qk_trace_results", None)
    if qk is None:
        return

    pruned_ids: set[str] = set(node_ids)

    def _to_cpu_list(tensor: Any) -> list[int]:
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        return tensor.tolist()

    def _node_offsets(dim: Any) -> list[int]:
        return _to_cpu_list(nodes.nodes_to_offsets(dim))

    def _activation_offsets(dim: Any) -> list[int]:
        return _to_cpu_list(ar.activations.dimensions[0].nodes_to_offsets(dim))

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

    def _resolve_contributor_ids(
        dim: Any,
        offsets_in_pruned: list[int],
        offsets_in_activations: list[int],
    ) -> list[str]:
        """Return node_ids for every entry in ``dim``.

        For contributors absent from the pruned graph, a descriptor is recorded
        in ``qk_only_nodes`` (deduped by ``node_id``) and that id is returned.
        """
        resolved: list[str] = []
        materialized: list[Any] | None = None
        for i, off in enumerate(offsets_in_pruned):
            if off >= 0:
                resolved.append(node_ids[off])
                continue
            if materialized is None:
                materialized = list(dim)
            ni = materialized[i]
            key = str(ni.key)
            indices_tuple = tuple(int(v) for v in ni.indices[0].tolist())
            act_off = offsets_in_activations[i]
            activation = (
                float(ar.activations.data[act_off].item()) if act_off >= 0 else 0.0
            )
            entry = make_node(key, indices_tuple, activation)
            entry_id = entry["node_id"]
            if entry_id not in pruned_ids and entry_id not in qk_only_nodes:
                qk_only_nodes[entry_id] = entry
            resolved.append(entry_id)
        return resolved

    target_offsets = _node_offsets(qk.pairs.dimensions[0])

    for target_offset, (pairs_ni, _target_info) in zip(target_offsets, qk.pairs):
        if target_offset < 0:
            continue
        q_dim = pairs_ni.dimensions[0]
        k_dim = pairs_ni.dimensions[1]
        q_ids = _resolve_contributor_ids(
            q_dim, _node_offsets(q_dim), _activation_offsets(q_dim)
        )
        k_ids = _resolve_contributor_ids(
            k_dim, _node_offsets(k_dim), _activation_offsets(k_dim)
        )
        values = pairs_ni.value.detach().cpu().tolist()
        contributors: list[tuple[str, str, float]] = [
            (q_id, k_id, float(v)) for q_id, k_id, v in zip(q_ids, k_ids, values)
        ]
        _ensure_entry(target_offset)["pair_wise_contributors"] = contributors

    for marginal, out_key in (
        (qk.q_marginal, "top_q_marginal_contributors"),
        (qk.k_marginal, "top_k_marginal_contributors"),
    ):
        marginal_target_offsets = _node_offsets(marginal.dimensions[0])
        for target_offset, (marg_ni, _target_info) in zip(marginal_target_offsets, marginal):
            if target_offset < 0:
                continue
            src_dim = marg_ni.dimensions[0]
            src_ids = _resolve_contributor_ids(
                src_dim, _node_offsets(src_dim), _activation_offsets(src_dim)
            )
            values = marg_ni.value.detach().cpu().tolist()
            contributors_1d: list[tuple[str, float]] = [
                (sid, float(v)) for sid, v in zip(src_ids, values)
            ]
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

        # Bias leaves (surfaced as QK-trace sources when `enable_qk_tracing=True`).
        # llamascopium exposes them under these hook-point suffixes:
        #   blocks.{i}.attn.hook_b_O        -> attention output bias
        #   blocks.{i}.mlp.hook_b_out       -> MLP output bias
        #   {hook_point_out}.sae.hook_b_Q   -> Lorsa Q bias
        #   {hook_point_out}.sae.hook_b_K   -> Lorsa K bias
        #   {hook_point_out}.sae.hook_b_D   -> SAE / Lorsa reconstruction bias
        bias_kind: str | None = None
        bias_hook_point_out: str | None = None
        for sae_suffix in (".sae.hook_b_Q", ".sae.hook_b_K", ".sae.hook_b_D"):
            if node_key.endswith(sae_suffix):
                bias_hook_point_out = node_key.removesuffix(sae_suffix)
                metadata = sae_metadata.get(bias_hook_point_out)
                is_lorsa = bool(metadata["is_lorsa"]) if metadata else False
                b_letter = sae_suffix[-1]  # "Q" / "K" / "D"
                bias_kind = f"{'lorsa' if is_lorsa else 'tc'}_b{b_letter}"
                break
        if bias_kind is None:
            if node_key.endswith(".attn.hook_b_O"):
                bias_kind = "attn_bO"
            elif node_key.endswith(".mlp.hook_b_out"):
                bias_kind = "mlp_bout"

        if bias_kind is not None:
            layer = (
                int(sae_metadata[bias_hook_point_out]["layer_idx"])
                if bias_hook_point_out and bias_hook_point_out in sae_metadata
                else _parse_hook_layer(node_key)
            )
            pos = indices[0] if len(indices) > 0 else 0
            node_id = f"{layer}_bias_{bias_kind}_{pos}"
            return {
                "node_id": node_id,
                "jsNodeId": node_id,
                "feature": None,
                "feature_type": "bias",
                "layer": layer,
                "ctx_idx": pos,
                "clerp": f"L{layer} Bias ({bias_kind})",
                "activation": None,
            }

        # Unknown source key: keep a raw, readable id so the UI doesn't silently
        # collapse distinct contributors, but still follow the `{layer}_..._{pos}`
        # shape so frontend parsers don't choke.
        layer = _parse_hook_layer(node_key)
        pos = indices[0] if len(indices) > 0 else 0
        safe_key = node_key.replace(".", "_")
        fallback_id = f"{layer}_unknown_{safe_key}_{pos}"
        return {
            "node_id": fallback_id,
            "jsNodeId": fallback_id,
            "feature": None,
            "feature_type": "unknown",
            "layer": layer,
            "ctx_idx": pos,
            "clerp": node_key,
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

    qk_only_nodes: dict[str, dict[str, Any]] = {}
    _attach_qk_tracing_results(
        ar, nodes, node_ids, node_entries, make_node, qk_only_nodes
    )

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

    result: dict[str, Any] = {
        "metadata": metadata,
        "qParams": {},
        "nodes": node_entries,
        "links": links,
    }
    if qk_only_nodes:
        result["qk_only_nodes"] = qk_only_nodes
    return result
