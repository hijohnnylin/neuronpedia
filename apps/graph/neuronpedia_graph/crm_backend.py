"""CRM (Complete Replacement Model) backend using lm-saes for attribution with Lorsa + Transcoders."""

from __future__ import annotations

import gzip
import json
import os
import time
from typing import Any

import requests as http_requests
import torch
from llamascopium.backend.attribution import AttributionResult, prune_attribution
from llamascopium.backend.language_model import (
    LanguageModelConfig,
    TransformerLensLanguageModel,
)
from llamascopium.models.sparse_dictionary import SparseDictionary

from .format_converter import _build_sae_metadata, convert_to_neuronpedia_graph

NP_MODEL_ID = os.getenv("NP_MODEL_ID", "qwen3-1.7b")
NP_TRANSCODER_SOURCE_SET = os.getenv("NP_TRANSCODER_SOURCE_SET")
NP_LORSA_SOURCE_SET = os.getenv("NP_LORSA_SOURCE_SET")


def get_device() -> torch.device:
    device_env = os.environ.get("DEVICE")
    if device_env:
        return torch.device(device_env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model_dtype() -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(os.environ.get("MODEL_DTYPE", "bfloat16"), torch.bfloat16)


def load_crm_model() -> tuple[
    TransformerLensLanguageModel, list[SparseDictionary], dict[str, dict[str, Any]]
]:
    """Load the CRM model and all SAE/Lorsa replacement modules. Returns (model, replacement_modules, sae_metadata)."""
    model_id = os.getenv("MODEL_ID")
    sae_repo = os.getenv("SAE_REPO", "OpenMOSS-Team/Llama-Scope-2-Qwen3-1.7B")
    sae_expansion = os.getenv("SAE_EXPANSION", "8x")
    sae_topk = os.getenv("SAE_TOPK", "k64")

    device = get_device()
    model_dtype = get_model_dtype()

    print(f"[CRM] Loading model: {model_id} on {device} with dtype {model_dtype}")
    cfg = LanguageModelConfig(
        model_name=model_id, dtype=model_dtype, device=str(device)
    )
    model = TransformerLensLanguageModel(cfg)

    n_layers = model.cfg.n_layers
    print(f"[CRM] Model loaded: {n_layers} layers")

    print(
        f"[CRM] Loading SAEs from {sae_repo} ({sae_expansion}/{sae_topk}) for {n_layers} layers..."
    )
    replacement_modules: list[SparseDictionary] = []

    for layer_idx in range(n_layers):
        tc_name = f"layer{layer_idx}_transcoder_{sae_expansion}_{sae_topk}"
        tc_path = f"{sae_repo}:transcoder/{sae_expansion}/{sae_topk}/{tc_name}"
        print(f"  Loading transcoder layer {layer_idx}")
        tc = SparseDictionary.from_pretrained(tc_path, device=str(device), dtype=model_dtype)
        replacement_modules.append(tc)

        lorsa_name = f"layer{layer_idx}_lorsa_{sae_expansion}_{sae_topk}"
        lorsa_path = f"{sae_repo}:lorsa/{sae_expansion}/{sae_topk}/{lorsa_name}"
        print(f"  Loading lorsa layer {layer_idx}")
        lorsa = SparseDictionary.from_pretrained(lorsa_path, device=str(device), dtype=model_dtype)
        replacement_modules.append(lorsa)

    sae_metadata = _build_sae_metadata(replacement_modules)
    print(
        f"[CRM] Loaded {len(replacement_modules)} replacement modules ({len(replacement_modules) // 2} layers x 2)"
    )

    return model, replacement_modules, sae_metadata


def generate_graph_crm(
    prompt: str,
    model: TransformerLensLanguageModel,
    replacement_modules: list[SparseDictionary],
    sae_metadata: dict[str, dict[str, Any]],
    *,
    slug_identifier: str,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 16,
    max_feature_nodes: int = 10000,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    signed_url: str | None = None,
    user_id: str | None = None,
    compress: bool = False,
    enable_qk_tracing: bool = False,
    qk_top_fraction: float = 0.6,
    qk_topk: int = 10,
) -> dict[str, Any]:
    """Run CRM attribution, prune, convert to Neuronpedia format, and optionally upload to S3."""
    total_start = time.time()

    attribution_start = time.time()
    ar: AttributionResult = model.attribute(
        inputs=prompt,
        replacement_modules=replacement_modules,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        batch_size=batch_size,
        max_features=max_feature_nodes,
        enable_qk_tracing=enable_qk_tracing,
        qk_top_fraction=qk_top_fraction,
        qk_topk=qk_topk,
    )
    attribution_ms = (time.time() - attribution_start) * 1000
    print(f"[CRM] Attribution completed in {attribution_ms:.0f}ms")

    # Note: `model.attribute(...)` already returns tensors on the model's
    # device, so there is no need to move them again. Previous versions called
    # `.to(device)` here, but that triggers llamascopium's PyTree cattrs
    # round-trip on `NodeIndexedMatrix` / `NodeDimension`, which can fail on
    # fields like `DiscreteMapper` or `torch.device | str` that have no
    # registered structure hooks.

    pruned = prune_attribution(
        ar.attribution,
        ar.probs,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )
    print("[CRM] Pruning completed")

    generation_settings: dict[str, Any] = {
        "max_n_logits": max_n_logits,
        "desired_logit_prob": desired_logit_prob,
        "batch_size": batch_size,
        "max_feature_nodes": max_feature_nodes,
    }
    if enable_qk_tracing:
        generation_settings["enable_qk_tracing"] = enable_qk_tracing
        generation_settings["qk_top_fraction"] = qk_top_fraction
        generation_settings["qk_topk"] = qk_topk

    output = convert_to_neuronpedia_graph(
        ar,
        pruned,
        sae_metadata,
        slug=slug_identifier,
        np_model_id=NP_MODEL_ID,
        prompt=prompt,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        np_transcoder_source_set=NP_TRANSCODER_SOURCE_SET,
        np_lorsa_source_set=NP_LORSA_SOURCE_SET,
        generation_settings=generation_settings,
    )

    if signed_url is None:
        return output

    output["metadata"]["info"]["creator_name"] = user_id or "Anonymous (CRM)"
    output["metadata"]["info"]["creator_url"] = "https://neuronpedia.org"
    output["metadata"]["info"]["create_time_ms"] = int(time.time() * 1000)

    model_json = json.dumps(output)

    if compress:
        data_to_upload = gzip.compress(model_json.encode("utf-8"), compresslevel=3)
        headers = {"Content-Type": "application/json", "Content-Encoding": "gzip"}
    else:
        data_to_upload = model_json.encode("utf-8")
        headers = {"Content-Type": "application/json"}

    upload_start = time.time()
    response = http_requests.put(signed_url, data=data_to_upload, headers=headers)
    upload_ms = (time.time() - upload_start) * 1000

    if response.status_code != 200:
        return {"error": "Failed to upload file"}

    total_ms = (time.time() - total_start) * 1000
    print(
        f"[CRM] Upload complete: {len(data_to_upload)} bytes in {upload_ms:.0f}ms (total {total_ms:.0f}ms)"
    )

    return {"success": f"Graph uploaded successfully to url: {signed_url}"}


def forward_pass_crm(
    prompt: str,
    model: TransformerLensLanguageModel,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
) -> dict[str, Any]:
    """Run a forward pass and return salient logits."""
    device = get_device()
    tokens = model.tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([tokens]).to(device)

    with torch.no_grad():
        output = model(input_ids)
        if hasattr(output, "logits"):
            output = output.logits
        logits = output[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        topk_probs, topk_indices = torch.topk(
            probs, min(max_n_logits * 3, probs.shape[0])
        )

        results = []
        cumulative = 0.0
        for idx, prob in zip(topk_indices.tolist(), topk_probs.tolist()):
            results.append(
                {
                    "token": model.tokenizer.decode([idx]),
                    "token_id": idx,
                    "probability": prob,
                }
            )
            cumulative += prob
            if cumulative >= desired_logit_prob and len(results) >= max_n_logits:
                break

    return {
        "prompt": prompt,
        "input_tokens": [model.tokenizer.decode([t]) for t in tokens],
        "salient_logits": results,
        "total_salient_tokens": len(results),
        "cumulative_probability": cumulative,
    }
