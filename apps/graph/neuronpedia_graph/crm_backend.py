"""CRM (Complete Replacement Model) backend using lm-saes for attribution with Lorsa + Transcoders."""

from __future__ import annotations

import gzip
import json
import os
import time
from typing import TYPE_CHECKING, Any

import requests as http_requests
import torch
from llamascopium.backend.attribution import AttributionResult, prune_attribution
from llamascopium.models.sparse_dictionary import SparseDictionary

from .format_converter import _build_sae_metadata, convert_to_neuronpedia_graph
from .runtime_env import get_device, get_model_dtype, get_model_engine
from .schemas import ForwardPassResponse, GraphGenerationResponse, SalientLogit

if TYPE_CHECKING:
    from llamascopium.backend.language_model import TransformerLensLanguageModel

    from .crm_interp_engine import InterpEngineLanguageModel

# Either implementation of the model the CRM attributes through. They are not related by
# inheritance -- one is a `HookedTransformer`, the other wraps a HuggingFace model --
# but the attribution only ever calls the handful of methods both provide. Each is imported in
# the branch that picks it, purely to keep the two symmetric: llamascopium depends on
# transformer-lens unconditionally, so this does not avoid importing TransformerLens.
type CRMModel = TransformerLensLanguageModel | InterpEngineLanguageModel

NP_MODEL_ID = os.getenv("NP_MODEL_ID", "qwen3-1.7b")
NP_TRANSCODER_SOURCE_SET = os.getenv("NP_TRANSCODER_SOURCE_SET")
NP_LORSA_SOURCE_SET = os.getenv("NP_LORSA_SOURCE_SET")


def load_crm_model() -> tuple[CRMModel, list[SparseDictionary], dict[str, dict[str, Any]]]:
    """Load the CRM model and all SAE/Lorsa replacement modules. Returns (model, replacement_modules, sae_metadata)."""
    model_id = os.getenv("MODEL_ID")
    sae_repo = os.getenv("SAE_REPO", "OpenMOSS-Team/Llama-Scope-2-Qwen3-1.7B")
    sae_expansion = os.getenv("SAE_EXPANSION", "8x")
    sae_topk = os.getenv("SAE_TOPK", "k64")

    device = get_device()
    model_dtype = get_model_dtype()
    model_engine = get_model_engine()

    print(f"[CRM] Loading model: {model_id} on {device} with dtype {model_dtype} via {model_engine}")

    # `interp_engine` (the default) hooks the HuggingFace model in place, so it reaches any
    # `AutoModelForCausalLM`; `transformerlens` converts the weights into TransformerLens'
    # convention, so it only reaches architectures TransformerLens has a conversion for. The two
    # are measured to agree to a few parts per million in float32
    # (`tests/test_crm_backend_parity.py`), which is what makes interp_engine safe to default to.
    # See `crm_interp_engine` for what the two do and do not share. `runtime_env` has already
    # rejected anything outside the three engines; the third is turned away below.
    model: CRMModel
    if model_engine == "interp_engine":
        from .crm_interp_engine import InterpEngineLanguageModel

        assert model_id is not None, "MODEL_ID must be set."
        model = InterpEngineLanguageModel(model_id, dtype=model_dtype, device=device)
    elif model_engine == "transformerlens":
        from llamascopium.backend.language_model import (
            LanguageModelConfig,
            TransformerLensLanguageModel,
        )

        cfg = LanguageModelConfig(model_name=model_id, dtype=model_dtype, device=str(device))
        model = TransformerLensLanguageModel(cfg)
    else:
        raise ValueError(
            f"--model-engine {model_engine!r} is not available for the lm-saes-crm attribution "
            "engine; it accepts 'interp_engine' or 'transformerlens'. (nnsight is circuit-tracer "
            "only: llamascopium's attribution reaches the model through methods nnsight's proxies "
            "do not provide.)"
        )

    n_layers = model.cfg.n_layers
    print(f"[CRM] Model loaded: {n_layers} layers")

    print(f"[CRM] Loading SAEs from {sae_repo} ({sae_expansion}/{sae_topk}) for {n_layers} layers...")
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
    print(f"[CRM] Loaded {len(replacement_modules)} replacement modules ({len(replacement_modules) // 2} layers x 2)")

    return model, replacement_modules, sae_metadata


def generate_graph_crm(
    prompt: str,
    model: CRMModel,
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
) -> GraphGenerationResponse | dict[str, Any]:
    """Run CRM attribution, prune, convert to Neuronpedia format, and optionally upload to S3.

    Returns an upload receipt, or -- when no ``signed_url`` is given -- the graph document
    itself, whose keys are the published graph-schema.json shape rather than a model of ours.
    """
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
        return GraphGenerationResponse(error="Failed to upload file")

    total_ms = (time.time() - total_start) * 1000
    print(f"[CRM] Upload complete: {len(data_to_upload)} bytes in {upload_ms:.0f}ms (total {total_ms:.0f}ms)")

    return GraphGenerationResponse(success=f"Graph uploaded successfully to url: {signed_url}")


def forward_pass_crm(
    prompt: str,
    model: CRMModel,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
) -> ForwardPassResponse:
    """Run a forward pass and return salient logits."""
    device = get_device()
    tokens = model.tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([tokens]).to(device)

    with torch.no_grad():
        output = model(input_ids)
        # The interp_engine path returns a HuggingFace ModelOutput; the TransformerLens path
        # returns the logits tensor directly.
        all_logits: torch.Tensor = getattr(output, "logits", output)
        logits = all_logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        topk_probs, topk_indices = torch.topk(probs, min(max_n_logits * 3, probs.shape[0]))

        results: list[SalientLogit] = []
        cumulative = 0.0
        for idx, prob in zip(topk_indices.tolist(), topk_probs.tolist()):
            results.append(SalientLogit(token=model.tokenizer.decode([idx]), token_id=idx, probability=prob))
            cumulative += prob
            if cumulative >= desired_logit_prob and len(results) >= max_n_logits:
                break

    return ForwardPassResponse(
        prompt=prompt,
        input_tokens=[model.tokenizer.decode([t]) for t in tokens],
        salient_logits=results,
        total_salient_tokens=len(results),
        cumulative_probability=cumulative,
    )
