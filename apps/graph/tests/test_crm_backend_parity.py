"""The interp_engine CRM backend must agree with the transformerlens one.

Both run llamascopium's attribution unmodified; they differ only in what runs the forward pass and
therefore in how the tensors the algorithm needs are produced. On TransformerLens they are hook
points on converted weights; on interp-engine several of them do not exist as module boundaries at
all and are reconstructed (see `neuronpedia_graph.crm_interp_engine`). This checks that the
reconstruction lands on the same numbers, including for QK tracing, whose bias leaves are the one
part of the algorithm that addresses the model through TransformerLens' module tree.

Deliberately run in float32. In bfloat16 this attribution is not reproducible to better than ~0.3
relative on the adjacency matrix *within a single backend* -- TransformerLens in bfloat16 disagrees
with TransformerLens in float32 by as much as the two backends disagree with each other -- so a
bfloat16 comparison measures rounding, not agreement.

Heavy: two models and two sets of replacement modules, a few GB of downloads. Opt in with
``RUN_CRM_PARITY=1``.
"""

import os

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("RUN_CRM_PARITY") != "1",
        reason="Set RUN_CRM_PARITY=1 to run (downloads several GB, needs a GPU)",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU"),
]

MODEL = "Qwen/Qwen3-1.7B"
SAE_REPO = "OpenMOSS-Team/Llama-Scope-2-Qwen3-1.7B"
EXPANSION, TOPK = "8x", "k64"
# Two layers is enough to exercise everything that differs: a transcoder and a Lorsa per layer, and
# at least one cross-layer edge between them. Deliberately from the middle of the 28 -- at layer 0
# the embedding dominates every attribution and the graph collapses to a couple of nodes, which
# hides disagreement rather than exposing it.
LAYERS = (14, 15)
PROMPT = "Fact: Michael Jordan played the sport of basketball, and Babe Ruth played the sport of"
DTYPE = torch.float32
# QK tracing requires batch_size >= qk_topk.
BATCH_SIZE = 16
MAX_N_LOGITS = 5
QK_TOPK = 10

# Relative to each tensor's own scale. float32 through an attribution this deep leaves a few parts
# per million between two different orderings of the same arithmetic.
REL_TOL = 1e-4


def _dump_qk_marginal(
    marginal,
) -> tuple[list[str], list[tuple[list[tuple[str, ...]], torch.Tensor]]]:
    """A QK marginal as ``(target keys, [(source keys per dimension, values)] per target)``.

    Keys are compared as well as values: QK tracing picks its own targets (the most influential
    Lorsa features) and its own top-k sources, so agreeing on numbers while disagreeing on *which*
    nodes they belong to would not be parity.
    """
    targets = [str(key) for key in marginal.dimensions[0].node_mappings]
    per_target = []
    for inner, _target_info in marginal:
        keys = [tuple(str(k) for k in dim.node_mappings) for dim in inner.dimensions]
        per_target.append((keys, inner.value.detach().float().cpu()))
    return targets, per_target


def _run(backend: str, enable_qk_tracing: bool = False) -> dict:
    from llamascopium.models.sparse_dictionary import SparseDictionary

    if backend == "transformerlens":
        from llamascopium.backend.language_model import (
            LanguageModelConfig,
            TransformerLensLanguageModel,
        )

        model = TransformerLensLanguageModel(LanguageModelConfig(model_name=MODEL, dtype=DTYPE, device="cuda"))
    else:
        from neuronpedia_graph.crm_interp_engine import InterpEngineLanguageModel

        model = InterpEngineLanguageModel(MODEL, dtype=DTYPE, device="cuda")

    modules = [
        SparseDictionary.from_pretrained(
            f"{SAE_REPO}:{kind}/{EXPANSION}/{TOPK}/layer{layer}_{kind}_{EXPANSION}_{TOPK}",
            device="cuda",
            dtype=DTYPE,
        )
        for layer in LAYERS
        for kind in ("transcoder", "lorsa")
    ]

    result = model.attribute(
        inputs=PROMPT,
        replacement_modules=modules,
        max_n_logits=MAX_N_LOGITS,
        batch_size=BATCH_SIZE,
        enable_qk_tracing=enable_qk_tracing,
        qk_top_fraction=1.0,
        qk_topk=QK_TOPK,
    )
    collected = {
        "logits": result.logits.detach().float().cpu(),
        "probs": result.probs.detach().float().cpu(),
        "adjacency": result.attribution.data.detach().float().cpu(),
        "activations": result.activations.data.detach().float().cpu(),
        "logit_token_ids": result.logit_token_ids,
        "prompt_token_ids": result.prompt_token_ids,
        "row_keys": [str(key) for key in result.attribution.dimensions[0].node_mappings],
        "col_keys": [str(key) for key in result.attribution.dimensions[1].node_mappings],
        "n_lorsa_modules": sum(1 for m in modules if m.cfg.sae_type == "lorsa"),
    }
    if enable_qk_tracing:
        qk = result.qk_trace_results
        assert qk is not None, "enable_qk_tracing=True produced no qk_trace_results"
        collected["qk"] = {name: _dump_qk_marginal(getattr(qk, name)) for name in ("pairs", "q_marginal", "k_marginal")}
    del model, modules, result
    torch.cuda.empty_cache()
    return collected


@pytest.fixture(scope="module")
def results() -> dict[str, dict]:
    """One at a time: two 1.7B models plus two module sets do not need to be resident together."""
    return {backend: _run(backend) for backend in ("transformerlens", "interp_engine")}


@pytest.fixture(scope="module")
def qk_results() -> dict[str, dict]:
    """A separate pass, so the plain path above stays the one the default settings exercise."""
    return {backend: _run(backend, enable_qk_tracing=True) for backend in ("transformerlens", "interp_engine")}


@pytest.fixture(scope="module")
def tl(results):
    return results["transformerlens"]


@pytest.fixture(scope="module")
def ie(results):
    return results["interp_engine"]


def _assert_close(a: torch.Tensor, b: torch.Tensor, name: str) -> None:
    assert a.shape == b.shape, f"{name} shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}"
    scale = b.abs().max().clamp(min=1.0)
    diff = (a - b).abs().max()
    assert diff <= REL_TOL * scale, (
        f"{name} differ by max {diff} against a scale of {scale} ({diff / scale:.2e} relative, limit {REL_TOL:.0e})"
    )


def test_lorsa_modules_were_actually_spliced(ie):
    """The Lorsa half is what makes this a CRM rather than a transcoder-only graph."""
    assert ie["n_lorsa_modules"] == len(LAYERS)


def test_same_tokens_and_logit_ranking(tl, ie):
    assert ie["prompt_token_ids"] == tl["prompt_token_ids"]
    assert ie["logit_token_ids"] == tl["logit_token_ids"]


def test_same_graph_nodes(tl, ie):
    """Node identity, not just node count: the greedy collection has to select the same features."""
    assert ie["row_keys"] == tl["row_keys"]
    assert ie["col_keys"] == tl["col_keys"]


def test_same_adjacency(tl, ie):
    _assert_close(ie["adjacency"], tl["adjacency"], "Adjacency")


def test_same_activations_and_probs(tl, ie):
    _assert_close(ie["activations"], tl["activations"], "Node activations")
    _assert_close(ie["probs"], tl["probs"], "Logit probabilities")


@pytest.mark.parametrize("marginal", ["pairs", "q_marginal", "k_marginal"])
def test_same_qk_tracing(qk_results, marginal):
    """QK tracing has to agree on which nodes it picks as well as on the numbers.

    Its second-order pass attributes to the biases as source nodes, which means turning each one
    into a batched leaf tensor. llamascopium does that through TransformerLens' module tree rather
    than through the model's own interface, so this is the part of the algorithm least likely to
    survive a change of backend -- see ``InterpEngineLanguageModel.hooks``.
    """
    tl_targets, tl_per_target = qk_results["transformerlens"]["qk"][marginal]
    ie_targets, ie_per_target = qk_results["interp_engine"]["qk"][marginal]

    assert ie_targets == tl_targets, f"{marginal} traced different target features"
    for target, (tl_keys, tl_values), (ie_keys, ie_values) in zip(tl_targets, tl_per_target, ie_per_target):
        assert ie_keys == tl_keys, f"{marginal} for {target} found different contributors"
        _assert_close(ie_values, tl_values, f"{marginal} for {target}")


def test_qk_tracing_attributes_to_bias_leaves(qk_results):
    """The bias leaves are the reason QK tracing needs anything special from the backend.

    If they were silently absent the tracing would still run and still return plausible numbers, so
    assert they are actually among the sources on both backends.
    """
    for backend, collected in qk_results.items():
        leaves = {
            key
            for _targets, per_target in collected["qk"].values()
            for keys, _values in per_target
            for dimension in keys
            for key in dimension
            if ".hook_b_" in key
        }
        assert leaves, f"{backend} attributed to no bias leaves at all"


def test_logits_agree_up_to_the_unembed_centering(tl, ie):
    """TransformerLens centers the unembedding, which shifts a row's logits by a constant.

    That is invisible to softmax and to the attribution, which mean-centers the logits before using
    them as targets -- so the two backends are expected to differ by exactly one constant per row,
    and by nothing else.
    """
    gap = tl["logits"] - ie["logits"]
    residual = (gap - gap.mean(dim=-1, keepdim=True)).abs().max()
    scale = tl["logits"].abs().max().clamp(min=1.0)
    assert residual <= REL_TOL * scale, (
        f"Logits differ by more than a per-row constant: {residual} remains after removing it"
    )
