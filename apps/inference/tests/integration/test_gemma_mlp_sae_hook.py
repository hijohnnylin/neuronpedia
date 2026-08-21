"""Which MLP tensor Gemma Scope's MLP SAEs read, measured on the shipping checkpoint.

TransformerLens has two names for the MLP output and they are different tensors on gemma-2:
block-level ``blocks.{i}.hook_mlp_out`` fires after the post-feedforward norm (the engine's
``mlp_out_post``), while ``blocks.{i}.mlp.hook_out`` is the raw module output (``mlp_out``). The
engine's mapper needs a model to tell them apart and ``engine_adapter`` cannot pass one, so it
resolves the block-level hooks to the ``*_post`` points unconditionally — see
``engine_adapter.tlens_hook_to_point``.

This is the evidence for that choice, and it is a numeric test rather than a comment because
nothing about getting it wrong raises. Reading ``gemmascope-mlp-16k`` off the raw output leaves the
whole source silently dead: the layer-4 feature whose published dashboard tops out at 23.5 on
``mass-production`` fires at no position in that text at all, which is how the mistake surfaced.

Marked ``xl``, so it never runs per-PR: it needs gemma-2-2b's weights and a ~300MB SAE. Run it with
``uv run pytest tests/integration/test_gemma_mlp_sae_hook.py -v -m xl``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from interp_engine import Address, EagerModel, run_with_cache

from neuronpedia_inference.engine_adapter import tlens_hook_to_point

# The text whose dashboard is the feature's top activation, generated with TransformerLens before
# the engine migration. Leading space and newline included: they are what was tokenized.
DASHBOARD_TEXT = (
    " For such a reason, throughput has conventionally been low, about several pieces per hour "
    "(in 8-inch wafer), proving the method to be unsuitable as a mass-production technology.\n"
    "As one of the measures to improve throughput of the electron beam lithography, for example "
    "as described in pp. 6897 to 6901, Japan Journal Applied Physics, vol., 39 (2000), electron "
    "projection lithography has been presented, which forms all patterns on a mask original "
    "plate (referred to as a reticle, hereinafter), and then projects/transfers the patterns by "
    "using electron beams."
)

MODEL_ID = "google/gemma-2-2b"
SAE_REPO = "google/gemma-scope-2b-pt-mlp"
SAE_FILE = "layer_4/width_16k/average_l0_85/params.npz"
LAYER = 4
FEATURE = 0
HOOK = f"blocks.{LAYER}.hook_mlp_out"

pytestmark = [pytest.mark.xl, pytest.mark.cuda]


@pytest.fixture(scope="module")
def model() -> EagerModel:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    try:
        return EagerModel(MODEL_ID, device="cuda", dtype="float32")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{MODEL_ID} weights unavailable: {exc}")


@pytest.fixture(scope="module")
def sae() -> dict[str, torch.Tensor]:
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo_id=SAE_REPO, filename=SAE_FILE)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{SAE_REPO}/{SAE_FILE} unavailable: {exc}")
    return {k: torch.tensor(v, device="cuda", dtype=torch.float32) for k, v in np.load(path).items()}


def _encode(acts: torch.Tensor, sae: dict[str, torch.Tensor]) -> torch.Tensor:
    """JumpReLU encode, as Gemma Scope defines it."""
    pre = acts.to(torch.float32) @ sae["W_enc"] + sae["b_enc"]
    return pre * (pre > sae["threshold"])


def _fvu(acts: torch.Tensor, sae: dict[str, torch.Tensor]) -> tuple[float, float]:
    """``(fraction of variance unexplained, mean L0)`` for the SAE reading ``acts``."""
    x = acts.to(torch.float32)
    features = _encode(x, sae)
    recon = features @ sae["W_dec"] + sae["b_dec"]
    fvu = ((x - recon).pow(2).sum() / (x - x.mean(0)).pow(2).sum()).item()
    return fvu, (features > 0).sum(-1).float().mean().item()


@pytest.fixture(scope="module")
def both_points(model: EagerModel) -> dict[str, torch.Tensor]:
    tokens = model.tok.to_tokens(DASHBOARD_TEXT)
    cache = run_with_cache(model, tokens, [("mlp_out", LAYER), ("mlp_out_post", LAYER)])
    return {name: cache[name, LAYER][0] for name in ("mlp_out", "mlp_out_post")}


def test_gemma_2_really_has_the_post_sublayer_norm_this_turns_on(model: EagerModel):
    # If this were False the two points would alias and the rest of the file would pass vacuously.
    assert model.arch.quirks.sandwich_norms is True


def test_the_adapter_maps_the_saes_hook_to_the_contribution(model: EagerModel):
    assert tlens_hook_to_point(HOOK) == Address("mlp_out_post", LAYER)


def test_the_two_candidates_are_genuinely_different_tensors(both_points: dict[str, torch.Tensor]):
    raw, post = both_points["mlp_out"], both_points["mlp_out_post"]
    cosine = torch.nn.functional.cosine_similarity(raw.flatten(), post.flatten(), dim=0).item()
    assert cosine < 0.9, f"expected the norm to matter, got cosine {cosine}"


def test_the_sae_reconstructs_the_contribution_and_not_the_raw_output(
    both_points: dict[str, torch.Tensor], sae: dict[str, torch.Tensor]
):
    """The decisive measurement: an SAE reconstructs its own training distribution and nothing else.

    Measured 0.26 vs 9.8 — the raw output is not merely worse, it is worse than predicting the mean
    (FVU > 1), which no SAE is on the tensor it was trained on. The bounds are loose because what
    matters is the order of magnitude between them, not either number.
    """
    post_fvu, post_l0 = _fvu(both_points["mlp_out_post"], sae)
    raw_fvu, raw_l0 = _fvu(both_points["mlp_out"], sae)

    assert post_fvu < 0.5, f"mlp_out_post should reconstruct well, got FVU {post_fvu}"
    assert raw_fvu > 1.0, f"mlp_out should reconstruct worse than the mean, got FVU {raw_fvu}"

    # L0 is the independent witness: the release is named `average_l0_85`, and only one of the two
    # tensors puts the SAE anywhere near its own declared sparsity.
    assert post_l0 == pytest.approx(85, abs=25), f"expected ~85 firing features, got {post_l0}"
    assert raw_l0 < 30, f"mlp_out should leave the SAE far off its declared L0, got {raw_l0}"


def test_the_dashboards_top_activation_reproduces(
    model: EagerModel, both_points: dict[str, torch.Tensor], sae: dict[str, torch.Tensor]
):
    """The user-visible symptom: this text is feature 0's top activation on the published dashboard.

    Pinned as a value rather than just "nonzero" because the failure being guarded against returned
    a perfectly well-formed all-zero column.
    """
    strs = model.tok.to_str_tokens(DASHBOARD_TEXT)

    post = _encode(both_points["mlp_out_post"], sae)[:, FEATURE]
    assert post.max().item() == pytest.approx(23.5, abs=1.0)
    assert strs[int(post.argmax())] == "production"

    raw = _encode(both_points["mlp_out"], sae)[:, FEATURE]
    assert int((raw > 0).sum()) == 0, "the raw output fired somewhere; the negative control is stale"
