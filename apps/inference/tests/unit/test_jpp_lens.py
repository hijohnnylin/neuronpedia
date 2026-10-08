"""The J++ Lens: a second fitted ``J_bar`` lens, held and served apart from the Jacobian lens.

It has the same form as the Jacobian lens (one ``J_bar`` per source layer), so it reuses
:class:`LoadedJacobianLens`. What must stay apart is everything around it: its own store, its
own file name, its own opt-in, and its own named set in the engine, so a ``JPP_LENS`` column
never reads the Jacobian lens's matrices or the reverse.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch

from neuronpedia_inference.endpoints.lens import lens_loader, prompt
from neuronpedia_inference.endpoints.lens.lens_loader import (
    JACOBIAN_LENS_KIND,
    JPP_LENS_KIND,
    JacobianLensStore,
    JppLensStore,
    LoadedJacobianLens,
    install_named_lens,
    load_jacobian_lens_at_startup,
)
from neuronpedia_inference.schemas import LensSteerToken, LensType

D_MODEL = 4


@pytest.fixture(autouse=True)
def _reset_stores():
    yield
    for store in (JacobianLensStore, JppLensStore):
        store._instance = None
        store._status = "not_loaded"
        store._error = None
        store._np_model_id = None


def _write_raw_jpp(path, layers=(0, 1)) -> dict[int, torch.Tensor]:
    """A J++ Lens file as the upstream fitter writes it."""
    jacobians = {layer: torch.randn(D_MODEL, D_MODEL) for layer in layers}
    torch.save(
        {
            "parameters": {"jacobians": jacobians},
            "source_layers": list(layers),
            "config": {"d_model": D_MODEL, "num_prompts_trained_on": 96},
        },
        path,
    )
    return jacobians


def _lens(scale: float, layers=(0, 1)) -> LoadedJacobianLens:
    return LoadedJacobianLens(
        jacobians={layer: torch.eye(D_MODEL) * scale for layer in layers},
        source_layers=list(layers),
        n_prompts=1,
        d_model=D_MODEL,
        dtype=torch.float32,
    )


def test_the_upstream_jpp_file_loads_like_a_jacobian_lens_file(tmp_path):
    path = tmp_path / "Qwen3.5-2B_jpp_lens.pt"
    jacobians = _write_raw_jpp(path)

    lens = LoadedJacobianLens.load(str(path), dtype=torch.float32)

    assert lens.source_layers == [0, 1]
    assert lens.d_model == D_MODEL
    assert lens.n_prompts == 96
    assert torch.equal(lens.jacobians[1], jacobians[1])


def test_the_jpp_lens_is_off_unless_asked_for(tmp_path):
    _write_raw_jpp(tmp_path / "m_jpp_lens.pt")
    args = SimpleNamespace(jpp_lens=False, jpp_source=str(tmp_path), neuronpedia_model_id="m")

    load_jacobian_lens_at_startup(SimpleNamespace(), args, JPP_LENS_KIND)

    assert JppLensStore.status() == "skipped"
    assert JppLensStore.get() is None


def test_the_jpp_lens_loads_into_its_own_store(tmp_path):
    # A Jacobian lens file beside it must not be picked up: the J++ kind looks for its own stem.
    torch.save(
        {"J": {0: torch.zeros(D_MODEL, D_MODEL)}, "d_model": D_MODEL, "source_layers": [0]},
        tmp_path / "m_jacobian_lens.pt",
    )
    _write_raw_jpp(tmp_path / "m_jpp_lens.pt")
    args = SimpleNamespace(jpp_lens=True, jpp_source=str(tmp_path), neuronpedia_model_id="m")

    load_jacobian_lens_at_startup(SimpleNamespace(), args, JPP_LENS_KIND)

    lens = JppLensStore.get()
    assert lens is not None and lens.n_prompts == 96
    assert JacobianLensStore.get() is None
    assert JacobianLensStore.status() == "not_loaded"


def test_the_jpp_download_uses_its_own_dataset_and_the_public_repo(tmp_path, monkeypatch):
    # The Jacobian lens of this model is fitted on another corpus. The J++ path must not use it.
    path = tmp_path / "m_jpp_lens.pt"
    _write_raw_jpp(path)
    calls = []

    def download(repo_id, np_model_id, dataset, hf_model_id, explicit_path, *, folder, stem):
        calls.append((repo_id, dataset, folder, stem))
        return str(path)

    monkeypatch.setattr(lens_loader, "_download_lens_from_hf", download)
    args = SimpleNamespace(
        jpp_lens=True,
        jpp_source=None,
        jpp_dataset="Salesforce-wikitext",
        jlens_dataset="NeelNanda-pile-10k",
        jpp_hf_repo=None,
        jpp_hf_path=None,
        neuronpedia_model_id="m",
    )

    load_jacobian_lens_at_startup(SimpleNamespace(), args, JPP_LENS_KIND)

    assert calls == [("neuronpedia/jacobian-lens", "Salesforce-wikitext", "jpp", "jpp_lens")]
    assert JppLensStore.status() == "loaded"


def test_each_j_bar_type_names_its_own_engine_set():
    jlens = prompt._lens_spec(LensType.JACOBIAN_LENS, [0])
    jpp = prompt._lens_spec(LensType.JPP_LENS, [0])
    logit = prompt._lens_spec(LensType.LOGIT_LENS, [0])

    assert (jlens.jacobian, jlens.jacobian_set) == (True, JACOBIAN_LENS_KIND.engine_set)
    assert (jpp.jacobian, jpp.jacobian_set) == (True, JPP_LENS_KIND.engine_set)
    assert not logit.jacobian


def test_a_jpp_lens_read_out_here_is_installed_by_name():
    calls = []

    class FakeModel:
        async def set_lens_jacobians(self, jacobians, *, name="default"):
            calls.append((sorted(jacobians), name))
            return 0

    JppLensStore.set_loaded(_lens(2.0), "m")
    asyncio.run(install_named_lens(FakeModel(), JPP_LENS_KIND))

    assert calls == [([0, 1], "jpp")]


def test_the_jacobian_lens_is_never_installed_by_name():
    # It rides along with each call, as it always has.
    class FakeModel:
        async def set_lens_jacobians(self, jacobians, *, name="default"):
            raise AssertionError("not expected")

    JacobianLensStore.set_loaded(_lens(1.0), "m")
    asyncio.run(install_named_lens(FakeModel(), JACOBIAN_LENS_KIND))


def test_a_steer_token_is_carried_by_the_lens_of_its_own_type(monkeypatch):
    # The J++ matrices flip the sign, the Jacobian lens's keep it, so each token's
    # direction shows which lens carried it.
    w = torch.tensor([1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(prompt, "_decoded_string_to_ids", lambda tokenizer: {})
    monkeypatch.setattr(prompt, "_resolve_steer_token_id", lambda index, token: 7)

    async def unembed(model, token_ids):
        return {7: w}

    monkeypatch.setattr(prompt, "_unembed_vectors_by_id", unembed)
    model = SimpleNamespace(tokenizer=object())
    lenses = {LensType.JACOBIAN_LENS: _lens(1.0), LensType.JPP_LENS: _lens(-1.0)}

    def deltas(lens_type):
        token = LensSteerToken(token="x", type=lens_type)
        return asyncio.run(prompt._build_steer_deltas(model, lenses, [token], [1]))[1]

    assert torch.allclose(deltas(LensType.JACOBIAN_LENS), w)
    assert torch.allclose(deltas(LensType.JPP_LENS), -w)
