"""Persona asset loading: everything specific to a fit comes out of the asset directory.

The layer, the component labels, and whether the system prompt must be blanked are all
properties of how a given PCA was fitted, so they live in that asset's ``persona.yaml``
and never in the code. These tests pin that, the symlink path a reused fit takes, and the
"no asset" cases that must leave persona monitoring cleanly unavailable rather than
half-loaded.
"""

import os
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from neuronpedia_inference.inference_utils.persona import persona_data as pd_module
from neuronpedia_inference.inference_utils.persona.persona_data import (
    MANIFEST_FILENAME,
    PersonaData,
)

LLAMA_ASSET = "meta-llama/Llama-3.3-70B-Instruct"
LLAMA_AWQ = "casperhansen/llama-3.3-70b-instruct-awq"
N_CONTRAST_LAYERS = 80
D_MODEL = 4
TITLE = "- Role-playing ↔\ufe0f + Assistant-like"


def _write_asset(
    root,
    model_id: str,
    layers: list[int],
    *,
    contrast: bool = True,
    manifest: str | None = None,
    blank_system_prompt: bool = False,
    pc_titles: list[str] | None = None,
) -> None:
    """Lay out an asset the way `initialize` expects to find it on disk."""
    asset_dir = root / model_id
    (asset_dir / "pca").mkdir(parents=True, exist_ok=True)
    if contrast:
        (asset_dir / "contrast_vectors.pt").write_bytes(b"stub")
    for layer in layers:
        (asset_dir / "pca" / f"roles_layer{layer}-min.pt").write_bytes(b"stub")

    if manifest is None:
        manifest = yaml.safe_dump(
            {
                "blank_system_prompt": blank_system_prompt,
                "pc_titles": pc_titles if pc_titles is not None else [TITLE],
                "contrast_vectors": "contrast_vectors.pt",
                "pca": [{"layer": layer, "file": f"pca/roles_layer{layer}-min.pt"} for layer in layers],
            },
            allow_unicode=True,
        )
    (asset_dir / MANIFEST_FILENAME).write_text(manifest, encoding="utf-8")


@pytest.fixture
def stub_torch_load(monkeypatch):
    """Stand in for the pickled assets, which need sklearn to unpickle for real."""

    def fake_load(path, **_kwargs):
        if str(path).endswith("contrast_vectors.pt"):
            return torch.ones((N_CONTRAST_LAYERS, D_MODEL))
        return {"pca": SimpleNamespace(components_=np.zeros((3, D_MODEL)))}

    monkeypatch.setattr(pd_module.torch, "load", fake_load)


@pytest.fixture
def asset_root(tmp_path, monkeypatch):
    monkeypatch.setattr(PersonaData, "_get_data_path", lambda _self: str(tmp_path))
    return tmp_path


class TestInitialize:
    def test_layer_comes_from_the_manifest_not_a_constant(self, asset_root, stub_torch_load):
        # A layer that is deliberately not 40: nothing may assume the old default.
        _write_asset(asset_root, "m", [31])
        data = PersonaData()
        data.initialize("m")
        assert data._initialized is True
        assert data.layers == [31]
        assert data.primary_layer == 31
        assert data.get_pca_data(31) is not None

    def test_primary_layer_is_the_lowest_loaded(self, asset_root, stub_torch_load):
        _write_asset(asset_root, "m", [20, 60])
        data = PersonaData()
        data.initialize("m")
        assert data.layers == [20, 60]
        assert data.primary_layer == 20

    def test_explicit_layers_override_the_manifest(self, asset_root, stub_torch_load):
        _write_asset(asset_root, "m", [20, 60])
        data = PersonaData()
        data.initialize("m", layers=[60])
        assert data.layers == [60]

    def test_explicit_layer_absent_from_the_manifest_is_skipped(self, asset_root, stub_torch_load):
        # The manifest is the only source of a layer's filename, so a layer it does not
        # describe cannot be loaded even if a caller names it.
        _write_asset(asset_root, "m", [40])
        data = PersonaData()
        data.initialize("m", layers=[41])
        assert data._initialized is False

    def test_pca_filename_is_read_from_the_manifest(self, asset_root, stub_torch_load):
        # Not derived from the layer number: an asset may name its files anything.
        asset_dir = asset_root / "m"
        (asset_dir / "pca").mkdir(parents=True)
        (asset_dir / "contrast_vectors.pt").write_bytes(b"stub")
        (asset_dir / "pca" / "custom-name.pt").write_bytes(b"stub")
        _write_asset(
            asset_root,
            "m",
            [],
            manifest=textwrap.dedent(f"""\
                pc_titles: ["{TITLE}"]
                pca:
                  - layer: 7
                    file: pca/custom-name.pt
                """),
        )
        data = PersonaData()
        data.initialize("m")
        assert data._initialized is True
        assert data.layers == [7]

    def test_unavailable_without_a_manifest(self, asset_root, stub_torch_load):
        asset_dir = asset_root / "m" / "pca"
        asset_dir.mkdir(parents=True)
        (asset_root / "m" / "contrast_vectors.pt").write_bytes(b"stub")
        (asset_dir / "roles_layer40-min.pt").write_bytes(b"stub")
        data = PersonaData()
        data.initialize("m")
        assert data._initialized is False
        assert data.primary_layer is None

    def test_unavailable_on_a_malformed_manifest(self, asset_root, stub_torch_load):
        # A broken manifest must not crash startup — persona monitoring goes unavailable
        # and the rest of the server runs.
        _write_asset(asset_root, "m", [40], manifest="pca: [{layer: nope}]\n")
        data = PersonaData()
        data.initialize("m")
        assert data._initialized is False

    def test_unavailable_without_contrast_vectors(self, asset_root, stub_torch_load):
        _write_asset(asset_root, "m", [40], contrast=False)
        data = PersonaData()
        data.initialize("m")
        assert data._initialized is False
        assert data.primary_layer is None

    def test_unavailable_when_the_manifest_declares_no_layers(self, asset_root, stub_torch_load):
        _write_asset(asset_root, "m", [])
        data = PersonaData()
        data.initialize("m")
        assert data._initialized is False
        assert data.primary_layer is None

    def test_failed_reload_does_not_keep_previous_state(self, asset_root, stub_torch_load):
        # A stale "ready" flag from a previous model would project onto the wrong PCs.
        _write_asset(asset_root, "m", [40], blank_system_prompt=True)
        data = PersonaData()
        data.initialize("m")
        assert data._initialized is True

        data.initialize("absent/model")
        assert data._initialized is False
        assert data.primary_layer is None
        assert data.fit.blank_system_prompt is False
        assert data.pc_titles == []

    def test_unavailable_when_no_layer_loads(self, asset_root, stub_torch_load):
        # A PCA layer deeper than the contrast vectors can't be used, so the asset
        # yields nothing and must not report itself as ready.
        _write_asset(asset_root, "m", [N_CONTRAST_LAYERS + 5])
        data = PersonaData()
        data.initialize("m")
        assert data._initialized is False
        assert data.primary_layer is None


class TestSymlinkedAsset:
    """A checkpoint that reuses another's fit is a symlink, not a table entry."""

    def test_symlinked_id_loads_the_target_asset(self, asset_root, stub_torch_load):
        _write_asset(asset_root, LLAMA_ASSET, [40], blank_system_prompt=True)
        link = asset_root / LLAMA_AWQ
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(asset_root / LLAMA_ASSET)

        data = PersonaData()
        data.initialize(LLAMA_AWQ)
        assert data._initialized is True
        assert data.primary_layer == 40
        assert data.get_pca_data(40) is not None

    def test_symlink_carries_the_targets_fit_conditions(self, asset_root, stub_torch_load):
        # The fit conditions travel with the asset, so both ids see the same rules.
        _write_asset(asset_root, LLAMA_ASSET, [40], blank_system_prompt=True)
        link = asset_root / LLAMA_AWQ
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(asset_root / LLAMA_ASSET)

        data = PersonaData()
        data.initialize(LLAMA_AWQ)
        assert data.fit.blank_system_prompt is True
        assert data.pc_titles == [TITLE]

    def test_model_id_stays_the_served_id(self, asset_root, stub_torch_load):
        _write_asset(asset_root, "m", [40])
        data = PersonaData()
        data.initialize("m")
        assert data.model_id == "m"


class TestFitConditions:
    def test_blank_system_prompt_is_read_from_the_manifest(self, asset_root, stub_torch_load):
        _write_asset(asset_root, "a", [40], blank_system_prompt=True)
        data = PersonaData()
        data.initialize("a")
        assert data.fit.blank_system_prompt is True

    def test_assets_default_to_keeping_the_system_prompt(self, asset_root, stub_torch_load):
        _write_asset(asset_root, "google/gemma-3-27b-it", [40])
        data = PersonaData()
        data.initialize("google/gemma-3-27b-it")
        assert data.fit.blank_system_prompt is False

    def test_no_asset_keeps_the_system_prompt(self):
        # The gate must be safe before/without initialization: never blank a caller's
        # system prompt just because assistant-axis was requested.
        assert PersonaData().fit.blank_system_prompt is False

    def test_pc_titles_come_from_the_manifest(self, asset_root, stub_torch_load):
        _write_asset(asset_root, "a", [40], pc_titles=["first", "second"])
        data = PersonaData()
        data.initialize("a")
        assert data.pc_titles == ["first", "second"]

    def test_pc_titles_empty_before_initialization(self):
        assert PersonaData().pc_titles == []


class TestShippedAsset:
    """The real asset in the repo, checked without unpickling it."""

    @pytest.fixture
    def shipped(self):
        root = PersonaData()._get_data_path()
        fit = pd_module._load_manifest(os.path.join(root, LLAMA_ASSET))
        assert fit is not None, f"{LLAMA_ASSET} ships no {MANIFEST_FILENAME}"
        return fit

    def test_declares_layer_40(self, shipped):
        assert shipped.layers == [40]
        assert shipped.file_for_layer(40) == "pca/roles_layer40-min.pt"

    def test_requires_a_blank_system_prompt(self, shipped):
        # Fitted with an empty system turn; a caller's system prompt must be dropped.
        assert shipped.blank_system_prompt is True

    def test_surfaces_one_pc(self, shipped):
        assert len(shipped.pc_titles) == 1

    def test_awq_id_resolves_to_the_same_asset(self):
        root = PersonaData()._get_data_path()
        awq = os.path.join(root, LLAMA_AWQ)
        assert os.path.islink(awq), f"{LLAMA_AWQ} should be a symlink"
        assert os.path.realpath(awq) == os.path.realpath(os.path.join(root, LLAMA_ASSET))

    def test_referenced_files_exist(self, shipped):
        root = os.path.join(PersonaData()._get_data_path(), LLAMA_ASSET)
        assert os.path.exists(os.path.join(root, shipped.contrast_vectors))
        for entry in shipped.pca:
            assert os.path.exists(os.path.join(root, entry.file))
