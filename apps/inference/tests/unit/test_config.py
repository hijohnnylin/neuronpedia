"""Tests for Config and the SAELens directory helpers it is built on.

The SAE config Config exposes is the SAELens directory narrowed down twice: once to the
requested sets, then again by the include/exclude patterns. Most of what is worth testing
is that second step, so these tests stub out the generated config and vary only the
patterns applied to it.
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from neuronpedia_inference.config import (
    EXTRA_SAES_PATH,
    Config,
    apply_bos_token,
    config_to_json,
    directory_for_model,
    extra_saes_directory_rows,
    get_sae_lens_ids_from_neuronpedia_id,
    get_saelens_neuronpedia_directory_df,
    required_bos_token,
    resolve_saelens_model_id,
)

RES_JB_SAES = [f"{i}-res-jb" for i in range(13)]


def sae_set(name: str, saes: list[str]) -> dict[str, Any]:
    """One entry shaped the way _generate_sae_config returns them, pre-filtering."""
    return {
        "model": "gpt2-small",
        "local": False,
        "set": name,
        "type": "saelens-1",
        "saes": saes,
    }


def build_config(generated: list[dict[str, Any]], **overrides: Any) -> Config:
    """Build a Config whose generated SAE config is fixed rather than looked up.

    The real _generate_sae_config reads the SAELens directory over the network. Stubbing
    it keeps these tests offline and deterministic, and leaves the filtering (which is
    what they actually exercise) running for real.
    """
    settings: dict[str, Any] = {
        "model_id": "gpt2-small",
        "sae_sets": ["res-jb"],
        "model_dtype": "float16",
        "sae_dtype": "float32",
        "secret": "test_secret",
        "token_limit": 100,
        "device": "cpu",
    }
    settings.update(overrides)
    with patch.object(Config, "_generate_sae_config", return_value=generated):
        return Config(**settings)


@pytest.fixture
def config() -> Config:
    return build_config(
        [sae_set("res-jb", RES_JB_SAES)],
        valid_completion_types=["DEFAULT", "STEERED"],
    )


class TestConfigInitialization:
    def test_settings_are_kept_as_given(self, config: Config):
        assert config.model_id == "gpt2-small"
        assert config.model_dtype == "float16"
        assert config.sae_dtype == "float32"
        assert config.secret == "test_secret"
        assert config.token_limit == 100
        assert config.device == "cpu"
        assert config.valid_completion_types == ["DEFAULT", "STEERED"]

    def test_port_falls_back_to_its_default(self, config: Config):
        assert config.port == 5000

    def test_sae_config_passes_through_untouched_without_patterns(self, config: Config):
        assert config.sae_config == [sae_set("res-jb", RES_JB_SAES)]


class TestSaeFiltering:
    @pytest.mark.parametrize(
        ("include", "exclude", "expected"),
        [
            pytest.param(None, None, RES_JB_SAES, id="no-patterns-keeps-everything"),
            pytest.param(
                [r"^[0-5]-res-jb"],
                None,
                [f"{i}-res-jb" for i in range(6)],
                id="include-keeps-only-what-matches",
            ),
            pytest.param(
                None,
                [r"^[0-5]-res-jb"],
                [f"{i}-res-jb" for i in range(6, 13)],
                id="exclude-drops-what-matches",
            ),
            pytest.param(
                [r"^[0-9]-res-jb"],
                [r"^[5-9]-res-jb"],
                [f"{i}-res-jb" for i in range(5)],
                id="exclude-overrides-include",
            ),
        ],
    )
    def test_patterns_select_saes(self, include: list[str] | None, exclude: list[str] | None, expected: list[str]):
        config = build_config(
            [sae_set("res-jb", RES_JB_SAES)],
            include_sae=include,
            exclude_sae=exclude,
        )
        assert [entry["saes"] for entry in config.sae_config] == [expected]

    def test_single_digit_patterns_do_not_catch_double_digit_ids(self):
        # `^[0-9]-` cannot match "10-res-jb": the character after the digit is "0",
        # not the hyphen. Worth pinning, since layers 10-12 silently going missing is
        # the kind of thing an over-eager pattern would cause.
        config = build_config([sae_set("res-jb", RES_JB_SAES)], include_sae=[r"^[0-9]-res-jb"])
        assert config.sae_config[0]["saes"] == [f"{i}-res-jb" for i in range(10)]

    def test_patterns_apply_across_every_set(self):
        config = build_config(
            [
                sae_set("res-jb", RES_JB_SAES),
                sae_set("res_fs768-jb", ["8-res_fs768-jb"]),
            ],
            sae_sets=["res-jb", "res_fs768-jb"],
            # "^8-res" spans both sets; the exclude then removes one id from the first.
            include_sae=[r"^[0-3]-res-jb", r"^8-res"],
            exclude_sae=[r"^2-res-jb"],
        )
        assert [entry["saes"] for entry in config.sae_config] == [
            ["0-res-jb", "1-res-jb", "3-res-jb", "8-res-jb"],
            ["8-res_fs768-jb"],
        ]

    def test_sets_keep_their_order_and_metadata(self):
        generated = [
            sae_set("res-jb", RES_JB_SAES),
            sae_set("res_fs768-jb", ["8-res_fs768-jb"]),
            sae_set("res_fs1536-jb", ["8-res_fs1536-jb"]),
        ]
        config = build_config(generated, sae_sets=["res-jb", "res_fs768-jb", "res_fs1536-jb"])

        assert config.sae_config == generated
        assert [entry["set"] for entry in config.sae_config] == [
            "res-jb",
            "res_fs768-jb",
            "res_fs1536-jb",
        ]


class TestSaeLensDirectory:
    def test_directory_has_the_columns_config_reads(self):
        directory_df = get_saelens_neuronpedia_directory_df()

        assert directory_df is not None
        assert directory_df.shape[0] > 0
        assert {
            "release",
            "model",
            "neuronpedia_id",
            "sae_lens_id",
            "neuronpedia_set",
        } <= set(directory_df.columns)

    def test_config_to_json_returns_each_requested_set(self):
        selected_sets = ["res-jb", "res_fs768-jb", "res_fs1536-jb"]

        json_output = config_to_json(
            get_saelens_neuronpedia_directory_df(),
            selected_sets_neuronpedia=selected_sets,
            selected_model="gpt2-small",
        )

        assert isinstance(json_output, list)
        assert {entry["set"] for entry in json_output} >= set(selected_sets)


def write_registry(tmp_path: Path, entries: dict[str, Any]) -> Path:
    path = tmp_path / "extra_saes.yaml"
    path.write_text(yaml.safe_dump(entries), encoding="utf-8")
    return path


def registry_entry(model: str, np_model: str, set_name: str, layers: list[int]) -> dict[str, Any]:
    return {
        "model": model,
        "saes": [
            {"path": f"blocks.{layer}.hook_resid_post", "neuronpedia": f"{np_model}/{layer}-{set_name}"}
            for layer in layers
        ],
    }


class TestExtraSaesRegistry:
    @pytest.mark.parametrize(
        ("hf_model", "np_model", "expected_key"),
        [
            # SAELens keys gpt2 and gemma-2-2b by short id, and Qwen3.5 by HF id.
            pytest.param("openai-community/gpt2", "gpt2-small", "gpt2-small", id="short-id"),
            pytest.param("Qwen/Qwen3.5-2B-Base", "qwen3.5-2b-pt", "Qwen/Qwen3.5-2B-Base", id="hf-id"),
            # An -it pod resolves to the base key; the row must follow, or the pod's
            # existing base SAEs would drop out of its resolution.
            pytest.param("google/gemma-2-2b-it", "gemma-2-2b-it", "gemma-2-2b", id="instruct-to-base"),
            # No SAELens entries at all: the pod keeps its configured HF id.
            pytest.param("Qwen/Qwen3.6-27B", "qwen3.6-27b", "Qwen/Qwen3.6-27B", id="unknown-model"),
        ],
    )
    def test_rows_take_the_key_the_pod_resolves_to(
        self, tmp_path: Path, hf_model: str, np_model: str, expected_key: str
    ):
        saelens_df = get_saelens_neuronpedia_directory_df()
        repo = f"neuronpedia/icalens-{np_model}-x"
        path = write_registry(tmp_path, {repo: registry_entry(hf_model, np_model, "icalens-x", [0, 3])})

        rows = extra_saes_directory_rows(saelens_df, path)

        assert [row["model"] for row in rows] == [expected_key, expected_key]
        assert resolve_saelens_model_id(hf_model, saelens_df) == expected_key
        merged = pd.concat([saelens_df, pd.DataFrame(rows)], ignore_index=True)
        merged["neuronpedia_set"] = merged["neuronpedia_id"].apply(
            lambda x: "-".join(x.split("/")[-1].split("-")[1:]) if x is not None else None
        )
        assert resolve_saelens_model_id(hf_model, merged) == expected_key
        assert get_sae_lens_ids_from_neuronpedia_id(expected_key, "3-icalens-x", merged) == (
            repo,
            "blocks.3.hook_resid_post",
        )

    def test_release_must_be_a_repo_id(self, tmp_path: Path):
        path = write_registry(tmp_path, {"icalens": registry_entry("openai-community/gpt2", "gpt2-small", "x", [0])})
        with pytest.raises(ValueError, match="repo id"):
            extra_saes_directory_rows(get_saelens_neuronpedia_directory_df(), path)

    def test_release_must_not_shadow_saelens(self, tmp_path: Path):
        path = write_registry(
            tmp_path, {"gpt2-small-res-jb": registry_entry("openai-community/gpt2", "gpt2-small", "x", [0])}
        )
        with pytest.raises(ValueError, match="also in SAELens"):
            extra_saes_directory_rows(get_saelens_neuronpedia_directory_df(), path)

    def test_committed_registry_resolves(self):
        directory_df = get_saelens_neuronpedia_directory_df()
        registry = yaml.safe_load(EXTRA_SAES_PATH.read_text(encoding="utf-8")) or {}
        for release, entry in registry.items():
            pod_df = directory_for_model(directory_df, entry["model"])
            model = resolve_saelens_model_id(entry["model"], pod_df)
            for sae in entry["saes"]:
                np_source = sae["neuronpedia"].split("/", 1)[1]
                assert get_sae_lens_ids_from_neuronpedia_id(model, np_source, pod_df) == (release, sae["path"])

    def test_base_and_instruct_rows_can_share_a_set_name(self, tmp_path: Path):
        # Both resolve to the gemma-2-2b key; each pod must see only its own model's rows.
        merged = directory_with_registry(
            tmp_path,
            {
                "neuronpedia/icalens-base": registry_entry("google/gemma-2-2b", "gemma-2-2b", "icalens-x", [0, 3]),
                "neuronpedia/icalens-it": registry_entry("google/gemma-2-2b-it", "gemma-2-2b-it", "icalens-x", [0, 3]),
            },
        )
        saelens_rows = int(merged["hf_model"].isna().sum())
        for pod_model, repo in [
            ("google/gemma-2-2b", "neuronpedia/icalens-base"),
            ("google/gemma-2-2b-it", "neuronpedia/icalens-it"),
        ]:
            pod_df = directory_for_model(merged, pod_model)
            assert int(pod_df["hf_model"].isna().sum()) == saelens_rows
            key = resolve_saelens_model_id(pod_model, pod_df)
            assert key == "gemma-2-2b"
            assert get_sae_lens_ids_from_neuronpedia_id(key, "3-icalens-x", pod_df) == (
                repo,
                "blocks.3.hook_resid_post",
            )
            [ica_set] = config_to_json(pod_df, selected_sets_neuronpedia=["icalens-x"], selected_model=key)
            assert ica_set["saes"] == ["0-icalens-x", "3-icalens-x"]

    def test_converted_folder_loads_through_sae_lens(self, tmp_path: Path):
        # What SaeLensSAE.load does for a registry row, minus the download: the release
        # is unknown to SAELens, so it is the repo id and the sae id is the folder.
        from sae_lens import SAE, StandardSAE, StandardSAEConfig
        from sae_lens.loading.pretrained_saes_directory import get_repo_id_and_folder_name
        from sae_lens.saes.sae import SAEMetadata

        assert get_repo_id_and_folder_name("neuronpedia/icalens-gpt2-small-x", "blocks.6.hook_resid_post") == (
            "neuronpedia/icalens-gpt2-small-x",
            "blocks.6.hook_resid_post",
        )
        cfg = StandardSAEConfig(
            d_in=4,
            d_sae=4,
            apply_b_dec_to_input=True,
            metadata=SAEMetadata(hook_name="blocks.6.hook_resid_post", neuronpedia_id="gpt2-small/6-icalens-x"),
        )
        StandardSAE(cfg).save_model(tmp_path / "blocks.6.hook_resid_post")
        loaded = SAE.load_from_disk(tmp_path / "blocks.6.hook_resid_post", device="cpu")
        assert loaded.cfg.metadata.neuronpedia_id == "gpt2-small/6-icalens-x"
        assert loaded.cfg.metadata.hook_name == "blocks.6.hook_resid_post"


QWEN_BASE = "Qwen/Qwen3.5-2B-Base"
ENDOFTEXT = "<|endoftext|>"


def directory_with_registry(tmp_path: Path, entries: dict[str, Any]) -> pd.DataFrame:
    """The SAELens directory plus the rows of a registry written to tmp_path."""
    saelens_df = get_saelens_neuronpedia_directory_df()
    rows = extra_saes_directory_rows(saelens_df, write_registry(tmp_path, entries))
    merged = pd.concat([saelens_df, pd.DataFrame(rows)], ignore_index=True)
    merged["neuronpedia_set"] = merged["neuronpedia_id"].apply(
        lambda x: "-".join(x.split("/")[-1].split("-")[1:]) if x is not None else None
    )
    return merged


class TestRequiredBosToken:
    @pytest.fixture
    def qwen_df(self, tmp_path: Path) -> pd.DataFrame:
        entry = {**registry_entry(QWEN_BASE, "qwen3.5-2b-pt", "icalens-x", [0, 4]), "bos_token": ENDOFTEXT}
        return directory_with_registry(tmp_path, {"neuronpedia/icalens-qwen-x": entry})

    def test_set_with_bos_token_returns_it(self, qwen_df: pd.DataFrame):
        assert required_bos_token(qwen_df, QWEN_BASE, ["icalens-x"]) == ENDOFTEXT

    def test_sets_without_bos_token_return_none(self, qwen_df: pd.DataFrame):
        saelens_sets = qwen_df.loc[(qwen_df["model"] == QWEN_BASE) & qwen_df["bos_token"].isna(), "neuronpedia_set"]
        assert not saelens_sets.empty
        assert required_bos_token(qwen_df, QWEN_BASE, [saelens_sets.iloc[0]]) is None

    def test_sets_that_disagree_cannot_share_a_server(self, qwen_df: pd.DataFrame):
        saelens_sets = qwen_df.loc[(qwen_df["model"] == QWEN_BASE) & qwen_df["bos_token"].isna(), "neuronpedia_set"]
        with pytest.raises(ValueError, match="different BOS tokens"):
            required_bos_token(qwen_df, QWEN_BASE, ["icalens-x", saelens_sets.iloc[0]])

    def test_committed_registry(self):
        directory_df = get_saelens_neuronpedia_directory_df()
        assert required_bos_token(directory_df, QWEN_BASE, ["icalens-res"]) == ENDOFTEXT
        assert required_bos_token(directory_df, "Qwen/Qwen3.5-9B-Base", ["icalens-res"]) == ENDOFTEXT
        assert required_bos_token(directory_df, "gpt2-small", ["icalens-res", "res-jb"]) is None


class FakeTokenizer:
    def __init__(self, bos_token: str | None):
        self.bos_token = bos_token

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [0]


class TestApplyBosToken:
    def test_none_leaves_the_tokenizer_alone(self):
        tokenizer = FakeTokenizer("<bos>")
        apply_bos_token(tokenizer, None)
        assert tokenizer.bos_token == "<bos>"

    def test_a_different_existing_bos_is_an_error(self):
        with pytest.raises(ValueError, match="already has BOS"):
            apply_bos_token(FakeTokenizer("<bos>"), ENDOFTEXT)

    def test_qwen_prompts_start_with_the_new_bos(self):
        transformers = pytest.importorskip("transformers")
        from interp_engine import Tokenize

        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(QWEN_BASE, local_files_only=True)
        except OSError:
            pytest.skip(f"{QWEN_BASE} tokenizer is not in the local Hugging Face cache")
        assert tokenizer.bos_token is None

        apply_bos_token(tokenizer, ENDOFTEXT)

        endoftext_id = tokenizer.convert_tokens_to_ids(ENDOFTEXT)
        assert tokenizer.bos_token_id == endoftext_id
        # The activation endpoints prepend bos_token to the text, then tokenize without BOS.
        text_ids = tokenizer(ENDOFTEXT + "Hello world", add_special_tokens=False)["input_ids"]
        engine_ids = Tokenize(tokenizer, device="cpu").to_tokens("Hello world", prepend_bos=True)[0].tolist()
        assert text_ids == engine_ids
        assert engine_ids[0] == endoftext_id
        assert endoftext_id not in engine_ids[1:]
