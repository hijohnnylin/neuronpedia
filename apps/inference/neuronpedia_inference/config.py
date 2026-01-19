import json
import logging
import re
from typing import Any

import pandas as pd
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

logger = logging.getLogger(__name__)


class Config:
    _instance = None  # Class variable to store the singleton instance

    @classmethod
    def get_instance(cls):
        """Get the global Config instance, creating it if it doesn't exist"""
        if cls._instance is None:
            cls._instance = Config()
        return cls._instance

    def __init__(
        self,
        model_id: str = "gpt2-small",
        custom_hf_model_id: str | None = None,
        sae_sets: list[str] = ["res-jb"],
        model_dtype: str = "float32",
        sae_dtype: str = "float32",
        secret: str | None = None,
        port: int = 5000,
        token_limit: int = 100,
        valid_completion_types: list[str] = ["default", "steered"],
        num_layers: int | None = None,
        device: str | None = None,
        override_model_id: str | None = None,
        include_sae: list[str] | None = None,
        exclude_sae: list[str] | None = None,
        model_from_pretrained_kwargs: str = "{}",
        max_loaded_saes: int = 100,
        steer_special_token_ids: list[int] | None = None,
        nnsight: bool = False,
        chatspace: bool = False,
    ):
        self.model_id = model_id
        self.custom_hf_model_id = custom_hf_model_id
        self.override_model_id = override_model_id or model_id
        self.model_dtype = model_dtype
        self.sae_dtype = sae_dtype
        self.secret = secret
        self.port = port
        self.token_limit = token_limit
        self.valid_completion_types = valid_completion_types
        self.num_layers = num_layers
        self.device = device
        self.sae_sets = sae_sets
        self.include_sae_patterns = include_sae
        self.exclude_sae_patterns = exclude_sae
        self.sae_config = self._filter_sae_config(self._generate_sae_config())
        self.model_kwargs = json.loads(model_from_pretrained_kwargs)
        self.max_loaded_saes = max_loaded_saes
        self.steer_special_token_ids = steer_special_token_ids
        self.nnsight = nnsight
        self.chatspace = chatspace

        # Log configuration details after initialization
        logger.info(
            f"Initialized Config with:\n"
            f"  model_id: {self.model_id}\n"
            f"  custom_hf_model_id: {self.custom_hf_model_id}\n"
            f"  override_model_id: {self.override_model_id}\n"
            f"  model_dtype: {self.model_dtype}\n"
            f"  sae_dtype: {self.sae_dtype}\n"
            f"  port: {self.port}\n"
            f"  token_limit: {self.token_limit}\n"
            f"  device: {self.device}\n"
            f"  sae_sets: {self.sae_sets}\n"
            f"  max_loaded_saes: {self.max_loaded_saes}\n"
            f"  include_sae_patterns: {self.include_sae_patterns}\n"
            f"  exclude_sae_patterns: {self.exclude_sae_patterns}\n"
            f"  nnsight: {self.nnsight}\n"
            f"  chatspace: {self.chatspace}\n"
        )

    def set_num_layers(self, num_layers: int) -> None:
        self.num_layers = num_layers

    def set_steer_special_token_ids(
        self, steer_special_token_ids: list[int] | set[int]
    ) -> None:
        self.steer_special_token_ids = steer_special_token_ids

    def get_valid_model_ids(self):
        return set([sae_set["model"] for sae_set in self.sae_config])

    def _generate_sae_config(self):
        directory_df = get_saelens_neuronpedia_directory_df()
        config_json = config_to_json(
            directory_df,
            selected_sets_neuronpedia=self.sae_sets,
            selected_model=(
                self.custom_hf_model_id if self.custom_hf_model_id else self.model_id
            ),
        )
        return config_json  # noqa: RET504

    def _filter_sae_config(
        self, sae_config: list[dict[str, str | list[str]]]
    ) -> list[dict[str, str | list[str]]]:
        filtered_config = []
        for sae_set in sae_config:
            sae_ids = sae_set["saes"]
            if isinstance(sae_ids, str):
                sae_ids = [sae_ids]
            filtered_saes = self._filter_saes(sae_ids)
            if filtered_saes:
                sae_set = sae_set.copy()
                sae_set["saes"] = filtered_saes
                filtered_config.append(sae_set)
        return filtered_config

    def _filter_saes(self, sae_ids: list[str]) -> list[str]:
        return [
            sae_id
            for sae_id in sae_ids
            if self._match_patterns(
                sae_id, self.include_sae_patterns, self.exclude_sae_patterns
            )
        ]

    def _match_patterns(
        self,
        sae_id: str,
        include_patterns: list[str] | None,
        exclude_patterns: list[str] | None,
    ) -> bool:
        if include_patterns and not any(
            re.search(pattern, sae_id) for pattern in include_patterns
        ):
            return False
        if exclude_patterns:
            return not any(re.search(pattern, sae_id) for pattern in exclude_patterns)
        return True


# this is an example of a Claude refactor gone wrong. way too confusing.
def get_saelens_neuronpedia_directory_df():
    df = pd.DataFrame.from_records(
        {k: v.__dict__ for k, v in get_pretrained_saes_directory().items()}
    ).T
    df.drop(
        columns=[
            "repo_id",
            "saes_map",
            "expected_var_explained",
            "expected_l0",
            "config_overrides",
            "conversion_func",
        ],
        inplace=True,
    )
    df["neuronpedia_id_list"] = df["neuronpedia_id"].apply(lambda x: list(x.items()))
    df_exploded = df.explode("neuronpedia_id_list")
    df_exploded[["sae_lens_id", "neuronpedia_id"]] = pd.DataFrame(
        df_exploded["neuronpedia_id_list"].tolist(), index=df_exploded.index
    )
    df_exploded = df_exploded.drop(columns=["neuronpedia_id_list"])
    df_exploded = df_exploded.reset_index(drop=True)
    df_exploded["neuronpedia_set"] = df_exploded["neuronpedia_id"].apply(
        lambda x: ("-".join(x.split("/")[-1].split("-")[1:]) if x is not None else None)
    )
    return df_exploded


def config_to_json(
    directory_df: pd.DataFrame,
    selected_sets_sae_lens: list[str] | None = None,
    selected_sets_neuronpedia: list[str] | None = None,
    selected_model: str | None = None,
) -> list[dict[str, Any]]:
    if selected_model:
        directory_df = directory_df.loc[directory_df["model"] == selected_model]
    if selected_sets_neuronpedia and selected_sets_sae_lens:
        directory_df = directory_df.loc[
            (directory_df["neuronpedia_set"].isin(selected_sets_neuronpedia))
            | (directory_df["release"].isin(selected_sets_sae_lens))
        ]
    elif selected_sets_sae_lens:
        directory_df = directory_df.loc[
            directory_df["release"].isin(selected_sets_sae_lens)
        ]
    elif selected_sets_neuronpedia:
        directory_df = directory_df.loc[
            directory_df["neuronpedia_set"].isin(selected_sets_neuronpedia)
        ]
    grouped = directory_df.groupby("model")
    config_json = []
    for model, group in grouped:
        # Get unique sets within the group (you can also use directory_df if that's intended)
        sets_to_include = group["neuronpedia_set"].unique()
        for set_name in sets_to_include:
            set_data = group.loc[group["neuronpedia_set"] == set_name]
            set_entry = {
                "model": model,
                "set": set_name,
                "type": "saelens-1",
                "local": False,
                "saes": [
                    sae.split("/")[-1] for sae in set_data["neuronpedia_id"].tolist()
                ],
            }
            config_json.append(set_entry)
    return config_json


def get_sae_lens_ids_from_neuronpedia_id(
    model_id: str, neuronpedia_id: str, df_exploded: pd.DataFrame
):
    # find where neuronpedia_id ends in /neuronpedia_id and df_exploded["model"] = model_id
    tmp_df = df_exploded[
        (df_exploded["model"] == model_id)
        & (df_exploded["neuronpedia_id"].str.endswith(f"/{neuronpedia_id}"))
    ]
    assert (
        tmp_df.shape[0] == 1
    ), f"Found {tmp_df.shape[0]} entries when searching for {model_id}/{neuronpedia_id}"
    sae_lens_release = tmp_df.release.values[0]
    sae_lens_id = tmp_df.sae_lens_id.values[0]
    return sae_lens_release, sae_lens_id
