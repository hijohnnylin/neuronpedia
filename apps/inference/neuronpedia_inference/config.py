import json
import logging
import re
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

logger = logging.getLogger(__name__)


def _repo_root_np_model_to_hf() -> Path | None:
    """Locate ``np_model_to_hf.json`` at the repo root, if present."""
    # apps/inference/neuronpedia_inference/config.py -> repo root is four parents up.
    candidate = Path(__file__).resolve().parents[3] / "np_model_to_hf.json"
    return candidate if candidate.exists() else None


def load_np_model_to_hf() -> dict[str, str]:
    """Neuronpedia short id → Hugging Face repo id mapping from the repo root."""
    path = _repo_root_np_model_to_hf()
    if path is None:
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read %s: %s", path, exc)
        return {}


def model_id_aliases(*model_ids: str | None) -> set[str]:
    """Expand configured model ids with their Neuronpedia ↔ HF aliases.

    Clients may send either the HF repo id (``google/gemma-2-2b``) or the
    Neuronpedia short id (``gemma-2-2b``); both must match the same pod.
    """
    aliases: set[str] = {m for m in model_ids if m}
    mapping = load_np_model_to_hf()
    if not mapping:
        return aliases
    for mid in list(aliases):
        if mid in mapping:
            aliases.add(mapping[mid])
        for np_id, hf_id in mapping.items():
            if mid == hf_id:
                aliases.add(np_id)
    return aliases


def resolve_saelens_model_id(model_id: str, directory_df: pd.DataFrame) -> str:
    """Map a configured HF (or short) model id to the key SAELens uses.

    Newer SAELens releases key some models by HF id (``meta-llama/...``) and
    others by Neuronpedia short id (``gemma-2-2b``, ``gpt2-small``). Prefer an
    exact directory hit, then the ``np_model_to_hf.json`` alias, then the
    instruct→base fallback used when an ``-it`` checkpoint reuses base SAEs.
    """
    candidates: list[str] = [model_id]
    mapping = load_np_model_to_hf()
    if mapping:
        if model_id in mapping and mapping[model_id] not in candidates:
            candidates.append(mapping[model_id])
        for np_id, hf_id in mapping.items():
            if model_id == hf_id and np_id not in candidates:
                candidates.append(np_id)

    for candidate in candidates:
        if (directory_df["model"] == candidate).any():
            return candidate

    for candidate in candidates:
        if candidate.endswith("-it"):
            base = candidate[: -len("-it")]
            if (directory_df["model"] == base).any():
                return base

    return model_id


@dataclass(eq=False, repr=False)
class Config:
    """Server-wide inference settings.

    Built once during startup from CLI args and environment (see ``server.py``)
    and read from everywhere else via ``Config.get_instance()``.

    ``eq``/``repr`` are left off deliberately: instances are compared by identity
    and must stay hashable, and a generated repr would print ``secret``.
    """

    # Hugging Face repo id. SAELens short ids ("gpt2-small") are derived from this via
    # resolve_saelens_model_id, not the other way round.
    model_id: str = "openai-community/gpt2"
    custom_hf_model_id: str | None = None
    sae_sets: list[str] = field(default_factory=lambda: ["res-jb"])
    model_dtype: str = "auto"
    sae_dtype: str = "float32"
    secret: str | None = None
    port: int = 5000
    token_limit: int = 100
    # Separate, higher cap that applies ONLY to the lens endpoints (logit/jacobian
    # lens). Independent of `token_limit` (which governs the completion/steer/tokenize
    # endpoints) so JLens can allow longer conversations without changing the other
    # endpoints' limits. The vLLM context is sized from whichever of the two is larger
    # (`server._engine_context_len`), so raising this alone is enough.
    lens_token_limit: int = 1024
    valid_completion_types: list[str] = field(default_factory=lambda: ["default", "steered"])
    num_layers: int | None = None
    device: str | None = None
    # Normalized to `model_id` when not supplied. See __post_init__.
    override_model_id: str | None = None
    max_loaded_saes: int = 100
    # Active interpretability backend: "vllm" (engine-owned vLLM) or "eager"
    # (EagerModel core). Resolved by interp_engine.select_backend at startup.
    backend: str = "eager"
    num_gpus: int = 1
    # Serve completions only, at the cost of everything hook-dependent (capture, steering, lens).
    # Set from GENERATION_ONLY; only meaningful on the vLLM backend, where it is what lets the
    # engine keep CUDA graphs on. Read by endpoints/capabilities.py and by the capture guard in
    # engine_adapter.py, so one flag decides both what is advertised and what is refused.
    generation_only: bool = False
    # SAE paging (sae_cache.py). Unset/None keeps every SAE GPU-resident for the life of
    # the process, which is what every pod did before paging existed; a GiB number or
    # "auto" instead keeps the master copies in host RAM and caches this many bytes of
    # them on the GPU. Resolved to bytes at startup by startup_memory.
    sae_gpu_budget_gib: str | None = None
    # Cap on page-locked host memory for those master copies. None = measure it.
    sae_pinned_host_gib: float | None = None
    # Human-readable summary of what STATIC_POINTS (+ STATIC_POINTS_EXTRA) asked to declare, for
    # the startup banner. A string rather than the addresses because this banner prints before the
    # SAEs load, so `sae` has no address list yet; the resolved set is in the ready banner.
    static_points_summary: str | None = None

    # Accepted as constructor arguments but stored under the derived names below.
    include_sae: InitVar[list[str] | None] = None
    exclude_sae: InitVar[list[str] | None] = None
    model_from_pretrained_kwargs: InitVar[str] = "{}"

    include_sae_patterns: list[str] | None = field(init=False, default=None)
    exclude_sae_patterns: list[str] | None = field(init=False, default=None)
    model_kwargs: dict[str, Any] = field(init=False, default_factory=dict)
    sae_config: list[dict[str, str | list[str]]] = field(init=False, default_factory=list)
    # Activation endpoints start at token_limit and may be lowered at the end of
    # startup from the measured VRAM budget + widest SAE (see
    # startup_memory.compute_activation_token_limit). Never raised above token_limit.
    activation_token_limit: int = field(init=False, default=0)
    # Memory-derived per-request sequence budget (prompt + generation); set at
    # startup by startup_memory.compute_serving_limits. None until configured.
    max_tokens: int | None = field(init=False, default=None)

    _instance: ClassVar["Config | None"] = None

    # Reported in the startup banner, in this order.
    _SUMMARY_FIELDS: ClassVar[tuple[str, ...]] = (
        "model_id",
        "custom_hf_model_id",
        "override_model_id",
        "model_dtype",
        "sae_dtype",
        "port",
        "token_limit",
        "lens_token_limit",
        "activation_token_limit",
        "device",
        "sae_sets",
        "max_loaded_saes",
        "sae_gpu_budget_gib",
        "include_sae_patterns",
        "exclude_sae_patterns",
        "backend",
        "num_gpus",
        "static_points_summary",
    )

    def __post_init__(
        self,
        include_sae: list[str] | None,
        exclude_sae: list[str] | None,
        model_from_pretrained_kwargs: str,
    ) -> None:
        self.override_model_id = self.override_model_id or self.model_id
        self.include_sae_patterns = include_sae
        self.exclude_sae_patterns = exclude_sae
        self.model_kwargs = json.loads(model_from_pretrained_kwargs)
        self.activation_token_limit = self.token_limit
        self.sae_config = self._filter_sae_config(self._generate_sae_config())
        logger.info(
            "Initialized Config with:\n%s",
            "".join(f"  {name}: {getattr(self, name)}\n" for name in self._SUMMARY_FIELDS),
        )

    @classmethod
    def get_instance(cls) -> "Config":
        """Get the global Config instance, creating it if it doesn't exist"""
        if cls._instance is None:
            cls._instance = Config()
        return cls._instance

    def set_num_layers(self, num_layers: int) -> None:
        self.num_layers = num_layers

    def set_max_tokens(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens

    def set_activation_token_limit(self, activation_token_limit: int) -> None:
        """Lower (never raise) the activation-endpoint prompt cap after budget measurement."""
        self.activation_token_limit = max(1, min(int(activation_token_limit), int(self.token_limit)))

    def clamp_completion_tokens(self, prompt_len: int, requested: int) -> int:
        """Clamp a requested generation length to the memory-safe sequence budget.

        Ensures ``prompt_len + returned <= max_tokens`` (when a budget is set), so a
        long prompt cannot also request a long generation past what fits in memory.
        """
        if self.max_tokens is None:
            return requested
        return max(0, min(requested, self.max_tokens - prompt_len))

    def get_valid_model_ids(self):
        """Model ids this pod is known to answer for.

        SAELens rows are keyed by short Neuronpedia ids (``gpt2-small``), while
        ``--model_id`` and request bodies may use the HF repo id
        (``openai-community/gpt2``). Expand both sides via ``np_model_to_hf.json``.

        Advisory only -- see :meth:`check_requested_model` for why requests are not
        rejected against this set.
        """
        # sae_config values are typed as str | list[str]; "model" is always a plain str.
        from_sae = {model_id for sae_set in self.sae_config if isinstance((model_id := sae_set["model"]), str)}
        return model_id_aliases(
            *from_sae,
            self.model_id,
            self.override_model_id,
            self.custom_hf_model_id,
        )

    def check_requested_model(self, model_id: str | None) -> None:
        """Log, but do not reject, a request naming a model this pod did not load.

        A pod serves exactly one model, so the ``model`` field never selects anything: it
        is at most a client-side assertion. Enforcing it meant every caller had to spell
        the id the way ``get_valid_model_ids`` happened to expand it, and that set is
        partly derived from the SAE config -- so a pod loaded for something other than
        SAEs (raw activation capture, lens) could refuse requests for the very model it
        had in memory.
        """
        if not model_id:
            return
        valid = self.get_valid_model_ids()
        if model_id not in valid:
            logger.warning(
                "Request names model %r, which is not among this pod's known ids %s. "
                "Serving it anyway against the loaded model.",
                model_id,
                sorted(valid),
            )

    def _generate_sae_config(self):
        # No sets requested means no SAEs at all, which is a supported way to run: the
        # capture/lens/steer endpoints need only the model. Returning early also skips
        # downloading and parsing the SAELens directory, so such a pod has no dependency
        # on it during startup.
        if not self.sae_sets:
            logger.info("No SAE sets configured; starting without SAEs")
            return []
        directory_df = get_saelens_neuronpedia_directory_df()
        configured = self.custom_hf_model_id if self.custom_hf_model_id else self.model_id
        selected_model = resolve_saelens_model_id(configured, directory_df)
        if selected_model != configured:
            logger.info("Resolved SAELens model id %r -> %r", configured, selected_model)
        config_json = config_to_json(
            directory_df,
            selected_sets_neuronpedia=self.sae_sets,
            selected_model=selected_model,
        )
        return config_json  # noqa: RET504

    def _filter_sae_config(self, sae_config: list[dict[str, str | list[str]]]) -> list[dict[str, str | list[str]]]:
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
            if self._match_patterns(sae_id, self.include_sae_patterns, self.exclude_sae_patterns)
        ]

    def _match_patterns(
        self,
        sae_id: str,
        include_patterns: list[str] | None,
        exclude_patterns: list[str] | None,
    ) -> bool:
        if include_patterns and not any(re.search(pattern, sae_id) for pattern in include_patterns):
            return False
        if exclude_patterns:
            return not any(re.search(pattern, sae_id) for pattern in exclude_patterns)
        return True


# this is an example of a Claude refactor gone wrong. way too confusing.
def get_saelens_neuronpedia_directory_df():
    df = pd.DataFrame.from_records({k: v.__dict__ for k, v in get_pretrained_saes_directory().items()}).T
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
        lambda x: "-".join(x.split("/")[-1].split("-")[1:]) if x is not None else None
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
        directory_df = directory_df.loc[directory_df["release"].isin(selected_sets_sae_lens)]
    elif selected_sets_neuronpedia:
        directory_df = directory_df.loc[directory_df["neuronpedia_set"].isin(selected_sets_neuronpedia)]
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
                "saes": [sae.split("/")[-1] for sae in set_data["neuronpedia_id"].tolist()],
            }
            config_json.append(set_entry)
    return config_json


def get_sae_lens_ids_from_neuronpedia_id(model_id: str, neuronpedia_id: str, df_exploded: pd.DataFrame):
    # find where neuronpedia_id ends in /neuronpedia_id and df_exploded["model"] = model_id
    tmp_df = df_exploded[
        (df_exploded["model"] == model_id) & (df_exploded["neuronpedia_id"].str.endswith(f"/{neuronpedia_id}"))
    ]
    assert tmp_df.shape[0] == 1, f"Found {tmp_df.shape[0]} entries when searching for {model_id}/{neuronpedia_id}"
    sae_lens_release = tmp_df.release.values[0]
    sae_lens_id = tmp_df.sae_lens_id.values[0]
    return sae_lens_release, sae_lens_id
