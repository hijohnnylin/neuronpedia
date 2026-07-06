# Converts a SAEDashboard NeuronpediaRunner to a Neuronpedia export so it can be imported into Neuronpedia by anyone.
# You don't need to run this if you generated dashboards using the generate-dashboards script in this directory.

# Profiling: poetry run python -m cProfile -o profile_output.prof neuronpedia_utils/convert-saedashboard-to-neuronpedia-export.py                 --saedashboard-output-dir="/Users/johnnylin/Documents/Projects/neuronpedia/utils/neuronpedia-utils/neuronpedia_utils/ignore-scripts/gdm_output/gemma-3-12b/12-gemmascope-2-res-16k"                 --creator-name='Google DeepMind'                 --release-id=gemma-scope-2                 --release-title='Gemma Scope 2'                 --url=http://huggingface.co/google/gemma-scope-2                 --model-name=gemma-3-12b                 --neuronpedia-source-set-id=gemmascope-2-res-16k                 --neuronpedia-source-set-description="Residual Stream - 16k"                 --hf-weights-repo-id=google/gemma-scope-2-12b-pt                 --hf-weights-path=resid_post/layer_12_width_16k_l0_medium                 --hook-point=hook_resid_post                 --layer-num=12                 --prompts-huggingface-dataset-path=monology/pile-uncopyrighted                 --n-prompts-total=392802                 --n-tokens-in-prompt=256                 --zero-out-bos-token && poetry run python -c "import pstats; p = pstats.Stats('profile_output.prof'); p.sort_stats('cumulative').print_stats()" > profile_results.txt

import gzip
import os
import time
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, List

import dotenv
import orjson
import typer
from neuronpedia_utils.db_models.activation import Activation
from neuronpedia_utils.db_models.feature import Feature
from neuronpedia_utils.db_models.model import Model
from neuronpedia_utils.db_models.source import Source
from neuronpedia_utils.db_models.source_release import SourceRelease
from neuronpedia_utils.db_models.source_set import SourceSet

try:
    import polars as pl
except ImportError:
    pl = None

dotenv.load_dotenv(".env.default")
dotenv.load_dotenv()

MODULE_DIR = Path(__file__).resolve().parent
dotenv.load_dotenv(MODULE_DIR / ".env.default")
dotenv.load_dotenv(MODULE_DIR / ".env")

DEFAULT_EXPORT_ROOT_ENV_VARS = ("NEURONPEDIA_EXPORT_ROOT",)


def _resolve_export_root(export_root: str | os.PathLike[str] | None = None) -> Path:
    if export_root is not None and str(export_root).strip():
        return Path(export_root).expanduser()
    for env_var in DEFAULT_EXPORT_ROOT_ENV_VARS:
        env_value = os.getenv(env_var)
        if env_value:
            return Path(env_value).expanduser()
    return Path("./exports")


def _configure_output_paths(
    model_name: str,
    export_root: str | os.PathLike[str] | None = None,
) -> tuple[str, str]:
    resolved_export_root = _resolve_export_root(export_root)
    global OUTPUT_DIR
    OUTPUT_DIR = str(resolved_export_root)
    global OUTPUT_PATH_BASE
    OUTPUT_PATH_BASE = str(resolved_export_root / model_name)
    Path(OUTPUT_PATH_BASE).mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR, OUTPUT_PATH_BASE


OUTPUT_DIR = str(_resolve_export_root())

creator_id = os.getenv("DEFAULT_CREATOR_ID")
if creator_id is None or creator_id == "":
    creator_id = "clkht01d40000jv08hvalcvly"

DEFAULT_CREATOR_ID = creator_id
CONVERSION_DEBUG_CALLBACK: Callable[[str, dict[str, Any]], None] | None = None
OUTPUT_PATH_BASE = OUTPUT_DIR
VECTOR_STEER_HOOK_NAME = ""
ZERO_OUT_BOS_TOKEN = False


class FastPseudoCuid:
    """Counter-based pseudo-CUID - ~100x faster than cuid2"""

    def __init__(self, length: int = 25):
        self.length = length
        self._counter = 0
        # Get time ONCE at startup, not per call
        self._prefix = f"c{os.getpid():04x}{int(time.time()):08x}"

    def generate(self) -> str:
        self._counter += 1
        result = f"{self._prefix}{self._counter:08x}"
        return result[: self.length]


CUID_GENERATOR = FastPseudoCuid(length=25)

created_at = datetime.now()


class HOOK_POINT_TYPE_CHOICES(str, Enum):
    hook_resid_pre = "hook_resid_pre"
    hook_resid_mid = "hook_resid_mid"
    hook_resid_post = "hook_resid_post"
    hook_mlp_in = "hook_mlp_in"
    hook_mlp_out = "hook_mlp_out"
    hook_attn_in = "hook_attn_in"
    hook_attn_out = "hook_attn_out"
    hook_z = "hook_z"


app = typer.Typer()


def _emit_conversion_debug(stage: str, **metadata: Any) -> None:
    if CONVERSION_DEBUG_CALLBACK is None:
        return
    try:
        CONVERSION_DEBUG_CALLBACK(stage, metadata)
    except Exception as exc:
        print(f"[conversion_cuda_debug] callback_error stage={stage} error={exc}")


def make_option(default: Any, *option_names: str, help_text: str, **kwargs) -> Any:
    """Create a Typer Option with the same text for both help and prompt."""
    return typer.Option(
        default,
        *option_names,
        help=help_text,
        prompt="\n" + help_text + "\n",
        **kwargs,
    )


# TODO: Use tokenizer to get BOS tokens
BOS_TOKENS = ["<bos>", "<|endoftext|>"]


@app.command()
def main(
    ctx: typer.Context,
    saedashboard_output_dir: str = make_option(
        ...,
        "--saedashboard-output-dir",
        help_text="[Input] SAEDashboard Output Directory: The directory containing the SAEDashboard output.",
    ),
    export_root: str | None = typer.Option(
        None,
        "--export-root",
        help="[Output] Export Root: Base directory for generated export bundles. Defaults to $NEURONPEDIA_EXPORT_ROOT, then ./exports.",
    ),
    creator_name: str = make_option(
        ...,
        "--creator-name",
        help_text="[Author] Name: Name of the creator (e.g., your organization/team name).",
    ),
    release_id: str = make_option(
        ...,
        "--release-id",
        help_text="[Release] Release ID: Enter the release id (eg gemma-scope). Must be alphanumeric, but can also include dashes. Your release will be available at https://[neuronpedia_domain]/[release_id]",
    ),
    release_title: str = make_option(
        ...,
        "--release-title",
        help_text="[Release] Release Title: Human-readable description of the release - probably a shortened version of your paper. Eg Exploring Gemma 2 with Gemma Scope.",
    ),
    url: str = make_option(
        ...,
        "--url",
        help_text="[Info] URL: URL associated with your paper/release. Include https://...",
    ),
    model_name: str = make_option(
        ...,
        "--model-name",
        help_text="[Model] Model Name: The TransformerLens name of the model to be used for the dashboard. Eg 'gemma-2-2b-it'. See https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html",
    ),
    model_layers: int = make_option(
        ...,
        "--model-layers",
        help_text="[Model] Model Layers: Total number of transformer layers in the backing model. Use the full model depth, not just the exported source-set coverage.",
    ),
    neuronpedia_source_set_id: str = make_option(
        ...,
        "--neuronpedia-source-set-id",
        help_text="[Source] Neuronpedia Source Set ID: All dashboards on Neuronpedia belong to a 'Source Set', which is an identifier. Please specify what the 'source set id' is for your data - it should be short and descriptive of the author, hook, and optionally, width (number of features/vectors per layer).\nFor example, an example source set ID is gemmascope-res-16k. The URL is then https://[neuronpedia_domain]/gemma-2-2b/gemmascope-res-16k.\nDo not include the layer number - that will automatically be prepended later.",
    ),
    neuronpedia_source_set_description: str = make_option(
        ...,
        "--neuronpedia-source-set-description",
        help_text="[Source] Neuronpedia Source Set Description: When this source set is displayed on Neuronpedia, this is the description that will be shown. Usually, it is a short human-readable hook and width for this source. Eg Residual Stream - 16k",
    ),
    hf_weights_repo_id: str = make_option(
        ...,
        "--hf-weights-repo-id",
        help_text="[Source] HuggingFace Repository ID: Huggingface repository ID for your weights/data in the form [user]/[repo_id], NOT INCLUDING the folder. Eg 'google/gemma-scope-2b-pt-res'",
    ),
    hf_weights_path: str = make_option(
        ...,
        "--hf-weights-path",
        help_text="[Source] HuggingFace Weights Path: Path to the weights on HuggingFace in the form 'layer_0/width_16k/average_l0_105/weights.pt'. Do not include the repo name.",
    ),
    hook_point: HOOK_POINT_TYPE_CHOICES = make_option(
        ...,
        "--hook-point",
        help_text=f"[Source] Hook Point: The TransformerLens hook point to use for the dashboard. Must be one of: {', '.join([f'{choice}' for choice in HOOK_POINT_TYPE_CHOICES])}.",
    ),
    layer_num: int = make_option(
        ...,
        "--layer-num",
        help_text="[Source] Layer Number: The layer number that this source/SAE is trained on. Eg 20.",
    ),
    prompts_huggingface_dataset_path: str = make_option(
        ...,
        "--prompts-huggingface-dataset-path",
        help_text="[Dashboard Gen Parameters] HuggingFace Dataset Path: The path to the HuggingFace dataset to use for prompts. Eg 'monology/pile-uncopyrighted'.",
    ),
    n_prompts_total: int = make_option(
        24576,
        "--n-prompts-total",
        help_text="[Dashboard Gen Parameters] Total Prompts: The number of prompts to use to generate activations for the dashboard. More will give you a wider breadth of activations, but requires more time and memory. 16,384 or 24,576 are common values.",
    ),
    n_tokens_in_prompt: int = make_option(
        128,
        "--n-tokens-in-prompt",
        help_text="[Dashboard Gen Parameters] Context Tokens per Prompt: The number of tokens per prompt to use for each activation in the dashboard. More requires more time and memory. We typically use 128.",
    ),
    # gemma 2 was not trained with BOS tokens, so we need to zero them out
    zero_out_bos_token: bool = typer.Option(
        False,
        "--zero-out-bos-token",
        help="[Dashboard Gen Parameters] Zero Out BOS Token: Whether to zero out the BOS token in the activations.",
    ),
    emit_arrow: bool = typer.Option(
        False,
        "--emit-arrow",
        help="[Export] Emit Arrow IPC tables and derive the current JSONL bundle from Polars while preserving the existing bundle contract.",
    ),
):
    print("Running with arguments:\n")
    for param, value in ctx.params.items():
        print(f"{param}: {value}")

    print("--------------------------------")
    print("Equivalent command is:")

    command = "python convert-saedashboard-to-neuronpedia-export.py"
    for name, value in ctx.params.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    command += f" --{name.replace('_', '-')}"
            else:
                # Quote strings if they contain spaces
                if isinstance(value, str) and (" " in value or "'" in value):
                    value = f"'{value}'"
                command += f" --{name.replace('_', '-')}={value}"
    command = command.replace(" --", " \\\n    --")
    print(command)

    try:
        print("Converting to Neuronpedia format for final output...")

        global VECTOR_STEER_HOOK_NAME
        VECTOR_STEER_HOOK_NAME = hook_point.value

        global ZERO_OUT_BOS_TOKEN
        ZERO_OUT_BOS_TOKEN = zero_out_bos_token

        # get the hf folder id from the hf weights path
        hf_folder_id = "/".join(hf_weights_path.split("/")[:-1])

        final_output_dir = ""
        _configure_output_paths(model_name, export_root)

        intermediate_output_dir_subdir = saedashboard_output_dir

        for file in sorted(
            f
            for f in os.listdir(intermediate_output_dir_subdir)
            if f.startswith("batch-")
        ):
            print("reading activations from batch file", file)
            batch_file_path = os.path.join(intermediate_output_dir_subdir, file)
            _emit_conversion_debug(
                "before_batch_json_read",
                batch_file=file,
                batch_file_bytes=os.path.getsize(batch_file_path),
            )
            batch_data = read_json_file(batch_file_path)
            _emit_conversion_debug(
                "after_batch_json_read",
                batch_file=file,
                feature_count=len(batch_data["features"]),
            )

            source_suffix = batch_data["sae_id_suffix"]

            source_id = (
                str(layer_num)
                + "-"
                + neuronpedia_source_set_id
                + ("__" + source_suffix if source_suffix else "")
            )

            final_output_dir = os.path.join(OUTPUT_PATH_BASE, source_id)
            if not os.path.exists(final_output_dir):
                os.makedirs(final_output_dir)
            if not emit_arrow:
                _remove_arrow_sidecars(final_output_dir)

            # make the release jsonl
            release_file_path = os.path.join(final_output_dir, "release.jsonl")
            release = SourceRelease(
                name=release_id,
                description=release_title,
                descriptionShort=release_title,
                urls=[url] if url else [],
                creatorNameShort=creator_name,
                creatorName=creator_name,
                creatorId=DEFAULT_CREATOR_ID,
                defaultSourceSetName=neuronpedia_source_set_id,
                createdAt=created_at,
            )
            _write_flat_table_rows(
                [_normalize_export_row(release.__dict__)],
                release_file_path,
                emit_arrow=emit_arrow,
            )

            # make the model jsonl
            model_file_path = os.path.join(final_output_dir, "model.jsonl")
            model = Model(
                id=model_name,
                instruct=model_name.endswith("-it"),
                displayNameShort=model_name,
                displayName=model_name,
                creatorId=DEFAULT_CREATOR_ID,
                defaultSourceSetName=neuronpedia_source_set_id,
                defaultGraphSourceSetName=neuronpedia_source_set_id,
                layers=model_layers,
                createdAt=created_at,
                updatedAt=created_at,
            )
            _write_flat_table_rows(
                [_normalize_export_row(model.__dict__)],
                model_file_path,
                emit_arrow=emit_arrow,
            )

            # make the sourceset jsonl
            sourceset_file_path = os.path.join(final_output_dir, "sourceset.jsonl")
            sourceset = SourceSet(
                modelId=model_name,
                name=neuronpedia_source_set_id,
                creatorId=DEFAULT_CREATOR_ID,
                createdAt=created_at,
                creatorName=creator_name,
                releaseName=release_id,
                description=neuronpedia_source_set_description,
                visibility="PUBLIC",
            )
            _write_flat_table_rows(
                [_normalize_export_row(sourceset.__dict__)],
                sourceset_file_path,
                emit_arrow=emit_arrow,
            )

            # make the source jsonl
            source_file_path = os.path.join(final_output_dir, "source.jsonl")
            source = Source(
                modelId=model_name,
                setName=neuronpedia_source_set_id,
                visibility="PUBLIC",
                dataset=prompts_huggingface_dataset_path,
                id=source_id,
                num_prompts=n_prompts_total,
                num_tokens_in_prompt=n_tokens_in_prompt,
                hfRepoId=hf_weights_repo_id,
                hfFolderId=hf_folder_id,
                creatorId=DEFAULT_CREATOR_ID,
            )
            _write_flat_table_rows(
                [_normalize_export_row(source.__dict__)],
                source_file_path,
                emit_arrow=emit_arrow,
            )

            process_data(
                batch_data,
                neuronpedia_source_set_id,
                file.replace(".json", ""),
                model_name,
                emit_arrow=emit_arrow,
            )
            _emit_conversion_debug(
                "after_process_data",
                batch_file=file,
                final_output_dir=final_output_dir,
            )

        print(
            "\n\n ==================== Dashboards generated successfully. ==================== \n"
        )
        print("The dashboards are available in the source output directory:")
        print(final_output_dir)

    except BaseException as e:
        print(f"\nError: {e}")
        print("\nTo run this job again, use this command (fixing any errors first):\n")

        print(command)
        raise typer.Abort()


def read_json_file(file_path):
    with open(file_path, "rb") as f:
        return orjson.loads(f.read())


def datetime_handler(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _normalize_export_row(row: dict[str, Any]) -> dict[str, Any]:
    return orjson.loads(orjson.dumps(row, default=datetime_handler))


def _arrow_path_for_jsonl(jsonl_path: str) -> str:
    if jsonl_path.endswith(".jsonl"):
        return jsonl_path[: -len(".jsonl")] + ".arrow"
    return jsonl_path + ".arrow"


def _gzip_jsonl_file(jsonl_path: str) -> str:
    gzip_path = jsonl_path + ".gz"
    with open(jsonl_path, "rb") as f_in:
        with open(gzip_path, "wb") as f_out:
            f_out.write(gzip.compress(f_in.read(), compresslevel=5))
    os.remove(jsonl_path)
    return gzip_path


def _write_legacy_jsonl_rows(rows: list[dict[str, Any]], jsonl_path: str) -> str:
    with open(jsonl_path, "wb") as f:
        for row in rows:
            f.write(orjson.dumps(row) + b"\n")
    return _gzip_jsonl_file(jsonl_path)


def _rows_to_polars_dataframe(rows: list[dict[str, Any]]) -> Any:
    if pl is None:
        raise RuntimeError("Arrow export helpers require polars to be installed")
    return pl.DataFrame(rows, strict=False)


def _write_arrow_rows(
    rows: list[dict[str, Any]],
    arrow_path: str,
    *,
    dataframe: Any | None = None,
) -> str:
    table = dataframe if dataframe is not None else _rows_to_polars_dataframe(rows)
    table.write_ipc(arrow_path)
    return arrow_path


def _write_polars_jsonl_rows(
    rows: list[dict[str, Any]],
    jsonl_path: str,
    *,
    dataframe: Any | None = None,
) -> str:
    table = dataframe if dataframe is not None else _rows_to_polars_dataframe(rows)
    table.write_ndjson(jsonl_path)
    return _gzip_jsonl_file(jsonl_path)


def benchmark_flat_table_write_modes(
    rows: list[dict[str, Any]],
    output_dir: str | os.PathLike[str],
    table_stem: str,
) -> dict[str, Any]:
    """Measure legacy JSONL and Arrow-oriented flat-row write modes from the same rows."""

    benchmark_dir = Path(output_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    legacy_jsonl_path = benchmark_dir / f"{table_stem}.legacy.jsonl"
    legacy_start_time = time.perf_counter()
    legacy_gzip_path = Path(_write_legacy_jsonl_rows(rows, str(legacy_jsonl_path)))
    results: dict[str, Any] = {
        "row_count": len(rows),
        "modes": {
            "legacy_jsonl_gzip": {
                "path": legacy_gzip_path,
                "elapsed_seconds": time.perf_counter() - legacy_start_time,
                "bytes": legacy_gzip_path.stat().st_size,
            }
        },
    }

    if not rows:
        return results

    arrow_only_path = benchmark_dir / f"{table_stem}.arrow"
    arrow_only_start_time = time.perf_counter()
    _write_arrow_rows(rows, str(arrow_only_path))
    results["modes"]["arrow_ipc_only"] = {
        "path": arrow_only_path,
        "elapsed_seconds": time.perf_counter() - arrow_only_start_time,
        "bytes": arrow_only_path.stat().st_size,
    }

    bundle_jsonl_path = benchmark_dir / f"{table_stem}.bundle.jsonl"
    bundle_arrow_path = Path(_arrow_path_for_jsonl(str(bundle_jsonl_path)))
    bundle_start_time = time.perf_counter()
    dataframe = _rows_to_polars_dataframe(rows)
    _write_arrow_rows(rows, str(bundle_arrow_path), dataframe=dataframe)
    bundle_gzip_path = Path(
        _write_polars_jsonl_rows(rows, str(bundle_jsonl_path), dataframe=dataframe)
    )
    results["modes"]["arrow_bundle"] = {
        "arrow_path": bundle_arrow_path,
        "jsonl_gzip_path": bundle_gzip_path,
        "elapsed_seconds": time.perf_counter() - bundle_start_time,
        "arrow_bytes": bundle_arrow_path.stat().st_size,
        "jsonl_gzip_bytes": bundle_gzip_path.stat().st_size,
    }
    return results


def _write_flat_table_rows(
    rows: list[dict[str, Any]],
    jsonl_path: str,
    *,
    emit_arrow: bool,
) -> tuple[str, str | None]:
    if not emit_arrow or not rows:
        arrow_path = _arrow_path_for_jsonl(jsonl_path)
        if os.path.exists(arrow_path):
            os.remove(arrow_path)
        return _write_legacy_jsonl_rows(rows, jsonl_path), None

    arrow_path = _arrow_path_for_jsonl(jsonl_path)
    dataframe = _rows_to_polars_dataframe(rows)
    _write_arrow_rows(rows, arrow_path, dataframe=dataframe)
    return _write_polars_jsonl_rows(rows, jsonl_path, dataframe=dataframe), arrow_path


def _remove_arrow_sidecars(root_dir: str) -> None:
    if not os.path.exists(root_dir):
        return
    for current_root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".arrow"):
                os.remove(os.path.join(current_root, filename))


def process_data(
    batch_data,
    source_set_name,
    batch_file_name,
    model_name_override: str | None = None,
    *,
    emit_arrow: bool = False,
):
    source_set = source_set_name
    model_id = model_name_override if model_name_override else batch_data["model_id"]
    layer_num = batch_data["layer"]
    source_suffix = batch_data["sae_id_suffix"]

    source_id = (
        str(layer_num)
        + "-"
        + source_set
        + ("__" + source_suffix if source_suffix else "")
    )

    source_dir = os.path.join(OUTPUT_PATH_BASE, source_id)

    activations: List[Activation] = []
    features: List[Feature] = []
    activation_count = 0
    token_value_count = 0
    vector_value_count = 0

    _emit_conversion_debug(
        "process_data_start",
        batch_file_name=batch_file_name,
        feature_count=len(batch_data["features"]),
        source_id=source_id,
    )

    for feature_data in batch_data["features"]:
        vector_value_count += len(feature_data.get("vector", []))
        new_feature = Feature(
            modelId=model_id,
            layer=source_id,
            index=feature_data["feature_index"],
            creatorId=DEFAULT_CREATOR_ID,
            createdAt=created_at,
            hasVector="vector" in feature_data and len(feature_data["vector"]) > 0,
            vector=feature_data["vector"] if "vector" in feature_data else [],
            vectorLabel=None,
            hookName=(
                f"blocks.{layer_num}.{VECTOR_STEER_HOOK_NAME}"
                if "vector" in feature_data
                else None
            ),
            topkCosSimIndices=[],
            topkCosSimValues=[],
            neuron_alignment_indices=feature_data["neuron_alignment_indices"],
            neuron_alignment_values=feature_data["neuron_alignment_values"],
            neuron_alignment_l1=feature_data["neuron_alignment_l1"],
            correlated_neurons_indices=feature_data["correlated_neurons_indices"],
            correlated_neurons_pearson=feature_data["correlated_neurons_pearson"],
            correlated_neurons_l1=feature_data["correlated_neurons_l1"],
            correlated_features_indices=feature_data["correlated_features_indices"],
            correlated_features_pearson=feature_data["correlated_features_pearson"],
            correlated_features_l1=feature_data["correlated_features_l1"],
            neg_str=feature_data["neg_str"],
            neg_values=feature_data["neg_values"],
            pos_str=feature_data["pos_str"],
            pos_values=feature_data["pos_values"],
            frac_nonzero=feature_data["frac_nonzero"],
            freq_hist_data_bar_heights=feature_data["freq_hist_data_bar_heights"],
            freq_hist_data_bar_values=feature_data["freq_hist_data_bar_values"],
            logits_hist_data_bar_heights=feature_data["logits_hist_data_bar_heights"],
            logits_hist_data_bar_values=feature_data["logits_hist_data_bar_values"],
            decoder_weights_dist=feature_data["decoder_weights_dist"],
        )
        max_act_approx = 0
        for activation_data in feature_data["activations"]:
            activation_count += 1
            token_value_count += len(activation_data["values"])
            if ZERO_OUT_BOS_TOKEN:
                for i, token in enumerate(activation_data["tokens"]):
                    if token in BOS_TOKENS and activation_data["values"][i] != 0:
                        print(
                            f"Zeroing out BOS token {token} at index {i}, source_id: {source_id}, feature_index: {feature_data['feature_index']}, file: {batch_file_name}"
                        )
                        activation_data["values"][i] = 0

            max_value = max(activation_data["values"])
            max_value_token_index = activation_data["values"].index(max_value)
            if max_value > max_act_approx:
                max_act_approx = max_value
            new_activation = Activation(
                id=CUID_GENERATOR.generate(),
                tokens=activation_data["tokens"],
                modelId=model_id,
                layer=source_id,
                index=feature_data["feature_index"],
                maxValue=max_value,
                maxValueTokenIndex=max_value_token_index,
                minValue=min(activation_data["values"]),
                values=activation_data["values"],
                dfaValues=(
                    activation_data["dfa_values"]
                    if "dfa_values" in activation_data
                    else []
                ),
                dfaTargetIndex=(
                    activation_data["dfa_targetIndex"]
                    if "dfa_targetIndex" in activation_data
                    else None
                ),
                dfaMaxValue=(
                    activation_data["dfa_maxValue"]
                    if "dfa_maxValue" in activation_data
                    else None
                ),
                creatorId=DEFAULT_CREATOR_ID,
                createdAt=created_at,
                lossValues=(
                    activation_data["loss_values"]
                    if "loss_values" in activation_data
                    else []
                ),
                logitContributions=(
                    activation_data["logit_contributions"]
                    if "logit_contributions" in activation_data
                    else None
                ),
                binContains=activation_data["bin_contains"],
                binMax=activation_data["bin_max"],
                binMin=activation_data["bin_min"],
                qualifyingTokenIndex=activation_data["qualifying_token_index"],
                dataIndex=None,
                dataSource=None,
            )

            activations.append(new_activation)
        new_feature.maxActApprox = max_act_approx
        if "vector" in feature_data:
            new_feature.vectorDefaultSteerStrength = new_feature.maxActApprox
        features.append(new_feature)

    _emit_conversion_debug(
        "after_object_materialization",
        batch_file_name=batch_file_name,
        feature_count=len(features),
        activation_count=activation_count,
        token_value_count=token_value_count,
        vector_value_count=vector_value_count,
    )

    # make features directory
    features_dir = os.path.join(source_dir, "features")
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    # write the features to a jsonl
    features_file_path = os.path.join(features_dir, f"{batch_file_name}.jsonl")
    _emit_conversion_debug(
        "before_features_jsonl_write",
        batch_file_name=batch_file_name,
        path=features_file_path,
    )
    feature_rows = [
        _normalize_export_row(feature.__dict__) if emit_arrow else feature.__dict__
        for feature in features
    ]
    features_gzip_path, features_arrow_path = _write_flat_table_rows(
        feature_rows,
        features_file_path,
        emit_arrow=emit_arrow,
    )
    if features_arrow_path is not None:
        _emit_conversion_debug(
            "after_features_arrow",
            batch_file_name=batch_file_name,
            arrow_path=features_arrow_path,
            arrow_bytes=os.path.getsize(features_arrow_path),
        )
    _emit_conversion_debug(
        "after_features_gzip",
        batch_file_name=batch_file_name,
        gzip_path=features_gzip_path,
        gzip_bytes=os.path.getsize(features_gzip_path),
    )

    activations_dir = os.path.join(source_dir, "activations")
    if not os.path.exists(activations_dir):
        os.makedirs(activations_dir)

    activations_file_path = os.path.join(activations_dir, f"{batch_file_name}.jsonl")
    _emit_conversion_debug(
        "before_activations_jsonl_write",
        batch_file_name=batch_file_name,
        path=activations_file_path,
    )
    activation_rows = [
        _normalize_export_row(activation.__dict__) if emit_arrow else activation.__dict__
        for activation in activations
    ]
    activations_gzip_path, activations_arrow_path = _write_flat_table_rows(
        activation_rows,
        activations_file_path,
        emit_arrow=emit_arrow,
    )
    if activations_arrow_path is not None:
        _emit_conversion_debug(
            "after_activations_arrow",
            batch_file_name=batch_file_name,
            arrow_path=activations_arrow_path,
            arrow_bytes=os.path.getsize(activations_arrow_path),
        )
    _emit_conversion_debug(
        "after_activations_gzip",
        batch_file_name=batch_file_name,
        gzip_path=activations_gzip_path,
        gzip_bytes=os.path.getsize(activations_gzip_path),
    )


if __name__ == "__main__":
    app()
