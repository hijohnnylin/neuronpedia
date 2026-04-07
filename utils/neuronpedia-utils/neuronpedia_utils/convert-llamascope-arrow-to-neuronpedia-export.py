"""
Converts a LlamaScope2 HuggingFace Arrow dataset to Neuronpedia export format.

The Arrow dataset is expected to have the following columns:
  - feature_index (int64)
  - frac_nonzero (double)
  - explanations (list<string>)
  - top_pos_logits_str (list<string>)
  - top_pos_logits_val (list<double>)
  - top_neg_logits_str (list<string>)
  - top_neg_logits_val (list<double>)
  - activations (list<struct<tokens, values, z_pattern_indices, z_pattern_values>>)

Usage:
  poetry run python neuronpedia_utils/convert-llamascope-arrow-to-neuronpedia-export.py \
    --arrow-dir /path/to/arrow/dataset \
    --model-name qwen3-1.7b \
    --layer-num 13 \
    --neuronpedia-source-set-id plt-8x-topk64 \
    --neuronpedia-source-set-description "PLT 8x TopK64" \
    --creator-name "LlamaScope" \
    --release-id llamascope-2 \
    --release-title "LlamaScope 2"
"""

import gzip
import math
import os
import time
from datetime import datetime
from typing import Annotated, Any, List

import dotenv
import orjson
import pyarrow.ipc as ipc
import typer
from neuronpedia_utils.db_models.activation import Activation
from neuronpedia_utils.db_models.explanation import Explanation
from neuronpedia_utils.db_models.feature import Feature
from neuronpedia_utils.db_models.model import Model
from neuronpedia_utils.db_models.source import Source
from neuronpedia_utils.db_models.source_release import SourceRelease
from neuronpedia_utils.db_models.source_set import SourceSet

dotenv.load_dotenv(".env.default")
dotenv.load_dotenv()

OUTPUT_DIR = "./exports"

# Truncate all exported float fields toward zero to at most this many decimal places.
FLOAT_DECIMAL_PLACES = 4


def truncate_float(x: float) -> float:
    """Truncate x toward zero to FLOAT_DECIMAL_PLACES decimal places."""
    if not math.isfinite(x):
        return x
    factor = 10**FLOAT_DECIMAL_PLACES
    return math.trunc(x * factor) / factor


creator_id = os.getenv("DEFAULT_CREATOR_ID")
if creator_id is None or creator_id == "":
    creator_id = "clkht01d40000jv08hvalcvly"

DEFAULT_CREATOR_ID = creator_id


class FastPseudoCuid:
    """Counter-based pseudo-CUID - ~100x faster than cuid2"""

    def __init__(self, length: int = 25):
        self.length = length
        self._counter = 0
        self._prefix = f"c{os.getpid():04x}{int(time.time()):08x}"

    def generate(self) -> str:
        self._counter += 1
        result = f"{self._prefix}{self._counter:08x}"
        return result[: self.length]


CUID_GENERATOR = FastPseudoCuid(length=25)
created_at = datetime.now()

app = typer.Typer()


def make_option(*option_names: str, help_text: str, **kwargs) -> Any:
    return typer.Option(
        *option_names,
        help=help_text,
        prompt="\n" + help_text + "\n",
        **kwargs,
    )


def load_arrow_dataset(arrow_dir: str) -> list[dict]:
    """Load all .arrow files from the directory and return rows as dicts."""
    arrow_files = sorted(f for f in os.listdir(arrow_dir) if f.endswith(".arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No .arrow files found in {arrow_dir}")

    all_rows = []
    for arrow_file in arrow_files:
        path = os.path.join(arrow_dir, arrow_file)
        print(f"Reading {arrow_file}...")
        reader = ipc.open_stream(path)
        table = reader.read_all()
        rows = table.to_pydict()
        n = table.num_rows
        for i in range(n):
            row = {col: rows[col][i] for col in table.column_names}
            all_rows.append(row)
        print(f"  -> {n} features loaded")

    print(f"Total features loaded: {len(all_rows)}")
    return all_rows


@app.command()
def main(
    ctx: typer.Context,
    arrow_dir: Annotated[
        str,
        make_option(
            "--arrow-dir",
            help_text="[Input] Arrow Dataset Directory: directory containing .arrow files from HuggingFace dataset.",
        ),
    ],
    creator_name: Annotated[
        str,
        make_option(
            "--creator-name",
            help_text="[Author] Name of the creator (e.g., your organization/team name).",
        ),
    ],
    release_id: Annotated[
        str,
        make_option(
            "--release-id",
            help_text="[Release] Release ID (e.g., llamascope-2). Must be alphanumeric with dashes.",
        ),
    ],
    release_title: Annotated[
        str,
        make_option(
            "--release-title",
            help_text="[Release] Human-readable release title.",
        ),
    ],
    model_name: Annotated[
        str,
        make_option(
            "--model-name",
            help_text="[Model] Model name (e.g., qwen3-1.7b).",
        ),
    ],
    neuronpedia_source_set_id: Annotated[
        str,
        make_option(
            "--neuronpedia-source-set-id",
            help_text="[Source] Source set ID (e.g., plt-8x-topk64). Do not include layer number.",
        ),
    ],
    neuronpedia_source_set_description: Annotated[
        str,
        make_option(
            "--neuronpedia-source-set-description",
            help_text="[Source] Human-readable source set description.",
        ),
    ],
    layer_num: Annotated[
        int,
        make_option(
            "--layer-num",
            help_text="[Source] Layer number this SAE is trained on.",
        ),
    ],
    url: Annotated[
        str,
        make_option(
            "--url",
            help_text="[Info] URL associated with paper/release.",
        ),
    ] = "",
    hf_repo_id: Annotated[
        str,
        make_option(
            "--hf-repo-id",
            help_text="[Source] HuggingFace repo ID for weights (e.g., user/repo).",
        ),
    ] = "",
    hf_folder_id: Annotated[
        str,
        make_option(
            "--hf-folder-id",
            help_text="[Source] HuggingFace folder path within the repo.",
        ),
    ] = "",
    batch_size: Annotated[
        int,
        make_option(
            "--batch-size",
            help_text="[Processing] Number of features per output batch file.",
        ),
    ] = 2048,
    explanation_model_name: Annotated[
        str,
        make_option(
            "--explanation-model-name",
            help_text="[Explanations] Model name that generated the explanations (if known).",
        ),
    ] = "unknown",
):
    print("Running with arguments:\n")
    for param, value in ctx.params.items():
        print(f"  {param}: {value}")
    print()

    source_id = f"{layer_num}-{neuronpedia_source_set_id}"
    output_base = os.path.join(OUTPUT_DIR, model_name)
    output_dir = os.path.join(output_base, source_id)
    os.makedirs(output_dir, exist_ok=True)

    # --- Write release.jsonl ---
    release_path = os.path.join(output_dir, "release.jsonl")
    if not os.path.exists(release_path):
        with open(release_path, "wb") as f:
            release = SourceRelease(
                name=release_id,
                description=release_title,
                descriptionShort=release_title,
                urls=[url] if url else [],
                creatorNameShort=creator_name,
                creatorName=creator_name,
                creatorId=DEFAULT_CREATOR_ID,
                createdAt=created_at,
            )
            f.write(orjson.dumps(release.__dict__) + b"\n")
        print(f"Wrote {release_path}")

    # --- Write model.jsonl ---
    model_path = os.path.join(output_dir, "model.jsonl")
    if not os.path.exists(model_path):
        with open(model_path, "wb") as f:
            model = Model(
                id=model_name,
                instruct=model_name.endswith("-it"),
                displayNameShort=model_name,
                displayName=model_name,
                creatorId=DEFAULT_CREATOR_ID,
                createdAt=created_at,
                updatedAt=created_at,
            )
            f.write(orjson.dumps(model.__dict__) + b"\n")
        print(f"Wrote {model_path}")

    # --- Write sourceset.jsonl ---
    sourceset_path = os.path.join(output_dir, "sourceset.jsonl")
    if not os.path.exists(sourceset_path):
        with open(sourceset_path, "wb") as f:
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
            f.write(orjson.dumps(sourceset.__dict__) + b"\n")
        print(f"Wrote {sourceset_path}")

    # --- Write source.jsonl ---
    source_path = os.path.join(output_dir, "source.jsonl")
    with open(source_path, "wb") as f:
        source = Source(
            modelId=model_name,
            setName=neuronpedia_source_set_id,
            visibility="PUBLIC",
            dataset="",
            id=source_id,
            num_prompts=None,
            num_tokens_in_prompt=None,
            hfRepoId=hf_repo_id or None,
            hfFolderId=hf_folder_id or None,
            creatorId=DEFAULT_CREATOR_ID,
        )
        f.write(orjson.dumps(source.__dict__) + b"\n")
    print(f"Wrote {source_path}")

    # --- Load arrow data ---
    all_rows = load_arrow_dataset(arrow_dir)

    # --- Process in batches ---
    features_dir = os.path.join(output_dir, "features")
    activations_dir = os.path.join(output_dir, "activations")
    explanations_dir = os.path.join(output_dir, "explanations")
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(activations_dir, exist_ok=True)
    os.makedirs(explanations_dir, exist_ok=True)

    total_features = 0
    total_activations = 0
    total_explanations = 0

    for batch_idx, batch_start in enumerate(range(0, len(all_rows), batch_size)):
        batch_end = min(batch_start + batch_size, len(all_rows))
        batch_rows = all_rows[batch_start:batch_end]
        batch_name = f"batch-{batch_idx}"

        features: List[Feature] = []
        activations: List[Activation] = []
        explanations: List[Explanation] = []

        for row in batch_rows:
            feature_index = row["feature_index"]

            max_act_approx = 0.0
            row_activations = row.get("activations") or []
            for act_data in row_activations:
                tokens = act_data.get("tokens") or []
                values = act_data.get("values") or []
                if not tokens or not values:
                    continue

                float_values = [truncate_float(float(v)) for v in values]
                max_val_elem = max(float_values)
                min_val_elem = min(float_values)
                max_value = truncate_float(max_val_elem)
                min_value = truncate_float(min_val_elem)
                max_value_token_index = float_values.index(max_val_elem)

                if max_value > max_act_approx:
                    max_act_approx = max_value

                raw_zpi = act_data.get("z_pattern_indices")
                raw_zpv = act_data.get("z_pattern_values")
                z_pattern_indices = (
                    [[int(x) for x in inner] for inner in raw_zpi]
                    if raw_zpi is not None
                    else None
                )
                z_pattern_values = (
                    [truncate_float(float(v)) for v in raw_zpv]
                    if raw_zpv is not None
                    else None
                )

                activation = Activation(
                    id=CUID_GENERATOR.generate(),
                    tokens=tokens,
                    modelId=model_name,
                    layer=source_id,
                    index=feature_index,
                    maxValue=max_value,
                    maxValueTokenIndex=max_value_token_index,
                    minValue=min_value,
                    values=float_values,
                    creatorId=DEFAULT_CREATOR_ID,
                    createdAt=created_at,
                    dfaValues=[],
                    dfaTargetIndex=None,
                    dfaMaxValue=None,
                    lossValues=[],
                    logitContributions=None,
                    binMin=None,
                    binMax=None,
                    binContains=None,
                    qualifyingTokenIndex=None,
                    dataIndex=None,
                    dataSource=None,
                    zIndices=z_pattern_indices,
                    zValues=z_pattern_values,
                )
                activations.append(activation)

            feature = Feature(
                modelId=model_name,
                layer=source_id,
                index=feature_index,
                creatorId=DEFAULT_CREATOR_ID,
                createdAt=created_at,
                maxActApprox=max_act_approx,
                hasVector=False,
                vector=[],
                vectorLabel=None,
                hookName=None,
                topkCosSimIndices=[],
                topkCosSimValues=[],
                neuron_alignment_indices=[],
                neuron_alignment_values=[],
                neuron_alignment_l1=[],
                correlated_neurons_indices=[],
                correlated_neurons_pearson=[],
                correlated_neurons_l1=[],
                correlated_features_indices=[],
                correlated_features_pearson=[],
                correlated_features_l1=[],
                neg_str=row.get("top_neg_logits_str") or [],
                neg_values=[
                    truncate_float(float(v))
                    for v in (row.get("top_neg_logits_val") or [])
                ],
                pos_str=row.get("top_pos_logits_str") or [],
                pos_values=[
                    truncate_float(float(v))
                    for v in (row.get("top_pos_logits_val") or [])
                ],
                frac_nonzero=float(row.get("frac_nonzero", 0)),
                freq_hist_data_bar_heights=[],
                freq_hist_data_bar_values=[],
                logits_hist_data_bar_heights=[],
                logits_hist_data_bar_values=[],
                decoder_weights_dist=[],
            )
            features.append(feature)

            for explanation_text in row.get("explanations") or []:
                if explanation_text:
                    explanation = Explanation(
                        id=CUID_GENERATOR.generate(),
                        modelId=model_name,
                        layer=source_id,
                        index=feature_index,
                        description=explanation_text,
                        authorId=DEFAULT_CREATOR_ID,
                        typeName="unknown",
                        explanationModelName="unknown",
                        createdAt=created_at,
                    )
                    explanations.append(explanation)

        # Write features
        features_path = os.path.join(features_dir, f"{batch_name}.jsonl")
        with open(features_path, "wb") as f:
            for feat in features:
                f.write(orjson.dumps(feat.__dict__) + b"\n")
        with open(features_path, "rb") as f_in:
            with open(features_path + ".gz", "wb") as f_out:
                f_out.write(gzip.compress(f_in.read(), compresslevel=5))
        os.remove(features_path)

        # Write activations
        activations_path = os.path.join(activations_dir, f"{batch_name}.jsonl")
        with open(activations_path, "wb") as f:
            for act in activations:
                f.write(orjson.dumps(act.__dict__) + b"\n")
        with open(activations_path, "rb") as f_in:
            with open(activations_path + ".gz", "wb") as f_out:
                f_out.write(gzip.compress(f_in.read(), compresslevel=5))
        os.remove(activations_path)

        # Write explanations
        if explanations:
            explanations_path = os.path.join(explanations_dir, f"{batch_name}.jsonl")
            with open(explanations_path, "wb") as f:
                for exp in explanations:
                    f.write(orjson.dumps(exp.__dict__) + b"\n")
            with open(explanations_path, "rb") as f_in:
                with open(explanations_path + ".gz", "wb") as f_out:
                    f_out.write(gzip.compress(f_in.read(), compresslevel=5))
            os.remove(explanations_path)

        total_features += len(features)
        total_activations += len(activations)
        total_explanations += len(explanations)

        print(
            f"  {batch_name}: {len(features)} features, "
            f"{len(activations)} activations, "
            f"{len(explanations)} explanations"
        )

    print(f"\nDone! Output written to: {output_dir}")
    print(f"  Total features:     {total_features}")
    print(f"  Total activations:  {total_activations}")
    print(f"  Total explanations: {total_explanations}")


if __name__ == "__main__":
    app()
