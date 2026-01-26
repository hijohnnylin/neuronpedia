import asyncio
import datetime
import glob
import gzip
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Dict, List

import boto3
import dotenv
import openai
import typer
from botocore import UNSIGNED
from botocore.config import Config
from cuid2 import Cuid
from neuronpedia_utils.db_models.activation import Activation
from neuronpedia_utils.db_models.explanation import Explanation
from neuronpedia_utils.db_models.feature import Feature
from tqdm import tqdm

# silence errors from neuron_explainer
logging.getLogger("neuron_explainer").setLevel(logging.CRITICAL)


# openai requires us to set the openai api key before the neuron_explainer imports
dotenv.load_dotenv()
from neuron_explainer.activations.activation_records import calculate_max_activation

# ruff: noqa: E402
# flake8: noqa: E402
from neuron_explainer.activations.activations import ActivationRecord
from neuron_explainer.api_client import ApiClient
from neuron_explainer.explanations.explainer import (
    AttentionHeadExplainer,
    MaxActivationAndLogitsExplainer,
    MaxActivationAndLogitsGeneralExplainer,
    MaxActivationExplainer,
    TokenActivationPairExplainer,
)
from neuron_explainer.explanations.prompt_builder import PromptFormat

UPLOAD_EXPLANATION_AUTHORID = os.getenv("DEFAULT_CREATOR_ID")
if UPLOAD_EXPLANATION_AUTHORID is None:
    UPLOAD_EXPLANATION_AUTHORID = "clkht01d40000jv08hvalcvly"

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = 256

VALID_EXPLAINER_TYPE_NAMES = [
    "oai_token-act-pair",
    "oai_attention-head",
    "np_max-act-logits",
    "np_max-act",
    "np_acts-logits-general",
]

# you can change this yourself if you want to experiment with other models
VALID_EXPLAINER_MODEL_NAMES = [
    "gpt-4o-mini",
    "gpt-4.1-nano",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

# GEMINI SUPPORT
GEMINI_MODEL_NAMES = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# we should use this one (ai studio, simpler) but we're super rate limited
GEMINI_BASE_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_VERTEX = False
# so we use vertex instead. when we are not rate limited on AI studio, remove the following 4 properties
# GEMINI_PROJECT_ID = os.getenv("GEMINI_PROJECT_ID")
# GEMINI_LOCATION = os.getenv("GEMINI_LOCATION")
# GEMINI_BASE_API_URL = f"https://{GEMINI_LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{GEMINI_PROJECT_ID}/locations/{GEMINI_LOCATION}/endpoints/openapi"
# GEMINI_VERTEX = True

# the following two are overwritten by the command line arguments
# the number of parallel autointerps to do
# this is two bottlenecks:
# 1. the rate limit of the explainer model API you're calling (tokens per minute, requests per minute, etc)
# 2. your local machine's network card and router - too many simultaneous calls will cause timeouts on requests
#  - for a normal macbook pro, 50-100 is a ok number
#  - for a machine with a beefier network card/memory, can go up to 300
#  - you may need to experiment to find the max number for your machine (you'll see timeout errors)
# consider setting these on your machine for higher performance
# ulimit -n 32768
# sudo sysctl -w net.inet.tcp.sendspace=2097152 net.inet.tcp.recvspace=2097152
AUTOINTERP_BATCH_SIZE = 128

# overridden by command line arguments
# the number of top activations to feed the explainer per feature
#  - 10 to 25 is what we usually use
#  - more activations = more $ spent
MAX_TOP_ACTIVATIONS_TO_SHOW_EXPLAINER_PER_FEATURE = 10

# we replace these characters during autointerp so that the explainer isn't confused/distracted by them
# HTML anomalies are weird tokenizer bugs
HTML_ANOMALY_AND_SPECIAL_CHARS_REPLACEMENTS = {
    "âĢĶ": "—",  # em dash
    "âĢĵ": "–",  # en dash
    "âĢľ": '"',  # left double curly quote
    "âĢĿ": '"',  # right double curly quote
    "âĢĺ": "'",  # left single curly quote
    "âĢĻ": "'",  # right single curly quote
    "âĢĭ": " ",  # zero width space
    "Ġ": " ",  # space
    "Ċ": "\n",  # line break
    "<0x0A>": "\n",
    "ĉ": "\t",  # tab
    "▁": " ",  # \u2581, gemma 2 uses this as a space
    "<|endoftext|>": " ",
    "<bos>": " ",
    "<|begin_of_text|>": " ",
    "<|end_of_text|>": " ",
}

CUID_GENERATOR: Cuid = Cuid(length=25)

# S3 constants for public bucket access
S3_BUCKET_NAME = "neuronpedia-datasets"
S3_REGION = "us-east-1"

queuedToSave: List[Explanation] = []

FAILED_FEATURE_INDEXES_QUEUED: List[int] | None = None
FAILED_FEATURE_INDEXES_OUTPUT: List[str] = []

IGNORE_FIRST_N_TOKENS: int = 0
GENERATE_EMBEDDINGS: bool = True


def normalize_s3_path(path: str) -> str:
    """Ensure S3 path starts with v1/ (without leading slash)."""
    # Remove leading slash if present
    path = path.lstrip("/")
    # Prepend v1/ if not already present
    if not path.startswith("v1/"):
        path = "v1/" + path
    return path


def download_s3_exports(s3_path: str, local_dir: str) -> str:
    """
    Download exports from the public S3 bucket to a local directory.

    Args:
        s3_path: Path within the bucket (will be normalized to start with v1/)
        local_dir: Local directory to download files to

    Returns:
        Path to the local directory containing the downloaded exports
    """
    s3_path = normalize_s3_path(s3_path)

    print(f"Downloading from s3://{S3_BUCKET_NAME}/{s3_path} to {local_dir}")

    # Create S3 client with anonymous access (public bucket)
    s3_client = boto3.client(
        "s3",
        region_name=S3_REGION,
        config=Config(signature_version=UNSIGNED),
    )

    # List all objects in the path
    paginator = s3_client.get_paginator("list_objects_v2")

    total_files = 0
    files_to_download = []

    for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=s3_path):
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            files_to_download.append(obj["Key"])
            total_files += 1

    if total_files == 0:
        raise ValueError(f"No files found at s3://{S3_BUCKET_NAME}/{s3_path}")

    print(f"Found {total_files} files to download")

    # Download each file
    for s3_key in tqdm(files_to_download, desc="Downloading from S3"):
        # Calculate relative path from the s3_path prefix
        relative_path = s3_key[len(s3_path) :].lstrip("/")
        local_file_path = os.path.join(local_dir, relative_path)

        # Create parent directories if needed
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download the file
        try:
            s3_client.download_file(S3_BUCKET_NAME, s3_key, local_file_path)
        except Exception as e:
            print(f"Error downloading {s3_key}: {e}")
            raise

    print(f"Successfully downloaded {total_files} files to {local_dir}")
    return local_dir


def replace_html_anomalies_and_special_chars_single(text: str) -> str:
    for old_char, new_char in HTML_ANOMALY_AND_SPECIAL_CHARS_REPLACEMENTS.items():
        text = text.replace(old_char, new_char)
    return text


def replace_html_anomalies_and_special_chars(texts: list[str]) -> list[str]:
    result = []
    for text in texts:
        result.append(replace_html_anomalies_and_special_chars_single(text))
    return result


async def call_autointerp_openai_for_activations(
    activations_sorted_by_max_value: List[Activation],
    feature: Feature,
):
    if len(activations_sorted_by_max_value) == 0:
        return

    top_activation = activations_sorted_by_max_value[0]

    if top_activation.maxValue == 0:
        # print("skipping dead feature: " + str(directionIndex))
        return

    feature_index = top_activation.index

    # only needed for vertex
    if GEMINI_VERTEX:
        model_name = (
            "google/" + EXPLAINER_MODEL_NAME
            if is_gemini_model(EXPLAINER_MODEL_NAME)
            else EXPLAINER_MODEL_NAME
        )
    else:
        model_name = EXPLAINER_MODEL_NAME
    base_api_url = (
        GEMINI_BASE_API_URL
        if is_gemini_model(EXPLAINER_MODEL_NAME)
        else ApiClient.BASE_API_URL
    )
    override_api_key = GEMINI_API_KEY if is_gemini_model(EXPLAINER_MODEL_NAME) else None

    global FAILED_FEATURE_INDEXES_OUTPUT

    try:
        activationRecords = []

        if EXPLAINER_TYPE_NAME == "oai_attention-head":
            for activation in activations_sorted_by_max_value:
                activationRecord = ActivationRecord(
                    tokens=replace_html_anomalies_and_special_chars(activation.tokens),
                    activations=activation.values,
                    dfa_values=activation.dfaValues,
                    dfa_target_index=activation.dfaTargetIndex,
                )
                activationRecords.append(activationRecord)
            explainer = AttentionHeadExplainer(
                model_name=model_name,
                prompt_format=PromptFormat.HARMONY_V4,
                max_concurrent=1,
                base_api_url=base_api_url,
                override_api_key=override_api_key,
            )
            explanations = await asyncio.wait_for(
                explainer.generate_explanations(
                    max_tokens=2000,
                    all_activation_records=activationRecords,
                    num_samples=1,
                    reasoning_effort=REASONING_EFFORT,
                ),
                timeout=20,
            )
        elif EXPLAINER_TYPE_NAME == "oai_token-act-pair":
            for activation in activations_sorted_by_max_value:
                activationRecord = ActivationRecord(
                    tokens=replace_html_anomalies_and_special_chars(activation.tokens),
                    activations=activation.values,
                )
                activationRecords.append(activationRecord)
            explainer = TokenActivationPairExplainer(
                model_name=model_name,
                prompt_format=PromptFormat.HARMONY_V4,
                max_concurrent=1,
                base_api_url=base_api_url,
                override_api_key=override_api_key,
            )
            explanations = await asyncio.wait_for(
                explainer.generate_explanations(
                    max_tokens=2000,
                    all_activation_records=activationRecords,
                    max_activation=calculate_max_activation(activationRecords),
                    num_samples=1,
                    reasoning_effort=REASONING_EFFORT,
                ),
                timeout=20,
            )
        elif EXPLAINER_TYPE_NAME == "np_acts-logits-general":
            for activation in activations_sorted_by_max_value:
                activationRecord = ActivationRecord(
                    tokens=replace_html_anomalies_and_special_chars(activation.tokens),
                    activations=activation.values,
                )
                activationRecords.append(activationRecord)
            explainer = MaxActivationAndLogitsGeneralExplainer(
                model_name=model_name,
                prompt_format=PromptFormat.HARMONY_V4,
                max_concurrent=1,
                base_api_url=base_api_url,
                override_api_key=override_api_key,
            )
            try:
                explanations = await asyncio.wait_for(
                    explainer.generate_explanations(
                        all_activation_records=activationRecords,
                        max_tokens=2000,
                        max_activation=calculate_max_activation(activationRecords),
                        top_positive_logits=replace_html_anomalies_and_special_chars(
                            feature.pos_str
                        ),
                        num_samples=1,
                        reasoning_effort=REASONING_EFFORT,
                    ),
                    timeout=20,
                )
            except Exception as e:
                print(
                    f"Error in MaxActivationAndLogitsGeneralExplainer.generate_explanations for feature index {feature_index}:"
                )
                print(f"Exception type: {type(e).__name__}")
                print(f"Exception message: {str(e)}")
                import traceback

                print(f"Traceback: {traceback.format_exc()}")
                raise  # Re-raise to be caught by the outer try-except block
        elif EXPLAINER_TYPE_NAME == "np_max-act-logits":
            for activation in activations_sorted_by_max_value:
                activationRecord = ActivationRecord(
                    tokens=replace_html_anomalies_and_special_chars(activation.tokens),
                    activations=activation.values,
                )
                activationRecords.append(activationRecord)
            explainer = MaxActivationAndLogitsExplainer(
                model_name=model_name,
                prompt_format=PromptFormat.HARMONY_V4,
                max_concurrent=1,
                base_api_url=base_api_url,
                override_api_key=override_api_key,
            )
            try:
                explanations = await asyncio.wait_for(
                    explainer.generate_explanations(
                        all_activation_records=activationRecords,
                        max_tokens=2000,
                        max_activation=calculate_max_activation(activationRecords),
                        top_positive_logits=replace_html_anomalies_and_special_chars(
                            feature.pos_str
                        ),
                        num_samples=1,
                        reasoning_effort=REASONING_EFFORT,
                    ),
                    timeout=20,
                )
            except Exception as e:
                print(
                    f"Error in MaxActivationAndLogitsExplainer.generate_explanations for feature index {feature_index}:"
                )
                print(f"Exception type: {type(e).__name__}")
                print(f"Exception message: {str(e)}")
                import traceback

                print(f"Traceback: {traceback.format_exc()}")
                raise  # Re-raise to be caught by the outer try-except block
        elif EXPLAINER_TYPE_NAME == "np_max-act":
            for activation in activations_sorted_by_max_value:
                activationRecord = ActivationRecord(
                    tokens=replace_html_anomalies_and_special_chars(activation.tokens),
                    activations=activation.values,
                )
                activationRecords.append(activationRecord)
            explainer = MaxActivationExplainer(
                model_name=model_name,
                prompt_format=PromptFormat.HARMONY_V4,
                max_concurrent=1,
                base_api_url=base_api_url,
                override_api_key=override_api_key,
            )
            try:
                explanations = await asyncio.wait_for(
                    explainer.generate_explanations(
                        all_activation_records=activationRecords,
                        max_tokens=2000,
                        max_activation=calculate_max_activation(activationRecords),
                        num_samples=1,
                        reasoning_effort=REASONING_EFFORT,
                    ),
                    timeout=20,
                )
            except Exception as e:
                print(
                    f"=== Error in MaxActivationExplainer.generate_explanations for feature index {feature_index} ==="
                )
                print(f"Exception type: {type(e).__name__}")
                print(f"Exception message: {str(e)}")
                import traceback

                print(f"Traceback: {traceback.format_exc()}")
                raise  # Re-raise to be caught by the outer try-except block

    except Exception as e:
        if isinstance(e, asyncio.TimeoutError):
            print("Timeout occurred, skipping index " + str(feature_index))
        else:
            print(f"=== Explain Error, skipping index {feature_index} ===")
            print(e)

        # processed at the end
        FAILED_FEATURE_INDEXES_OUTPUT.append(feature_index)
        return

    if len(explanations) == 0:
        FAILED_FEATURE_INDEXES_OUTPUT.append(feature_index)
        return

    explanation = explanations[0]
    if explanation.endswith("."):
        explanation = explanation[:-1]
    explanation = explanation.replace("\n", "").replace("\r", "")

    # print(f"Explanation: {explanation}  {feature.layer} {feature.index}")

    global queuedToSave
    # if the explanation is just "first token", use the top activation token
    if (
        "unclear" in explanation.strip().lower()
        or "unsure" in explanation.strip().lower()
        or "first token" in explanation.strip().lower()
        or len(explanation.strip()) == 0
    ):
        # use the top activation token
        explanation = replace_html_anomalies_and_special_chars_single(
            top_activation.tokens[
                top_activation.values.index(max(top_activation.values))
            ].strip()
        )
        if len(explanation.strip()) == 0:
            # top activating token is empty, skip this feature
            FAILED_FEATURE_INDEXES_OUTPUT.append(feature_index)
            pass
        else:
            queuedToSave.append(
                Explanation(
                    id=CUID_GENERATOR.generate(),
                    modelId=top_activation.modelId,
                    layer=top_activation.layer,
                    index=str(feature_index),
                    description=explanation,
                    typeName=EXPLAINER_TYPE_NAME,
                    explanationModelName=EXPLAINER_MODEL_NAME,
                    authorId=UPLOAD_EXPLANATION_AUTHORID or "",
                )
            )
            # print(
            #     f"Using top activation token {explanation} for feature index {feature_index}\n"
            # )
    else:
        queuedToSave.append(
            Explanation(
                id=CUID_GENERATOR.generate(),
                modelId=top_activation.modelId,
                layer=top_activation.layer,
                index=str(feature_index),
                description=explanation,
                typeName=EXPLAINER_TYPE_NAME,
                explanationModelName=EXPLAINER_MODEL_NAME,
                authorId=UPLOAD_EXPLANATION_AUTHORID or "",
            )
        )


semaphore = asyncio.Semaphore(AUTOINTERP_BATCH_SIZE)


async def enqueue_autointerp_openai_task_with_activations(
    activations: List[Activation], feature: Feature
):
    async with semaphore:
        return await call_autointerp_openai_for_activations(activations, feature)


async def start(activations_dir: str):
    autointerp_tasks = []

    def get_batch_number(filename):
        # Extract batch number from filename like "batch-123.jsonl.gz"
        try:
            return int(os.path.basename(filename).split("-")[1].split(".")[0])
        except (IndexError, ValueError):
            return 0  # Default value if parsing fails

    # don't check subdirectories (eg)
    activations_files = sorted(
        [
            f
            for f in glob.glob(os.path.join(activations_dir, "*.gz"))
            if os.path.dirname(f) == activations_dir
        ],
        key=get_batch_number,
    )

    print(f"got activations files: {len(activations_files)} files")

    # for each gz file, decompress it into memory
    for activations_file in tqdm(activations_files, desc="Processing files"):
        # print(f"processing activations file: {activations_file}")
        with gzip.open(activations_file, "rt") as f:
            # also decompress its associated features file
            features_file = activations_file.replace("activations", "features")
            with gzip.open(features_file, "rt") as f_features:
                # read the features file line by line
                features: List[Feature] = []
                for line in f_features:
                    feature_json = json.loads(line)
                    feature = Feature.from_dict(feature_json)
                    features.append(feature)
                features_by_index: Dict[str, Feature] = {}
                for feature in features:
                    if feature.index not in features_by_index:
                        features_by_index[feature.index] = feature

                # read activations jsonl file line by line
                activations: List[Activation] = []
                read_activation_texts: List[str] = []
                for line in f:
                    activation_json = json.loads(line)
                    activation = Activation.from_dict(activation_json)
                    if int(activation.index) < START_INDEX:
                        # print(f"Skipping activation {activation.index} because it's less than START_INDEX {START_INDEX}")
                        continue
                    if END_INDEX is not None and int(activation.index) > END_INDEX:
                        # print(f"Skipping activation {activation.index} because it's greater than END_INDEX {END_INDEX}")
                        continue
                    if IGNORE_FIRST_N_TOKENS > 0:
                        activation.tokens = activation.tokens[IGNORE_FIRST_N_TOKENS:]
                        activation.values = activation.values[IGNORE_FIRST_N_TOKENS:]
                        if activation.dfaValues is not None:
                            activation.dfaValues = activation.dfaValues[
                                IGNORE_FIRST_N_TOKENS:
                            ]
                    global FAILED_FEATURE_INDEXES_QUEUED
                    if (
                        FAILED_FEATURE_INDEXES_QUEUED is not None
                        and len(FAILED_FEATURE_INDEXES_QUEUED) > 0
                        and int(activation.index) not in FAILED_FEATURE_INDEXES_QUEUED
                    ):
                        # print(f"Skipping activation {activation.index} because it's not in FAILED_FEATURE_INDEXES_QUEUED {FAILED_FEATURE_INDEXES_QUEUED}")
                        continue
                    # turn activation into a string and check if it's duplicate
                    activation_text = "".join(activation.tokens)
                    if activation_text in read_activation_texts:
                        continue
                    read_activation_texts.append(activation_text)
                    activations.append(activation)
                activations_by_index: Dict[str, List[Activation]] = {}
                for activation in activations:
                    if activation.index not in activations_by_index:
                        activations_by_index[activation.index] = []
                    activations_by_index[activation.index].append(activation)
                # sort each activations_by_index by maxAct, largest to smallest
                # then run them in batches
                for index in tqdm(
                    activations_by_index,
                    desc=f"Auto-Interping activations in {os.path.basename(activations_file)}",
                    leave=False,
                ):
                    # Sort and take top MAX_TOP_ACTIVATIONS_TO_SHOW_EXPLAINER_PER_FEATURE activations
                    activations_by_index[index] = sorted(
                        activations_by_index[index],
                        key=lambda x: x.maxValue,
                        reverse=True,
                    )[:MAX_TOP_ACTIVATIONS_TO_SHOW_EXPLAINER_PER_FEATURE]

                    # enqueue it
                    # if features_by_index doesn't have "index", skip
                    if index not in features_by_index:
                        # print(f"Skipping activation {index} because it's not in features_by_index {features_by_index}")
                        continue
                    task = asyncio.create_task(
                        enqueue_autointerp_openai_task_with_activations(
                            activations_by_index[index],
                            features_by_index[index],
                        )
                    )
                    autointerp_tasks.append(task)
                    # if we have enough tasks, run them
                    if len(autointerp_tasks) >= AUTOINTERP_BATCH_SIZE:
                        # print(f"Enqueuing {len(autointerp_tasks)} tasks")
                        await asyncio.gather(*autointerp_tasks)
                        autointerp_tasks.clear()
                        generate_embeddings_and_flush_explanations_to_file(queuedToSave)
                        queuedToSave.clear()

    # do the last batch
    await asyncio.gather(*autointerp_tasks)
    autointerp_tasks.clear()
    generate_embeddings_and_flush_explanations_to_file(queuedToSave)
    queuedToSave.clear()


@dataclass
class AutoInterpConfig:
    input_dir_with_source_exports: str
    s3_exports_path: str | None
    start_index: int
    end_index: int | None
    explainer_model_name: str
    explainer_type_name: str
    reasoning_effort: str | None
    max_top_activations_to_show_explainer_per_feature: int
    autointerp_batch_size: int
    gzip_output: bool
    ignore_first_n_tokens: int
    generate_embeddings: bool


def is_gemini_model(model_name: str) -> bool:
    return model_name in GEMINI_MODEL_NAMES


def main(
    input_dir_with_source_exports: str | None = typer.Option(
        None,
        help="The directory where you exported your activations and features. Either this or --s3-exports-path must be provided.",
    ),
    s3_exports_path: str | None = typer.Option(
        None,
        help=f"S3 path in the {S3_BUCKET_NAME} bucket (e.g., 'v1/model/layer'). Will be downloaded to a local temp directory. Either this or --input-dir-with-source-exports must be provided.",
    ),
    start_index: int = typer.Option(
        0, help="The starting index to process", prompt=True
    ),
    end_index: int | None = typer.Option(
        None,
        help="The ending index to process - if not provided, we'll just do all of them.",
        prompt=True,
        prompt_required=False,
    ),
    explainer_model_name: str = typer.Option(
        "gpt-4.1-nano",
        help="The name of the explainer model eg gpt-4o-mini",
        prompt=True,
    ),
    explainer_type_name: str = typer.Option(
        "oai_token-act-pair",
        help="The type name of the explainer - oai_token-act-pair or oai_attention-head",
        prompt=True,
    ),
    reasoning_effort: str | None = typer.Option(
        None,
        help="The reasoning effort to use for the explainer, only works if the model is a reasoning model. Minimal, low, medium, high.",
    ),
    max_top_activations_to_show_explainer_per_feature: int = typer.Option(
        20, help="Number of top activations to use for explanation"
    ),
    autointerp_batch_size: int = typer.Option(
        50,
        help="Batch size for autointerp - this is the max parallel connections to the explainer model",
        prompt=True,
    ),
    output_dir: str | None = typer.Option(
        default=None, help="The path to the output directory"
    ),
    gzip_output: bool = typer.Option(
        False, help="Whether to gzip the output file", prompt=True
    ),
    only_failed_features: bool = typer.Option(
        False, help="Whether to only auto-interp failed features", prompt=True
    ),
    ignore_first_n_tokens: int = typer.Option(
        0,
        help="Optional number of tokens to ignore from the beginning of the text so that autointerp doesn't see it",
        prompt=True,
    ),
    generate_embeddings: bool = typer.Option(
        True,
        help="Whether to generate OpenAI embeddings for explanations. If False, skips embedding generation and doesn't require OPENAI_API_KEY.",
        prompt=True,
    ),
):
    if explainer_type_name not in VALID_EXPLAINER_TYPE_NAMES:
        raise ValueError(f"Invalid explainer type name: {explainer_type_name}")

    if explainer_model_name not in VALID_EXPLAINER_MODEL_NAMES:
        raise ValueError(f"Invalid explainer model name: {explainer_model_name}")

    if is_gemini_model(explainer_model_name) and GEMINI_API_KEY is None:
        raise ValueError(
            "GEMINI_API_KEY is not set even though you're using a Gemini model"
        )

    global \
        FAILED_FEATURE_INDEXES_QUEUED, \
        INPUT_DIR_WITH_SOURCE_EXPORTS, \
        START_INDEX, \
        END_INDEX, \
        EXPLAINER_MODEL_NAME, \
        EXPLAINER_TYPE_NAME, \
        MAX_TOP_ACTIVATIONS_TO_SHOW_EXPLAINER_PER_FEATURE, \
        AUTOINTERP_BATCH_SIZE, \
        EXPLANATIONS_OUTPUT_DIR, \
        GZIP_OUTPUT, \
        IGNORE_FIRST_N_TOKENS, \
        REASONING_EFFORT, \
        GENERATE_EMBEDDINGS

    GENERATE_EMBEDDINGS = generate_embeddings
    if GENERATE_EMBEDDINGS and not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is not set. Please set it in the .env file or export it as an environment variable. "
            "Or set --generate-embeddings=False to skip embedding generation."
        )

    # Validate that either input_dir_with_source_exports or s3_exports_path is provided
    if input_dir_with_source_exports is None and s3_exports_path is None:
        raise ValueError(
            "Either --input-dir-with-source-exports or --s3-exports-path must be provided."
        )
    if input_dir_with_source_exports is not None and s3_exports_path is not None:
        raise ValueError(
            "Only one of --input-dir-with-source-exports or --s3-exports-path should be provided, not both."
        )
    if s3_exports_path is not None and output_dir is None:
        raise ValueError(
            "--output-dir is required when using --s3-exports-path, otherwise outputs would be written to a temp directory that gets deleted."
        )

    # If s3_exports_path is provided, download to a temp directory
    temp_dir = None
    if s3_exports_path is not None:
        temp_dir = tempfile.mkdtemp(prefix="neuronpedia_autointerp_")
        print(f"Created temporary directory: {temp_dir}")
        try:
            download_s3_exports(s3_exports_path, temp_dir)
            INPUT_DIR_WITH_SOURCE_EXPORTS = temp_dir
        except Exception as e:
            # Clean up temp dir on failure
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ValueError(f"Failed to download from S3: {e}")
    else:
        INPUT_DIR_WITH_SOURCE_EXPORTS = input_dir_with_source_exports

    if not os.path.exists(INPUT_DIR_WITH_SOURCE_EXPORTS):
        raise ValueError(
            f"Input directory does not exist: {INPUT_DIR_WITH_SOURCE_EXPORTS}"
        )
    activations_dir = os.path.join(INPUT_DIR_WITH_SOURCE_EXPORTS, "activations")
    if not os.path.exists(activations_dir):
        raise ValueError(f"Activations directory does not exist: {activations_dir}")

    START_INDEX = start_index
    if START_INDEX < 0:
        raise ValueError(f"Start index must be greater than 0: {START_INDEX}")
    END_INDEX = end_index
    if END_INDEX is not None and END_INDEX < START_INDEX:
        raise ValueError(
            f"End index must be greater than start index: {END_INDEX} < {START_INDEX}"
        )

    EXPLAINER_MODEL_NAME = explainer_model_name
    EXPLAINER_TYPE_NAME = explainer_type_name
    REASONING_EFFORT = reasoning_effort
    MAX_TOP_ACTIVATIONS_TO_SHOW_EXPLAINER_PER_FEATURE = (
        max_top_activations_to_show_explainer_per_feature
    )
    AUTOINTERP_BATCH_SIZE = autointerp_batch_size
    IGNORE_FIRST_N_TOKENS = ignore_first_n_tokens
    EXPLANATIONS_OUTPUT_DIR = output_dir
    if not EXPLANATIONS_OUTPUT_DIR:
        EXPLANATIONS_OUTPUT_DIR = os.path.join(
            INPUT_DIR_WITH_SOURCE_EXPORTS, "explanations"
        )
    if not os.path.exists(EXPLANATIONS_OUTPUT_DIR):
        # print(f"Creating explanations output directory: {EXPLANATIONS_OUTPUT_DIR}")
        os.makedirs(EXPLANATIONS_OUTPUT_DIR)

    GZIP_OUTPUT = gzip_output

    config = AutoInterpConfig(
        input_dir_with_source_exports=INPUT_DIR_WITH_SOURCE_EXPORTS,
        s3_exports_path=s3_exports_path,
        start_index=START_INDEX,
        end_index=END_INDEX,
        explainer_model_name=EXPLAINER_MODEL_NAME,
        explainer_type_name=EXPLAINER_TYPE_NAME,
        reasoning_effort=REASONING_EFFORT,
        max_top_activations_to_show_explainer_per_feature=MAX_TOP_ACTIVATIONS_TO_SHOW_EXPLAINER_PER_FEATURE,
        autointerp_batch_size=AUTOINTERP_BATCH_SIZE,
        gzip_output=gzip_output,
        ignore_first_n_tokens=IGNORE_FIRST_N_TOKENS,
        generate_embeddings=GENERATE_EMBEDDINGS,
    )

    print("Auto-Interp Config\n", json.dumps(asdict(config), indent=2))

    failed_file_path = os.path.join(
        EXPLANATIONS_OUTPUT_DIR, "failed_explanation_indexes.txt"
    )
    if only_failed_features is False:
        with open(os.path.join(EXPLANATIONS_OUTPUT_DIR, "config.json"), "w") as f:
            json.dump(asdict(config), f, indent=2)
    else:
        print("Only auto-interping failed features")
        # read failed_feature_explanation_indexes from file
        with open(failed_file_path, "r") as f:
            FAILED_FEATURE_INDEXES_QUEUED = sorted(
                [int(line.strip()) for line in f.readlines()]
            )
            print(
                f"Number of failed features to auto-interp: {len(FAILED_FEATURE_INDEXES_QUEUED)}"
            )
            if len(FAILED_FEATURE_INDEXES_QUEUED) == 0:
                print("No failed features to auto-interp")
                return

    total_start_time = time.time()

    asyncio.run(start(activations_dir))

    global FAILED_FEATURE_INDEXES_OUTPUT
    print(
        f"{len(FAILED_FEATURE_INDEXES_OUTPUT)} indexes failed to auto-interp: {FAILED_FEATURE_INDEXES_OUTPUT}"
    )
    print(f"Writing failed indexes to {failed_file_path}")
    mode = "w" if only_failed_features else "a"
    with open(failed_file_path, mode) as f:
        for index in FAILED_FEATURE_INDEXES_OUTPUT:
            f.write(f"{index}\n")

    print("--- %s seconds total ---" % (time.time() - total_start_time))
    total_time_seconds = time.time() - total_start_time
    total_time_minutes = total_time_seconds / 60
    print(f"--- {total_time_minutes:.2f} minutes total ---")

    # Clean up temp directory if we created one from S3 download
    if temp_dir is not None and os.path.exists(temp_dir):
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_next_batch_number() -> int:
    existing_batch_numbers = [
        int(os.path.basename(f).split("-")[1].split(".")[0])
        for f in glob.glob(os.path.join(EXPLANATIONS_OUTPUT_DIR or "", "*.jsonl*"))
    ]
    # no files yet, this is batch 0
    if len(existing_batch_numbers) == 0:
        return 0

    # get the highest batch number
    highest_batch_number = max(existing_batch_numbers)
    return highest_batch_number + 1


def generate_embeddings_and_flush_explanations_to_file(explanations: List[Explanation]):
    explanations.sort(key=lambda x: x.index)
    # remove all explanations with empty descriptions
    explanations = [exp for exp in explanations if exp.description.strip() != ""]

    global FAILED_FEATURE_INDEXES_OUTPUT

    if GENERATE_EMBEDDINGS:
        descriptions = [exp.description for exp in explanations]
        try:
            embeddings = openai.embeddings.create(
                model=DEFAULT_EMBEDDING_MODEL,
                input=descriptions,
                dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            )
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            print(f"Descriptions: {json.dumps(descriptions)}")
            print(f"Length of descriptions: {len(descriptions)}")
            # add all the description indexes to the failed_feature_indexes
            FAILED_FEATURE_INDEXES_OUTPUT.extend([exp.index for exp in explanations])
            return
        if len(embeddings.data) != len(explanations):
            raise Exception("Number of embeddings doesn't match number of explanations")
        for exp, emb in zip(explanations, embeddings.data):
            exp.embedding = [round(value, 9) for value in emb.embedding]
        # print(f"Generated {len(embeddings.data)} embeddings")

    batch_number = get_next_batch_number()
    filename = f"batch-{batch_number}.jsonl"
    filepath = os.path.join(EXPLANATIONS_OUTPUT_DIR or "", filename)

    with open(filepath, "wt") as f:
        for explanation in explanations:
            explanation_dict = asdict(explanation)
            for key, value in explanation_dict.items():
                if isinstance(value, datetime.datetime):
                    explanation_dict[key] = value.isoformat()
            # Remove embedding field if embeddings were not generated
            if not GENERATE_EMBEDDINGS and "embedding" in explanation_dict:
                del explanation_dict["embedding"]
            json.dump(explanation_dict, f)
            f.write("\n")

    if GZIP_OUTPUT:
        with open(filepath, "rb") as f_in:
            with gzip.open(filepath + ".gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(filepath)

    # print(f"Saved {len(explanations)} explanations to {filepath}")


if __name__ == "__main__":
    typer.run(main)
