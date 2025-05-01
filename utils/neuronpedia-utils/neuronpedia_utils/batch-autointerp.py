import asyncio
import datetime
import glob
import gzip
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from typing import Dict, List

import dotenv
import openai
import typer
from cuid2 import Cuid
from neuronpedia_utils.db_models.activation import Activation
from neuronpedia_utils.db_models.explanation import Explanation
from tqdm import tqdm

# openai requires us to set the openai api key before the neuron_explainer imports
dotenv.load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY is not set. Please set it in the .env file or export it as an environment variable."
    )
from neuron_explainer.activations.activation_records import calculate_max_activation

# ruff: noqa: E402
# flake8: noqa: E402
from neuron_explainer.activations.activations import ActivationRecord
from neuron_explainer.api_client import ApiClient
from neuron_explainer.explanations.explainer import (
    AttentionHeadExplainer,
    TokenActivationPairExplainer,
)
from neuron_explainer.explanations.prompt_builder import PromptFormat

SAVE_DIR_BASE = "./export-autointerp"
UPLOAD_EXPLANATION_AUTHORID = os.getenv("DEFAULT_CREATOR_ID")
if UPLOAD_EXPLANATION_AUTHORID is None:
    raise ValueError(
        "UPLOAD_EXPLANATION_AUTHORID is not set. Please set it in the .env file or export it as an environment variable."
    )

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = 256

VALID_EXPLAINER_TYPE_NAMES = ["oai_token-act-pair", "oai_attention-head"]

# you can change this yourself if you want to experiment with other models
VALID_EXPLAINER_MODEL_NAMES = ["gpt-4o-mini", "gpt-4.1-nano", "gemini-2.0-flash"]

# GEMINI SUPPORT
GEMINI_MODEL_NAMES = ["gemini-2.0-flash"]
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# we should use this one (ai studio, simpler) but we're super rate limited
# GEMINI_BASE_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
# so we use vertex instead. when we are not rate limited on AI studio, remove the following 4 properties
GEMINI_PROJECT_ID = os.getenv("GEMINI_PROJECT_ID")
GEMINI_LOCATION = os.getenv("GEMINI_LOCATION")
GEMINI_BASE_API_URL = f"https://{GEMINI_LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{GEMINI_PROJECT_ID}/locations/{GEMINI_LOCATION}/endpoints/openapi"
GEMINI_VERTEX = True

# the number of parallel autointerps to do
# this is two bottlenecks:
# 1. the rate limit of the explainer model API you're calling (tokens per minute, requests per minute, etc)
# 2. your local machine's network card and router - too many simultaneous calls will cause timeouts on requests
#  - for a normal macbook pro, 50-100 is a ok number
#  - for a machine with a beefier network card/memory, can go up to 300
#  - you may need to experiment to find the max number for your machine (you'll see timeout errors)
AUTOINTERP_BATCH_SIZE = 150

# the number of top activations to feed the explainer per feature
#  - 10 to 25 is what we usually use
#  - more activations = more $ spent
MAX_TOP_ACTIVATIONS_TO_SHOW_EXPLAINER_PER_FEATURE = 20

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
    "▁": " ",
    "<|endoftext|>": " ",
    "<bos>": " ",
    "<|begin_of_text|>": " ",
    "<|end_of_text|>": " ",
}

CUID_GENERATOR: Cuid = Cuid(length=25)

queuedToSave: List[Explanation] = []

FAILED_FEATURE_INDEXES_QUEUED: List[int] | None = None
FAILED_FEATURE_INDEXES_OUTPUT: List[str] = []


def replace_html_anomalies_and_special_chars(texts: list[str]) -> list[str]:
    return [
        "".join(
            HTML_ANOMALY_AND_SPECIAL_CHARS_REPLACEMENTS.get(char, char) for char in text
        )
        for text in texts
    ]


async def call_autointerp_openai_for_activations(
    activations_sorted_by_max_value: List[Activation],
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

    try:
        activationRecords = []

        if EXPLAINER_TYPE_NAME == "oai_attention-head":
            for activation in activations_sorted_by_max_value:
                activationRecord = ActivationRecord(
                    tokens=activation.tokens,
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
                    all_activation_records=activationRecords,
                    num_samples=1,
                ),
                timeout=10,
            )
        elif EXPLAINER_TYPE_NAME == "oai_token-act-pair":
            for activation in activations_sorted_by_max_value:
                activationRecord = ActivationRecord(
                    tokens=activation.tokens,
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
                    all_activation_records=activationRecords,
                    max_activation=calculate_max_activation(activationRecords),
                    num_samples=1,
                ),
                timeout=10,
            )

    except Exception as e:
        if isinstance(e, asyncio.TimeoutError):
            # print("Timeout occurred, skipping index " + str(feature_index))
            pass
        else:
            print("Explain Error, skipping index " + str(feature_index))
            print(e)

        # print this at the end
        global FAILED_FEATURE_INDEXES_OUTPUT
        FAILED_FEATURE_INDEXES_OUTPUT.append(feature_index)
        return
    assert len(explanations) == 1
    explanation = explanations[0]
    if explanation.endswith("."):
        explanation = explanation[:-1]
    explanation = explanation.replace("\n", "").replace("\r", "")
    # print(f"{explanation=}")

    global queuedToSave
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


async def enqueue_autointerp_openai_task_with_activations(activations):
    async with semaphore:
        return await call_autointerp_openai_for_activations(activations)


async def start(activations_dir: str):
    autointerp_tasks = []

    def get_batch_number(filename):
        # Extract batch number from filename like "batch-123.jsonl.gz"
        try:
            return int(os.path.basename(filename).split("-")[1].split(".")[0])
        except (IndexError, ValueError):
            return 0  # Default value if parsing fails

    activations_files = sorted(
        glob.glob(os.path.join(activations_dir, "*.gz")), key=get_batch_number
    )

    print(f"got activations files: {len(activations_files)} files")

    # for each gz file, decompress it into memory
    for activations_file in tqdm(activations_files, desc="Processing files"):
        # print(f"processing activations file: {activations_file}")
        with gzip.open(activations_file, "rt") as f:
            # read jsonl file line by line
            activations: List[Activation] = []
            for line in f:
                activation_json = json.loads(line)
                activation = Activation.from_dict(activation_json)
                if int(activation.index) < START_INDEX:
                    continue
                if END_INDEX is not None and int(activation.index) > END_INDEX:
                    continue
                global FAILED_FEATURE_INDEXES_QUEUED
                if (
                    FAILED_FEATURE_INDEXES_QUEUED is not None
                    and len(FAILED_FEATURE_INDEXES_QUEUED) > 0
                    and int(activation.index) not in FAILED_FEATURE_INDEXES_QUEUED
                ):
                    continue
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
                    activations_by_index[index], key=lambda x: x.maxValue, reverse=True
                )[:MAX_TOP_ACTIVATIONS_TO_SHOW_EXPLAINER_PER_FEATURE]

                # enqueue it
                task = asyncio.create_task(
                    enqueue_autointerp_openai_task_with_activations(
                        activations_by_index[index]
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
    start_index: int
    end_index: int | None
    explainer_model_name: str
    explainer_type_name: str
    max_top_activations_to_show_explainer_per_feature: int
    autointerp_batch_size: int
    gzip_output: bool


def is_gemini_model(model_name: str) -> bool:
    return model_name in GEMINI_MODEL_NAMES


def main(
    input_dir_with_source_exports: str = typer.Option(
        ...,
        help="The directory where you exported your activations and features",
        prompt=True,
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
        GZIP_OUTPUT
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
    MAX_TOP_ACTIVATIONS_TO_SHOW_EXPLAINER_PER_FEATURE = (
        max_top_activations_to_show_explainer_per_feature
    )
    AUTOINTERP_BATCH_SIZE = autointerp_batch_size

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
        start_index=START_INDEX,
        end_index=END_INDEX,
        explainer_model_name=EXPLAINER_MODEL_NAME,
        explainer_type_name=EXPLAINER_TYPE_NAME,
        max_top_activations_to_show_explainer_per_feature=MAX_TOP_ACTIVATIONS_TO_SHOW_EXPLAINER_PER_FEATURE,
        autointerp_batch_size=AUTOINTERP_BATCH_SIZE,
        gzip_output=gzip_output,
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

    total_start_time = time.time()

    asyncio.run(start(activations_dir))

    global FAILED_FEATURE_INDEXES_OUTPUT
    print(
        f"{len(FAILED_FEATURE_INDEXES_OUTPUT)} indexes failed to auto-interp: {FAILED_FEATURE_INDEXES_OUTPUT}"
    )
    print(f"Writing failed indexes to {failed_file_path}")
    with open(failed_file_path, "w") as f:
        for index in FAILED_FEATURE_INDEXES_OUTPUT:
            f.write(f"{index}\n")

    print("--- %s seconds total ---" % (time.time() - total_start_time))
    total_time_seconds = time.time() - total_start_time
    total_time_minutes = total_time_seconds / 60
    print(f"--- {total_time_minutes:.2f} minutes total ---")


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

    descriptions = [exp.description for exp in explanations]
    try:
        embeddings = openai.embeddings.create(
            model=DEFAULT_EMBEDDING_MODEL,
            input=descriptions,
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
        )
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        print(f"Descriptions: {descriptions}")
        print(f"Length of descriptions: {len(descriptions)}")
        # add all the description indexes to the failed_feature_indexes
        global FAILED_FEATURE_INDEXES_OUTPUT
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
