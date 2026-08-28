# Description: this script dumps the database (all sources/SAEs, models, explanations, activations, features) into ./exports
# This is useful so that others can import it into their own instance
# Usage: python export-database.py

# TODO: should NOT re-export if files exist already
# TODO: should be able to export a single model/source/release

import json
import os
from psycopg2.extras import RealDictCursor
from tqdm import tqdm
import datetime
import gzip
import dotenv
import typer
import psycopg2
from concurrent.futures import ThreadPoolExecutor
import threading
import concurrent.futures

dotenv.load_dotenv(".env.default")
dotenv.load_dotenv()

FEATURES_EXPLANATIONS_BATCH_SIZE = 1024
ACTIVATIONS_BATCH_SIZE = 1024


DEFAULT_CREATOR_ID = os.getenv("DEFAULT_CREATOR_ID")

# for our neuronpedia database, these are the activations that are publicly visible
# we don't want to export activations that users cached for themselves or anonymous ones
PUBLIC_ACTIVATION_USER_IDS = [
    "clsxqq2xd0000vvp2k5itlhqj",
    "clkht01d40000jv08hvalcvly",
    "cljqfoqm1000776wmbr1f5mux",
    "cljj57d3c000076ei38vwnv35",
]

OUTPUT_PATH_CONFIG = "./exports/config"
if not os.path.exists(OUTPUT_PATH_CONFIG):
    os.makedirs(OUTPUT_PATH_CONFIG)


# Add function to create new connection
def create_connection():
    conn = psycopg2.connect(
        dbname=os.getenv("DATABASE_NAME"),
        user=os.getenv("DATABASE_USERNAME"),
        password=os.getenv("DATABASE_PASSWORD"),
        host=os.getenv("DATABASE_HOST"),
        port=os.getenv("DATABASE_PORT"),
    )
    # Set work_mem for this session
    with conn.cursor() as cur:
        cur.execute("SET work_mem = '2048MB'")  # Adjust size as needed
    conn.commit()
    return conn


def write_batch_to_jsonl(
    table_name, columns, where_clause, output_file, expected_num_lines=None
):
    # Create new connection instead of getting from pool
    conn = create_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if columns is not None:
                column_list = ", ".join(f'"{c}"' for c in columns)
            else:
                column_list = "*"

            # # Use COPY command for faster data transfer
            # copy_command = f"""
            # COPY (
            #     SELECT row_to_json(t)
            #     FROM (
            #         SELECT {column_list}
            #         FROM "{table_name}"
            #         WHERE {where_clause}
            #     ) t
            # ) TO STDOUT
            # """

            # with open(output_file, "w", buffering=8192 * 1024) as f:
            #     cur.copy_expert(copy_command, f)
            command = f"""
            SELECT {column_list}
            FROM "{table_name}"
            WHERE {where_clause}
            """
            # print("Executing query...")
            cur.execute(command)
            # print("query executed")

            # Open file for writing
            with open(output_file, "w") as f:
                # print("opened file")
                total_rows = 0

                # Fetch all rows at once
                rows = cur.fetchall()
                total_rows = len(rows)
                # print(f"Writing {total_rows} rows...")

                # Write all rows at once instead of line by line
                f.write("\n".join(json.dumps(row) for row in rows))
                if rows:  # Add final newline if there are rows
                    f.write("\n")

                # print(f"Finished writing {total_rows} rows to {output_file}")

                if (
                    expected_num_lines is not None
                    and total_rows < expected_num_lines
                    and total_rows != 1  # empty source
                ):
                    raise Exception(
                        f"Expected {expected_num_lines} lines in {output_file}, but got {total_rows}"
                    )
    finally:
        conn.close()


def getFormattedJsonlRow(row):
    row_dict = dict(row)
    for key, value in row_dict.items():
        if isinstance(value, datetime.datetime):
            row_dict[key] = value.isoformat()
        if key == "creatorId":
            row_dict[key] = DEFAULT_CREATOR_ID
    return row_dict


def writeRowsToJsonlFile(rows, file):
    to_write = ""
    for row in rows:
        to_write += json.dumps(getFormattedJsonlRow(row)) + "\n"
    file.write(to_write)


def export_model(model_id, source, OUTPUT_PATH_BASE):
    conn = create_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        print(f"Exporting model {model_id}")
        cur.execute(
            """
            SELECT * FROM "Model" WHERE "id" = %s
            """,
            (model_id,),
        )

        row = cur.fetchone()

        with open(f"{OUTPUT_PATH_BASE}/{source}/model.jsonl", "w") as f:
            writeRowsToJsonlFile([row], f)
    finally:
        cur.close()
        conn.close()


def export_source(model_id, sourceset_name, source_id, OUTPUT_PATH_BASE):
    path = f"{OUTPUT_PATH_BASE}/{source_id}/source.jsonl"
    if os.path.exists(path):
        return
    conn = create_connection()
    try:
        print(f"Exporting source {source_id} for model {model_id}")
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT * FROM "Source" WHERE "modelId" = %s AND "id" = %s
            """,
            (model_id, source_id),
        )

        row = cur.fetchone()

        if row is not None:
            with open(path, "w") as f:
                writeRowsToJsonlFile([row], f)

        print(f"Exporting sourceset {sourceset_name} for model {model_id}")
        cur.execute(
            """
            SELECT * FROM "SourceSet" WHERE "modelId" = %s AND "name" = %s
            """,
            (model_id, sourceset_name),
        )

        row = cur.fetchone()

        if row is not None:
            with open(f"{OUTPUT_PATH_BASE}/{source_id}/sourceset.jsonl", "w") as f:
                writeRowsToJsonlFile([row], f)
    finally:
        cur.close()
        conn.close()


def export_release(release_id, source, OUTPUT_PATH_BASE):
    path = f"{OUTPUT_PATH_BASE}/{source}/release.jsonl"
    if os.path.exists(path):
        return
    conn = create_connection()
    try:
        print(f"Exporting release {release_id}")
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT * FROM "SourceRelease" WHERE "name" = %s
            """,
            (release_id,),
        )

        row = cur.fetchone()

        if row is not None:
            with open(path, "w") as f:
                writeRowsToJsonlFile([row], f)
    finally:
        cur.close()
        conn.close()


def export_config():
    conn = create_connection()
    try:
        print("Exporting config")
        if not os.path.exists(OUTPUT_PATH_CONFIG):
            os.makedirs(OUTPUT_PATH_CONFIG)

        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute(
            """
            SELECT * FROM "ExplanationType"
            """
        )
        rows = cur.fetchall()
        with open(f"{OUTPUT_PATH_CONFIG}/explanation_type.jsonl", "w") as f:
            writeRowsToJsonlFile(rows, f)

        cur.execute(
            """
            SELECT * FROM "ExplanationModelType"
            """
        )
        rows = cur.fetchall()
        with open(f"{OUTPUT_PATH_CONFIG}/explanation_model_type.jsonl", "w") as f:
            writeRowsToJsonlFile(rows, f)

        cur.execute(
            """
            SELECT * FROM "ExplanationScoreModel"
            """
        )
        rows = cur.fetchall()
        with open(f"{OUTPUT_PATH_CONFIG}/explanation_score_model.jsonl", "w") as f:
            writeRowsToJsonlFile(rows, f)

        cur.execute(
            """
            SELECT * FROM "ExplanationScoreType"
            """
        )
        rows = cur.fetchall()
        with open(f"{OUTPUT_PATH_CONFIG}/explanation_score_type.jsonl", "w") as f:
            writeRowsToJsonlFile(rows, f)

        cur.execute(
            """
            SELECT * FROM "EvalType"
            """
        )
        rows = cur.fetchall()
        with open(f"{OUTPUT_PATH_CONFIG}/eval_type.jsonl", "w") as f:
            writeRowsToJsonlFile(rows, f)
    finally:
        cur.close()
        conn.close()


def export_features(model_id, source, OUTPUT_PATH_BASE):
    conn = create_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        export_path = f"{OUTPUT_PATH_BASE}/{source}/features"
        if not os.path.exists(export_path):
            os.makedirs(export_path)

        cur.execute(
            """
            SELECT COUNT(*) as count
            FROM "Neuron"
            WHERE "modelId" = %s AND "layer" = %s
            """,
            (model_id, source),
        )
        result = cur.fetchone()

        if result is None:
            total_count = 0
        else:
            total_count = result["count"]

        print(f"Total count: {total_count}")
        num_batches = (
            total_count + FEATURES_EXPLANATIONS_BATCH_SIZE - 1
        ) // FEATURES_EXPLANATIONS_BATCH_SIZE

        print(f"Number of batches: {num_batches}")
    finally:
        cur.close()
        conn.close()

    for batch in tqdm(range(num_batches)):
        print(f"Exporting batch {batch + 1}/{num_batches}")
        offset = batch * FEATURES_EXPLANATIONS_BATCH_SIZE

        output_file = f"{export_path}/batch-{batch}.jsonl"
        # if the zipped file exists, skip
        if os.path.exists(output_file + ".gz"):
            # print(f"{output_file} Skipping batch {batch + 1}/{num_batches} because it already exists")
            continue
        if os.path.exists(output_file):
            # print(f"{output_file} Skipping batch {batch + 1}/{num_batches} because it already exists")
            continue
        write_batch_to_jsonl(
            "Neuron",
            [
                "modelId",
                "layer",
                "index",
                "maxActApprox",
                "creatorId",
                "frac_nonzero",
                "freq_hist_data_bar_heights",
                "freq_hist_data_bar_values",
                "logits_hist_data_bar_heights",
                "logits_hist_data_bar_values",
                "neuron_alignment_indices",
                "neuron_alignment_l1",
                "neuron_alignment_values",
                "correlated_neurons_indices",
                "correlated_neurons_l1",
                "correlated_neurons_pearson",
                "decoder_weights_dist",
                "neg_str",
                "neg_values",
                "pos_str",
                "pos_values",
                "sourceSetName",
                "topkCosSimIndices",
                "topkCosSimValues",
                "vector",
                "vectorLabel",
                "hookName",
                "vectorDefaultSteerStrength",
                "hasVector",
            ],
            f"\"modelId\" = '{model_id}' AND \"layer\" = '{source}' ORDER BY index LIMIT {FEATURES_EXPLANATIONS_BATCH_SIZE} OFFSET {offset}",
            output_file,
            expected_num_lines=(
                FEATURES_EXPLANATIONS_BATCH_SIZE if batch < num_batches - 1 else None
            ),
        )
        replace_author_and_creator_with_default_creator_id(output_file)

        print(f"{output_file} Exported batch {batch + 1}/{num_batches}")


def replace_author_and_creator_with_default_creator_id(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

        filtered_lines = []
        for line in lines:
            data = json.loads(line)
            if "creatorId" in data:
                data["creatorId"] = DEFAULT_CREATOR_ID
            if "authorId" in data:
                data["authorId"] = DEFAULT_CREATOR_ID
            filtered_lines.append(json.dumps(data) + "\n")

    with open(file_path, "w") as f:
        f.writelines(filtered_lines)


def filter_author_and_creator_in_activations(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

        filtered_lines = []
        for line in lines:
            data = json.loads(line)
            if (
                data.get("creatorId") in PUBLIC_ACTIVATION_USER_IDS
                or data.get("authorId") in PUBLIC_ACTIVATION_USER_IDS
            ):
                if "creatorId" in data:
                    data["creatorId"] = DEFAULT_CREATOR_ID
                if "authorId" in data:
                    data["authorId"] = DEFAULT_CREATOR_ID
                filtered_lines.append(json.dumps(data) + "\n")

    with open(file_path, "w") as f:
        f.writelines(filtered_lines)


def export_explanations(model_id, source, OUTPUT_PATH_BASE):

    # ============ EXPLANATIONS EXPORT ============

    export_path = f"{OUTPUT_PATH_BASE}/{source}/explanations"
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    conn = create_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Get total count first
    cur.execute(
        """
        SELECT COUNT(*)
        FROM "Explanation"
        WHERE "modelId" = %s AND "layer" = %s
        """,
        (model_id, source),
    )
    result = cur.fetchone()
    if result is None:
        total_count = 0
    else:
        total_count = result["count"]

    print(f"Total count: {total_count}")
    num_batches = (
        total_count + FEATURES_EXPLANATIONS_BATCH_SIZE - 1
    ) // FEATURES_EXPLANATIONS_BATCH_SIZE

    conn.close()

    for batch in tqdm(range(num_batches)):
        offset = batch * FEATURES_EXPLANATIONS_BATCH_SIZE
        output_file = f"{export_path}/batch-{batch}.jsonl"
        # if the zipped file exists, skip
        if os.path.exists(output_file + ".gz"):
            # print(f"Skipping batch {batch + 1}/{num_batches} because it already exists")
            continue
        if os.path.exists(output_file):
            # print(f"Skipping batch {batch + 1}/{num_batches} because it already exists")
            continue
        write_batch_to_jsonl(
            "Explanation",
            [
                "id",
                "modelId",
                "layer",
                "index",
                "authorId",
                "description",
                "embedding",
                "typeName",
                "explanationModelName",
                "umap_x",
                "umap_y",
                "umap_cluster",
                "umap_log_feature_sparsity",
            ],
            f"\"modelId\" = '{model_id}' AND \"layer\" = '{source}' ORDER BY index LIMIT {FEATURES_EXPLANATIONS_BATCH_SIZE} OFFSET {offset}",
            output_file,
            expected_num_lines=(
                FEATURES_EXPLANATIONS_BATCH_SIZE if batch < num_batches - 1 else None
            ),
        )
        replace_author_and_creator_with_default_creator_id(output_file)
        print(f"{output_file} Exported batch {batch + 1}/{num_batches}")


def export_activations(model_id, source, OUTPUT_PATH_BASE):
    print(
        f"\n[Thread {threading.current_thread().name}] Starting export_activations for model {model_id} and source {source}"
    )

    export_path = f"{OUTPUT_PATH_BASE}/{source}/activations"
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    conn = create_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute(
            """
            SELECT COUNT(*)
            FROM "Neuron"
            WHERE "modelId" = %s AND "layer" = %s
            """,
            (model_id, source),
        )
        result = cur.fetchone()
        if result is None:
            total_count = 0
        else:
            total_count = result["count"]

        print(f"Total count: {total_count}")
        num_batches = (
            total_count + ACTIVATIONS_BATCH_SIZE - 1
        ) // ACTIVATIONS_BATCH_SIZE

        print(f"Number of batches: {num_batches}")

    conn.close()

    for batch in tqdm(range(num_batches)):
        offset = batch * ACTIVATIONS_BATCH_SIZE
        output_file = f"{export_path}/batch-{batch}.jsonl"
        # if the zipped file exists, skip
        if os.path.exists(output_file + ".gz"):
            # print(f"Skipping batch {batch + 1}/{num_batches} because it already exists")
            continue
        if os.path.exists(output_file):
            # print(f"Skipping batch {batch + 1}/{num_batches} because it already exists")
            continue
        write_batch_to_jsonl(
            "Activation",
            [
                "id",
                "modelId",
                "layer",
                "index",
                "creatorId",
                "dataSource",
                "tokens",
                "maxValue",
                "minValue",
                "values",
                "maxValueTokenIndex",
                "dfaValues",
                "dfaMaxValue",
                "dfaTargetIndex",
                "binContains",
                "binMin",
                "binMax",
            ],
            f"\"modelId\" = '{model_id}' AND \"layer\" = '{source}' AND index IN (SELECT generate_series::text AS text FROM generate_series({offset}, {min(offset + ACTIVATIONS_BATCH_SIZE - 1, total_count - 1)}))",
            output_file,
            expected_num_lines=(
                ACTIVATIONS_BATCH_SIZE if batch < num_batches - 1 else None
            ),
        )
        filter_author_and_creator_in_activations(output_file)
        print(f"{output_file} Exported batch {batch + 1}/{num_batches}")


def compress_files(model_id, source, OUTPUT_PATH_BASE):
    print(f"Compressing files for model {model_id} and source {source}")
    export_path = f"{OUTPUT_PATH_BASE}/{source}"

    # iterate through subdirectories of export_path
    def compress_dir(dir_path):
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path):
                compress_dir(item_path)
            elif item.endswith(".jsonl") and os.path.dirname(item_path) != export_path:
                print(f"Compressing file {item_path}")
                with open(item_path, "rb") as f:
                    with open(item_path + ".gz", "wb") as f_out:
                        f_out.write(gzip.compress(f.read(), compresslevel=5))
                os.remove(item_path)

    compress_dir(export_path)


def get_all_releases():
    import requests

    response = requests.post("https://neuronpedia.org/api/global")
    return response.json()["releases"]


def process_source_sync(release, sourceset, source):
    """Process a single source synchronously"""
    try:
        RELEASE_ID = release["name"]
        SOURCE_SET = sourceset["name"]
        MODEL_ID = source["modelId"]
        source_id = source["id"]

        print(
            f"\n[Process {os.getpid()}] Starting export for source {source_id} in model {MODEL_ID}"
        )

        OUTPUT_PATH_BASE = f"./exports/{MODEL_ID}"
        os.makedirs(f"{OUTPUT_PATH_BASE}/{source_id}", exist_ok=True)

        # Run operations synchronously since we're in a separate process
        export_model(MODEL_ID, source_id, OUTPUT_PATH_BASE)
        if RELEASE_ID is not None:
            export_release(RELEASE_ID, source_id, OUTPUT_PATH_BASE)
        export_source(MODEL_ID, SOURCE_SET, source_id, OUTPUT_PATH_BASE)
        # GPU hosts are not exported. They belong to the environment that runs
        # them, and an importer registers its own with `make host-add`.
        export_features(MODEL_ID, source_id, OUTPUT_PATH_BASE)
        export_explanations(MODEL_ID, source_id, OUTPUT_PATH_BASE)
        export_activations(MODEL_ID, source_id, OUTPUT_PATH_BASE)
        compress_files(MODEL_ID, source_id, OUTPUT_PATH_BASE)

    except Exception as e:
        print(f"Error processing source {source_id}: {str(e)}")
        raise e


def main(
    num_parallel_export_jobs: int = typer.Option(
        1,
        "--parallel-jobs",
        "-j",
        help="Number of parallel export jobs to run",
        prompt=True,
    ),
    limit_to_release: str = typer.Option(
        None,
        "--release",
        "-r",
        help="Limit export to a specific release name",
        prompt=True,
        prompt_required=False,
    ),
    limit_to_model_id: str = typer.Option(
        None,
        "--model-id",
        "-m",
        help="Limit export to a specific model id",
        prompt=True,
        prompt_required=False,
    ),
    limit_to_source_id: str = typer.Option(
        None,
        "--source-id",
        "-s",
        help="Limit export to a specific source id",
        prompt=True,
        prompt_required=False,
    ),
):
    """
    Export the database contents.
    """
    if not os.path.exists("./exports"):
        os.makedirs("./exports")

    export_config()
    releases = get_all_releases()

    if limit_to_release:
        releases = [r for r in releases if r["name"] == limit_to_release]
        print("\nAvailable releases:")
        for r in releases:
            print(f"\nRelease: {r['name']}")
            print(json.dumps(r, indent=2, default=str))
        if not releases:
            print(f"No release found with name: {limit_to_release}")
            return

    # Create a list of all source processing tasks
    tasks = []
    for release in sorted(releases, key=lambda x: x["name"]):
        for sourceset in sorted(release["sourceSets"], key=lambda x: x["name"]):
            if limit_to_model_id:
                if sourceset["modelId"] != limit_to_model_id:
                    continue
            for source in sorted(sourceset["sources"], key=lambda x: x["id"]):
                if limit_to_source_id:
                    if source["id"] != limit_to_source_id:
                        continue
                tasks.append((release, sourceset, source))

    with ThreadPoolExecutor(max_workers=num_parallel_export_jobs) as executor:
        futures = [
            executor.submit(process_source_sync, release, sourceset, source)
            for release, sourceset, source in tasks
        ]

        # Wait for all tasks to complete
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing sources",
        ):
            try:
                future.result()  # This will raise any exceptions that occurred
            except Exception as e:
                print(f"Task failed with error: {str(e)}")


if __name__ == "__main__":
    typer.run(main)
