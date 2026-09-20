# SPDX-License-Identifier: Apache-2.0
#
# Backfill the `environment` block into config.yaml files already published on
# the Hub. run-all-fit-lens.py writes it for new fits; the fits before
# 2026-09 did not record which library versions produced them, and that is
# the fact hijohnnylin/neuronpedia#235 needed (OLMo-3 under transformers
# 5.11.0 ran a different forward pass than the served model).
#
# Downloads only config.yaml (and a convergence.csv where the config's results
# block disagrees with it), never the weights. Writes into the local exports
# mirror, and with --create-pr opens one pull request on the Hub repo.
#
# Usage:
#   uv run python backfill-config-environment.py --dry-run      # show diffs
#   uv run python backfill-config-environment.py                # write locally
#   uv run python backfill-config-environment.py --create-pr    # and open a PR
"""Add an ``environment`` block to published jlens config.yaml files."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import difflib
import io
import re
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPORTS_DIR = SCRIPT_DIR.parent / "exports"
DEFAULT_REPO = "neuronpedia/jacobian-lens"

# Only configs written by run-all-fit-lens.py carry this line. Converted
# external lenses (convert-external-lens.py) were fit elsewhere, so their
# environment is unknown and they are left alone.
FIT_MARKER = "Fit via Neuronpedia run-all-fit-lens.py"

# What neuronpedia_utils/jlens/uv.lock resolved to when these lenses were fit
# (committed as a69f3394, fits generated 2026-06-11 to 2026-06-16; transformers
# 5.11.0 was released 2026-06-10 and 5.12.0 on 2026-06-12, after the lock).
HISTORICAL_ENVIRONMENT: dict[str, str | None] = {
    "transformers": "5.11.0",
    "torch": "2.11.0+cu128",
    "accelerate": "1.14.0",
    "datasets": "5.0.0",
    "python": None,
    "jlens_commit": None,
}

# Per-model facts that a reader of the config needs and that the fit could not
# have recorded. Keyed by np_model_id.
KNOWN_ISSUES: dict[str, str] = {
    "olmo-3-1025-7b": (
        "Fitted under transformers 5.11.0, which applied the YaRN rope_scaling to "
        "OLMo-3 sliding-window layers as well as full-attention layers. transformers "
        "5.13.0 (huggingface/transformers#46911) restored per-layer-type RoPE, so this "
        "lens encodes a forward pass that transformers >= 5.13 does not run. A refit "
        "is planned. See hijohnnylin/neuronpedia#235 and anthropics/jacobian-lens#15."
    ),
    "olmo-3-1125-32b": (
        "Fitted under transformers 5.11.0, which applied the YaRN rope_scaling to "
        "OLMo-3 sliding-window layers as well as full-attention layers. transformers "
        "5.13.0 (huggingface/transformers#46911) restored per-layer-type RoPE, so this "
        "lens encodes a forward pass that transformers >= 5.13 does not run. A refit "
        "is planned. See hijohnnylin/neuronpedia#235 and anthropics/jacobian-lens#15."
    ),
    "qwen3-32b": (
        "PARTIAL CHECKPOINT. Qwen3-32B_jacobian_lens.pt is a mid-fit jlens checkpoint, "
        "not a finished lens: keys jacobian_sum / n_done / next_idx / source_layers, "
        "raw float32 sums over n_done = 80 prompts (divide jacobian_sum by n_done for "
        "J). The fit stopped before min_prompts and did not converge. JacobianLens.load "
        "rejects the file. A full refit is planned. See "
        "https://huggingface.co/neuronpedia/jacobian-lens/discussions/3."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--exports-dir", default=str(DEFAULT_EXPORTS_DIR))
    parser.add_argument(
        "--dry-run", action="store_true", help="print diffs, write nothing"
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="open one pull request on the Hub with every changed config.yaml",
    )
    return parser.parse_args()


def yaml_scalar(value: Any) -> str:
    """Same scalar formatting as run-all-fit-lens.py, so both agree."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def read_scalar(text: str, key: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(key)}:\s*(.+?)\s*$", text, re.MULTILINE)
    if match is None:
        return None
    return match.group(1).strip().strip('"')


def csv_results(csv_text: str) -> dict[str, Any] | None:
    """The results block run-all-fit-lens.py derives from a convergence CSV."""
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    if not rows:
        return None
    last = rows[-1]
    return {
        "prompts_fitted": int(last["n_done"]),
        "final_identity_distance": float(last["identity_distance"]),
        "final_mean_rel_change": float(last["mean_rel_change"]),
    }


def replace_results_block(text: str, results: dict[str, Any]) -> str:
    """Swap the three-line ``results:`` block for one built from ``results``."""
    block = "results:\n" + "\n".join(
        f"  {key}: {yaml_scalar(value)}" for key, value in results.items()
    )
    pattern = re.compile(r"^results:\n(?:  .+\n)+", re.MULTILINE)
    if pattern.search(text) is None:
        raise ValueError("no results block to replace")
    return pattern.sub(block + "\n", text, count=1)


def environment_block(recorded: str) -> str:
    lines = ["environment:"]
    for key, value in HISTORICAL_ENVIRONMENT.items():
        lines.append(f"  {key}: {yaml_scalar(value)}")
    lines.append(f"  recorded: {yaml_scalar(recorded)}")
    return "\n".join(lines) + "\n"


def insert_before(text: str, anchor_key: str, block: str) -> str:
    """Insert ``block`` before the top-level ``anchor_key:`` line."""
    match = re.search(rf"^{re.escape(anchor_key)}:", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"no top-level {anchor_key!r} key")
    return text[: match.start()] + block + text[match.start() :]


def backfill(path: str, text: str, fetch_csv, today: str) -> tuple[str, list[str]]:
    """Return the updated config text and a list of what changed."""
    if FIT_MARKER not in text:
        return text, []
    if re.search(r"^environment:", text, re.MULTILINE):
        return text, []
    np_model_id = read_scalar(text, "np_model_id")
    if np_model_id is None:
        raise ValueError(f"{path}: no np_model_id")
    changes: list[str] = []
    header_notes: list[str] = []

    # A results block that disagrees with its own CSV is rewritten from the CSV,
    # which is what run-all-fit-lens.py would have produced.
    csv_text = fetch_csv(path)
    if csv_text is not None:
        results = csv_results(csv_text)
        recorded_n = read_scalar(text, "prompts_fitted")
        if results is not None and recorded_n != str(results["prompts_fitted"]):
            text = replace_results_block(text, results)
            changes.append(
                f"results rewritten from convergence.csv "
                f"(was prompts_fitted={recorded_n}, csv has {results['prompts_fitted']})"
            )
            header_notes.append(
                f"# Reconstructed {today}: the results block disagreed with the "
                f"convergence CSV and was rewritten from it."
            )

    recorded = (
        f"backfilled {today} from neuronpedia_utils/jlens/uv.lock as committed "
        f"with these fits (a69f3394); the fit did not record its environment"
    )
    text = insert_before(text, "results", environment_block(recorded))
    changes.append("environment block added")

    issue = KNOWN_ISSUES.get(np_model_id)
    if issue is not None:
        text = insert_before(text, "command", f"known_issue: {yaml_scalar(issue)}\n")
        changes.append("known_issue added")

    if not text.lstrip().startswith("#"):
        header_notes.insert(
            0,
            "# Jacobian lens fit — config.yaml for a Neuronpedia run-all-fit-lens.py fit. "
            "The original file had no generator header.",
        )
    if header_notes:
        text = "\n".join(header_notes) + "\n" + text
        changes.append("header note added")
    return text, changes


def main() -> None:
    args = parse_args()
    api = HfApi()
    exports_dir = Path(args.exports_dir).expanduser().resolve()
    today = dt.datetime.now(tz=dt.timezone.utc).date().isoformat()

    files = api.list_repo_files(args.repo_id, revision=args.revision)
    config_paths = sorted(
        f for f in files if re.fullmatch(r"[^/]+/jlens/[^/]+/config\.yaml", f)
    )
    csv_by_dir = {
        f.rsplit("/", 1)[0]: f for f in files if f.endswith("_convergence.csv")
    }
    print(f"{len(config_paths)} config.yaml files in {args.repo_id}@{args.revision}")

    def download(repo_path: str) -> str:
        local = hf_hub_download(
            args.repo_id, repo_path, revision=args.revision, repo_type="model"
        )
        return Path(local).read_text()

    def fetch_csv(config_path: str) -> str | None:
        csv_path = csv_by_dir.get(config_path.rsplit("/", 1)[0])
        return download(csv_path) if csv_path else None

    operations: list[CommitOperationAdd] = []
    for repo_path in config_paths:
        original = download(repo_path)
        try:
            updated, changes = backfill(repo_path, original, fetch_csv, today)
        except ValueError as exc:
            print(f"SKIP {repo_path}: {exc}", file=sys.stderr)
            continue
        if not changes:
            print(f"unchanged  {repo_path}")
            continue
        print(f"UPDATE     {repo_path}: {'; '.join(changes)}")
        if args.dry_run:
            sys.stdout.writelines(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    updated.splitlines(keepends=True),
                    fromfile=f"a/{repo_path}",
                    tofile=f"b/{repo_path}",
                )
            )
            continue
        local_path = exports_dir / repo_path
        if local_path.parent.is_dir():
            local_path.write_text(updated)
        else:
            print(f"  (no local mirror dir for {repo_path}; not written locally)")
        operations.append(
            CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=updated.encode())
        )

    if args.dry_run or not args.create_pr or not operations:
        return
    info = api.create_commit(
        repo_id=args.repo_id,
        repo_type="model",
        operations=operations,
        commit_message="Record fit environment in every config.yaml",
        commit_description=(
            "Adds an `environment` block (transformers, torch, accelerate, datasets) "
            "to each config.yaml written by run-all-fit-lens.py, backfilled from the "
            "uv.lock committed with these fits. New fits record it at fit time.\n\n"
            "Also marks the two OLMo-3 lenses as fitted under transformers 5.11.0, "
            "which applied YaRN to their sliding-window layers "
            "(hijohnnylin/neuronpedia#235), and rewrites qwen3-32b's config from its "
            "own convergence.csv, labelling the .pt as a partial checkpoint "
            "(discussion #3). No weights are changed."
        ),
        create_pr=True,
        revision=args.revision,
    )
    print(f"\nPull request: {info.pr_url}")


if __name__ == "__main__":
    main()
