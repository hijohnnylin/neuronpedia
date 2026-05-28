# Based on HeadVis by Luger, Kamath et al (Anthropic 2026)
# https://transformer-circuits.pub/2026/headvis/index.html
# https://github.com/anthropics/headvis
#
# Usage Example:
# poetry run python compute-head-metrics.py \
# --model-name google/gemma-3-1b-pt \
# --dataset-name monology/pile-uncopyrighted \
# --n-sequences 16384
# --batch-size 16

import argparse
import json
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for ad-hoc utility runs
    tqdm = None


DEFAULT_OUTPUT_DIR = "head-metrics"
DEFAULT_MIN_FREE_VRAM_GB = 12.0
DEFAULT_INDUCTION_ATTENTION_THRESHOLD = 0.01


@dataclass
class HeadMetricsConfig:
    model_name: str
    dataset_name: str
    dataset_config_name: str | None
    dataset_split: str
    dataset_text_field: str
    n_sequences: int
    seq_len: int
    batch_size: int
    output_dir: str
    output_file: str
    device: str
    dtype: str
    attn_implementation: str
    induction_attention_threshold: float
    trust_remote_code: bool
    min_free_vram_gb: float
    streaming: bool
    model_cache_dir: str
    delete_model_cache: bool
    revision: str | None
    dataset_revision: str | None
    created_at: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute previous-token and induction attention scores for every "
            "attention head in a causal LM."
        )
    )
    parser.add_argument("--model-name", required=True, help="Hugging Face model name.")
    parser.add_argument(
        "--dataset-name", required=True, help="Hugging Face dataset name."
    )
    parser.add_argument(
        "--dataset-config-name",
        default=None,
        help="Optional Hugging Face dataset config/subset name.",
    )
    parser.add_argument("--dataset-split", default="train", help="Dataset split to use.")
    parser.add_argument(
        "--dataset-text-field",
        default="text",
        help="Field containing text in each dataset row.",
    )
    parser.add_argument(
        "--n-sequences",
        type=int,
        default=2000,
        help="Number of non-empty sequences to process.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Maximum tokenized sequence length.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of sequences per model forward pass.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the metrics JSON is written.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run on. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype. 'auto' lets transformers choose.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help="Transformers attention implementation. Use eager to return weights.",
    )
    parser.add_argument(
        "--induction-attention-threshold",
        type=float,
        default=DEFAULT_INDUCTION_ATTENTION_THRESHOLD,
        help=(
            "Zero induction attention values below this threshold before summing. "
            "Use 0 to disable thresholding."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to transformers/datasets loaders.",
    )
    parser.add_argument(
        "--min-free-vram-gb",
        type=float,
        default=DEFAULT_MIN_FREE_VRAM_GB,
        help="Warn when free CUDA VRAM is below this threshold.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable Hugging Face dataset streaming.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help=(
            "Directory for Hugging Face model and tokenizer downloads. By default, "
            "uses a temporary directory that is deleted when the run finishes."
        ),
    )
    parser.add_argument(
        "--keep-hf-cache",
        action="store_true",
        help="Keep the temporary Hugging Face cache directory after the run.",
    )
    parser.add_argument(
        "--revision", default=None, help="Optional model revision to load."
    )
    parser.add_argument(
        "--dataset-revision", default=None, help="Optional dataset revision to load."
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print top heads for each metric after writing the JSON output.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def sanitize_filename_part(value: str) -> str:
    value = value.replace("/", "-")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str) -> torch.dtype | str:
    if dtype_arg == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_arg]


def get_model_config_value(model: AutoModelForCausalLM, field_name: str) -> Any:
    for config in (model.config, getattr(model.config, "text_config", None)):
        if config is not None and hasattr(config, field_name):
            return getattr(config, field_name)
    raise AttributeError(
        f"Could not find {field_name!r} on model config or nested text_config."
    )


def warn_if_low_vram(device: torch.device, min_free_vram_gb: float) -> None:
    if device.type != "cuda":
        print("WARNING: CUDA is not available or not selected; this will be very slow.")
        return

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_gb = free_bytes / 1024**3
    total_gb = total_bytes / 1024**3
    print(f"CUDA VRAM: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
    if free_gb < min_free_vram_gb:
        print(
            "WARNING: free GPU VRAM is below "
            f"{min_free_vram_gb:.1f} GB. This may OOM; lower --seq-len or use a "
            "smaller model."
        )


def prepare_model_cache(args: argparse.Namespace) -> tuple[str, bool]:
    if args.hf_cache_dir is not None:
        cache_dir = os.path.abspath(os.path.expanduser(args.hf_cache_dir))
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Using Hugging Face model cache directory, will delete after this run: {cache_dir}")
        return cache_dir, True

    cache_dir = tempfile.mkdtemp(prefix="headvis-model-cache-")
    if args.keep_hf_cache:
        print(f"Using temporary Hugging Face model cache directory: {cache_dir}")
        print("Keeping temporary Hugging Face model cache after this run.")
        return cache_dir, False

    print(f"Using temporary Hugging Face model cache directory: {cache_dir}")
    print("Temporary Hugging Face model cache will be deleted after this run.")
    return cache_dir, True


def configure_model_cache_environment(cache_dir: str) -> dict[str, str | None]:
    cache_env_vars = ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE")
    previous_values = {name: os.environ.get(name) for name in cache_env_vars}
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "hub")
    os.environ["HF_XET_CACHE"] = os.path.join(cache_dir, "xet")
    return previous_values


def restore_cache_environment(previous_values: dict[str, str | None]) -> None:
    for name, value in previous_values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def iter_texts(dataset: Iterable[dict[str, Any]], text_field: str) -> Iterable[str]:
    for row in dataset:
        text = row.get(text_field)
        if isinstance(text, str) and text.strip():
            yield text


def build_induction_indices(
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    token_positions: dict[int, list[int]] = defaultdict(list)
    query_indices: list[int] = []
    shifted_key_indices: list[int] = []
    repeated_query_positions = 0

    for query_position, token_id in enumerate(tokens.tolist()):
        previous_positions = token_positions[token_id]
        has_prior_repeat = False
        for previous_position in previous_positions:
            shifted_key_position = previous_position + 1
            if shifted_key_position < len(tokens):
                query_indices.append(query_position)
                shifted_key_indices.append(shifted_key_position)
                has_prior_repeat = True
        if has_prior_repeat:
            repeated_query_positions += 1
        previous_positions.append(query_position)

    if not query_indices:
        empty = torch.empty(0, dtype=torch.long, device=tokens.device)
        return empty, empty, repeated_query_positions

    return (
        torch.tensor(query_indices, dtype=torch.long, device=tokens.device),
        torch.tensor(shifted_key_indices, dtype=torch.long, device=tokens.device),
        repeated_query_positions,
    )


def compute_attention_metrics_for_batch(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    self_attention_sum: torch.Tensor,
    prev_token_sum: torch.Tensor,
    pattern_entropy_sum: torch.Tensor,
    qk_distance_sum: torch.Tensor,
    qk_distance_squared_sum: torch.Tensor,
    induction_sum: torch.Tensor,
    attention_position_count: torch.Tensor,
    induction_count: torch.Tensor,
    induction_attention_threshold: float,
) -> tuple[int, int]:
    with torch.inference_mode():
        output = model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )

    attentions = output.attentions
    if attentions is None:
        raise RuntimeError(
            "Model did not return attention weights. Try "
            "--attn-implementation eager."
        )

    sequence_lengths = attention_mask.sum(dim=1)
    valid_sequence_mask = sequence_lengths >= 2
    valid_sequences = int(valid_sequence_mask.sum().item())
    if valid_sequences == 0:
        del output, attentions
        return 0, 0

    total_attention_positions = sequence_lengths[valid_sequence_mask].sum()
    total_prev_positions = int((sequence_lengths[valid_sequence_mask] - 1).sum().item())
    attention_position_count += total_attention_positions

    _, sequence_length = attention_mask.shape
    positions = torch.arange(sequence_length, device=input_ids.device)
    query_mask = (
        (positions[None, :] < sequence_lengths[:, None])
        & valid_sequence_mask[:, None]
    )
    pair_mask = query_mask[:, :, None] & query_mask[:, None, :]
    prev_position_mask = positions[1:][None, :] < sequence_lengths[:, None]
    qk_distance = (positions[:, None] - positions[None, :]).abs().float()
    qk_distance_squared = qk_distance.square()

    for layer_idx, attention in enumerate(attentions):
        layer_attention_float = attention.float()
        self_attention_sum[layer_idx] += (
            attention.diagonal(dim1=-2, dim2=-1)
            .float()
            .masked_fill(~query_mask[:, None, :], 0.0)
            .sum(dim=(0, 2))
        )
        prev_token_sum[layer_idx] += (
            attention.diagonal(offset=-1, dim1=-2, dim2=-1)
            .float()
            .masked_fill(~prev_position_mask[:, None, :], 0.0)
            .sum(dim=(0, 2))
        )
        pattern_entropy_sum[layer_idx] += (
            -(
                layer_attention_float
                * torch.log(layer_attention_float.clamp_min(torch.finfo(torch.float32).tiny))
            )
            .masked_fill(~pair_mask[:, None, :, :], 0.0)
            .sum(dim=(0, 2, 3))
        )
        qk_distance_sum[layer_idx] += (
            layer_attention_float
            .masked_fill(~pair_mask[:, None, :, :], 0.0)
            .mul(qk_distance)
            .sum(dim=(0, 2, 3))
        )
        qk_distance_squared_sum[layer_idx] += (
            layer_attention_float
            .masked_fill(~pair_mask[:, None, :, :], 0.0)
            .mul(qk_distance_squared)
            .sum(dim=(0, 2, 3))
        )

    sequence_lengths_list = sequence_lengths.tolist()
    for batch_idx, item_sequence_length in enumerate(sequence_lengths_list):
        item_sequence_length = int(item_sequence_length)
        if item_sequence_length < 2:
            continue

        tokens = input_ids[batch_idx, :item_sequence_length]
        (
            induction_queries,
            induction_shifted_keys,
            repeated_query_positions,
        ) = build_induction_indices(tokens)
        induction_count += repeated_query_positions
        if induction_queries.numel() == 0:
            continue

        for layer_idx, attention in enumerate(attentions):
            layer_attention = attention[
                batch_idx, :, :item_sequence_length, :item_sequence_length
            ]
            induction_attention = layer_attention[
                :, induction_queries, induction_shifted_keys
            ]
            if induction_attention_threshold > 0:
                induction_attention = induction_attention.masked_fill(
                    induction_attention < induction_attention_threshold, 0.0
                )
            induction_sum[layer_idx] += (
                induction_attention.sum(dim=-1).float()
            )

    del output, attentions
    return valid_sequences, total_prev_positions


def load_text_dataset(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    dataset_kwargs: dict[str, Any] = {
        "path": args.dataset_name,
        "name": args.dataset_config_name,
        "split": args.dataset_split,
        "streaming": not args.no_streaming,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.dataset_revision is not None:
        dataset_kwargs["revision"] = args.dataset_revision
    return load_dataset(**dataset_kwargs)


def build_config(
    args: argparse.Namespace,
    device: torch.device,
    output_file: str,
    model_cache_dir: str,
    delete_model_cache: bool,
) -> HeadMetricsConfig:
    return HeadMetricsConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_split=args.dataset_split,
        dataset_text_field=args.dataset_text_field,
        n_sequences=args.n_sequences,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        output_file=output_file,
        device=str(device),
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        induction_attention_threshold=args.induction_attention_threshold,
        trust_remote_code=args.trust_remote_code,
        min_free_vram_gb=args.min_free_vram_gb,
        streaming=not args.no_streaming,
        model_cache_dir=model_cache_dir,
        delete_model_cache=delete_model_cache,
        revision=args.revision,
        dataset_revision=args.dataset_revision,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def nan_if_zero_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    result = torch.full_like(numerator, math.nan, dtype=torch.float32)
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1), result)


def print_top_heads(metric_name: str, scores: torch.Tensor, top_k: int = 3) -> None:
    flat_scores = scores.flatten()
    valid_scores = torch.nan_to_num(flat_scores, nan=-math.inf)
    top_values, top_indices = torch.topk(valid_scores, k=min(top_k, flat_scores.numel()))

    print(f"Top {len(top_values)} {metric_name}:")
    for rank, (value, flat_index) in enumerate(zip(top_values, top_indices), start=1):
        layer_idx = int(flat_index // scores.size(1))
        head_idx = int(flat_index % scores.size(1))
        print(
            f"  {rank}. layer={layer_idx}, head={head_idx}, "
            f"score={float(value):.6f}"
        )


def run_head_metrics(
    args: argparse.Namespace, model_cache_dir: str, delete_model_cache: bool
) -> None:
    device = resolve_device(args.device)
    warn_if_low_vram(device, args.min_free_vram_gb)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        (
            f"{sanitize_filename_part(args.model_name)}-"
            f"{sanitize_filename_part(args.dataset_name)}.json"
        ),
    )

    torch_dtype = resolve_dtype(args.dtype)
    model_kwargs: dict[str, Any] = {
        "attn_implementation": args.attn_implementation,
        "torch_dtype": torch_dtype,
        "trust_remote_code": args.trust_remote_code,
        "cache_dir": os.path.join(model_cache_dir, "hub"),
    }
    if args.revision is not None:
        model_kwargs["revision"] = args.revision

    tokenizer_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "cache_dir": os.path.join(model_cache_dir, "hub"),
    }
    if args.revision is not None:
        tokenizer_kwargs["revision"] = args.revision

    previous_cache_environment = configure_model_cache_environment(model_cache_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tokenizer_kwargs)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    finally:
        restore_cache_environment(previous_cache_environment)

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError(
            "Tokenizer does not define a pad token or eos token. Set a pad token "
            "before running with padded batches."
        )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    n_layers = get_model_config_value(model, "num_hidden_layers")
    n_heads = get_model_config_value(model, "num_attention_heads")
    accumulator_kwargs = {"device": device, "dtype": torch.float32}
    self_attention_sum = torch.zeros(n_layers, n_heads, **accumulator_kwargs)
    prev_token_sum = torch.zeros(n_layers, n_heads, **accumulator_kwargs)
    pattern_entropy_sum = torch.zeros(n_layers, n_heads, **accumulator_kwargs)
    qk_distance_sum = torch.zeros(n_layers, n_heads, **accumulator_kwargs)
    qk_distance_squared_sum = torch.zeros(n_layers, n_heads, **accumulator_kwargs)
    induction_sum = torch.zeros(n_layers, n_heads, **accumulator_kwargs)
    attention_position_count = torch.zeros(n_layers, 1, **accumulator_kwargs)
    prev_token_count = torch.zeros(n_layers, 1, **accumulator_kwargs)
    induction_count = torch.zeros(n_layers, 1, **accumulator_kwargs)

    dataset = load_text_dataset(args)
    text_iter = iter_texts(dataset, args.dataset_text_field)
    progress = (
        tqdm(total=args.n_sequences, desc="Computing head metrics", unit="seq")
        if tqdm is not None
        else None
    )
    processed_sequences = 0

    try:
        while processed_sequences < args.n_sequences:
            batch_texts: list[str] = []
            while (
                len(batch_texts) < args.batch_size
                and processed_sequences + len(batch_texts) < args.n_sequences
            ):
                try:
                    batch_texts.append(next(text_iter))
                except StopIteration:
                    break

            if not batch_texts:
                break

            tokenized = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.seq_len,
            )
            input_ids = tokenized.input_ids.to(device)
            attention_mask = tokenized.attention_mask.to(device)

            valid_sequences, prev_positions_count = compute_attention_metrics_for_batch(
                model,
                input_ids,
                attention_mask,
                self_attention_sum,
                prev_token_sum,
                pattern_entropy_sum,
                qk_distance_sum,
                qk_distance_squared_sum,
                induction_sum,
                attention_position_count,
                induction_count,
                args.induction_attention_threshold,
            )
            prev_token_count += prev_positions_count
            processed_sequences += valid_sequences

            if progress is not None:
                progress.update(valid_sequences)
            elif processed_sequences % 10 == 0:
                print(f"Processed {processed_sequences}/{args.n_sequences} sequences")

            del tokenized, input_ids, attention_mask
    finally:
        if progress is not None:
            progress.close()

    self_attention_score = nan_if_zero_divide(
        self_attention_sum, attention_position_count
    )
    prev_token_score = nan_if_zero_divide(prev_token_sum, prev_token_count)
    pattern_entropy = nan_if_zero_divide(pattern_entropy_sum, attention_position_count)
    qk_distance = nan_if_zero_divide(qk_distance_sum, attention_position_count)
    qk_distance_second_moment = nan_if_zero_divide(
        qk_distance_squared_sum, attention_position_count
    )
    qk_distance_variance = qk_distance_second_moment - qk_distance.square()
    induction_score = nan_if_zero_divide(induction_sum, induction_count)

    config = asdict(
        build_config(args, device, output_file, model_cache_dir, delete_model_cache)
    )
    config["actual_sequences_processed"] = processed_sequences
    config["num_hidden_layers"] = n_layers
    config["num_attention_heads"] = n_heads

    metrics = {
        "self_attention_score": self_attention_score.tolist(),
        "prev_token_score": prev_token_score.tolist(),
        "pattern_entropy": pattern_entropy.tolist(),
        "qk_distance": qk_distance.tolist(),
        "qk_distance_variance": qk_distance_variance.tolist(),
        "induction_score": induction_score.tolist(),
        "counts": {
            "attention_positions": attention_position_count.squeeze(-1).tolist(),
            "prev_token_positions": prev_token_count.squeeze(-1).tolist(),
            "induction_positions": induction_count.squeeze(-1).tolist(),
        },
    }

    with open(output_file, "w") as f:
        json.dump({"config": config, "metrics": metrics}, f, indent=2)
        f.write("\n")

    print(f"Wrote head metrics to {output_file}")
    if args.print_summary:
        print_top_heads("self_attention_score", self_attention_score)
        print_top_heads("prev_token_score", prev_token_score)
        print_top_heads("pattern_entropy", pattern_entropy)
        print_top_heads("qk_distance", qk_distance)
        print_top_heads("qk_distance_variance", qk_distance_variance)
        print_top_heads("induction_score", induction_score)


def main() -> None:
    args = parse_args()
    model_cache_dir, delete_model_cache = prepare_model_cache(args)
    try:
        run_head_metrics(args, model_cache_dir, delete_model_cache)
    finally:
        if delete_model_cache:
            print(f"Deleting temporary Hugging Face model cache: {model_cache_dir}")
            shutil.rmtree(model_cache_dir, ignore_errors=True)
 

if __name__ == "__main__":
    main()
