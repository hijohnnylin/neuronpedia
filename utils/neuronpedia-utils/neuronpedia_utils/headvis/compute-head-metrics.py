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
import gc
import json
import math
import os
import re
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
    trust_remote_code: bool
    min_free_vram_gb: float
    streaming: bool
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
        default=1024,
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


def iter_texts(dataset: Iterable[dict[str, Any]], text_field: str) -> Iterable[str]:
    for row in dataset:
        text = row.get(text_field)
        if isinstance(text, str) and text.strip():
            yield text


def build_induction_indices(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    token_positions: dict[int, list[int]] = defaultdict(list)
    query_indices: list[int] = []
    shifted_key_indices: list[int] = []

    for query_position, token_id in enumerate(tokens.tolist()):
        previous_positions = token_positions[token_id]
        for previous_position in previous_positions:
            shifted_key_position = previous_position + 1
            if shifted_key_position < len(tokens):
                query_indices.append(query_position)
                shifted_key_indices.append(shifted_key_position)
        previous_positions.append(query_position)

    if not query_indices:
        empty = torch.empty(0, dtype=torch.long, device=tokens.device)
        return empty, empty

    return (
        torch.tensor(query_indices, dtype=torch.long, device=tokens.device),
        torch.tensor(shifted_key_indices, dtype=torch.long, device=tokens.device),
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
) -> tuple[int, int]:
    with torch.no_grad():
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

    sequence_lengths = attention_mask.sum(dim=1).tolist()
    valid_sequences = 0
    total_prev_positions = 0

    for batch_idx, sequence_length in enumerate(sequence_lengths):
        sequence_length = int(sequence_length)
        if sequence_length < 2:
            continue

        valid_sequences += 1
        total_prev_positions += sequence_length - 1
        tokens = input_ids[batch_idx, :sequence_length]
        previous_positions = torch.arange(
            0, sequence_length - 1, device=input_ids.device, dtype=torch.long
        )
        current_positions = torch.arange(
            1, sequence_length, device=input_ids.device, dtype=torch.long
        )
        positions = torch.arange(sequence_length, device=input_ids.device)
        qk_distance = (positions[:, None] - positions[None, :]).abs().float()
        qk_distance_squared = qk_distance.square()
        induction_queries, induction_shifted_keys = build_induction_indices(tokens)

        for layer_idx, attention in enumerate(attentions):
            layer_attention = attention[batch_idx, :, :sequence_length, :sequence_length]
            layer_attention_float = layer_attention.float()
            self_attention_sum[layer_idx] += (
                layer_attention[:, positions, positions]
                .sum(dim=-1)
                .detach()
                .float()
                .cpu()
            )
            prev_token_sum[layer_idx] += (
                layer_attention[:, current_positions, previous_positions]
                .sum(dim=-1)
                .detach()
                .float()
                .cpu()
            )
            pattern_entropy_sum[layer_idx] += (
                -(
                    layer_attention_float
                    * torch.log(layer_attention_float.clamp_min(torch.finfo(torch.float32).tiny))
                )
                .sum(dim=-1)
                .sum(dim=-1)
                .detach()
                .cpu()
            )
            qk_distance_sum[layer_idx] += (
                (layer_attention_float * qk_distance)
                .sum(dim=(-1, -2))
                .detach()
                .cpu()
            )
            qk_distance_squared_sum[layer_idx] += (
                (layer_attention_float * qk_distance_squared)
                .sum(dim=(-1, -2))
                .detach()
                .cpu()
            )

            if induction_queries.numel() > 0:
                induction_sum[layer_idx] += (
                    layer_attention[:, induction_queries, induction_shifted_keys]
                    .sum(dim=-1)
                    .detach()
                    .float()
                    .cpu()
                )
            attention_position_count[layer_idx] += sequence_length
            induction_count[layer_idx] += sequence_length

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


def build_config(args: argparse.Namespace, device: torch.device, output_file: str) -> HeadMetricsConfig:
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
        trust_remote_code=args.trust_remote_code,
        min_free_vram_gb=args.min_free_vram_gb,
        streaming=not args.no_streaming,
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


def main() -> None:
    args = parse_args()
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
    }
    if args.revision is not None:
        model_kwargs["revision"] = args.revision

    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if args.revision is not None:
        tokenizer_kwargs["revision"] = args.revision

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tokenizer_kwargs)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    self_attention_sum = torch.zeros(n_layers, n_heads, dtype=torch.float32)
    prev_token_sum = torch.zeros(n_layers, n_heads, dtype=torch.float32)
    pattern_entropy_sum = torch.zeros(n_layers, n_heads, dtype=torch.float32)
    qk_distance_sum = torch.zeros(n_layers, n_heads, dtype=torch.float32)
    qk_distance_squared_sum = torch.zeros(n_layers, n_heads, dtype=torch.float32)
    induction_sum = torch.zeros(n_layers, n_heads, dtype=torch.float32)
    attention_position_count = torch.zeros(n_layers, 1, dtype=torch.float32)
    prev_token_count = torch.zeros(n_layers, 1, dtype=torch.float32)
    induction_count = torch.zeros(n_layers, 1, dtype=torch.float32)

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
            )
            prev_token_count += prev_positions_count
            processed_sequences += valid_sequences

            if progress is not None:
                progress.update(valid_sequences)
            elif processed_sequences % 10 == 0:
                print(f"Processed {processed_sequences}/{args.n_sequences} sequences")

            del tokenized, input_ids, attention_mask
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
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

    config = asdict(build_config(args, device, output_file))
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
 

if __name__ == "__main__":
    main()
