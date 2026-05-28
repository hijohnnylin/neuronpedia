# Based on HeadVis by Luger, Kamath et al (Anthropic 2026)
# https://transformer-circuits.pub/2026/headvis/index.html
# https://github.com/anthropics/headvis
#
# Usage Example:
# uv run python compute-head-metrics.py \
# --model-name google/gemma-3-1b-pt \
# --dataset-name monology/pile-uncopyrighted \
# --n-sequences 4096 \
# --batch-size 16

import argparse
import bisect
import heapq
import json
import math
import os
import random
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for ad-hoc utility runs
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPORTS_DIR = SCRIPT_DIR.parent / "exports"
DEFAULT_MODEL_MAP_PATH = SCRIPT_DIR / "np_model_to_hf.json"
DEFAULT_MIN_FREE_VRAM_GB = 12.0
DEFAULT_INDUCTION_ATTENTION_THRESHOLD = 0.01

DEFAULT_N_INTERVALS = 5
DEFAULT_SAMPLES_PER_INTERVAL = 3
DEFAULT_SAMPLES_PER_TOP_INTERVAL = 10
DEFAULT_SPARSE_TOPK_PER_ROW = 8
DEFAULT_SPARSE_THRESHOLD = 0.005
DEFAULT_SAMPLE_SEED = 0
DEFAULT_WARMUP_CAP = 2000

ACTIVATION_HISTOGRAM_BINS = 50
QK_DISTANCE_BIN_EDGES = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
TOP_TOKENS_PER_SIDE = 50

METRIC_NAMES = [
    "self_attention_score",
    "prev_token_score",
    "pattern_entropy",
    "qk_distance",
    "qk_distance_variance",
    "induction_score",
]

METRIC_DESCRIPTIONS = {
    "self_attention_score": (
        "Mean attention from each query position to itself (the diagonal). "
        "Heads near 1 dump nearly all attention onto the current token."
    ),
    "prev_token_score": (
        "Mean attention to the immediately preceding token (first subdiagonal). "
        "High values mark previous-token / shift heads."
    ),
    "pattern_entropy": (
        "Mean Shannon entropy of each row of the attention matrix. "
        "High = dispersed attention; low = highly peaked."
    ),
    "qk_distance": (
        "Mean |q - k| weighted by attention mass. Captures how far back "
        "a head looks on average."
    ),
    "qk_distance_variance": (
        "Variance of |q - k| weighted by attention mass."
    ),
    "induction_score": (
        "Average attention from a query token to the position after a prior "
        "occurrence of the same token. High = induction head."
    ),
}


@dataclass
class HeadMetricsConfig:
    model_name: str
    np_model_id: str
    dataset_name: str
    dataset_config_name: str | None
    dataset_split: str
    dataset_text_field: str
    n_sequences: int
    seq_len: int
    batch_size: int
    exports_dir: str
    output_run_dir: str
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
    n_intervals: int
    samples_per_interval: int
    samples_per_top_interval: int
    sparse_topk_per_row: int
    sparse_threshold: float
    sample_seed: int
    warmup_size: int
    created_at: str


@dataclass
class SampledRecord:
    sequence_id: int
    max_activation: float
    flat_indices: np.ndarray
    values: np.ndarray
    seq_len: int


@dataclass
class SamplerConfig:
    n_intervals: int
    samples_per_interval: int
    samples_per_top_interval: int
    seed: int

    def capacity(self, bucket_idx: int) -> int:
        if self.n_intervals == 1:
            return max(self.samples_per_interval, self.samples_per_top_interval)
        if bucket_idx == self.n_intervals - 1:
            return self.samples_per_top_interval
        return self.samples_per_interval

    def is_top_bucket(self, bucket_idx: int) -> bool:
        return self.n_intervals == 1 or bucket_idx == self.n_intervals - 1


class BucketSlot:
    """Per-bucket sample container.

    For the top bucket: deterministic top-K by max_activation, kept in a
    min-heap so that the heap root is the weakest current sample. Ties
    broken by lower sequence_id (more stable).

    For other buckets: reservoir sampling (Algorithm R).
    """

    def __init__(self, capacity: int, deterministic: bool, rng: random.Random):
        self.capacity = capacity
        self.deterministic = deterministic
        self.rng = rng
        self.heap: list[tuple[float, int, int, SampledRecord]] = []
        self.items: list[SampledRecord] = []
        self.n_seen = 0
        self._push_counter = 0

    def add(self, record: SampledRecord) -> None:
        self.n_seen += 1
        if self.deterministic:
            self._push_counter += 1
            entry = (
                record.max_activation,
                -record.sequence_id,
                self._push_counter,
                record,
            )
            if len(self.heap) < self.capacity:
                heapq.heappush(self.heap, entry)
            elif entry > self.heap[0]:
                heapq.heapreplace(self.heap, entry)
        else:
            if len(self.items) < self.capacity:
                self.items.append(record)
            else:
                idx = self.rng.randrange(self.n_seen)
                if idx < self.capacity:
                    self.items[idx] = record

    def get_records(self) -> list[SampledRecord]:
        if self.deterministic:
            return [item[3] for item in sorted(self.heap, key=lambda x: -x[0])]
        return list(self.items)


class HeadSampler:
    """Streaming sampler for one (layer, head) pair."""

    def __init__(self, sampler_config: SamplerConfig, layer: int, head: int):
        self.cfg = sampler_config
        self.layer = layer
        self.head = head
        self.boundaries: list[float] | None = None
        seed_base = sampler_config.seed + 31 * (1000 * layer + 100 * head)
        self.bucket_slots = [
            BucketSlot(
                capacity=sampler_config.capacity(i),
                deterministic=sampler_config.is_top_bucket(i),
                rng=random.Random(seed_base + i),
            )
            for i in range(sampler_config.n_intervals)
        ]

    def set_boundaries(self, boundaries: list[float]) -> None:
        if self.boundaries is not None:
            raise RuntimeError("Boundaries already set for this sampler")
        self.boundaries = boundaries

    def add(self, record: SampledRecord) -> None:
        if self.boundaries is None:
            return
        if self.cfg.n_intervals == 1:
            bucket_idx = 0
        else:
            bucket_idx = bisect.bisect_right(self.boundaries, record.max_activation)
            bucket_idx = min(bucket_idx, self.cfg.n_intervals - 1)
        self.bucket_slots[bucket_idx].add(record)

    def all_records(self) -> list[tuple[int, SampledRecord]]:
        """Returns list of (interval, record) where interval is 1-indexed (1 = lowest)."""
        out: list[tuple[int, SampledRecord]] = []
        for i, slot in enumerate(self.bucket_slots):
            interval = i + 1
            for record in slot.get_records():
                out.append((interval, record))
        return out


class SequenceSamplingState:
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        sampler_config: SamplerConfig,
        warmup_size: int,
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.sampler_config = sampler_config
        self.warmup_size = warmup_size
        self.warmup_buffer = np.zeros(
            (warmup_size, n_layers, n_heads), dtype=np.float32
        )
        self.warmup_filled = 0
        self.boundaries_set = False
        self.samplers: list[list[HeadSampler]] = [
            [HeadSampler(sampler_config, layer, head) for head in range(n_heads)]
            for layer in range(n_layers)
        ]
        self.histogram_counts = np.zeros(
            (n_layers, n_heads, ACTIVATION_HISTOGRAM_BINS), dtype=np.int64
        )
        self.token_cache: dict[int, list[str]] = {}
        self.n_valid_seen = 0
        self.activations_per_head_sum = np.zeros(
            (n_layers, n_heads), dtype=np.float64
        )
        self.activations_per_head_count = np.zeros(
            (n_layers, n_heads), dtype=np.int64
        )

    def maybe_finalize_warmup(self) -> bool:
        if self.boundaries_set:
            return False
        if self.warmup_filled < self.warmup_size and self.n_valid_seen < self.warmup_size:
            return False
        self._compute_boundaries()
        self.boundaries_set = True
        self.warmup_buffer = np.zeros((0, 0, 0), dtype=np.float32)
        return True

    def _compute_boundaries(self) -> None:
        n = self.warmup_filled
        if n == 0:
            for layer_samplers in self.samplers:
                for sampler in layer_samplers:
                    sampler.set_boundaries([0.0] * (self.sampler_config.n_intervals - 1))
            return
        n_intervals = self.sampler_config.n_intervals
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                values = self.warmup_buffer[:n, layer, head]
                sorted_values = np.sort(values)
                if n_intervals <= 1:
                    boundaries: list[float] = []
                else:
                    boundaries = []
                    for i in range(1, n_intervals):
                        idx = min(int(round(i * n / n_intervals)), n - 1)
                        boundaries.append(float(sorted_values[idx]))
                self.samplers[layer][head].set_boundaries(boundaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-head attention metrics and HeadVis-style sampled "
            "sequences for a causal LM, writing the canonical HeadVis data tree."
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
        "--exports-dir",
        default=str(DEFAULT_EXPORTS_DIR),
        help=(
            "Root exports directory. The HeadVis tree is written to "
            "<exports-dir>/<np_model_id>/headvis/<dataset>/."
        ),
    )
    parser.add_argument(
        "--np-model-map",
        default=str(DEFAULT_MODEL_MAP_PATH),
        help=(
            "Path to np_model_to_hf.json. Used to reverse-map "
            "--model-name to a Neuronpedia model id."
        ),
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
    parser.add_argument(
        "--n-intervals",
        type=int,
        default=DEFAULT_N_INTERVALS,
        help="Number of activation intervals for stratified sequence sampling.",
    )
    parser.add_argument(
        "--samples-per-interval",
        type=int,
        default=DEFAULT_SAMPLES_PER_INTERVAL,
        help="Reservoir-sampled sequences per non-top interval.",
    )
    parser.add_argument(
        "--samples-per-top-interval",
        type=int,
        default=DEFAULT_SAMPLES_PER_TOP_INTERVAL,
        help=(
            "Top-K sequences (by max attention) kept in the highest-activation "
            "interval. When --n-intervals is 1, the single bucket uses "
            "max(samples-per-interval, samples-per-top-interval)."
        ),
    )
    parser.add_argument(
        "--sparse-topk-per-row",
        type=int,
        default=DEFAULT_SPARSE_TOPK_PER_ROW,
        help="Keep at most this many key positions per query row in the sparse COO.",
    )
    parser.add_argument(
        "--sparse-threshold",
        type=float,
        default=DEFAULT_SPARSE_THRESHOLD,
        help="Drop COO entries whose attention value is below this threshold.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help="Random seed for reservoir sampling within non-top intervals.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.n_intervals < 1:
        parser.error("--n-intervals must be at least 1")
    if args.samples_per_interval < 1:
        parser.error("--samples-per-interval must be at least 1")
    if args.samples_per_top_interval < 1:
        parser.error("--samples-per-top-interval must be at least 1")
    if args.sparse_topk_per_row < 1:
        parser.error("--sparse-topk-per-row must be at least 1")
    if not (0.0 <= args.sparse_threshold < 1.0):
        parser.error("--sparse-threshold must be in [0, 1)")
    return args


def sanitize_filename_part(value: str) -> str:
    value = value.replace("/", "-")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def resolve_np_model_id(hf_model_name: str, model_map_path: str) -> str:
    """Reverse-lookup an HF model name to a Neuronpedia model id.

    Hard error if the HF name is missing from the map or maps to multiple ids.
    """
    path = Path(model_map_path).expanduser().resolve()
    with open(path) as f:
        model_map = json.load(f)
    if not isinstance(model_map, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    matches = [
        np_id
        for np_id, mapped_hf in model_map.items()
        if mapped_hf == hf_model_name
    ]
    if not matches:
        raise ValueError(
            f"Hugging Face model '{hf_model_name}' not found in {path}. "
            "Add it to np_model_to_hf.json before running."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Hugging Face model '{hf_model_name}' maps to multiple Neuronpedia "
            f"ids in {path}: {matches}. Cannot pick one."
        )
    return matches[0]


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


def _decode_tokens(tokenizer: AutoTokenizer, ids: list[int]) -> list[str]:
    """Decode each token id individually so spaces and punctuation render correctly."""
    return [tokenizer.decode([int(i)]) for i in ids]


def _build_sparse_records_for_layer(
    layer_idx: int,
    n_heads: int,
    topk_values_cpu: np.ndarray,
    topk_indices_cpu: np.ndarray,
    max_act_cpu: np.ndarray,
    sequence_lengths_list: list[int],
    valid_sequence_mask_list: list[bool],
    batch_seq_ids: list[int],
    state: SequenceSamplingState,
    sparse_threshold: float,
) -> None:
    """Build per-(batch, head) COO records and feed them to samplers."""
    batch_size = topk_values_cpu.shape[0]

    for b in range(batch_size):
        if not valid_sequence_mask_list[b]:
            continue
        actual_len = sequence_lengths_list[b]
        if actual_len < 2:
            continue
        seq_id = batch_seq_ids[b]
        for h in range(n_heads):
            row_values = topk_values_cpu[b, h, :actual_len, :]
            row_keys = topk_indices_cpu[b, h, :actual_len, :]

            keep_mask = (row_values >= sparse_threshold) & (row_keys < actual_len)
            if not keep_mask.any():
                continue

            q_grid = np.broadcast_to(
                np.arange(actual_len, dtype=np.int64)[:, None],
                row_values.shape,
            )
            kept_q = q_grid[keep_mask]
            kept_k = row_keys[keep_mask].astype(np.int64)
            kept_v = row_values[keep_mask].astype(np.float64)
            kept_v = np.round(kept_v, 4)

            order = np.lexsort((kept_k, kept_q))
            kept_q = kept_q[order]
            kept_k = kept_k[order]
            kept_v = kept_v[order]

            flat_indices = (kept_q * actual_len + kept_k).astype(np.int32)

            record = SampledRecord(
                sequence_id=seq_id,
                max_activation=float(max_act_cpu[b, h]),
                flat_indices=flat_indices,
                values=kept_v.astype(np.float32),
                seq_len=actual_len,
            )
            state.samplers[layer_idx][h].add(record)


def _update_warmup_buffer(
    layer_idx: int,
    max_act_cpu: np.ndarray,
    sequence_lengths_list: list[int],
    valid_sequence_mask_list: list[bool],
    state: SequenceSamplingState,
    seq_indices_in_warmup: list[int],
) -> None:
    """Stash per-(layer, head, seq) max_activation in the warmup buffer."""
    for b, warmup_idx in enumerate(seq_indices_in_warmup):
        if warmup_idx < 0:
            continue
        if not valid_sequence_mask_list[b]:
            continue
        if sequence_lengths_list[b] < 2:
            continue
        if warmup_idx >= state.warmup_size:
            continue
        state.warmup_buffer[warmup_idx, layer_idx, :] = max_act_cpu[b, :]


def _update_histogram(
    layer_idx: int,
    max_act_cpu: np.ndarray,
    sequence_lengths_list: list[int],
    valid_sequence_mask_list: list[bool],
    state: SequenceSamplingState,
) -> None:
    """Increment per-(layer, head) histogram counts and running mean stats."""
    bin_indices = np.clip(
        (max_act_cpu * ACTIVATION_HISTOGRAM_BINS).astype(np.int64),
        0,
        ACTIVATION_HISTOGRAM_BINS - 1,
    )
    for b in range(max_act_cpu.shape[0]):
        if not valid_sequence_mask_list[b]:
            continue
        if sequence_lengths_list[b] < 2:
            continue
        head_indices = np.arange(state.n_heads)
        np.add.at(
            state.histogram_counts[layer_idx],
            (head_indices, bin_indices[b]),
            1,
        )
        state.activations_per_head_sum[layer_idx] += max_act_cpu[b].astype(np.float64)
        state.activations_per_head_count[layer_idx] += 1


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
    sequence_state: SequenceSamplingState,
    batch_seq_ids: list[int],
    seq_indices_in_warmup: list[int],
    sparse_topk_per_row: int,
    sparse_threshold: float,
    tokenizer: AutoTokenizer,
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

    batch_size, sequence_length = attention_mask.shape
    positions = torch.arange(sequence_length, device=input_ids.device)
    query_mask = (
        (positions[None, :] < sequence_lengths[:, None])
        & valid_sequence_mask[:, None]
    )
    pair_mask = query_mask[:, :, None] & query_mask[:, None, :]
    prev_position_mask = positions[1:][None, :] < sequence_lengths[:, None]
    qk_distance = (positions[:, None] - positions[None, :]).abs().float()
    qk_distance_squared = qk_distance.square()

    not_pos0_q = (positions != 0)[None, :, None]
    not_pos0_k = (positions != 0)[None, None, :]
    max_pair_mask = pair_mask & not_pos0_q & not_pos0_k

    sequence_lengths_list = [int(x) for x in sequence_lengths.tolist()]
    valid_sequence_mask_list = [bool(x) for x in valid_sequence_mask.tolist()]
    n_heads_runtime = attentions[0].shape[1]
    in_warmup = not sequence_state.boundaries_set

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

        max_act = (
            layer_attention_float
            .masked_fill(~max_pair_mask[:, None, :, :], 0.0)
            .amax(dim=(-2, -1))
        )
        max_act_cpu = max_act.cpu().numpy()

        _update_histogram(
            layer_idx,
            max_act_cpu,
            sequence_lengths_list,
            valid_sequence_mask_list,
            sequence_state,
        )

        if in_warmup:
            _update_warmup_buffer(
                layer_idx,
                max_act_cpu,
                sequence_lengths_list,
                valid_sequence_mask_list,
                sequence_state,
                seq_indices_in_warmup,
            )
        else:
            attn_for_topk = layer_attention_float.masked_fill(
                ~query_mask[:, None, None, :], 0.0
            )
            topk_v, topk_i = torch.topk(attn_for_topk, k=sparse_topk_per_row, dim=-1)
            topk_v_cpu = topk_v.cpu().numpy()
            topk_i_cpu = topk_i.cpu().numpy()
            _build_sparse_records_for_layer(
                layer_idx,
                n_heads_runtime,
                topk_v_cpu,
                topk_i_cpu,
                max_act_cpu,
                sequence_lengths_list,
                valid_sequence_mask_list,
                batch_seq_ids,
                sequence_state,
                sparse_threshold,
            )
            del topk_v, topk_i, attn_for_topk
        del max_act

    for batch_idx, item_sequence_length in enumerate(sequence_lengths_list):
        if item_sequence_length < 2:
            continue
        if not valid_sequence_mask_list[batch_idx]:
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

    seq_id_iter = iter(batch_seq_ids)
    for b in range(batch_size):
        seq_id = next(seq_id_iter)
        if not valid_sequence_mask_list[b]:
            continue
        actual_len = sequence_lengths_list[b]
        if actual_len < 2:
            continue
        if seq_id in sequence_state.token_cache:
            continue
        ids = input_ids[b, :actual_len].tolist()
        sequence_state.token_cache[seq_id] = _decode_tokens(tokenizer, ids)

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
    np_model_id: str,
    device: torch.device,
    output_run_dir: str,
    model_cache_dir: str,
    delete_model_cache: bool,
    warmup_size: int,
) -> HeadMetricsConfig:
    return HeadMetricsConfig(
        model_name=args.model_name,
        np_model_id=np_model_id,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_split=args.dataset_split,
        dataset_text_field=args.dataset_text_field,
        n_sequences=args.n_sequences,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        exports_dir=args.exports_dir,
        output_run_dir=output_run_dir,
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
        n_intervals=args.n_intervals,
        samples_per_interval=args.samples_per_interval,
        samples_per_top_interval=args.samples_per_top_interval,
        sparse_topk_per_row=args.sparse_topk_per_row,
        sparse_threshold=args.sparse_threshold,
        sample_seed=args.sample_seed,
        warmup_size=warmup_size,
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


def _round_value(value: float, decimals: int = 4) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return round(float(value), decimals)


def _compute_qk_distance_histogram(
    sequences: list[dict[str, Any]],
) -> dict[str, list[float | int]]:
    """Histogram of |q - k| weighted by attention mass over all kept COO entries."""
    edges = QK_DISTANCE_BIN_EDGES
    bin_values = [0.0] * (len(edges) - 1)
    for seq in sequences:
        seq_len = seq["seq_len"]
        if seq_len <= 0:
            continue
        for flat_idx, value in zip(seq["attention_indices"], seq["attention_values"]):
            q = flat_idx // seq_len
            k = flat_idx % seq_len
            distance = abs(q - k)
            bin_idx = bisect.bisect_right(edges, distance) - 1
            if 0 <= bin_idx < len(bin_values):
                bin_values[bin_idx] += float(value)
    return {
        "bin_edges": list(edges),
        "bin_values": [round(v, 4) for v in bin_values],
    }


def _compute_top_tokens(
    sequences: list[dict[str, Any]],
    side: str,
    top_n: int = TOP_TOKENS_PER_SIDE,
) -> list[dict[str, Any]]:
    """Rank tokens by marginal attention mass on the query or key side."""
    if side not in ("query", "key"):
        raise ValueError(f"side must be 'query' or 'key', got {side!r}")
    accumulator: dict[str, float] = defaultdict(float)
    for seq in sequences:
        tokens = seq["tokens"]
        seq_len = seq["seq_len"]
        if seq_len <= 0:
            continue
        for flat_idx, value in zip(seq["attention_indices"], seq["attention_values"]):
            q = flat_idx // seq_len
            k = flat_idx % seq_len
            pos = q if side == "query" else k
            if 0 <= pos < len(tokens):
                accumulator[tokens[pos]] += float(value)
    ranked = sorted(accumulator.items(), key=lambda x: -x[1])[:top_n]
    return [{"token": token, "weight": round(weight, 4)} for token, weight in ranked]


def _records_to_sequence_dicts(
    sampler: HeadSampler,
    token_cache: dict[int, list[str]],
) -> list[dict[str, Any]]:
    sequences: list[dict[str, Any]] = []
    for interval, record in sampler.all_records():
        tokens = token_cache.get(record.sequence_id)
        if tokens is None:
            continue
        sequences.append(
            {
                "sequence_id": int(record.sequence_id),
                "interval": int(interval),
                "tokens": tokens,
                "attention_indices": [int(x) for x in record.flat_indices.tolist()],
                "attention_values": [float(x) for x in record.values.tolist()],
                "seq_len": int(record.seq_len),
                "max_activation": _round_value(float(record.max_activation)),
            }
        )
    return sequences


def _activation_histogram_payload(counts_for_head: np.ndarray) -> dict[str, list[Any]]:
    edges = np.linspace(0.0, 1.0, ACTIVATION_HISTOGRAM_BINS + 1)
    return {
        "bin_edges": [round(float(e), 4) for e in edges.tolist()],
        "bin_values": [int(v) for v in counts_for_head.tolist()],
    }


def write_head_json(
    output_run_dir: str,
    layer: int,
    head: int,
    sampler: HeadSampler,
    token_cache: dict[int, list[str]],
    histogram_counts: np.ndarray,
    mean_max_activation: float | None,
) -> None:
    sequences = _records_to_sequence_dicts(sampler, token_cache)
    payload = {
        "sequences": sequences,
        "qk_distance_histogram": _compute_qk_distance_histogram(sequences),
        "top_query_tokens": _compute_top_tokens(sequences, side="query"),
        "top_key_tokens": _compute_top_tokens(sequences, side="key"),
        "histogram": _activation_histogram_payload(histogram_counts),
        "statistics": {
            "n_samples": len(sequences),
            "mean_max_activation": _round_value(mean_max_activation)
            if mean_max_activation is not None
            else None,
        },
    }
    heads_dir = os.path.join(output_run_dir, "heads")
    os.makedirs(heads_dir, exist_ok=True)
    head_path = os.path.join(heads_dir, f"L{layer}H{head}.json")
    with open(head_path, "w") as f:
        json.dump(payload, f)
        f.write("\n")


def write_scatter_data(
    output_run_dir: str,
    n_layers: int,
    n_heads: int,
    metric_values: dict[str, torch.Tensor],
) -> None:
    rows: list[dict[str, Any]] = []
    for layer in range(n_layers):
        for head in range(n_heads):
            row: dict[str, Any] = {"layer": layer, "head": head}
            for metric_name in METRIC_NAMES:
                value = float(metric_values[metric_name][layer, head].item())
                row[metric_name] = _round_value(value)
            rows.append(row)
    scatter_path = os.path.join(output_run_dir, "scatter_data.json")
    with open(scatter_path, "w") as f:
        json.dump(rows, f)
        f.write("\n")


def write_run_config_json(
    output_run_dir: str,
    args: argparse.Namespace,
    metrics_config: HeadMetricsConfig,
    n_layers: int,
    n_heads: int,
    actual_sequences_processed: int,
) -> None:
    """Write data/config.json for the HeadVis frontend.

    Also embeds the original metrics-pipeline config under "pipeline_config"
    so downstream loaders and reproducibility tooling have full provenance.
    """
    heads = [[layer, head] for layer in range(n_layers) for head in range(n_heads)]
    pipeline_config = asdict(metrics_config)
    pipeline_config["actual_sequences_processed"] = actual_sequences_processed
    pipeline_config["num_hidden_layers"] = n_layers
    pipeline_config["num_attention_heads"] = n_heads

    config_payload = {
        "model_name": args.model_name,
        "np_model_id": metrics_config.np_model_id,
        "dataset_name": args.dataset_name,
        "heads": heads,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_intervals": args.n_intervals,
        "metrics": list(METRIC_NAMES),
        "metric_descriptions": dict(METRIC_DESCRIPTIONS),
        "head_data_pattern": "L{layer}H{head}.json",
        "has_umap": False,
        "umap_heads": [],
        "custom_plots": [
            {"key": "qk_distance_histogram", "label": "Q-K distance"},
            {"key": "histogram", "label": "Activation distribution"},
        ],
        "custom_tables": [
            {"key": "top_query_tokens", "label": "Top query tokens"},
            {"key": "top_key_tokens", "label": "Top key tokens"},
        ],
        "statistics_display": {
            "n_samples": {"label": "Samples kept", "format": "default"},
            "mean_max_activation": {
                "label": "Mean max activation",
                "format": "default",
            },
        },
        "pipeline_config": pipeline_config,
    }
    with open(os.path.join(output_run_dir, "config.json"), "w") as f:
        json.dump(config_payload, f, indent=2)
        f.write("\n")


def write_static_only_stubs(output_run_dir: str) -> None:
    with open(os.path.join(output_run_dir, "server_config.json"), "w") as f:
        json.dump({}, f)
        f.write("\n")
    attributions_dir = os.path.join(output_run_dir, "attributions")
    os.makedirs(attributions_dir, exist_ok=True)
    with open(os.path.join(attributions_dir, "manifest.json"), "w") as f:
        json.dump({}, f)
        f.write("\n")


def _resolve_warmup_size(n_sequences: int) -> int:
    return max(1, min(DEFAULT_WARMUP_CAP, max(50, n_sequences // 10)))


def run_head_metrics(
    args: argparse.Namespace, model_cache_dir: str, delete_model_cache: bool
) -> None:
    device = resolve_device(args.device)
    warn_if_low_vram(device, args.min_free_vram_gb)

    np_model_id = resolve_np_model_id(args.model_name, args.np_model_map)
    print(f"Resolved Neuronpedia model id: {np_model_id}")

    exports_dir = os.path.abspath(os.path.expanduser(args.exports_dir))
    os.makedirs(exports_dir, exist_ok=True)
    output_run_dir = os.path.join(
        exports_dir,
        np_model_id,
        "headvis",
        sanitize_filename_part(args.dataset_name),
    )
    os.makedirs(output_run_dir, exist_ok=True)
    os.makedirs(os.path.join(output_run_dir, "heads"), exist_ok=True)

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

    warmup_size = _resolve_warmup_size(args.n_sequences)
    sampler_config = SamplerConfig(
        n_intervals=args.n_intervals,
        samples_per_interval=args.samples_per_interval,
        samples_per_top_interval=args.samples_per_top_interval,
        seed=args.sample_seed,
    )
    sequence_state = SequenceSamplingState(
        n_layers=n_layers,
        n_heads=n_heads,
        sampler_config=sampler_config,
        warmup_size=warmup_size,
    )
    print(
        f"Sequence sampling: n_intervals={args.n_intervals}, "
        f"samples_per_interval={args.samples_per_interval}, "
        f"samples_per_top_interval={args.samples_per_top_interval}, "
        f"warmup_size={warmup_size}, sparse_topk_per_row={args.sparse_topk_per_row}, "
        f"sparse_threshold={args.sparse_threshold}"
    )

    dataset = load_text_dataset(args)
    text_iter = iter_texts(dataset, args.dataset_text_field)
    progress = (
        tqdm(total=args.n_sequences, desc="Computing head metrics", unit="seq")
        if tqdm is not None
        else None
    )
    processed_sequences = 0
    next_seq_id = 0

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

            sequence_lengths_for_batch = attention_mask.sum(dim=1).tolist()
            batch_seq_ids: list[int] = []
            seq_indices_in_warmup: list[int] = []
            for length in sequence_lengths_for_batch:
                if int(length) >= 2:
                    batch_seq_ids.append(next_seq_id)
                    if not sequence_state.boundaries_set:
                        seq_indices_in_warmup.append(next_seq_id)
                    else:
                        seq_indices_in_warmup.append(-1)
                    next_seq_id += 1
                else:
                    batch_seq_ids.append(-1)
                    seq_indices_in_warmup.append(-1)

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
                sequence_state,
                batch_seq_ids,
                seq_indices_in_warmup,
                args.sparse_topk_per_row,
                args.sparse_threshold,
                tokenizer,
            )
            prev_token_count += prev_positions_count
            processed_sequences += valid_sequences
            sequence_state.n_valid_seen += valid_sequences
            sequence_state.warmup_filled = min(
                sequence_state.n_valid_seen, sequence_state.warmup_size
            )

            if (
                not sequence_state.boundaries_set
                and sequence_state.n_valid_seen >= sequence_state.warmup_size
            ):
                if sequence_state.maybe_finalize_warmup():
                    print(
                        f"Sampling boundaries computed after "
                        f"{sequence_state.warmup_filled} warmup sequences."
                    )

            if progress is not None:
                progress.update(valid_sequences)
            elif processed_sequences % 10 == 0:
                print(f"Processed {processed_sequences}/{args.n_sequences} sequences")

            del tokenized, input_ids, attention_mask
    finally:
        if progress is not None:
            progress.close()

    if not sequence_state.boundaries_set:
        sequence_state.maybe_finalize_warmup()
        print(
            "Warmup boundaries computed at end of run; sample slots may be empty "
            "because the dataset was smaller than the warmup window."
        )

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

    metric_tensors = {
        "self_attention_score": self_attention_score,
        "prev_token_score": prev_token_score,
        "pattern_entropy": pattern_entropy,
        "qk_distance": qk_distance,
        "qk_distance_variance": qk_distance_variance,
        "induction_score": induction_score,
    }

    write_scatter_data(output_run_dir, n_layers, n_heads, metric_tensors)

    for layer in range(n_layers):
        for head in range(n_heads):
            counts = sequence_state.histogram_counts[layer, head]
            count = int(sequence_state.activations_per_head_count[layer, head])
            if count > 0:
                mean_max = float(
                    sequence_state.activations_per_head_sum[layer, head] / count
                )
            else:
                mean_max = None
            write_head_json(
                output_run_dir,
                layer,
                head,
                sequence_state.samplers[layer][head],
                sequence_state.token_cache,
                counts,
                mean_max,
            )

    write_static_only_stubs(output_run_dir)

    metrics_config = build_config(
        args,
        np_model_id,
        device,
        output_run_dir,
        model_cache_dir,
        delete_model_cache,
        warmup_size,
    )
    write_run_config_json(
        output_run_dir, args, metrics_config, n_layers, n_heads, processed_sequences
    )

    print(f"Wrote HeadVis tree to {output_run_dir}")
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
