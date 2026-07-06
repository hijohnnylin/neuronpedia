# Based on the Jacobian lens ("jlens") reference implementation by Anthropic PBC.
# Companion code for the "Verbalizable Workspace" paper.
# https://github.com/anthropics/jlens
# SPDX-License-Identifier: Apache-2.0
#
# This is a Neuronpedia-adapted copy of the upstream ``demo/fit_lens.py``: the
# interactive slice-visualisation has been removed (Neuronpedia only needs the
# fitted lens), and multi-GPU loading via ``device_map`` plus an optional HF
# cache directory have been added so large models can be fit across GPUs.
"""Fit a Jacobian lens for *any* HuggingFace model on *any* text dataset.

The model id and the fitting corpus are both command-line arguments. The fit
reports a convergence metric (``Δmean``) at every prompt — the relative
Frobenius change of the running-mean Jacobian contributed by the latest prompt,
averaged over fitted layers. It decays roughly like ``1/n``; the prompt count at
which it flattens below a small threshold is where extra prompts stop improving
the lens. Pass ``--stop_at_delta`` to stop automatically once it converges.

Run::

    python fit_lens.py Qwen/Qwen3.5-0.8B --out_dir out/
    python fit_lens.py meta-llama/Llama-3.1-8B --n_prompts 1000 --stop_at_delta 1e-3

    # A different corpus (any HF dataset with a text column):
    python fit_lens.py Qwen/Qwen3.5-0.8B \\
        --dataset stas/openwebtext-10k --dataset_config none --text_field text

Fitting is checkpointed after every prompt, so it is safe to interrupt/resume.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from collections import deque

import torch
import transformers

import jlens


def load_prompts(
    *,
    dataset: str,
    config: str | None,
    split: str,
    text_field: str,
    n_prompts: int,
    max_chars: int = 2000,
    min_chars: int = 200,
    trust_remote_code: bool = False,
) -> list[str]:
    """Stream ``n_prompts`` text chunks of ~``max_chars`` from a HF dataset.

    Records are concatenated and re-chunked to ``max_chars`` so that short rows
    (e.g. WikiText lines) and long documents (e.g. web text) both yield usable
    prompts. Blank rows and obvious section headers are skipped. The dataset is
    streamed from the HuggingFace Hub at call time; nothing is bundled here.

    Args:
        dataset: HF dataset id (e.g. ``"Salesforce/wikitext"``).
        config: Dataset config/subset name, or ``None`` for datasets without one.
        split: Split to read (e.g. ``"train"``).
        text_field: Name of the text column on each record.
        n_prompts: Number of prompts to return.
        max_chars: Target chunk length (also the hard truncation).
        min_chars: Drop any final/partial chunk shorter than this.
        trust_remote_code: Forwarded to ``datasets.load_dataset``.

    Returns:
        A list of up to ``n_prompts`` text prompts.
    """
    from datasets import load_dataset

    stream = load_dataset(
        dataset,
        config,
        split=split,
        streaming=True,
        trust_remote_code=trust_remote_code,
    )

    prompts: list[str] = []
    buffer = ""
    for record in stream:
        text = str(record.get(text_field, "")).strip()
        if not text or text.startswith("="):
            continue
        buffer += " " + text
        while len(buffer) > max_chars:
            prompts.append(buffer[:max_chars].strip())
            buffer = buffer[max_chars:]
            if len(prompts) >= n_prompts:
                return prompts
    if buffer.strip() and len(buffer.strip()) >= min_chars and len(prompts) < n_prompts:
        prompts.append(buffer.strip())
    return prompts


def _slug(model: str) -> str:
    """Filesystem-safe stem derived from a model id or path."""
    base = model.rstrip("/").split("/")[-1]
    return re.sub(r"[^0-9A-Za-z._-]+", "-", base).strip("-") or "model"


def peak_vram_gb() -> float:
    """Peak allocated CUDA memory summed across all visible devices, in GiB."""
    if not torch.cuda.is_available():
        return 0.0
    return sum(
        torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
    ) / 1024**3


class ConvergenceTracker:
    """Writes the per-prompt convergence metric to CSV, records milestones, and
    optionally requests early stop once the lens has converged.

    Pass :meth:`record` as ``metrics_callback`` to :func:`jlens.fit`. ``record``
    returns ``True`` to ask :func:`jlens.fit` to stop early.

    Early stop fires only when *all* of these hold:
      * ``stop_at_delta`` is set, and
      * at least ``min_prompts`` prompts have been accumulated, and
      * the mean of the last ``window`` ``Δmean`` values is below
        ``stop_at_delta`` (smoothing avoids tripping on a single noisy step).
    """

    def __init__(
        self,
        csv_path: str,
        thresholds: tuple[float, ...],
        *,
        stop_at_delta: float | None = None,
        min_prompts: int = 100,
        window: int = 10,
    ) -> None:
        self.csv_path = csv_path
        self.thresholds = thresholds
        self.stop_at_delta = stop_at_delta
        self.min_prompts = min_prompts
        self.window = max(1, window)
        self.history: list[tuple[int, float]] = []
        self.stopped_at: int | None = None
        self._crossed: dict[float, int] = {}
        self._recent: deque[float] = deque(maxlen=self.window)
        self._file = open(csv_path, "w", newline="")  # noqa: SIM115 (closed in close())
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "n_done",
                "prompt_idx",
                "seq_len",
                "n_valid_positions",
                "elapsed_s",
                "identity_distance",
                "mean_rel_change",
            ]
        )

    def record(self, p: jlens.FitProgress) -> bool:
        self._writer.writerow(
            [
                p.n_done,
                p.prompt_idx,
                p.seq_len,
                p.n_valid_positions,
                f"{p.elapsed_s:.3f}",
                f"{p.identity_distance:.6f}",
                f"{p.mean_rel_change:.8f}",
            ]
        )
        self._file.flush()
        if p.mean_rel_change != p.mean_rel_change:  # NaN (first prompt)
            return False
        self.history.append((p.n_done, p.mean_rel_change))
        self._recent.append(p.mean_rel_change)
        for thr in self.thresholds:
            if thr not in self._crossed and p.mean_rel_change < thr:
                self._crossed[thr] = p.n_done

        if (
            self.stop_at_delta is not None
            and p.n_done >= self.min_prompts
            and len(self._recent) == self.window
            and (sum(self._recent) / self.window) < self.stop_at_delta
        ):
            smoothed = sum(self._recent) / self.window
            self.stopped_at = p.n_done
            print(
                f"Converged: {self.window}-prompt mean Δmean={smoothed:.2e} < "
                f"{self.stop_at_delta:g} at {p.n_done} prompts — stopping early."
            )
            return True
        return False

    def close(self) -> None:
        self._file.close()

    def summary(self) -> str:
        lines = ["Convergence (Δmean = relative change of the running-mean Jacobian):"]
        for thr in self.thresholds:
            n = self._crossed.get(thr)
            if n is None:
                lines.append(f"  Δmean < {thr:g}: not reached within this run")
            else:
                lines.append(f"  Δmean < {thr:g}: first reached at {n} prompts")
        if self.history:
            last_n, last_v = self.history[-1]
            lines.append(f"  last: Δmean={last_v:.2e} at {last_n} prompts")
            if self.stopped_at is not None:
                lines.append(f"  stopped early at {self.stopped_at} prompts (--stop_at_delta)")
            lines.append(f"  full curve written to {self.csv_path}")
        return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", help="HF model id or local path (any decoder LM)")
    parser.add_argument("--out_dir", default="out", help="output directory for the lens")
    parser.add_argument("--n_prompts", type=int, default=200, help="prompts to average over")
    parser.add_argument("--dim_batch", type=int, default=8, help="output dims per backward pass")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument(
        "--target_layer",
        type=int,
        default=None,
        help="layer to take gradients w.r.t. (default: final; negative indexes from end)",
    )
    parser.add_argument(
        "--text_module", default=None, help="dotted path to the text decoder (auto-detected)"
    )
    parser.add_argument("--no_compile", action="store_true", help="disable per-layer torch.compile")
    parser.add_argument(
        "--device_map",
        default="cuda",
        help=(
            "how to place the model: 'cuda' = single GPU (.cuda()); "
            "'auto' (or any accelerate device_map) = shard across all visible GPUs "
            "for models too large for one card"
        ),
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="model dtype",
    )
    parser.add_argument(
        "--hf_cache_dir",
        default=None,
        help=(
            "HuggingFace cache dir for weights. When set, ALL HF downloads are "
            "confined here and the directory is deleted after the fit (so disk "
            "isn't filled up across many models) unless --keep_hf_cache is given."
        ),
    )
    parser.add_argument(
        "--keep_hf_cache",
        action="store_true",
        help="do not delete --hf_cache_dir after fitting (default: delete it)",
    )

    # Dataset selection. Defaults reproduce the WikiText-103 corpus.
    parser.add_argument("--dataset", default="Salesforce/wikitext", help="HF dataset id")
    parser.add_argument(
        "--dataset_config",
        default="wikitext-103-raw-v1",
        help="dataset config/subset; pass 'none' for datasets without one",
    )
    parser.add_argument("--dataset_split", default="train", help="dataset split")
    parser.add_argument("--text_field", default="text", help="text column name on each record")
    parser.add_argument("--max_chars", type=int, default=2000, help="target prompt length (chars)")
    parser.add_argument(
        "--trust_remote_code", action="store_true", help="pass through to HF loaders"
    )

    # Convergence reporting.
    parser.add_argument(
        "--metrics_csv",
        default=None,
        help="per-prompt convergence curve (default: <out_dir>/<model>_convergence.csv)",
    )
    parser.add_argument(
        "--levels",
        default="1e-2,5e-3,1e-3",
        help="comma-separated Δmean thresholds to report 'levelled-off' prompt counts for",
    )
    parser.add_argument(
        "--stop_at_delta",
        type=float,
        default=None,
        help="stop early once the smoothed Δmean drops below this (e.g. 1e-3); off by default",
    )
    parser.add_argument(
        "--min_prompts",
        type=int,
        default=100,
        help="never stop early before this many prompts (only with --stop_at_delta)",
    )
    parser.add_argument(
        "--stop_window",
        type=int,
        default=10,
        help="number of recent prompts averaged for the --stop_at_delta test",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for fitting.")

    jlens.configure_logging()
    os.makedirs(args.out_dir, exist_ok=True)
    slug = _slug(args.model)
    lens_path = os.path.join(args.out_dir, f"{slug}_jacobian_lens.pt")
    checkpoint_path = os.path.join(args.out_dir, f"{slug}_checkpoint.pt")
    metrics_csv = args.metrics_csv or os.path.join(args.out_dir, f"{slug}_convergence.csv")
    config = None if args.dataset_config.lower() in ("none", "", "null") else args.dataset_config
    thresholds = tuple(sorted((float(x) for x in args.levels.split(",") if x.strip()), reverse=True))
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        args.dtype
    ]

    # Confine every HF download (weights + hub metadata) to the cache dir so it
    # can be wiped wholesale afterwards. Without this, downloads also leak into
    # the default ~/.cache/huggingface and survive cleanup.
    cache_root: str | None = None
    if args.hf_cache_dir:
        cache_root = os.path.abspath(os.path.expanduser(args.hf_cache_dir))
        os.makedirs(cache_root, exist_ok=True)
        os.environ["HF_HOME"] = cache_root
        os.environ["HF_HUB_CACHE"] = os.path.join(cache_root, "hub")
        os.environ["HF_XET_CACHE"] = os.path.join(cache_root, "xet")
        os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_root, "datasets")

    hub_cache = os.path.join(cache_root, "hub") if cache_root else None
    load_kwargs: dict = {"torch_dtype": dtype, "trust_remote_code": args.trust_remote_code}
    if hub_cache:
        load_kwargs["cache_dir"] = hub_cache
    single_gpu = args.device_map.lower() == "cuda"
    if not single_gpu:
        load_kwargs["device_map"] = args.device_map

    try:
        print(f"Loading {args.model} ({args.dtype}, device_map={args.device_map}) ...")
        hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        if single_gpu:
            hf = hf.cuda()
        tok = transformers.AutoTokenizer.from_pretrained(
            args.model, cache_dir=hub_cache, trust_remote_code=args.trust_remote_code
        )
        model = jlens.from_hf(hf, tok, text_module=args.text_module, compile=not args.no_compile)
        print(f"Wrapped: {model!r}")

        print(
            f"Loading {args.n_prompts} prompts from {args.dataset}"
            + (f" ({config})" if config else "")
            + f" [{args.dataset_split}::{args.text_field}] ..."
        )
        prompts = load_prompts(
            dataset=args.dataset,
            config=config,
            split=args.dataset_split,
            text_field=args.text_field,
            n_prompts=args.n_prompts,
            max_chars=args.max_chars,
            trust_remote_code=args.trust_remote_code,
        )
        if not prompts:
            raise SystemExit("no prompts loaded — check --dataset/--dataset_config/--text_field")

        tracker = ConvergenceTracker(
            metrics_csv,
            thresholds,
            stop_at_delta=args.stop_at_delta,
            min_prompts=args.min_prompts,
            window=args.stop_window,
        )
        print(f"Fitting lens over {len(prompts)} prompts (first call compiles, ~1-2 min) ...")
        try:
            lens = jlens.fit(
                model,
                prompts,
                target_layer=args.target_layer,
                dim_batch=args.dim_batch,
                max_seq_len=args.max_seq_len,
                checkpoint_path=checkpoint_path,
                metrics_callback=tracker.record,
            )
        finally:
            tracker.close()
        lens.save(lens_path)

        # The checkpoint only exists to resume an interrupted fit; once the lens
        # is saved it is dead weight, so drop it.
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        print(f"Peak CUDA memory during fit (all GPUs): {peak_vram_gb():.2f} GB")
        print(f"Done. Saved lens -> {lens_path}\n{lens!r}")
        print(tracker.summary())
    finally:
        # Free disk: drop the downloaded weights so they don't accumulate when
        # fitting many models in sequence. The lens (in out_dir) is unaffected.
        if cache_root and not args.keep_hf_cache:
            print(f"Deleting HuggingFace cache to free disk: {cache_root}")
            shutil.rmtree(cache_root, ignore_errors=True)


if __name__ == "__main__":
    main()
