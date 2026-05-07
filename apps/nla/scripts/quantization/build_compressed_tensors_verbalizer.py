#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "transformers>=4.45,<5",
#     "llmcompressor>=0.5",
#     "compressed-tensors>=0.7",
#     "accelerate>=1.0",
#     "huggingface_hub",
#     "safetensors",
#     "packaging",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///
"""Build a `compressed-tensors` FP8 NLA verbalizer for sglang.

This is the sglang-compatible counterpart to `build_quantized_models.py
--target verbalizer-fp8`. That sibling script uses `torchao`, which sglang
0.5.x does NOT support — the resulting checkpoint cannot be loaded by
`sgl.Engine`. This script uses `llmcompressor` to produce a
`compressed-tensors` FP8 checkpoint, which sglang loads natively.

PEP 723 inline-metadata script: `uv run` will spin up an isolated venv
with the deps declared above. That isolation matters because the main
`apps/nla` project pins `sglang>=0.5.6` (which in turn pins
`torchao==0.9.0` and other floors); `llmcompressor`'s deps would conflict.

Run it from anywhere:

    HF_TOKEN=hf_xxx uv run build_compressed_tensors_verbalizer.py

    # Or with explicit upstream / target name:
    HF_TOKEN=hf_xxx uv run build_compressed_tensors_verbalizer.py \\
        --verbalizer-model kitft/Llama-3.3-70B-NLA-av \\
        --target-name Llama-3.3-70B-NLA-av-FP8

The output:
  - `pytorch_model-*.safetensors` shards with FP8 weights via
    `compressed-tensors` packing (sglang loads these directly into
    fp8 buffers; no bf16 transient peak).
  - `config.json` with `quantization_config.quant_method="compressed-tensors"`
    so sglang's auto-detection identifies it correctly.
  - All upstream NLA-specific files (`nla_meta.yaml`, prompt templates,
    tokenizer extras, generation_config.json, license, …) preserved.
  - The upstream model card is renamed to `README_upstream.md` and a
    fresh README is written documenting the recipe + how to launch
    `apps/nla/server.py` against it.

To consume the output with `apps/nla/server.py`:

    NLA_VERBALIZER_MODEL=<your-org>/<target-name> \\
    uv run server.py --truncate-source --fp8-source --int4-reconstructor

`NLA_FP8_VERBALIZER` must be off — the checkpoint is already quantized,
and sglang auto-detects the `compressed-tensors` recipe from
`config.json`'s `quantization_config` field at engine init.

System requirements:
  - GPU(s) with ≥ (bf16 model size) of total VRAM. For Llama-3.3-70B
    in bf16 (~141 GB) you need 2× 80 GB or larger; pass `--device-map
    auto` (default) and llmcompressor will shard across visible GPUs.
  - ≥ ~150 GB CPU RAM for `low_cpu_mem_usage=True` staging buffers.
  - `uv` to run this script (PEP 723 inline-metadata isolation).

The recipe defaults to FP8_DYNAMIC (per-token dynamic activation scales,
per-channel weight scales, no calibration data required). Pass
`--scheme FP8` for the static-activation variant (needs calibration —
not implemented here yet) or `--scheme W8A16` for weight-only FP8 with
bf16 activations (most conservative, no calibration).
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import os
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
log = logging.getLogger("build_ct_verb")


DEFAULT_VERBALIZER = "kitft/Llama-3.3-70B-NLA-av"
DEFAULT_SCHEME = "FP8_DYNAMIC"
SUPPORTED_SCHEMES = ("FP8_DYNAMIC", "FP8", "W8A16")


# Mirrors build_quantized_models._WEIGHT_FILE_PATTERNS — files we never copy
# from the upstream snapshot dir because our save regenerates them with a
# different sharding layout.
_WEIGHT_FILE_PATTERNS = (
    "model-*.safetensors",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model-*.bin",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "consolidated.*.pth",
    "consolidated.*.bin",
)


def _is_weight_file(name: str) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in _WEIGHT_FILE_PATTERNS)


def _resolve_checkpoint_path(model_path: str) -> str:
    """Return a local path — `snapshot_download` from HF Hub if needed."""
    if Path(model_path).is_dir():
        return model_path
    from huggingface_hub import snapshot_download

    log.info("[hf] downloading %s …", model_path)
    return snapshot_download(model_path)


def _copy_upstream_extras(src_dir: Path, out_dir: Path, *, label: str) -> None:
    """Copy every non-weight file from `src_dir` into `out_dir` exactly once.

    Mirrors `build_quantized_models._copy_upstream_extras` so the produced
    checkpoint has nla_meta.yaml, the verbalizer prompt template, license,
    chat template, etc. alongside the freshly-written quantized weights.
    """
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    written = {p.name for p in out_dir.iterdir() if p.is_file()}

    n_copied = 0
    n_skipped_written = 0
    n_skipped_weights = 0
    for src_file in sorted(src_dir.iterdir()):
        if not src_file.is_file():
            continue
        if _is_weight_file(src_file.name):
            n_skipped_weights += 1
            continue

        dst_name = src_file.name
        if dst_name in written:
            if dst_name == "README.md":
                dst_name = "README_upstream.md"
                if dst_name in written:
                    n_skipped_written += 1
                    continue
            else:
                n_skipped_written += 1
                continue

        shutil.copy2(src_file, out_dir / dst_name)
        n_copied += 1
        log.info("[%s] copied %s%s", label, src_file.name, "" if dst_name == src_file.name else f" -> {dst_name}")

    log.info(
        "[%s] upstream extras: %d copied, %d already-written skipped, "
        "%d weight files skipped",
        label,
        n_copied,
        n_skipped_written,
        n_skipped_weights,
    )


def _write_verbalizer_readme(
    out_dir: Path, verbalizer_id: str, scheme: str
) -> None:
    text = f"""---
library_name: transformers
base_model: {verbalizer_id}
tags:
- quantized
- fp8
- compressed-tensors
- llmcompressor
- nla
---

# {verbalizer_id} (FP8, compressed-tensors)

NLA action verbalizer with `{scheme}` weight-only FP8 quantization via
`llmcompressor`. Saved in **`compressed-tensors` format** with FP8
weights (`*.safetensors`) and `quantization_config` in `config.json`,
so `sgl.Engine` loads it directly with no bf16 staging transient.

- **`{scheme}`** quantization on every `nn.Linear` weight in the
  backbone (`lm_head` is NOT quantized — output projection numerics
  matter).
- All upstream files (`nla_meta.yaml`, prompt templates, model card,
  license, generation config, tokenizer extras, …) are preserved
  verbatim. The upstream `README.md` is renamed to `README_upstream.md`.

## Usage with `apps/nla/server.py`

```bash
export NLA_VERBALIZER_MODEL=<this repo>
# Leave NLA_FP8_VERBALIZER OFF — that flag is sglang's runtime bf16->fp8
# conversion path; this checkpoint is already fp8 at rest. Sglang
# auto-detects the `compressed-tensors` recipe from this repo's
# config.json `quantization_config` field at engine init.
uv run server.py --truncate-source --fp8-source --int4-reconstructor
```

`apps/nla/server.py` will pass `quantization=None` to `sgl.Engine`;
sglang's config-auto-detect then routes through the
`compressed-tensors` quantizer and loads the pre-quantized FP8 weights
with no bf16 load-time peak. See the apps/nla README's "Multi-GPU >
Layout E" for the full per-GPU memory budget.

## Comparison vs `apps/nla/build_quantized_models.py --target verbalizer-fp8`

That sibling script uses **torchao** (`AffineQuantizedTensor`-backed) FP8.
Sglang 0.5.x has no `"torchao"` entry in its quantization registry, so
that output is **not loadable by `apps/nla/server.py`**. Use this
checkpoint for the sglang verbalizer; reserve the torchao path for the
HF source/reconstructor (which transformers consumes natively).
"""
    (out_dir / "README.md").write_text(text)


def _make_recipe(scheme: str, ignore_modules: list[str]):
    """Build a llmcompressor `QuantizationModifier` for `scheme`.

    `FP8_DYNAMIC` (default): per-channel weight scales + per-token dynamic
    activation scales. No calibration data required. Best general default
    on Hopper+ for autoregressive decoding through sglang.

    `FP8`: same weights but static per-tensor activation scales. Requires
    calibration data; slightly better accuracy on heavy-tail
    distributions. Not implemented in this script (would need a calibration
    dataset hook).

    `W8A16`: FP8 weights + bf16 activations. Most conservative. No
    calibration. Loses the FP8 tensor-core compute speedup on Hopper+ but
    is portable to Ampere/Ada.
    """
    from llmcompressor.modifiers.quantization import (  # type: ignore[import-not-found]
        QuantizationModifier,
    )

    if scheme == "FP8":
        raise SystemExit(
            "scheme=FP8 (static activation scales) requires calibration data; "
            "this script doesn't wire that up. Use FP8_DYNAMIC (default) or "
            "W8A16, or extend the script with a calibration dataset hook."
        )
    if scheme not in ("FP8_DYNAMIC", "W8A16"):
        raise SystemExit(
            f"unknown --scheme {scheme!r}; choices: {', '.join(SUPPORTED_SCHEMES)}"
        )

    return QuantizationModifier(
        targets="Linear",
        scheme=scheme,
        ignore=ignore_modules,
    )


def build_compressed_tensors_verbalizer(
    verbalizer_id: str,
    out_dir: Path,
    *,
    scheme: str = DEFAULT_SCHEME,
    device_map: str = "auto",
    ignore_modules: list[str] | None = None,
) -> None:
    """Quantize the verbalizer with llmcompressor + compressed-tensors.

    Loads the upstream model in bf16 (sharded across visible GPUs via
    `device_map="auto"`), runs `oneshot` quantization in place, saves to
    `out_dir` in compressed-tensors format, then copies upstream NLA
    extras (`nla_meta.yaml`, prompt templates, etc.) verbatim.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from llmcompressor import oneshot  # type: ignore[import-not-found]
    except ImportError:
        # Older llmcompressor versions exposed oneshot under a different path.
        from llmcompressor.transformers import oneshot  # type: ignore[import-not-found]

    if ignore_modules is None:
        # Default: keep the LM head in its original dtype. The verbalizer's
        # lm_head distribution can be heavy-tailed, and the storage savings
        # are modest (~256 MB on Llama-3.3-70B's 128k×8192 head). Match
        # llmcompressor's own README recommendation.
        ignore_modules = ["lm_head"]

    log.info("[verbalizer] resolving %s (snapshot_download)…", verbalizer_id)
    src_dir = Path(_resolve_checkpoint_path(verbalizer_id))

    log.info(
        "[verbalizer] loading %s with device_map=%r (bf16)…",
        verbalizer_id,
        device_map,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(src_dir),
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(src_dir), trust_remote_code=True
    )

    recipe = _make_recipe(scheme, ignore_modules)

    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "[verbalizer] running llmcompressor oneshot with scheme=%s, ignore=%r…",
        scheme,
        ignore_modules,
    )
    oneshot(
        model=model,
        recipe=recipe,
        output_dir=str(out_dir),
        save_compressed=True,
        tokenizer=tokenizer,
    )

    # llmcompressor's `oneshot(output_dir=...)` already calls
    # save_pretrained with `save_compressed=True`, so weights + config.json
    # are written. Ensure the tokenizer is saved too — different
    # llmcompressor versions vary on this.
    log.info("[verbalizer] saving tokenizer (idempotent)…")
    tokenizer.save_pretrained(str(out_dir))

    log.info("[verbalizer] writing README…")
    _write_verbalizer_readme(out_dir, verbalizer_id, scheme)

    log.info(
        "[verbalizer] copying upstream extras "
        "(nla_meta.yaml, prompt templates, model card, …)…"
    )
    _copy_upstream_extras(src_dir, out_dir, label="verbalizer")

    del model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("[verbalizer] done -> %s", out_dir)


def upload_to_hf(
    local_dir: Path,
    repo_id: str,
    *,
    token: str,
    private: bool = True,
    commit_message: str = "Upload compressed-tensors FP8 verbalizer",
) -> None:
    from huggingface_hub import HfApi, create_repo

    log.info("[upload] ensuring repo %s exists (private=%s)…", repo_id, private)
    create_repo(
        repo_id,
        token=token,
        private=private,
        exist_ok=True,
        repo_type="model",
    )
    log.info("[upload] uploading %s -> %s …", local_dir, repo_id)
    HfApi(token=token).upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    log.info("[upload] https://huggingface.co/%s", repo_id)


def _resolve_username(token: str | None, override: str | None) -> str:
    if override:
        return override
    if not token:
        raise SystemExit(
            "--hf-username not provided and no token available to look it up. "
            "Pass --hf-username or --hf-token (or $HF_TOKEN)."
        )
    from huggingface_hub import HfApi

    info = HfApi(token=token).whoami()
    name = info.get("name") or info.get("fullname")
    if not name:
        raise SystemExit(f"could not infer HF username from whoami(): {info!r}")
    return name


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--verbalizer-model",
        default=DEFAULT_VERBALIZER,
        help=f"Upstream verbalizer (HF repo or local dir). Default: {DEFAULT_VERBALIZER}.",
    )
    p.add_argument(
        "--scheme",
        default=DEFAULT_SCHEME,
        choices=SUPPORTED_SCHEMES,
        help=(
            "Quantization scheme. FP8_DYNAMIC = per-channel weight + per-token "
            "dynamic activation FP8 (recommended for sglang/Hopper+). "
            "W8A16 = FP8 weights + bf16 activations (Ampere/Ada portable). "
            "FP8 (static) requires calibration data and is not implemented here."
        ),
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Local output dir. Default: ./quantized_models/<target-name>. "
            "Created if missing."
        ),
    )
    p.add_argument(
        "--target-name",
        default=None,
        help=(
            "Repo name (basename) for the produced checkpoint. Default: "
            "<upstream-basename>-FP8 (e.g. Llama-3.3-70B-NLA-av-FP8)."
        ),
    )
    p.add_argument(
        "--device-map",
        default="auto",
        help=(
            "device_map passed to from_pretrained. 'auto' shards across "
            "visible GPUs (recommended for 70B-class on multi-GPU). Use "
            "'cuda:0' if the model fits on one GPU in bf16."
        ),
    )
    p.add_argument(
        "--upload",
        action="store_true",
        help="Upload the produced checkpoint to HuggingFace after building.",
    )
    p.add_argument(
        "--hf-username",
        default=None,
        help=(
            "HuggingFace username/org for the upload repo. If omitted, derived "
            "from the token via whoami()."
        ),
    )
    p.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token. Default: $HF_TOKEN.",
    )
    p.add_argument(
        "--hub-private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload as a private repo (default) or public.",
    )
    args = p.parse_args()

    upstream_basename = args.verbalizer_model.rsplit("/", 1)[-1]
    target_name = args.target_name or f"{upstream_basename}-FP8"
    out_dir = Path(args.output_dir or f"./quantized_models/{target_name}").resolve()

    log.info(
        "[plan] upstream=%s scheme=%s output=%s upload=%s",
        args.verbalizer_model,
        args.scheme,
        out_dir,
        args.upload,
    )

    build_compressed_tensors_verbalizer(
        args.verbalizer_model,
        out_dir,
        scheme=args.scheme,
        device_map=args.device_map,
    )

    if args.upload:
        if not args.hf_token:
            raise SystemExit(
                "--upload requires --hf-token or $HF_TOKEN to be set."
            )
        username = _resolve_username(args.hf_token, args.hf_username)
        repo_id = f"{username}/{target_name}"
        upload_to_hf(
            out_dir,
            repo_id,
            token=args.hf_token,
            private=args.hub_private,
        )
    else:
        log.info(
            "[done] checkpoint built at %s (use --upload to push to HF Hub).",
            out_dir,
        )


if __name__ == "__main__":
    sys.exit(main())
