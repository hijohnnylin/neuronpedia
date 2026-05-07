#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "transformers>=4.45,<5",
#     "llmcompressor>=0.6",
#     "compressed-tensors>=0.9",
#     "accelerate>=1.0",
#     "huggingface_hub",
#     "safetensors",
#     "packaging",
#     "hf_transfer"
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
"""Build a `compressed-tensors` FP8 NLA source model for `apps/nla/server.py`.

Pre-bakes three pieces of surgery the runtime would otherwise do every
time `apps/nla/server.py` boots:

  1. Vision tower / multi-modal projector stripped (`vision_tower`,
     `multi_modal_projector`, `image_newline`, `vision_resampler`).
  2. Decoder truncated to layers `0..LAYER_INDEX` inclusive (default 41
     -> 42 layers kept of the original 62 for Gemma-3-27B-IT).
  3. Re-saved as `Gemma3ForCausalLM` instead of
     `Gemma3ForConditionalGeneration`. `vision_config` is stripped from
     `config.json`, `architectures` is set to `["Gemma3ForCausalLM"]`,
     and per-layer config lists (`layer_types`, sliding-window
     patterns) are truncated to match the kept layer count.
  4. **FP8 weight-only quantization** via `llmcompressor` +
     `compressed-tensors` packing.

This is the `compressed-tensors` counterpart to `build_quantized_models.py
--target source`. That sibling script uses `torchao.Float8WeightOnlyConfig`
+ `safe_serialization=False` (pickled `pytorch_model-*.bin` shards).
The torchao path has surfaced loading regressions across HF / torchao
version pairs in this stack and is not loadable by sglang. This script
avoids torchao entirely: weights ship as canonical compressed-tensors
`*.safetensors`, loadable by HF transformers natively (and by sglang on
hardware that supports the chosen scheme).

Why weight-only quant for the source model specifically:
  - The source model exists to extract activations at a known layer.
    Those activations are then fed to (a) the verbalizer, which produces
    a description, and (b) the reconstructor, which scores the
    description. Any drift in the residual stream at the extraction
    point degrades **both** halves of the round-trip.
  - The default scheme `W8A16` keeps activations in bf16 and only
    quantizes weights — same numerics as the legacy
    `torchao.Float8WeightOnlyConfig` path. The activation captured at
    the hooked layer is `Linear(W_fp8) @ x_bf16` accumulated in fp32 →
    bf16 cast: bit-for-bit equivalent to the torchao path within FP
    rounding noise.
  - `FP8_DYNAMIC` (W8A8) is offered as an opt-in for cases where the
    server's source-model decode is the bottleneck. Activations get
    quantized to fp8 and back at every linear, which DOES drift the
    residual stream noticeably (~0.5-1.5% L2 vs bf16 source on Gemma3-27B
    at layer 41). Don't pick W8A8 if reconstruction MSE matters.

Run it from anywhere:

    HF_TOKEN=hf_xxx uv run build_compressed_tensors_source.py

    # Or with explicit settings:
    HF_TOKEN=hf_xxx uv run build_compressed_tensors_source.py \\
        --source-model google/gemma-3-27b-it \\
        --layer-index 41 \\
        --scheme W8A16 \\
        --target-name gemma-3-27b-it-FP8-trunc41

The output:
  - `*.safetensors` shards with FP8-quantized weights via
    `compressed-tensors` packing.
  - `config.json` for `Gemma3ForCausalLM` (text-only, truncated to the
    kept layer count) with `quantization_config.quant_method` set to
    `"compressed-tensors"`.
  - All upstream non-weight files (license, tokenizer extras,
    generation_config.json, README -> README_upstream.md, …) preserved.

To consume the output with `apps/nla/server.py`:

    NLA_SOURCE_MODEL=<your-org>/<target-name> \\
    NLA_TRUNCATE_SOURCE=0 \\
    NLA_FP8_SOURCE=0 \\
    NLA_OVERRIDE_EXTRACTION_LAYER=<layer-index> \\
    uv run server.py

(Both `NLA_TRUNCATE_SOURCE` and `NLA_FP8_SOURCE` MUST be off — the
truncation and quantization are already baked in. Setting either back
on tries to re-do the surgery on an already-quantized model and will
fail.)

System requirements:
  - GPU with >= bf16 model size (post-truncation). For Gemma-3-27B-IT
    truncated to layer 41 in bf16: ~36 GB; one 48 GB A40 / 80 GB H100 /
    96 GB Blackwell is fine.
  - >=80 GB CPU RAM (the full multimodal bf16 source is resident before
    truncation and class swap).
  - `uv` to run this script (PEP 723 inline-metadata isolation).

Schemes (`--scheme`):
  - `W8A16` (default): FP8 weights (per-channel) + bf16 activations.
    Lowest activation drift; matches legacy
    `torchao.Float8WeightOnlyConfig`. Loadable by sglang via
    `CompressedTensorsW8A16Fp8`.
  - `FP8_DYNAMIC`: W8A8 FP8. Faster decode on Hopper+, but activations
    quantize to fp8 and back — drifts the extracted residual. Only
    pick this if you don't care about reconstruction quality.
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import gc
import logging
import os
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
log = logging.getLogger("build_ct_source")


DEFAULT_SOURCE = "google/gemma-3-27b-it"
DEFAULT_LAYER_INDEX = 41

# Source model is for activation extraction; weight-only is the safe
# default. FP8_DYNAMIC offered for users explicitly trading drift for
# decode speed.
SUPPORTED_SCHEMES = ("W8A16", "FP8_DYNAMIC")
DEFAULT_SCHEME = "W8A16"
_SCHEME_TAG = {
    "W8A16": "FP8",
    "FP8_DYNAMIC": "FP8-W8A8",
}


_VISION_SUBMODULE_ATTRS = (
    "vision_tower",
    "vision_model",
    "multi_modal_projector",
    "image_newline",
    "vision_resampler",
)


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
    """Return a local path — `snapshot_download` from HF Hub if needed.

    Skips Meta's `original/consolidated.*.pth` shards: these are
    duplicates of the HF-format `*.safetensors` weights we actually load,
    but ~17.6 GB each on Llama-3.x-style repos. Gemma3-27B doesn't ship
    those, but the filter is harmless.
    """
    if Path(model_path).is_dir():
        return model_path
    from huggingface_hub import snapshot_download

    log.info("[hf] downloading %s …", model_path)
    return snapshot_download(
        model_path,
        ignore_patterns=["original/*", "consolidated.*"],
    )


def _strip_vision_components(backbone) -> list[tuple[str, float]]:
    """Replace vision/multi-modal submodules with `nn.Identity`, drop their
    weights.

    Mirrors `nla_inference._strip_vision_components` so the runtime
    doesn't have to redo it. Returns `(attr, gb_freed)` pairs for log
    accounting.
    """
    from torch import nn

    inner = getattr(backbone, "model", backbone)
    freed: list[tuple[str, float]] = []
    for attr in _VISION_SUBMODULE_ATTRS:
        for owner in (backbone, inner):
            mod = getattr(owner, attr, None)
            if mod is None or isinstance(mod, nn.Identity):
                continue
            bytes_total = sum(p.numel() * p.element_size() for p in mod.parameters())
            # Drop from `_modules` registry so `parameters()` no longer
            # traverses these weights — `setattr` to Identity alone leaves
            # the original Module ref alive in some torch versions.
            if hasattr(owner, "_modules") and attr in owner._modules:
                del owner._modules[attr]
            setattr(owner, attr, nn.Identity())
            freed.append((attr, bytes_total / (1024**3)))
            break
    return freed


def _text_decoder(inner):
    """Drill into the text decoder of a (possibly multimodal) HF backbone."""
    return getattr(inner, "language_model", inner)


def _get_text_config(model):
    """Return text-decoder config (`text_config` for multimodal, else `config`)."""
    cfg = model.config
    return getattr(cfg, "text_config", cfg)


def _truncate_per_layer_lists(cfg, old_n: int, new_n: int) -> None:
    """In-place: trim any list/tuple-valued attribute whose length matches
    `old_n` to length `new_n`.

    Gemma-3 stores `layer_types` (`["sliding_attention", "full_attention",
    …]`) and validates `len(layer_types) == num_hidden_layers`. Future HF
    architectures may add similar per-layer metadata; rather than enumerate
    every name, we trim everything that looks per-layer (length ==
    old_n). Anything else is left alone.
    """
    if old_n == new_n:
        return
    for name, value in list(vars(cfg).items()):
        if isinstance(value, (list, tuple)) and len(value) == old_n:
            trimmed = list(value)[:new_n]
            if isinstance(value, tuple):
                trimmed = tuple(trimmed)
            setattr(cfg, name, trimmed)
            log.info("  trimmed config.%s: %d -> %d", name, old_n, new_n)


def _copy_upstream_extras(src_dir: Path, out_dir: Path, *, label: str) -> None:
    """Copy every non-weight file from `src_dir` into `out_dir` exactly once.

    Mirrors `build_compressed_tensors_verbalizer._copy_upstream_extras`.
    Files already written by our save (config.json, tokenizer files, the
    README we wrote ourselves) are skipped, except upstream `README.md`
    which is preserved as `README_upstream.md`.

    For multimodal upstream sources we deliberately keep `preprocessor_config.json`
    and friends even though the saved checkpoint is text-only — they're harmless
    when transformers loads `Gemma3ForCausalLM` (the AutoProcessor route just
    isn't taken) and keeping them avoids surprising any downstream tooling
    that expects them.
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
        log.info(
            "[%s] copied %s%s",
            label,
            src_file.name,
            "" if dst_name == src_file.name else f" -> {dst_name}",
        )

    log.info(
        "[%s] upstream extras: %d copied, %d already-written skipped, "
        "%d weight files skipped",
        label,
        n_copied,
        n_skipped_written,
        n_skipped_weights,
    )


def _write_source_readme(
    out_dir: Path,
    source_id: str,
    layer_index: int,
    n_layers_kept: int,
    scheme: str,
) -> None:
    tag = _SCHEME_TAG[scheme]
    sglang_note = {
        "W8A16": (
            "loadable by sglang via `CompressedTensorsW8A16Fp8` (lazy-imports "
            "vllm only on sm_<89; native sm_89+)"
        ),
        "FP8_DYNAMIC": (
            "loadable by sglang via `CompressedTensorsW8A8Fp8` (native, no "
            "vllm dep). FP8 tensor-core compute on Hopper sm_90+; Ada/Ampere "
            "fall back to dequant->bf16 GEMM"
        ),
    }[scheme]

    text = f"""---
library_name: transformers
base_model: {source_id}
tags:
- quantized
- {tag.lower()}
- compressed-tensors
- llmcompressor
- nla
- gemma3
---

# {source_id} ({tag}, vision-stripped, layer-truncated, compressed-tensors)

NLA-server source-model variant of `{source_id}`:

- **Vision tower stripped** (`vision_tower`, `multi_modal_projector`,
  etc. removed — replaced with `nn.Identity` and dropped from the
  module registry).
- **Decoder truncated** to layers `0..{layer_index}` ({n_layers_kept} of
  the original kept).
- **Saved as `Gemma3ForCausalLM`** — `vision_config` removed from
  `config.json`, `architectures` set accordingly. Per-layer config
  lists (`layer_types`, sliding-window patterns) are truncated to match
  the kept layer count.
- **`{scheme}` quantization** on every backbone `nn.Linear` weight via
  `llmcompressor`. Saved in **`compressed-tensors` format**
  (`*.safetensors`, no pickle), so HF transformers loads it directly
  with no torchao dependency.

This checkpoint is **only for activation extraction** at layer
{layer_index} via `apps/nla/server.py`. It cannot generate coherent
completions because the post-extraction layers are gone (`/completion`
will return 503).

This checkpoint is {sglang_note}.

## Usage with `apps/nla/server.py`

```bash
export NLA_SOURCE_MODEL=<this repo>
# Both flags MUST be off — the surgery is already baked in.
export NLA_TRUNCATE_SOURCE=0
export NLA_FP8_SOURCE=0
export NLA_OVERRIDE_EXTRACTION_LAYER={layer_index}
uv run server.py
```

## Standalone load (HF transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("<this repo>")
model = AutoModelForCausalLM.from_pretrained(
    "<this repo>", torch_dtype=torch.bfloat16, device_map="cuda"
).eval()
# model.forward() returns hidden states at the truncation layer; the
# final-norm and lm_head are NOT useful (they were not the trained ones).
```

Loading requires `compressed-tensors` (a transitive dep of
`transformers >= 4.45` when `quant_method == "compressed-tensors"`).

## Comparison vs `apps/nla/build_quantized_models.py --target source`

That sibling script uses **torchao** (`AffineQuantizedTensor`-backed)
FP8 + pickled `pytorch_model-*.bin` shards. Same architectural surgery,
but the resulting checkpoint:
  - has had recurring loading regressions across HF / torchao version
    pairs in this stack;
  - is not loadable by sglang at any version we've tested.

This `compressed-tensors` checkpoint is the canonical pre-quantized
source going forward.
"""
    (out_dir / "README.md").write_text(text)


def _make_recipe(scheme: str, ignore_modules: list[str]):
    from llmcompressor.modifiers.quantization import (  # type: ignore[import-not-found]
        QuantizationModifier,
    )

    if scheme not in SUPPORTED_SCHEMES:
        raise SystemExit(
            f"unknown --scheme {scheme!r}; choices: {', '.join(SUPPORTED_SCHEMES)}"
        )

    return QuantizationModifier(
        targets="Linear",
        scheme=scheme,
        ignore=ignore_modules,
    )


def build_compressed_tensors_source(
    source_id: str,
    layer_index: int,
    out_dir: Path,
    *,
    scheme: str = DEFAULT_SCHEME,
    device: str = "cuda",
    ignore_modules: list[str] | None = None,
) -> None:
    """Vision-strip, layer-truncate, FP8-quantize, and save the source model
    as a text-only `Gemma3ForCausalLM` in compressed-tensors format.

    The strip + truncate + class swap is done on CPU (low_cpu_mem_usage)
    to avoid spending GPU memory on weights we're about to drop.
    Quantization runs on GPU (or whatever `--device` you pass) via
    `llmcompressor.oneshot`; CPU is supported but slow.
    """
    import torch
    from torch import nn
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from llmcompressor import oneshot  # type: ignore[import-not-found]
    except ImportError:
        from llmcompressor.transformers import oneshot  # type: ignore[import-not-found]

    if ignore_modules is None:
        # Default: keep `lm_head` un-quantized. For Gemma3 with
        # `tie_word_embeddings=True`, lm_head shares storage with
        # `embed_tokens` (an `nn.Embedding` that's NOT a Linear and
        # therefore not targeted by `targets="Linear"` either) — quantizing
        # lm_head would mutate the shared tied storage and break the
        # embedding lookup. Keeping lm_head unquantized preserves tying.
        # The lm_head weight itself goes unused at runtime (we hook the
        # layer at `layer_index` and never actually reach lm_head), but
        # the saved file size win is marginal and not worth the tying
        # footgun.
        ignore_modules = ["lm_head"]

    log.info("[source] resolving %s (snapshot_download)…", source_id)
    src_dir = Path(_resolve_checkpoint_path(source_id))

    log.info("[source] loading %s on CPU (bf16)…", source_id)
    full = AutoModelForCausalLM.from_pretrained(
        str(src_dir),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    upstream_generation_config = getattr(full, "generation_config", None)

    log.info("[source] stripping vision components…")
    freed = _strip_vision_components(full)
    for attr, gb in freed:
        log.info("  freed %s: %.2f GB", attr, gb)
    gc.collect()

    inner = _text_decoder(full.model if hasattr(full, "model") else full)
    n_layers = len(inner.layers)
    if not 0 <= layer_index < n_layers:
        raise ValueError(
            f"layer_index={layer_index} out of range for {n_layers}-layer model"
        )

    log.info(
        "[source] truncating decoder to layers 0..%d (%d/%d kept)",
        layer_index,
        layer_index + 1,
        n_layers,
    )
    inner.layers = nn.ModuleList(list(inner.layers)[: layer_index + 1])
    gc.collect()

    log.info("[source] reconstructing as text-only Gemma3ForCausalLM…")
    text_config = copy.deepcopy(_get_text_config(full))
    text_config.num_hidden_layers = layer_index + 1
    text_config.architectures = ["Gemma3ForCausalLM"]
    text_config.torch_dtype = "bfloat16"
    if not hasattr(text_config, "tie_word_embeddings"):
        text_config.tie_word_embeddings = bool(
            getattr(full.config, "tie_word_embeddings", True)
        )
    # Per-layer config lists (`layer_types`, sliding-window patterns) MUST
    # match `num_hidden_layers` — newer transformers / huggingface_hub
    # validators raise:
    #   ValueError: `num_hidden_layers` (42) must be equal to the number
    #   of layer types (62)
    # Walk the config and trim any list/tuple whose length matches the
    # original layer count.
    _truncate_per_layer_lists(text_config, n_layers, layer_index + 1)

    # Allocate the new model on the meta device so we don't pay a second
    # 27B-worth of CPU RAM for randomly-initialized weights — `assign=True`
    # below replaces each meta parameter with the loaded tensor.
    with init_empty_weights():
        text_only = AutoModelForCausalLM.from_config(text_config)

    # `Gemma3ForConditionalGeneration.model.language_model` and
    # `Gemma3ForCausalLM.model` are both `Gemma3TextModel`, so their
    # state_dict keys line up directly.
    inner_sd = inner.state_dict()
    missing, unexpected = text_only.model.load_state_dict(
        inner_sd, strict=False, assign=True
    )
    if unexpected:
        log.warning("[source] unexpected keys when loading inner state: %s", unexpected)
    if missing:
        log.warning("[source] missing keys when loading inner state: %s", missing)

    lm_head_src = getattr(full, "lm_head", None)
    if text_config.tie_word_embeddings:
        text_only.tie_weights()
    elif lm_head_src is not None and not isinstance(lm_head_src, nn.Identity):
        text_only.lm_head.load_state_dict(
            lm_head_src.state_dict(), strict=True, assign=True
        )
    else:
        log.warning(
            "[source] tie_word_embeddings=False but no lm_head on source — "
            "saved checkpoint will have a randomly-initialised lm_head."
        )

    if upstream_generation_config is not None:
        text_only.generation_config = upstream_generation_config

    # Drop refs to the multimodal-shaped wrapper so its tensors can be
    # collected before we move text_only onto GPU.
    del full, inner, inner_sd
    gc.collect()

    log.info("[source] moving to %s for quantization…", device)
    text_only = text_only.to(device).eval()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(str(src_dir), trust_remote_code=True)

    recipe = _make_recipe(scheme, ignore_modules)

    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "[source] running llmcompressor oneshot with scheme=%s, ignore=%r…",
        scheme,
        ignore_modules,
    )
    oneshot(
        model=text_only,
        recipe=recipe,
        output_dir=str(out_dir),
        save_compressed=True,
        tokenizer=tokenizer,
    )

    log.info("[source] saving tokenizer (idempotent)…")
    tokenizer.save_pretrained(str(out_dir))

    log.info("[source] writing README…")
    _write_source_readme(out_dir, source_id, layer_index, layer_index + 1, scheme)

    log.info("[source] copying upstream extras (config-adjacent files)…")
    _copy_upstream_extras(src_dir, out_dir, label="source")

    del text_only
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("[source] done -> %s", out_dir)


def upload_to_hf(
    local_dir: Path,
    repo_id: str,
    *,
    token: str,
    private: bool = True,
    commit_message: str = "Upload compressed-tensors NLA source model",
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
        "--source-model",
        default=DEFAULT_SOURCE,
        help=f"Upstream source model. Default: {DEFAULT_SOURCE}.",
    )
    p.add_argument(
        "--layer-index",
        type=int,
        default=DEFAULT_LAYER_INDEX,
        help=(
            "Truncate the decoder to layers 0..LAYER_INDEX (inclusive). "
            f"Default: {DEFAULT_LAYER_INDEX} (42 layers kept of 62)."
        ),
    )
    p.add_argument(
        "--scheme",
        default=DEFAULT_SCHEME,
        choices=SUPPORTED_SCHEMES,
        help=(
            "Quantization scheme. W8A16 (default) = FP8 weights + bf16 "
            "activations (lowest activation drift, recommended for the "
            "source model). FP8_DYNAMIC = W8A8 FP8 (faster on Hopper+, "
            "but activations drift through fp8)."
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
            "<source-basename>-<scheme-tag>-trunc<layer-index> "
            "(e.g. gemma-3-27b-it-FP8-trunc41)."
        ),
    )
    p.add_argument(
        "--device",
        default="cuda",
        help=(
            "Device for quantization (default: cuda). The full surgery "
            "and class swap are done on CPU regardless; only the final "
            "oneshot quantization runs on the target device. CPU works "
            "but is slow (~10x)."
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
            "HuggingFace username/org for the upload repo. If omitted, "
            "derived from the token via whoami()."
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

    tag = _SCHEME_TAG[args.scheme]
    upstream_basename = args.source_model.rsplit("/", 1)[-1].lower()
    target_name = (
        args.target_name or f"{upstream_basename}-{tag}-trunc{args.layer_index}"
    )
    out_dir = Path(args.output_dir or f"./quantized_models/{target_name}").resolve()

    log.info(
        "[plan] upstream=%s scheme=%s layer_index=%d output=%s upload=%s",
        args.source_model,
        args.scheme,
        args.layer_index,
        out_dir,
        args.upload,
    )

    build_compressed_tensors_source(
        args.source_model,
        args.layer_index,
        out_dir,
        scheme=args.scheme,
        device=args.device,
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
            commit_message=(
                f"Upload {args.source_model} (vision-stripped, "
                f"truncated to layer {args.layer_index}, {args.scheme}, "
                f"compressed-tensors via llmcompressor)"
            ),
        )
    else:
        log.info(
            "[done] checkpoint built at %s (use --upload to push to HF Hub).",
            out_dir,
        )


if __name__ == "__main__":
    sys.exit(main())
