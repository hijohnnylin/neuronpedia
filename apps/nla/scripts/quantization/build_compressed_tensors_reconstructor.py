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
"""Build a `compressed-tensors` FP8 / FP4 NLA reconstructor for `apps/nla/server.py`.

This is the `compressed-tensors` counterpart to `build_quantized_models.py
--target reconstructor-fp8 / reconstructor-int4`. That sibling script uses
`torchao` (`AffineQuantizedTensor`-backed quantized weights serialized as
pickled `pytorch_model-*.bin` shards). The torchao path has shipped real
loading regressions in our stack — sglang 0.5.x has no `"torchao"` entry
in its quantization registry, so the same checkpoint cannot be reused if
you ever decide to host the reconstructor through sglang, and several
HF/torchao version combinations have surfaced subtle deserialization
errors against `safe_serialization=False` shards. This script avoids
torchao entirely: `llmcompressor` performs the weight-only quantization
in place and serializes via `compressed-tensors` (canonical
`*.safetensors`, no pickle).

Why this is safe for the reconstructor specifically:
  - The reconstructor backbone is loaded by `nla_inference.NLAReconstructor`
    via plain `transformers.AutoModelForCausalLM.from_pretrained`. HF
    transformers consumes the embedded `quantization_config` block in
    `config.json` natively (compressed-tensors auto-detection), reloading
    the quantized weights with no extra deps beyond `compressed-tensors`
    (already a transitive dep of `llmcompressor`).
  - The default scheme is **weight-only** (`W8A16` for FP8, `NVFP4A16` for
    FP4). Activations stay in bf16 — same numerics as the existing
    `torchao.Float8WeightOnlyConfig` path — so the `value_head` projection
    sees the same residual stream values it did at training time. This
    minimizes activation drift on the d-dim score the reconstructor
    produces.
  - `value_head.safetensors` is **not** quantized: it ships as a sibling
    file that's copied verbatim from upstream (see "upstream extras"
    below). The build only quantizes the backbone.
  - `lm_head` is in the default `ignore` list. `NLAReconstructor.__init__`
    replaces `backbone.lm_head` with `nn.Identity()` at load time anyway —
    the lm_head's quantized weight (if quantized) would be allocated then
    immediately freed. Keeping it in the ignore list saves an extra
    quantize/dequantize pass and keeps tied embedding semantics intact
    when the upstream model has `tie_word_embeddings=True`.

Sglang compatibility (forward-looking):
  - Even though `apps/nla/server.py` currently loads the reconstructor
    through HF transformers, this checkpoint is also loadable by recent
    sglang for free:
      - `W8A16` (FP8 weight-only) -> `CompressedTensorsW8A16Fp8`
        (sglang #4852, ported from vLLM; lazy-imports `vllm` only on
        sm_<89 hosts).
      - `FP8_DYNAMIC` (W8A8 FP8) -> `CompressedTensorsW8A8Fp8`
        (native sglang, no vllm dep). Hopper+ for the FP8 compute path.
      - `W4A16` (INT4 weight-only) -> `CompressedTensorsWNA16` /
        `CompressedTensorsWNA16TritonMoE`. Portable Ampere+.
      - `NVFP4A16` (FP4 weight-only) -> `CompressedTensorsW4A16Fp4`
        (sglang #18116). Requires sm_100 (Blackwell) for the FP4 GEMM
        path; on older GPUs sglang will refuse to load it. Use `W4A16`
        for portability if you need INT4-class compression today.
  - The torchao path the legacy `build_quantized_models.py` uses is
    fundamentally not loadable by sglang at any version we've seen.

Run it from anywhere:

    HF_TOKEN=hf_xxx uv run build_compressed_tensors_reconstructor.py

    # Or with explicit upstream / target name / scheme:
    HF_TOKEN=hf_xxx uv run build_compressed_tensors_reconstructor.py \\
        --reconstructor-model kitft/nla-gemma3-27b-ar \\
        --scheme NVFP4A16 \\
        --target-name nla-gemma3-27b-ar-NVFP4A16

The output:
  - `*.safetensors` shards with quantized backbone weights via
    `compressed-tensors` packing.
  - `config.json` with
    `quantization_config.quant_method="compressed-tensors"` so HF
    transformers (and sglang) auto-detect on load.
  - `value_head.safetensors` preserved verbatim alongside the new
    backbone shards (bf16, quality-critical projection).
  - `nla_meta.yaml`, prompt templates, tokenizer extras,
    `generation_config.json`, license, … all preserved.
  - The upstream model card is renamed to `README_upstream.md` and a
    fresh README is written documenting the recipe + how to point
    `apps/nla/server.py` at the result.

To consume the output with `apps/nla/server.py`:

    NLA_RECONSTRUCTOR_MODEL=<your-org>/<target-name> \\
    uv run server.py --truncate-source --fp8-source
    # NLA_FP8_RECONSTRUCTOR / NLA_INT4_RECONSTRUCTOR must be off — the
    # backbone is already quantized at rest.

System requirements:
  - GPU with ~(bf16 model size) of total VRAM. For a 27B Gemma3
    reconstructor in bf16 that's ~54 GB; one 80 GB H100 / RTX Pro 6000
    is fine. Pass `--device-map auto` (default) and llmcompressor will
    shard across visible GPUs if needed (e.g. 2× 48 GB A40).
  - ~80 GB CPU RAM for `low_cpu_mem_usage=True` staging buffers.
  - `uv` to run this script (PEP 723 inline-metadata isolation).

Schemes (`--scheme`):
  - `W8A16` (default for `--quant fp8`): FP8 weights (per-channel) +
    bf16 activations. No calibration. Lowest activation drift; matches
    legacy `torchao.Float8WeightOnlyConfig` behavior. Recommended for
    the reconstructor.
  - `FP8_DYNAMIC`: per-channel FP8 weights + per-token dynamic FP8
    activations (W8A8). No calibration. ~2× faster decode on Hopper+
    via the FP8 tensor cores, BUT activations drift to fp8 and back —
    has a measurable downstream effect on the reconstruction
    cosine/MSE. Use only if decode latency dominates.
  - `NVFP4A16` (default for `--quant fp4`): NVFP4 weights (group=16) +
    bf16 activations. Weight-only FP4. No calibration. Loadable by HF
    transformers anywhere; sglang only on Blackwell sm_100+.
  - `W4A16`: INT4 weights (group=128) + bf16 activations. No
    calibration. Most portable 4-bit option (Ampere+). Compresses to
    ~3.7×, comparable to NVFP4A16 in size but with INT4 numerics
    instead of FP4. Pick this over `NVFP4A16` if you need to load on
    pre-Blackwell hardware via sglang.
  - `FP8` (static W8A8): not implemented (requires calibration data).
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
log = logging.getLogger("build_ct_recon")


DEFAULT_RECONSTRUCTOR = "kitft/nla-gemma3-27b-ar"

# Schemes we know how to drive without a calibration dataset, grouped by the
# coarse `--quant` shorthand the user passes. The default within each group
# is the weight-only variant (`*A16`) because that's what minimizes activation
# drift and matches the existing torchao-FP8 baseline numerics.
_FP8_SCHEMES = ("W8A16", "FP8_DYNAMIC")
_FP4_SCHEMES = ("NVFP4A16", "W4A16")
SUPPORTED_SCHEMES = _FP8_SCHEMES + _FP4_SCHEMES
DEFAULT_FP8_SCHEME = "W8A16"
DEFAULT_FP4_SCHEME = "NVFP4A16"

# Pretty short labels used in target-name suffixes / README tags.
_SCHEME_TAG = {
    "W8A16": "FP8",
    "FP8_DYNAMIC": "FP8-W8A8",
    "NVFP4A16": "FP4",
    "W4A16": "INT4",
}


# Mirrors build_compressed_tensors_verbalizer._WEIGHT_FILE_PATTERNS — files we
# never copy from the upstream snapshot dir because our save regenerates them
# with a different sharding layout / format.
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

    Mirrors `build_compressed_tensors_verbalizer._copy_upstream_extras` so
    the produced checkpoint has nla_meta.yaml, value_head.safetensors,
    license, chat template, etc. alongside the freshly-written quantized
    weights. Files already written by our save (e.g. config.json,
    tokenizer files, the README we wrote ourselves) are skipped, except
    upstream `README.md` which is preserved as `README_upstream.md`.
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


def _write_reconstructor_readme(
    out_dir: Path, reconstructor_id: str, scheme: str
) -> None:
    tag = _SCHEME_TAG[scheme]
    sglang_note = {
        "W8A16": (
            "loadable by sglang via `CompressedTensorsW8A16Fp8` (lazy-imports "
            "vllm only on sm_<89 hosts; native sm_89+)"
        ),
        "FP8_DYNAMIC": (
            "loadable by sglang via `CompressedTensorsW8A8Fp8` (native, no "
            "vllm dependency). FP8 tensor-core compute on Hopper sm_90+; "
            "Ada/Ampere fall back to dequant->bf16 GEMM"
        ),
        "NVFP4A16": (
            "loadable by sglang via `CompressedTensorsW4A16Fp4` (sglang "
            "PR #18116). **Requires Blackwell sm_100+** for the FP4 GEMM "
            "path; older GPUs will refuse to load. HF transformers can "
            "load it anywhere"
        ),
        "W4A16": (
            "loadable by sglang via `CompressedTensorsWNA16` (portable, "
            "Ampere+). HF transformers loads it anywhere"
        ),
    }[scheme]

    text = f"""---
library_name: transformers
base_model: {reconstructor_id}
tags:
- quantized
- {tag.lower()}
- compressed-tensors
- llmcompressor
- nla
---

# {reconstructor_id} ({tag}, compressed-tensors)

NLA reconstructor with **`{scheme}`** quantization on the backbone via
`llmcompressor`. Saved in **`compressed-tensors` format** (canonical
`*.safetensors`, no pickle), so HF transformers loads it directly with
no torchao dependency.

- **`{scheme}`** quantization on every `nn.Linear` weight in the backbone.
- `lm_head` is **not** quantized — `NLAReconstructor` replaces it with
  `nn.Identity()` at load time, so the lm_head weight is dropped after
  load anyway. Skipping it from quant keeps tied-embedding semantics
  intact for upstream models with `tie_word_embeddings=True`.
- `value_head.safetensors` is preserved in **bf16** alongside the
  quantized backbone — output projection, quality-critical.
- All other upstream files (`nla_meta.yaml`, model card, license,
  generation config, tokenizer extras, …) preserved verbatim. The
  upstream `README.md` is renamed to `README_upstream.md`.

This checkpoint is {sglang_note}.

## Usage with `apps/nla/server.py`

```bash
export NLA_RECONSTRUCTOR_MODEL=<this repo>
# Do NOT set NLA_FP8_RECONSTRUCTOR or NLA_INT4_RECONSTRUCTOR — the
# backbone is already quantized at rest. Setting them on top would
# either be a no-op (transformers loads the embedded quantization_config
# first) or cause a quantize-on-quantized failure.
uv run server.py --truncate-source --fp8-source
```

## Standalone load (HF transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("<this repo>")
model = AutoModelForCausalLM.from_pretrained(
    "<this repo>", torch_dtype=torch.bfloat16, device_map="cuda"
).eval()
```

Loading requires `compressed-tensors` (already a transitive dep of
`transformers >= 4.45` when `quantization_config.quant_method == "compressed-tensors"`).

## Comparison vs `apps/nla/build_quantized_models.py --target reconstructor-fp8`

That sibling script uses **torchao** (`AffineQuantizedTensor`-backed)
FP8 + pickled `pytorch_model-*.bin` shards. The torchao path has had
recurring loading regressions across HF / torchao version pairs in
this stack and is not loadable by sglang at any version we've tested.
Use this `compressed-tensors` checkpoint as the canonical pre-quantized
reconstructor.
"""
    (out_dir / "README.md").write_text(text)


def _make_recipe(scheme: str, ignore_modules: list[str]):
    """Build a llmcompressor `QuantizationModifier` for `scheme`.

    All four supported schemes are calibration-free (RTN / dynamic). The
    static `FP8` (W8A8 with calibrated per-tensor activation scales)
    variant is rejected here because it requires a calibration dataset
    we don't wire up — pass `W8A16` (weight-only FP8) instead, which is
    the recommended low-drift option for the reconstructor.
    """
    from llmcompressor.modifiers.quantization import (  # type: ignore[import-not-found]
        QuantizationModifier,
    )

    if scheme == "FP8":
        raise SystemExit(
            "scheme=FP8 (static W8A8 with calibrated activation scales) "
            "requires a calibration dataset; this script doesn't wire that "
            "up. Use W8A16 (weight-only FP8, default for --quant fp8) for "
            "minimal activation drift, or FP8_DYNAMIC (per-token activation "
            "scales, no calibration) if decode latency dominates."
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


def build_compressed_tensors_reconstructor(
    reconstructor_id: str,
    out_dir: Path,
    *,
    scheme: str = DEFAULT_FP8_SCHEME,
    device_map: str = "auto",
    ignore_modules: list[str] | None = None,
) -> None:
    """Quantize the reconstructor backbone with llmcompressor + compressed-tensors.

    Loads the upstream model in bf16 (sharded via `device_map="auto"`),
    runs `oneshot` quantization in place, saves to `out_dir` in
    compressed-tensors format, then copies upstream NLA extras
    (`nla_meta.yaml`, `value_head.safetensors`, prompt templates, etc.)
    verbatim.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from llmcompressor import oneshot  # type: ignore[import-not-found]
    except ImportError:
        # Older llmcompressor versions exposed oneshot under a different path.
        from llmcompressor.transformers import oneshot  # type: ignore[import-not-found]

    if ignore_modules is None:
        ignore_modules = ["lm_head"]

    log.info("[reconstructor] resolving %s (snapshot_download)…", reconstructor_id)
    src_dir = Path(_resolve_checkpoint_path(reconstructor_id))

    log.info(
        "[reconstructor] loading %s with device_map=%r (bf16)…",
        reconstructor_id,
        device_map,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(src_dir),
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(src_dir), trust_remote_code=True)

    recipe = _make_recipe(scheme, ignore_modules)

    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "[reconstructor] running llmcompressor oneshot with scheme=%s, "
        "ignore=%r…",
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

    # llmcompressor's `oneshot(output_dir=...)` already calls save_pretrained
    # with `save_compressed=True`. Tokenizer save is sometimes implicit,
    # sometimes not — re-save explicitly to be version-independent.
    log.info("[reconstructor] saving tokenizer (idempotent)…")
    tokenizer.save_pretrained(str(out_dir))

    log.info("[reconstructor] writing README…")
    _write_reconstructor_readme(out_dir, reconstructor_id, scheme)

    log.info(
        "[reconstructor] copying upstream extras "
        "(nla_meta.yaml, value_head.safetensors, prompt templates, …)…"
    )
    _copy_upstream_extras(src_dir, out_dir, label="reconstructor")

    # Sanity-check the value_head landed in the output dir. The
    # reconstructor is unusable without it; if upstream renamed or moved
    # it, fail loudly here rather than at server load time.
    if not (out_dir / "value_head.safetensors").exists():
        raise SystemExit(
            f"value_head.safetensors not present in {out_dir} after copying "
            f"upstream extras from {src_dir}. The reconstructor cannot be "
            f"loaded without it. Inspect upstream layout."
        )

    del model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("[reconstructor] done -> %s", out_dir)


def upload_to_hf(
    local_dir: Path,
    repo_id: str,
    *,
    token: str,
    private: bool = True,
    commit_message: str = "Upload compressed-tensors NLA reconstructor",
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


def _resolve_scheme(quant: str | None, scheme: str | None) -> str:
    """Resolve the explicit `--scheme` (preferred) or `--quant fp8/fp4` shorthand.

    `--scheme` always wins. `--quant` is a friendly shorthand: `fp8` -> the
    weight-only FP8 default (W8A16, low-drift), `fp4` -> the weight-only FP4
    default (NVFP4A16). If neither is given, default to W8A16 (matches the
    legacy torchao Float8WeightOnlyConfig path).
    """
    if scheme is not None:
        return scheme
    if quant is None:
        return DEFAULT_FP8_SCHEME
    quant = quant.lower()
    if quant == "fp8":
        return DEFAULT_FP8_SCHEME
    if quant == "fp4":
        return DEFAULT_FP4_SCHEME
    raise SystemExit(
        f"--quant must be 'fp8' or 'fp4' (got {quant!r}); pass --scheme to "
        f"select an explicit recipe out of {', '.join(SUPPORTED_SCHEMES)}."
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--reconstructor-model",
        default=DEFAULT_RECONSTRUCTOR,
        help=(
            f"Upstream reconstructor (HF repo or local dir). "
            f"Default: {DEFAULT_RECONSTRUCTOR}."
        ),
    )
    p.add_argument(
        "--quant",
        default=None,
        choices=("fp8", "fp4"),
        help=(
            "Friendly shorthand for picking a scheme: 'fp8' -> "
            f"{DEFAULT_FP8_SCHEME} (weight-only FP8, recommended for "
            "minimal activation drift); 'fp4' -> "
            f"{DEFAULT_FP4_SCHEME} (weight-only FP4). Use --scheme to pin "
            "an exact recipe."
        ),
    )
    p.add_argument(
        "--scheme",
        default=None,
        choices=SUPPORTED_SCHEMES,
        help=(
            "Explicit llmcompressor scheme. Overrides --quant. Choices: "
            "W8A16 (FP8 weight-only, bf16 acts; default for --quant fp8), "
            "FP8_DYNAMIC (W8A8 FP8 with per-token dynamic act scales), "
            "NVFP4A16 (FP4 weight-only, bf16 acts; default for --quant fp4), "
            "W4A16 (INT4 weight-only, bf16 acts; portable to Ampere+)."
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
            "<upstream-basename>-<scheme-tag> (e.g. "
            "nla-gemma3-27b-ar-FP8 for W8A16, "
            "nla-gemma3-27b-ar-FP4 for NVFP4A16)."
        ),
    )
    p.add_argument(
        "--device-map",
        default="auto",
        help=(
            "device_map passed to from_pretrained. 'auto' shards across "
            "visible GPUs (recommended for 27B-class on multi-GPU). Use "
            "'cuda:0' if the model fits on one GPU in bf16 (~54 GB for "
            "Gemma3-27B)."
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

    scheme = _resolve_scheme(args.quant, args.scheme)
    tag = _SCHEME_TAG[scheme]

    upstream_basename = args.reconstructor_model.rsplit("/", 1)[-1]
    target_name = args.target_name or f"{upstream_basename}-{tag}"
    out_dir = Path(args.output_dir or f"./quantized_models/{target_name}").resolve()

    log.info(
        "[plan] upstream=%s scheme=%s output=%s upload=%s",
        args.reconstructor_model,
        scheme,
        out_dir,
        args.upload,
    )

    build_compressed_tensors_reconstructor(
        args.reconstructor_model,
        out_dir,
        scheme=scheme,
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
            commit_message=(
                f"Upload {args.reconstructor_model} ({scheme}, "
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
