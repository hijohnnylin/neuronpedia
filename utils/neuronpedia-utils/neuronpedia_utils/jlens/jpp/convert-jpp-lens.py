# J++ Lens by Ayonrinde & Lindsey -- companion code for "J++ Lens: Jacobian Filtering
# Enables More Faithful Workspace Lenses" (2026), Apache-2.0.
#
# Write a J++ Lens in the format the inference server reads, with a config.yaml.
#
# The J++ fit writes `parameters.jacobians` in fp32 under its own keys. This script keys the
# maps as `J`, casts them to fp16, and records which activation they read, as
# convert-external-lens.py does for an external Jacobian lens. The maps do not change.
#
# Usage:
# hf download koayon/jpp-lenses qwen3.5-9b/lens.pt --local-dir /tmp/koayon
# uv run python jpp/convert-jpp-lens.py /tmp/koayon/qwen3.5-9b/lens.pt \
#   --np-model-id qwen3.5-9b --source "koayon/jpp-lenses@<sha>:qwen3.5-9b/lens.pt"
#
# An R-Lens (the pooled lens of an LRP fit, merge-experts --pooled-lens-out) takes
# --method rlens and goes to <np_model_id>/rlens/<dataset-dir>/<slug>_rlens.pt.
# A Jacobian lens fitted with this code (lrp_mode 'none') takes --method jlens.
#
# DeepSeek-V4-Flash has 4 residual streams. The J++ fit reads their mean:
#   ... --stream-reduce mean --derived-from "safety-research/jpp_lens workspace_lens/residual_streams.py"

import argparse
import datetime as dt
import importlib
import importlib.util
import json
import pickle
import re
import shlex
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
JLENS_DIR = SCRIPT_DIR.parent

_spec = importlib.util.spec_from_file_location(
    "convert_external_lens", JLENS_DIR / "convert-external-lens.py"
)
assert _spec is not None and _spec.loader is not None
_external = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_external)
CAPTURE_POINTS = _external.CAPTURE_POINTS
STREAM_REDUCTIONS = _external.STREAM_REDUCTIONS
_yaml_dump = _external._yaml_dump

ATTRIBUTION = (
    "J++ Lens by Ayonrinde & Lindsey -- companion code for 'J++ Lens: Jacobian Filtering "
    "Enables More Faithful Workspace Lenses' (2026), Apache-2.0. Converted via Neuronpedia "
    "convert-jpp-lens.py; the maps are not changed."
)
RLENS_ATTRIBUTION = (
    "R-Lens fitted with the J++ Lens code by Ayonrinde & Lindsey (safety-research/jpp_lens, "
    "2026), Apache-2.0: the pooled lens of an LRP fit (lrp_mode 'rlens'). Converted via "
    "Neuronpedia convert-jpp-lens.py; the maps are not changed."
)
JLENS_ATTRIBUTION = (
    "Jacobian lens ('jlens') by Anthropic PBC (https://github.com/anthropics/jlens), "
    "fitted with the J++ Lens code by Ayonrinde & Lindsey (safety-research/jpp_lens, "
    "2026), Apache-2.0: the pooled lens of a plain fit (lrp_mode 'none'). Converted via "
    "Neuronpedia convert-jpp-lens.py; the maps are not changed."
)
# --method: (lens name, folder and file suffix, attribution).
METHODS = {
    "jpp": ("J++ Lens", "jpp", "_jpp_lens.pt", ATTRIBUTION),
    "rlens": ("R-Lens", "rlens", "_rlens.pt", RLENS_ATTRIBUTION),
    "jlens": ("Jacobian Lens", "jlens", "_jacobian_lens.pt", JLENS_ATTRIBUTION),
}

# fp16 keeps the relative error near 2e-4. Layers with very small values (Gemma 4 31B) reach
# 1.1e-3. The server holds the maps in bf16 (about 1.7e-3), so below this limit the fp16
# file loses less than serving does.
MAX_FP16_REL_ERR = 2e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a J++ Lens (.pt from safety-research/jpp_lens) in the Neuronpedia format."
    )
    parser.add_argument(
        "lens", help="the J++ fit's .pt (parameters.jacobians + config)"
    )
    parser.add_argument("--np-model-id", required=True, help="e.g. qwen3.6-27b")
    parser.add_argument(
        "--method",
        default="jpp",
        choices=sorted(METHODS),
        help=(
            "'rlens' for the pooled lens of an LRP fit (merge-experts --pooled-lens-out), "
            "'jlens' for a plain fit"
        ),
    )
    parser.add_argument(
        "--hf-model-name",
        default=None,
        help="HF model id; defaults to the fit config's hf_model_name",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="where the .pt came from, e.g. 'koayon/jpp-lenses@<sha>:qwen3.6-27b/lens.pt'",
    )
    parser.add_argument(
        "--out-dir",
        default="exports",
        help="writes <out-dir>/<np_model_id>/<method>/<dataset-dir>/<slug>_<method>[_lens].pt",
    )
    parser.add_argument(
        "--dataset-dir",
        default="Salesforce-wikitext",
        help="the fit corpus folder name (the J++ recipe fits on WikiText-103)",
    )
    parser.add_argument(
        "--capture-point", default="block_output", choices=CAPTURE_POINTS
    )
    parser.add_argument(
        "--stream-reduce",
        default="none",
        choices=STREAM_REDUCTIONS,
        help="'none' on a single-stream trunk; 'mean' for DeepSeek-V4",
    )
    parser.add_argument("--stream-index", type=int, default=None)
    parser.add_argument(
        "--derived-from",
        default=None,
        help="where the capture point comes from, when the fit did not record it",
    )
    parser.add_argument(
        "--no-hf-check",
        action="store_true",
        help="do not compare the layers and d_model with the model's config.json",
    )
    parser.add_argument(
        "--allow-global",
        action="append",
        default=[],
        metavar="MODULE.NAME",
        help=(
            "let the weights-only load build this object, e.g. pathlib.PosixPath. "
            "Repeat for each. Use only after the load names it and you trust it."
        ),
    )
    return parser.parse_args()


def _resolve_global(dotted: str) -> Any:
    module, _, name = dotted.rpartition(".")
    if not module:
        raise SystemExit(f"--allow-global needs MODULE.NAME, not {dotted!r}")
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise SystemExit(f"--allow-global {dotted}: {exc}") from exc


def load_fit(path: Path, allow_globals: list[str]) -> dict[str, Any]:
    """Load the fit without running code from the file.

    Lens files come from third parties, and a full unpickle runs any code they hold.
    A weights-only load admits only tensors and plain containers, which is all a
    J++ fit needs. When a file holds another object, stop and name it.
    """
    allowed = [_resolve_global(name) for name in allow_globals]
    try:
        with torch.serialization.safe_globals(allowed):
            return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except pickle.UnpicklingError as exc:
        # Torch's message also suggests weights_only=False, so show only the line that
        # names the object.
        detail = next(
            (
                line.strip()
                for line in str(exc).splitlines()
                if "WeightsUnpickler error" in line
            ),
            str(exc),
        )
        raise SystemExit(
            f"{path} holds an object that a weights-only load refuses:\n  {detail}\n"
            "Make sure that you trust this object, then add --allow-global MODULE.NAME for it."
        ) from exc


def _slug(hf_model_id: str) -> str:
    """The file stem. Same as `_slug` in the inference lens_loader."""
    base = hf_model_id.rstrip("/").split("/")[-1]
    return re.sub(r"[^0-9A-Za-z._-]+", "-", base).strip("-") or "model"


def validate(raw: dict[str, Any], path: Path) -> tuple[list[int], int, dict[str, Any]]:
    """Check what the server needs. Returns (layers, d_model, config)."""
    jacobians = (raw.get("parameters") or {}).get("jacobians")
    config = raw.get("config")
    if not isinstance(jacobians, dict) or not isinstance(config, dict):
        raise SystemExit(f"{path} is not a J++ fit file (keys: {sorted(raw)!r})")
    layers = sorted(int(layer) for layer in jacobians)
    if layers != list(range(len(layers))):
        raise SystemExit(f"source layers are not contiguous from 0: {layers}")
    recorded = raw.get("source_layers")
    if recorded is not None and sorted(int(x) for x in recorded) != layers:
        raise SystemExit(
            f"source_layers {list(recorded)} disagrees with the maps {layers}"
        )
    if config.get("relative_end_transport_layer", -1) != -1:
        raise SystemExit(
            "the maps transport to block "
            f"{config['relative_end_transport_layer']}, not the final block"
        )
    d_model = int(config["d_model"])
    for layer in layers:
        shape = tuple(jacobians[layer].shape)
        if shape != (d_model, d_model):
            raise SystemExit(
                f"layer {layer} is {shape}, expected ({d_model}, {d_model})"
            )
    return layers, d_model, config


def check_hf_config(hf_model_name: str, layers: list[int], d_model: int) -> None:
    """The maps cover every block but the final one, at the model's width."""
    from huggingface_hub import hf_hub_download

    try:
        with open(hf_hub_download(hf_model_name, "config.json")) as f:
            cfg = json.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"  warning: no config.json for {hf_model_name} ({exc}); not checked")
        return
    text = cfg.get("text_config") or cfg
    n_layers, hidden = int(text["num_hidden_layers"]), int(text["hidden_size"])
    if hidden != d_model or layers != list(range(n_layers - 1)):
        raise SystemExit(
            f"{hf_model_name} has {n_layers} layers at width {hidden}; the lens has "
            f"layers {layers[0]}..{layers[-1]} at width {d_model}"
        )
    print(f"  matches {hf_model_name}: {n_layers} layers, d_model={hidden}")


def to_fp16(
    jacobians: dict[Any, torch.Tensor],
) -> tuple[dict[int, torch.Tensor], float]:
    out, worst = {}, 0.0
    for layer, jac in jacobians.items():
        full = jac.to(torch.float32)
        half = full.to(torch.float16).contiguous()
        if not torch.isfinite(half).all():
            raise SystemExit(f"layer {layer} overflows fp16")
        worst = max(worst, ((half.float() - full).norm() / full.norm()).item())
        out[int(layer)] = half
    if worst > MAX_FP16_REL_ERR:
        raise SystemExit(
            f"fp16 relative error {worst:.2e} is above {MAX_FP16_REL_ERR:.0e}"
        )
    return out, worst


def capture_fields(args: argparse.Namespace) -> dict[str, Any]:
    if (args.stream_reduce == "select") != (args.stream_index is not None):
        raise SystemExit(
            "--stream-index goes with --stream-reduce select, and only with it"
        )
    fields: dict[str, Any] = {
        "capture_point": args.capture_point,
        "stream_reduce": args.stream_reduce,
        "stream_index": args.stream_index,
    }
    if args.derived_from:
        fields["capture_point_derived_from"] = (
            f"{args.derived_from} (recorded by convert-jpp-lens.py; the fit did not record it)"
        )
    return fields


def write_config_yaml(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    lens_path: Path,
    hf_model_name: str,
    config: dict[str, Any],
    capture: dict[str, Any],
    layers: list[int],
    d_model: int,
    fp16_rel_err: float,
) -> Path:
    """config.yaml in the shape the Jacobian lens exports have."""
    lens_name, _, _, attribution = METHODS[args.method]
    header = [
        f"# {lens_name} -- converted by Neuronpedia convert-jpp-lens.py",
        f"# {attribution}",
        "#",
        "# This lens was NOT fitted by run-all-fit-lens.py. The `fit` block is the J++ fit's",
        "# config mapped onto our field names (null where it has no counterpart), and",
        "# `extra_metadata` carries the rest of it.",
        "#",
        "# The .pt's own `provenance` is authoritative. This copy is for reading; the",
        "# inference server loads the .pt and never opens this file.",
        "#",
        f"# Exact command used:\n#   {shlex.join(sys.argv)}",
        "#",
        f"# Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
    ]
    mapped = {
        "hf_model_name",
        "d_model",
        "num_prompts_trained_on",
        "max_seq_len",
        "skip_first_n_positions",
        "source_layers",
    }
    extra = {
        key: value
        if isinstance(value, (str, int, float, bool)) or value is None
        else str(value)
        for key, value in sorted(config.items())
        if key not in mapped
    }
    body: dict[str, Any] = {
        "np_model_id": args.np_model_id,
        "hf_model_name": hf_model_name,
        "dataset": {
            "name": "Salesforce/wikitext"
            if args.dataset_dir == "Salesforce-wikitext"
            else None,
            "config": None,
            "split": None,
            "text_field": None,
            "max_chars": None,
        },
        "fit": {
            "method": args.method,
            "n_prompts": config.get("num_prompts_trained_on"),
            "max_seq_len": config.get("max_seq_len"),
            "target_layer": layers[-1] + 1,
            "skip_first": config.get("skip_first_n_positions"),
            "lrp_mode": config.get("lrp_mode"),
        },
        "lens": {
            "file": lens_path.name,
            "source": args.source,
            "dtype": "float16",
            "fp16_max_rel_err": fp16_rel_err,
            "d_model": d_model,
            "source_layers_first": layers[0],
            "source_layers_last": layers[-1],
            "n_source_layers": len(layers),
            "capture_point": capture["capture_point"],
            "stream_reduce": capture["stream_reduce"],
            "stream_index": capture["stream_index"],
            "capture_point_derived_from": capture.get("capture_point_derived_from"),
        },
        "extra_metadata": extra or None,
        "command": shlex.join(sys.argv),
        "attribution": attribution,
    }
    out_path = out_dir / "config.yaml"
    with open(out_path, "w") as f:
        f.write("\n".join(header))
        f.write(_yaml_dump(body))
        f.write("\n")
    return out_path


def main() -> None:
    args = parse_args()
    src = Path(args.lens).expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"no such lens: {src}")
    print(f"== reading {src.name} ({src.stat().st_size / 2**30:.2f} GiB) ==")
    raw = load_fit(src, args.allow_global)
    layers, d_model, config = validate(raw, src)
    _, method_dir, file_suffix, _ = METHODS[args.method]
    if args.method == "rlens" and config.get("lrp_mode") != "rlens":
        raise SystemExit(
            f"--method rlens needs an lrp_mode 'rlens' fit, not {config.get('lrp_mode')!r}"
        )
    if args.method == "jlens" and config.get("lrp_mode") not in (None, "none"):
        raise SystemExit(
            "--method jlens needs a plain fit (lrp_mode 'none'), "
            f"not {config.get('lrp_mode')!r}"
        )
    hf_model_name = args.hf_model_name or config.get("hf_model_name")
    if not hf_model_name:
        raise SystemExit("no --hf-model-name, and the fit config has none")
    print(
        f"  {hf_model_name}: d_model={d_model} layers={layers[0]}..{layers[-1]} ({len(layers)})"
    )
    if not args.no_hf_check:
        check_hf_config(hf_model_name, layers, d_model)

    capture = capture_fields(args)
    J, worst = to_fp16(raw["parameters"]["jacobians"])
    print(f"  fp16 max relative error {worst:.2e}")
    out = {
        "J": J,
        "source_layers": layers,
        "d_model": d_model,
        "n_prompts": int(config.get("num_prompts_trained_on", 0)),
        "provenance": {
            "source": args.source,
            "method": args.method,
            "lrp_mode": config.get("lrp_mode"),
            "source_config": json.loads(json.dumps(config, default=str)),
            **capture,
        },
    }

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        / args.np_model_id
        / method_dir
        / args.dataset_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_slug(hf_model_name)}{file_suffix}"
    # Temp-then-rename, so a stopped save leaves no short file at the real path.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    print(f"\n== writing {out_path} ==")
    torch.save(out, tmp_path)
    tmp_path.replace(out_path)
    print(f"  {out_path.stat().st_size / 2**30:.2f} GiB")
    config_path = write_config_yaml(
        out_dir,
        args=args,
        lens_path=out_path,
        hf_model_name=hf_model_name,
        config=config,
        capture=capture,
        layers=layers,
        d_model=d_model,
        fp16_rel_err=worst,
    )
    print(f"  {config_path}")


if __name__ == "__main__":
    main()
