# Based on the Jacobian lens ("jlens") reference implementation by Anthropic PBC.
# Companion code for the "Verbalizable Workspace" paper.
# https://github.com/anthropics/jlens
# SPDX-License-Identifier: Apache-2.0
#
# Bring a lens fitted OUTSIDE this repo up to the format the inference server reads,
# and give it the config.yaml every run-all-fit-lens.py export has.
#
# There is no format conversion here despite the name: an external lens written by
# jlens' own `JacobianLens.save` already has the keys our loader wants (`J`,
# `source_layers`, `n_prompts`, `d_model`). What it lacks is a record of WHICH
# activation `J_bar` decodes, and on a multi-stream trunk that is not recoverable
# from the file -- every candidate reduction of the stream stack yields the same
# `d_model` and the same matrix shape. Read it wrong and the served tokens are
# plausible and wrong, so the server refuses rather than guesses, and this script is
# how the answer gets written down.
#
# Usage:
# uv run python convert-external-lens.py \
#   ../exports/deepseek-v4-flash/jlens/NeelNanda-pile-10k/deepseek-v4-flash_jacobian_lens_cblank.pt \
#   --np-model-id deepseek-v4-flash \
#   --capture-point block_output --stream-reduce mean \
#   --derived-from "rlens global_workspace/relp_jlens/mhc_fit.py" \
#   --backup-dir ~/old_pts

import argparse
import datetime as dt
import json
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_MODEL_MAP_PATH = REPO_ROOT / "np_model_to_hf.json"

# The closed vocabulary for `provenance["capture_point"]`: WHERE the fit hooked the
# model, named architecturally rather than as an interp-engine address. The server maps
# these to an address, because which address serves a boundary depends on the model's
# stream count and the backend -- `block_output` is `resid_post` on a conventional trunk
# and `resid_streams` on a hyper-connection one. Keeping engine names out of the artifact
# also keeps published lenses readable by consumers that do not have the engine, and
# survives a canonical rename there.
#
# DUPLICATED in apps/inference/neuronpedia_inference/endpoints/lens/lens_loader.py, which
# is a separate uv project and cannot import this. Change both or neither.
CAPTURE_POINTS = ("block_output", "attn_out", "mlp_out", "attn_in", "mlp_in")

# How a `[tokens, streams, d_model]` stack is reduced to the `[tokens, d_model]` vector
# `J_bar` was fitted on. `none` is the only valid value on a single-stream trunk, where
# there is no stream axis to reduce.
#
# `mean` and `sum` are the same lens: scaling source and target alike leaves the Jacobian
# unchanged, and the resulting factor dies in the final norm before the unembed. `select`
# is genuinely different, which is why this cannot be inferred.
STREAM_REDUCTIONS = ("none", "mean", "sum", "select")

ATTRIBUTION = (
    "Jacobian lens ('jlens') by Anthropic PBC — companion code for the "
    "'Verbalizable Workspace' paper (https://github.com/anthropics/jlens), "
    "Apache-2.0. Fitted externally; converted via Neuronpedia convert-external-lens.py."
)

# Provenance keys that map onto a field config.yaml already has, so two lenses stay
# comparable under one field name instead of one of them hiding in extra_metadata.
# Everything else in the source provenance is carried through verbatim.
_MAPPED_PROVENANCE_KEYS = frozenset(
    {"model_id", "dataset_id", "t_max", "n_prompts", "target_layer", "skip_first"}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record which activation an externally fitted Jacobian lens decodes, and "
            "write the config.yaml our own exports carry."
        )
    )
    parser.add_argument("lens", help="path to the .pt to convert")
    parser.add_argument(
        "--capture-point",
        required=True,
        choices=CAPTURE_POINTS,
        help="where the fit hooked the model (architectural name, not an engine address)",
    )
    parser.add_argument(
        "--stream-reduce",
        required=True,
        choices=STREAM_REDUCTIONS,
        help="how the stream stack was reduced to d_model ('none' on a single-stream trunk)",
    )
    parser.add_argument(
        "--stream-index",
        type=int,
        default=None,
        help="which stream, required with --stream-reduce select and forbidden otherwise",
    )
    parser.add_argument(
        "--np-model-id",
        required=True,
        help="neuronpedia model id, e.g. deepseek-v4-flash",
    )
    parser.add_argument(
        "--hf-model-name",
        default=None,
        help="HF model id; defaults to the source provenance's model_id, else np_model_to_hf.json",
    )
    parser.add_argument(
        "--derived-from",
        default=None,
        help=(
            "where the capture point was established, when the fit did not record it. "
            "Written to provenance so a later reader can tell a reconstructed claim from "
            "a recorded one."
        ),
    )
    parser.add_argument(
        "--backup-dir",
        default=None,
        help="move the original .pt here before writing the converted one in its place",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="write here instead of over the input (skips --backup-dir)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite a capture point the file already declares differently",
    )
    return parser.parse_args()


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _yaml_dump(obj: dict[str, Any], indent: int = 0) -> str:
    """Minimal YAML emitter for nested dicts of scalars (no external deps).

    Deliberately the same emitter run-all-fit-lens.py uses, so the two write the same
    dialect. It has no list support: a list would fall through to `_yaml_scalar` and come
    out as a quoted string, which is why `levels` is stored as "1e-2,5e-3,1e-3" there and
    why extra_metadata below must stay scalars and nested dicts of scalars.
    """
    pad = "  " * indent
    lines: list[str] = []
    for key, value in obj.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.append(_yaml_dump(value, indent + 1))
        else:
            lines.append(f"{pad}{key}: {_yaml_scalar(value)}")
    return "\n".join(line for line in lines if line)


def validate_structure(
    checkpoint: dict[str, Any], path: Path
) -> tuple[list[int], int, int]:
    """Check the invariants the inference loader depends on. Returns (layers, d_model, target).

    Cheaper to fail here, once, than to have the server discover any of these on a request.
    """
    if "J" not in checkpoint:
        raise SystemExit(
            f"{path} is not a Jacobian lens file (keys: {sorted(checkpoint)!r})"
        )
    jacobians = checkpoint["J"]
    if not isinstance(jacobians, dict):
        raise SystemExit(
            f"J is {type(jacobians).__name__}, expected a dict keyed by layer"
        )
    layers = sorted(jacobians)

    # `stacked_jacobians` and the server's per-layer indexing both assume rows are layers
    # 0..n-1, so a gap would silently misalign every read-out rather than fail.
    if layers != list(range(len(layers))):
        raise SystemExit(f"source_layers are not contiguous from 0: {layers}")

    d_model = int(checkpoint["d_model"])
    for layer in layers:
        shape = tuple(jacobians[layer].shape)
        if shape != (d_model, d_model):
            raise SystemExit(f"J[{layer}] is {shape}, expected ({d_model}, {d_model})")

    recorded = checkpoint.get("source_layers")
    if recorded is not None and list(recorded) != layers:
        raise SystemExit(
            f"source_layers {list(recorded)} disagrees with J's keys {layers}"
        )

    # The target row transports the target to itself, so it must be the identity. If it is
    # not, `target_layer` does not mean what the readout takes it to mean.
    target = int(checkpoint.get("provenance", {}).get("target_layer", layers[-1]))
    if target in jacobians:
        anchor = jacobians[target].to(torch.float32)
        err = (anchor - torch.eye(d_model)).abs().max().item()
        if err > 1e-2:
            raise SystemExit(
                f"J[{target}] is not the identity (max|J - I| = {err:.2e})"
            )
        print(f"  anchor: max|J[{target}] - I| = {err:.2e}")
    else:
        print(
            f"  note: target_layer {target} has no row; the readout treats it as J = I"
        )

    return layers, d_model, target


def resolve_hf_model_name(
    args: argparse.Namespace, provenance: dict[str, Any]
) -> str | None:
    """The HF model id, preferring the flag, then the source provenance, then the repo map."""
    if args.hf_model_name:
        return args.hf_model_name
    recorded = provenance.get("model_id")
    if recorded:
        return str(recorded)
    if DEFAULT_MODEL_MAP_PATH.is_file():
        with open(DEFAULT_MODEL_MAP_PATH) as f:
            return json.load(f).get(args.np_model_id)
    return None


def build_capture_fields(
    args: argparse.Namespace, provenance: dict[str, Any]
) -> dict[str, Any]:
    """The residual-definition fields to add, refusing to silently overwrite a different one."""
    if args.stream_reduce == "select":
        if args.stream_index is None:
            raise SystemExit("--stream-reduce select needs --stream-index")
    elif args.stream_index is not None:
        raise SystemExit(
            f"--stream-index is meaningless with --stream-reduce {args.stream_reduce}"
        )

    existing = {
        key: provenance.get(key)
        for key in ("capture_point", "stream_reduce", "stream_index")
    }
    wanted = {
        "capture_point": args.capture_point,
        "stream_reduce": args.stream_reduce,
        "stream_index": args.stream_index,
    }
    declared = {key: value for key, value in existing.items() if value is not None}
    if (
        declared
        and declared != {k: v for k, v in wanted.items() if v is not None}
        and not args.force
    ):
        raise SystemExit(
            f"this file already declares {declared!r}, which differs from {wanted!r}. "
            "A lens knows what it was fitted on better than a command line does -- "
            "re-run with --force only if the recorded value is known to be wrong."
        )

    fields: dict[str, Any] = dict(wanted)
    if args.derived_from:
        fields["capture_point_derived_from"] = (
            f"{args.derived_from} (reconstructed by convert-external-lens.py; "
            "the original fit did not record it)"
        )
    return fields


def write_config_yaml(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    lens_path: Path,
    hf_model_name: str | None,
    provenance: dict[str, Any],
    capture: dict[str, Any],
    layers: list[int],
    d_model: int,
    target: int,
) -> Path:
    """config.yaml in run-all-fit-lens.py's shape, with the source fit's own fields carried through."""
    header = [
        "# Jacobian lens — converted by Neuronpedia convert-external-lens.py",
        f"# {ATTRIBUTION}",
        "#",
        "# This lens was NOT fitted by this repo, so the `fit` block below is the source",
        "# fit's parameters mapped onto our field names (null where it recorded nothing),",
        "# and `extra_metadata` carries everything of its provenance that has no counterpart.",
        "#",
        "# The .pt's own `provenance` is authoritative for the `lens` block. This copy is for",
        "# reading; the inference server loads the .pt and never opens this file.",
        "#",
        f"# Exact command used:\n#   {shlex.join(sys.argv)}",
        "#",
        f"# Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
    ]

    config_json = provenance.get("config_json")
    extra: dict[str, Any] = {
        key: value
        for key, value in sorted(provenance.items())
        if key not in _MAPPED_PROVENANCE_KEYS
    }
    if isinstance(config_json, str):
        try:
            parsed = json.loads(config_json)
        except json.JSONDecodeError:
            parsed = None
        # Only lift a flat dict: the emitter has no list support, and a nested rules dict
        # would come out as a quoted repr rather than YAML.
        if isinstance(parsed, dict) and all(
            not isinstance(v, (list, dict)) for v in parsed.values()
        ):
            extra |= {f"config_{key}": value for key, value in sorted(parsed.items())}

    body: dict[str, Any] = {
        "np_model_id": args.np_model_id,
        "hf_model_name": hf_model_name,
        "dataset": {
            "name": provenance.get("dataset_id"),
            "config": None,
            "split": None,
            "text_field": None,
            "max_chars": None,
        },
        "fit": {
            "n_prompts": provenance.get("n_prompts"),
            "dim_batch": None,
            "max_seq_len": provenance.get("t_max"),
            "target_layer": target,
            "skip_first": provenance.get("skip_first"),
            "dtype": None,
            "device_map": None,
            "compile": None,
            "trust_remote_code": None,
            "stop_at_delta": None,
            "min_prompts": None,
            "stop_window": None,
            "levels": None,
        },
        "lens": {
            "file": lens_path.name,
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
        "attribution": ATTRIBUTION,
    }

    out_path = out_dir / "config.yaml"
    with open(out_path, "w") as f:
        f.write("\n".join(header))
        f.write(_yaml_dump(body))
        f.write("\n")
    return out_path


def main() -> None:
    args = parse_args()
    lens_path = Path(args.lens).expanduser().resolve()
    if not lens_path.is_file():
        raise SystemExit(f"no such lens: {lens_path}")

    print(
        f"== reading {lens_path.name} ({lens_path.stat().st_size / 2**30:.2f} GiB) =="
    )
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    layers, d_model, target = validate_structure(checkpoint, lens_path)
    provenance: dict[str, Any] = dict(checkpoint.get("provenance") or {})
    dtypes = {str(checkpoint["J"][layer].dtype) for layer in layers}
    print(
        f"  d_model={d_model} layers={layers[0]}..{layers[-1]} ({len(layers)}) "
        f"target={target} dtype={'/'.join(sorted(dtypes))} "
        f"provenance={'present' if provenance else 'absent'}"
    )

    capture = build_capture_fields(args, provenance)
    hf_model_name = resolve_hf_model_name(args, provenance)
    checkpoint["provenance"] = provenance | capture
    # Keep these consistent with J rather than trusting what was recorded; `validate_structure`
    # has already refused a genuine disagreement.
    checkpoint["source_layers"] = layers

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = lens_path
        if args.backup_dir:
            backup_dir = Path(args.backup_dir).expanduser().resolve()
            backup_dir.mkdir(parents=True, exist_ok=True)
            destination = backup_dir / lens_path.name
            if destination.exists():
                raise SystemExit(
                    f"backup already exists, refusing to clobber: {destination}"
                )
            print(f"\n== backing up original -> {destination} ==")
            shutil.move(str(lens_path), str(destination))

    # Temp-then-rename so an interrupted save cannot leave a truncated lens at the real path.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    print(f"\n== writing {out_path} ==")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(out_path)
    print(f"  {out_path.stat().st_size / 2**30:.2f} GiB")

    config_path = write_config_yaml(
        out_path.parent,
        args=args,
        lens_path=out_path,
        hf_model_name=hf_model_name,
        provenance=provenance,
        capture=capture,
        layers=layers,
        d_model=d_model,
        target=target,
    )
    print(f"  {config_path}")
    print("\n== recorded ==")
    for key, value in capture.items():
        print(f"  {key}: {value!r}")


if __name__ == "__main__":
    main()
