"""Convert a legacy pickled persona asset into per-axis safetensors directories.

Run by hand, once per asset; the output is committed and this script is not part of any
build. It exists because the legacy assets are pickled sklearn ``PCA`` objects, so something
has to unpickle them exactly once to get the numbers out.

The legacy format put every component of a fit in one directory and read the first one out of
``contrast_vectors.pt``, which meant a fit could only declare one layer. The new format is one
directory per axis, each holding the three vectors a projection actually needs plus the
scalars that calibrate it, so an asset whose components live at different layers needs no
manifest extension.

What the runtime projects onto is ``components_`` *after* the loader's
``components_[0] = normalize(contrast_vectors[layer]) * -1`` overwrite, so this replicates that
overwrite rather than second-guessing it. For an asset that stored the negated unit component 0
to make the overwrite a no-op, that is the identity; for one whose fitted PC1 was genuinely
discarded, it is the contrast vector. Either way the extracted direction is what was being
served.

Usage:

    python convert_to_safetensors.py SRC DST --author NAME [--ids a,b,c] [--verify]

``SRC`` is a legacy asset directory (``persona.yaml`` + ``contrast_vectors.pt`` + ``pca/*.pt``,
plus ``quantiles.pt`` when the fit produced quantile tables) and ``DST`` is the model directory to
write axis subdirectories into. ``--verify`` re-runs the legacy projection through the original
sklearn objects and fails if the new path disagrees, and checks that the quantile tables stay
inside [-1, 1].

``--author`` is who fitted the asset, and applies to the whole run: one legacy asset is one
fit by one group. Directories come out named ``<author>_<name>``, and the author is written
into each manifest as well, since the loader checks the two agree.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import save_file

# Tolerance for the --verify parity check. The two paths differ only in float32 accumulation
# order (sklearn's matmul against ours), so anything above this is a real discrepancy.
PARITY_ATOL = 1e-5

# Matches L2MeanScaler's default. The new loader hardcodes it, so an asset that used a
# different value would silently change meaning and is rejected below instead.
EXPECTED_EPS = 1e-12


class MeanScaler:
    """Unpickling shim: the legacy assets pickled their scalers out of ``__main__``."""

    def __init__(self, mean=None):
        self.mean = mean

    def transform(self, X):
        return np.asarray(X) - self.mean


class L2MeanScaler:
    """Unpickling shim; see :class:`MeanScaler`."""

    def __init__(self, mean=None, eps: float = 1e-12):
        self.mean = mean
        self.eps = eps

    def transform(self, X):
        centered = np.asarray(X) - self.mean
        norms = np.linalg.norm(centered, ord=2, axis=-1, keepdims=True)
        return centered / np.maximum(norms, self.eps)


NORMALIZE_BY_SCALER = {"MeanScaler": "none", "L2MeanScaler": "l2"}


def _install_shims() -> None:
    """Make the pickles resolvable, exactly as ``persona_data.py`` used to."""
    main_module = sys.modules["__main__"]
    for name, cls in (("MeanScaler", MeanScaler), ("L2MeanScaler", L2MeanScaler)):
        if not hasattr(main_module, name):
            setattr(main_module, name, cls)


def slugify(title: str) -> str:
    """A filesystem- and URL-safe id from a display title, as a fallback for --ids."""
    decomposed = unicodedata.normalize("NFKD", title)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", ascii_only.lower())).strip("-")


def as_vector(value: Any, hidden: int) -> np.ndarray:
    """Broadcast a scalar/None/array mean to a ``[hidden]`` float32 vector.

    A scaler fitted with a zero mean pickles that mean as a bare ``0`` in some assets and as a
    zero vector in others; both describe the same transform, so both become a vector here.
    """
    if value is None:
        return np.zeros(hidden, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return np.full(hidden, float(array), dtype=np.float32)
    if array.shape != (hidden,):
        raise ValueError(
            f"expected a scalar or [{hidden}] vector, got shape {array.shape}"
        )
    return array


def load_legacy(src: Path) -> tuple[dict, dict[int, dict], torch.Tensor]:
    """Read a legacy asset: its manifest, its per-layer pickles, and its contrast vectors."""
    with open(src / "persona.yaml", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    contrast = torch.load(
        src / manifest.get("contrast_vectors", "contrast_vectors.pt"),
        weights_only=False,
    )
    by_layer: dict[int, dict] = {}
    for entry in manifest["pca"]:
        layer = int(entry["layer"])
        by_layer[layer] = torch.load(src / entry["file"], weights_only=False)
    return manifest, by_layer, contrast


def served_components(
    pca_results: dict, contrast: torch.Tensor, layer: int
) -> np.ndarray:
    """``components_`` as the server actually projects onto them, overwrite included.

    The normalize-then-cast order is the loader's, and it matters: the 70B contrast vectors are
    stored bfloat16, so normalizing before the cast keeps the bfloat16 rounding the served
    direction has always had. Casting first would give a subtly different vector.
    """
    components = np.array(pca_results["pca"].components_, dtype=np.float32, copy=True)
    contrast_vector = F.normalize(contrast[layer], dim=0)
    components[0] = contrast_vector.float().numpy() * -1
    return components


def axis_layers(manifest: dict, n_axes: int) -> list[int]:
    """One layer per component.

    ``pc_layers`` is the extension a multi-layer asset carries; without it every component
    belongs to the single layer the manifest declares, which is what the old loader assumed.
    """
    declared = manifest.get("pc_layers")
    if declared:
        if len(declared) != n_axes:
            raise ValueError(
                f"pc_layers has {len(declared)} entries for {n_axes} titles"
            )
        return [int(layer) for layer in declared]
    layers = sorted(int(entry["layer"]) for entry in manifest["pca"])
    return [layers[0]] * n_axes


def scalar_list(manifest: dict, key: str, n_axes: int, default: float) -> list[float]:
    values = manifest.get(key)
    if not values:
        return [default] * n_axes
    if len(values) != n_axes:
        raise ValueError(f"{key} has {len(values)} entries for {n_axes} titles")
    return [float(value) for value in values]


def pole_scales(manifest: dict, n_axes: int) -> tuple[list[float], list[float]]:
    """One divisor per pole per axis: ``(pc_scales_pos, pc_scales_neg)``.

    These axes are lopsided, and a single divisor has to be the larger of the two spreads, so
    it squeezes the tighter pole towards zero -- ``erudite`` shipped that way and its positive
    pole could not read past +0.21. ``pc_scales`` is retained in the source as ``max(pos, neg)``,
    so an asset predating the per-pole keys falls back to exactly the behaviour it had rather
    than to something wrong.
    """
    shared = scalar_list(manifest, "pc_scales", n_axes, 1.0)
    per_pole: list[list[float]] = []
    for key in ("pc_scales_pos", "pc_scales_neg"):
        values = manifest.get(key)
        if not values:
            per_pole.append(list(shared))
            continue
        if len(values) != n_axes:
            raise ValueError(f"{key} has {len(values)} entries for {n_axes} titles")
        per_pole.append([float(value) for value in values])
    return per_pole[0], per_pole[1]


def pole_quantile_tables(
    manifest: dict, src: Path, n_axes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """``(levels, pos, neg)`` quantile tables, or ``None`` for an asset that ships none.

    These are what let a reading be reported as a percentile instead of a ratio. The ratio is a
    multiple of the p99 landmark, so it passes 1 for roughly 2% of turns by construction --
    accurate, and unreadable on a gauge that a person expects to stop at 100%. A percentile cannot
    leave [-1, 1] at all, so both go into the asset: the ratio as the measurement, the percentile as
    what a display shows.

    Optional because ``lu_assistant-axis`` predates them. An axis without tables simply has no
    percentile to report, which the loader and the UI both handle by falling back to the ratio.
    """
    declared = manifest.get("pc_quantiles")
    if not declared:
        return None
    tables = torch.load(src / declared["file"], weights_only=False)
    levels = np.asarray(tables["levels"], dtype=np.float32)
    pos = np.asarray(tables["pos"], dtype=np.float32)
    neg = np.asarray(tables["neg"], dtype=np.float32)
    for name, table in (("pos", pos), ("neg", neg)):
        if table.shape != (n_axes, levels.shape[0]):
            raise ValueError(
                f"{declared['file']}: {name} has shape {table.shape}, "
                f"expected {(n_axes, levels.shape[0])}"
            )
        # A table that is not sorted would make the interpolation non-monotone, which would
        # reorder readings rather than merely misplace them.
        if np.any(np.diff(table, axis=1) < 0):
            raise ValueError(f"{declared['file']}: {name} is not nondecreasing")
    if np.any(np.diff(levels) <= 0) or levels[0] != 0.0 or levels[-1] != 1.0:
        raise ValueError(f"{declared['file']}: levels must increase from 0 to 1")
    return levels, pos, neg


def percentile(
    raw: np.ndarray,
    center: float,
    levels: np.ndarray,
    table_pos: np.ndarray,
    table_neg: np.ndarray,
) -> np.ndarray:
    """The new loader's percentile map, restated so --verify exercises it against the tables."""
    d = raw - center
    forward = np.interp(d, table_pos, levels)
    backward = -np.interp(-d, table_neg, levels)
    return np.where(d >= 0, forward, backward)


def load_annotations(path: Path | None) -> dict[str, dict]:
    """Per-axis ``caveat`` and ``provenance``, keyed by axis name without the author prefix.

    Held in a file beside the converter rather than typed at the command line so that a rerun
    reproduces the committed YAML byte for byte. Neither key affects a projection: ``caveat`` is
    for the UI to surface, and ``provenance`` is recorded for humans and read by nothing.
    """
    if path is None:
        return {}
    with open(path, encoding="utf-8") as handle:
        annotations = json.load(handle)
    if not isinstance(annotations, dict):
        raise ValueError(f"{path}: expected an object keyed by axis name")
    return annotations


def render_block(manifest: dict) -> dict:
    """The rendering conditions the fit assumed, as one block.

    A lone ``blank_system_prompt`` was enough while one asset existed. Llama 3.1's template
    injects the current date into the system block, so a fit on that model is only reproducible
    with the date pinned too -- and anything else a future fit assumes about rendering belongs
    in the same place, where the server can compare two axes for agreement.
    """
    render: dict[str, Any] = {
        "blank_system_prompt": bool(manifest.get("blank_system_prompt", False))
    }
    template_kwargs = manifest.get("template_kwargs")
    if template_kwargs:
        render["template_kwargs"] = dict(template_kwargs)
    return render


def calibrate(
    raw: np.ndarray, center: float, scale_pos: float, scale_neg: float
) -> np.ndarray:
    """The new loader's per-pole map, restated so --verify exercises both branches."""
    d = raw - center
    return np.where(d >= 0, d / scale_pos, d / scale_neg)


def project_new(
    acts: np.ndarray,
    direction,
    scaler_mean,
    pca_mean,
    normalize: str,
    center: float,
    scale_pos: float,
    scale_neg: float,
):
    """The new loader's projection, restated here so --verify compares two implementations."""
    x = acts.astype(np.float32) - scaler_mean
    if normalize == "l2":
        norms = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
        x = x / np.maximum(norms, EXPECTED_EPS)
    raw = (x - pca_mean) @ direction
    return calibrate(raw, center, scale_pos, scale_neg)


def project_legacy(
    acts: np.ndarray, pca_results: dict, components: np.ndarray, index: int
) -> np.ndarray:
    """The legacy path: the asset's own scaler, then its PCA transform, then one column."""
    scaled = pca_results["scaler"].transform(acts.astype(np.float32))
    centered = scaled - pca_results["pca"].mean_
    return (centered @ components.T)[:, index]


def convert(
    src: Path,
    dst: Path,
    author: str,
    ids: list[str] | None,
    verify: bool,
    annotations_path: Path | None,
) -> None:
    _install_shims()
    manifest, by_layer, contrast = load_legacy(src)
    annotations = load_annotations(annotations_path)

    titles = list(manifest["pc_titles"])
    n_axes = len(titles)
    layers = axis_layers(manifest, n_axes)
    centers = scalar_list(manifest, "pc_centers", n_axes, 0.0)
    scales_pos, scales_neg = pole_scales(manifest, n_axes)
    quantiles = pole_quantile_tables(manifest, src, n_axes)
    render = render_block(manifest)

    if ids is None:
        ids = [slugify(title) for title in titles]
    # A typo'd key would otherwise drop a caveat silently, which is the one annotation whose
    # absence a reader cannot notice. Keys starting with "_" are commentary. Annotations are
    # keyed by the unprefixed name, since one run has one author and repeating it per key
    # would only be a second place to get it wrong.
    unknown = sorted(k for k in annotations if not k.startswith("_") and k not in ids)
    if unknown:
        raise ValueError(f"annotations name unknown axes {unknown}; ids are {ids}")
    if len(ids) != n_axes:
        raise ValueError(f"--ids has {len(ids)} entries for {n_axes} titles")
    if len(set(ids)) != n_axes:
        raise ValueError(f"--ids must be unique, got {ids}")

    if not re.fullmatch(r"[a-z0-9]+", author):
        raise ValueError(f"--author must be lowercase alphanumeric, got {author!r}")
    base_ids = ids
    ids = [f"{author}_{base_id}" for base_id in base_ids]

    tables_note = (
        f"quantile tables at {quantiles[0].shape[0]} levels"
        if quantiles is not None
        else "no quantile tables (ratio only)"
    )
    print(
        f"{src}: {n_axes} axis/axes at layer(s) {sorted(set(layers))}, "
        f"contrast {tuple(contrast.shape)}, {tables_note}"
    )

    rng = np.random.default_rng(0)
    for index, (axis_id, base_id, title, layer) in enumerate(
        zip(ids, base_ids, titles, layers, strict=True)
    ):
        if layer not in by_layer:
            raise ValueError(
                f"{axis_id}: manifest declares no pca entry for layer {layer}"
            )
        pca_results = by_layer[layer]
        scaler = pca_results["scaler"]
        scaler_name = type(scaler).__name__
        if scaler_name not in NORMALIZE_BY_SCALER:
            raise ValueError(f"{axis_id}: unknown scaler {scaler_name}")
        normalize = NORMALIZE_BY_SCALER[scaler_name]
        if normalize == "l2" and not np.isclose(
            getattr(scaler, "eps", EXPECTED_EPS), EXPECTED_EPS
        ):
            raise ValueError(f"{axis_id}: scaler eps {scaler.eps} != {EXPECTED_EPS}")

        components = served_components(pca_results, contrast, layer)
        hidden = components.shape[1]
        direction = np.ascontiguousarray(components[index], dtype=np.float32)
        scaler_mean = as_vector(getattr(scaler, "mean", None), hidden)
        pca_mean = as_vector(pca_results["pca"].mean_, hidden)

        norm = float(np.linalg.norm(direction))
        print(
            f"  {axis_id:<20} layer {layer:>2}  |d|={norm:.6f}  normalize={normalize}"
            f"  center={centers[index]:+.6f}  scale_pos={scales_pos[index]:.6f}"
            f"  scale_neg={scales_neg[index]:.6f}"
        )

        if verify:
            acts = rng.normal(size=(64, hidden)).astype(np.float32) * 3.0
            legacy = project_legacy(acts, pca_results, components, index)
            # The same per-pole map on the legacy projection: this check is for the direction and
            # the two means, so calibrating one side only would report a difference it caused.
            legacy_calibrated = calibrate(
                legacy, centers[index], scales_pos[index], scales_neg[index]
            )
            new = project_new(
                acts,
                direction,
                scaler_mean,
                pca_mean,
                normalize,
                centers[index],
                scales_pos[index],
                scales_neg[index],
            )
            worst = float(np.abs(legacy_calibrated - new).max())
            if worst > PARITY_ATOL:
                raise SystemExit(
                    f"  {axis_id}: parity FAILED, max |legacy - new| = {worst:.3e}"
                )
            print(f"  {' ' * 20} parity ok, max |legacy - new| = {worst:.3e}")

            if quantiles is not None:
                levels, table_pos, table_neg = quantiles
                # The property the percentile exists for, checked on the asset rather than assumed
                # from the tables: no input, however far off distribution, leaves [-1, 1].
                far = np.array([-1e6, -1.0, 0.0, centers[index], 1.0, 1e6])
                ranked = percentile(
                    far, centers[index], levels, table_pos[index], table_neg[index]
                )
                if np.abs(ranked).max() > 1.0:
                    raise SystemExit(
                        f"  {axis_id}: percentile left [-1, 1], reaching {ranked.max():+.6f}"
                    )
                at_center = percentile(
                    np.array([centers[index]]),
                    centers[index],
                    levels,
                    table_pos[index],
                    table_neg[index],
                )
                if at_center[0] != 0.0:
                    raise SystemExit(
                        f"  {axis_id}: percentile is {at_center[0]:+.6f} at the centre, not 0"
                    )
                print(f"  {' ' * 20} percentile bounded to [-1, 1], zero at centre")

        axis_dir = dst / axis_id
        axis_dir.mkdir(parents=True, exist_ok=True)
        tensors = {
            "direction": torch.from_numpy(direction),
            "scaler_mean": torch.from_numpy(scaler_mean),
            "pca_mean": torch.from_numpy(pca_mean),
        }
        if quantiles is not None:
            levels, table_pos, table_neg = quantiles
            # The levels ride in every axis directory rather than being implied by the table
            # length, so one directory stays enough to interpret its own numbers.
            tensors["quantile_levels"] = torch.from_numpy(np.ascontiguousarray(levels))
            tensors["quantiles_pos"] = torch.from_numpy(
                np.ascontiguousarray(table_pos[index])
            )
            tensors["quantiles_neg"] = torch.from_numpy(
                np.ascontiguousarray(table_neg[index])
            )
        save_file(tensors, str(axis_dir / "axis.safetensors"))

        axis_yaml: dict[str, Any] = {
            "author": author,
            "title": title,
            "layer": layer,
            "normalize": normalize,
            "center": centers[index],
            "scale_pos": scales_pos[index],
            "scale_neg": scales_neg[index],
            "render": render,
        }
        annotation = annotations.get(base_id, {})
        if annotation.get("caveat"):
            axis_yaml["caveat"] = annotation["caveat"]
        if annotation.get("provenance"):
            axis_yaml["provenance"] = annotation["provenance"]
        body = yaml.safe_dump(axis_yaml, allow_unicode=True, sort_keys=False)
        header = (
            f"# Readout axis '{axis_id}', converted from the legacy pickled asset by\n"
            f"# utils/persona-assets/convert_to_safetensors.py. Do not hand-edit: rerun the\n"
            f"# converter, or hand-write a new asset directory.\n"
            f"#\n"
            f"# A projection is ((x - scaler_mean) [L2-normalized if normalize: l2] - pca_mean)\n"
            f"# . direction, then d = raw - center and d / scale_pos if d >= 0 else d / scale_neg.\n"
            f"# One divisor per pole because these axes are lopsided: a single divisor has to be\n"
            f"# the larger spread, which squeezes the tighter pole towards zero. The map is\n"
            f"# continuous at center and monotone, so it preserves rank order exactly. Values are\n"
            f"# never clipped -- a value outside [-1, 1] is a real reading.\n"
            f"#\n"
            f"# axis.safetensors also carries quantile_levels / quantiles_pos / quantiles_neg\n"
            f"# when the fit produced them: entry i of a table is the distance from center at\n"
            f"# quantile_levels[i] of that pole's half of the fitting corpus. They give the same\n"
            f"# reading as a percentile -- interp(d, quantiles_pos, quantile_levels) for d >= 0, and\n"
            f"# -interp(-d, quantiles_neg, quantile_levels) below -- which cannot leave [-1, 1],\n"
            f"# because the interpolation holds its end values. The ratio above is the\n"
            f"# measurement and stays unclipped; the percentile is what a display can show without\n"
            f"# reading '102%'. An axis without these tables has no percentile, only the ratio.\n"
            f"#\n"
            f"# 'render' is what the fit assumed about how the conversation is templated. An\n"
            f"# axis is only meaningful against activations rendered the same way, so the server\n"
            f"# refuses a request mixing axes that disagree here.\n"
            f"#\n"
            f"# 'author' is who fitted this axis, and must match the '<author>_' prefix of the\n"
            f"# directory name. Two groups fitting the same trait are different measurements, so\n"
            f"# both the prefix and this field name the one that produced these numbers.\n"
        )
        (axis_dir / "axis.yaml").write_text(header + "\n" + body, encoding="utf-8")

    print(f"wrote {n_axes} axis director{'y' if n_axes == 1 else 'ies'} under {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("src", type=Path, help="legacy asset directory")
    parser.add_argument(
        "dst", type=Path, help="model directory to write axis subdirectories into"
    )
    parser.add_argument(
        "--author",
        required=True,
        help="who fitted this asset; ids come out as <author>_<name>",
    )
    parser.add_argument(
        "--ids",
        help="comma-separated axis names, in component order, without the author prefix",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check the new path against the legacy one",
    )
    parser.add_argument(
        "--print-manifest",
        action="store_true",
        help="dump the legacy manifest and exit",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="JSON of per-axis caveat/provenance, keyed by axis name without the author prefix",
    )
    args = parser.parse_args()

    if args.print_manifest:
        _install_shims()
        with open(args.src / "persona.yaml", encoding="utf-8") as handle:
            print(json.dumps(yaml.safe_load(handle), indent=2, ensure_ascii=False))
        return

    convert(
        args.src,
        args.dst,
        args.author,
        args.ids.split(",") if args.ids else None,
        args.verify,
        args.annotations,
    )


if __name__ == "__main__":
    main()
