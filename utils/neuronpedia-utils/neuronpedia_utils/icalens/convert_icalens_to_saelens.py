"""Convert an ICA Lens artifact (icalens format v4) to SAELens StandardSAE folders.

ICA Lens: Liu & Han et al., "Non-Gaussianity Reveals Interpretable Directions in
Language Models" (arXiv 2606.11722), https://github.com/liusida/ica-lens-paper.

Per layer the lens stores `center`, `reading_matrix` R and `writing_matrix` W, with
`scores = (x - center) @ R.T` and `x = scores @ W.T + center`. A component is a
direction, not a feature with a sign, so its interpretable tail can be positive or
negative. The profile records that tail as `tail_direction`. We fold the sign
`s_j = +1 | -1` into both matrices so that ReLU keeps the interpretable tail:

    act_j = ReLU(s_j * n_j * score_j)        n_j = ||W[:, j]||
    W_enc = (s * n * R).T,  b_enc = 0,  W_dec = (W * s / n).T,  b_dec = center

W_dec rows are unit norm, so `fold_W_dec_norm()` in inference and SAEDashboard has no
effect. An activation is the norm of the component's write into the residual stream.
`act_j / n_j` is the standardized ICA score (unit variance on the fit data).

Usage (from this directory):
    uv run python convert_icalens_to_saelens.py \
        --source-repo sida/icalens-gpt2-small-pile10k \
        --np-model-id gpt2-small --hf-model-id openai-community/gpt2 \
        --model-name gpt2-small --np-set-name icalens-res \
        --upload-repo neuronpedia/icalens-gpt2-small
"""

import argparse
import gzip
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import HfApi, snapshot_download
from sae_lens import SAE, StandardSAE, StandardSAEConfig
from sae_lens.saes.sae import SAEMetadata
from safetensors.torch import load_file
from transformers import AutoTokenizer

SIDECAR_FILENAME = "icalens_components.json"
ACTIVATION_SITE = "resid_post"
SIGN_SOURCE_PROFILE = "profile.tail_direction"


def hook_name_for_layer(layer: int) -> str:
    """Return the TransformerLens-style hook name for the block output of a layer."""
    return f"blocks.{layer}.hook_{ACTIVATION_SITE}"


def load_profile(lens_dir: Path, layer_entry: dict[str, Any]) -> dict[str, Any]:
    """Read the gzipped component profile of one layer."""
    profile_path = layer_entry.get("component_profile")
    if not profile_path:
        raise ValueError(
            "layer has no component profile, so the tail direction is unknown"
        )
    with gzip.open(lens_dir / profile_path, "rt", encoding="utf-8") as f:
        return json.load(f)


def component_signs(
    profile: dict[str, Any], d: int
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Return the per-component sign and the statistics to keep in the sidecar file."""
    components = profile["components"]
    if len(components) != d:
        raise ValueError(f"profile has {len(components)} components, lens has {d}")
    signs = torch.empty(d, dtype=torch.float32)
    stats = []
    for j, comp in enumerate(components):
        if comp["component"] != j:
            raise ValueError(f"profile component {j} has id {comp['component']}")
        tail = comp["tail_direction"]
        if tail not in ("positive", "negative"):
            raise ValueError(f"component {j} has tail_direction {tail!r}")
        signs[j] = 1.0 if tail == "positive" else -1.0
        score_stats = comp.get("score_statistics", {})
        stats.append(
            {
                "skewness": score_stats.get("skewness"),
                "excess_kurtosis": score_stats.get("excess_kurtosis"),
            }
        )
    return signs, stats


def build_sae(
    tensors: dict[str, torch.Tensor],
    signs: torch.Tensor,
    metadata: dict[str, Any],
) -> tuple[StandardSAE, torch.Tensor]:
    """Build the sign-oriented StandardSAE with unit-norm decoder rows."""
    reading = tensors["reading_matrix"].float()
    writing = tensors["writing_matrix"].float()
    center = tensors["center"].float()
    d = reading.shape[1]
    if reading.shape != (d, d) or writing.shape != (d, d) or center.shape != (d,):
        raise ValueError(
            f"unexpected shapes R={reading.shape} W={writing.shape} c={center.shape}"
        )
    norms = writing.norm(dim=0)
    if not torch.all(norms > 0):
        raise ValueError("writing matrix has a zero column")
    scale = signs * norms
    cfg = StandardSAEConfig(
        d_in=d,
        d_sae=d,
        dtype="float32",
        device="cpu",
        apply_b_dec_to_input=True,
        metadata=SAEMetadata(**metadata),
    )
    sae = StandardSAE(cfg)
    sae.load_state_dict(
        {
            "W_enc": (scale[:, None] * reading).T.contiguous(),
            "b_enc": torch.zeros(d),
            "W_dec": (writing * (signs / norms)[None, :]).T.contiguous(),
            "b_dec": center.clone(),
        }
    )
    return sae, norms


def check_sae(
    sae: StandardSAE,
    tensors: dict[str, torch.Tensor],
    signs: torch.Tensor,
    norms: torch.Tensor,
) -> None:
    """Check the SAE against the lens on random inputs, before and after a save round trip."""
    reading = tensors["reading_matrix"].float()
    writing = tensors["writing_matrix"].float()
    center = tensors["center"].float()
    gen = torch.Generator().manual_seed(0)
    x = center + torch.randn(
        64, center.shape[0], generator=gen
    ) * center.abs().mean().clamp_min(1.0)
    scores = (x - center) @ reading.T
    expected = torch.relu(signs * norms * scores)
    with torch.no_grad():
        acts = sae.encode(x)
        pre = (x - sae.b_dec) @ sae.W_enc + sae.b_enc
        recon = pre @ sae.W_dec + sae.b_dec
    torch.testing.assert_close(
        acts, expected, rtol=1e-4, atol=1e-4 * float(expected.abs().max())
    )
    torch.testing.assert_close(
        recon, scores @ writing.T + center, rtol=1e-3, atol=1e-3 * float(x.abs().max())
    )
    torch.testing.assert_close(
        sae.W_dec.norm(dim=1), torch.ones_like(norms), rtol=1e-5, atol=1e-5
    )


def convert_layer(
    lens_dir: Path,
    manifest: dict[str, Any],
    layer: int,
    out_dir: Path,
    metadata: dict[str, Any],
) -> None:
    """Convert one lens layer into `out_dir/blocks.{layer}.hook_resid_post/`."""
    entry = manifest["layers"][str(layer)]
    tensors = load_file(lens_dir / entry["file"])
    d = tensors["reading_matrix"].shape[1]
    profile = load_profile(lens_dir, entry)
    signs, stats = component_signs(profile, d)
    dataset = (profile.get("provenance") or {}).get("dataset") or {}
    metadata = {**metadata, "dataset_path": dataset.get("repo_id")}
    sae, norms = build_sae(tensors, signs, metadata)
    check_sae(sae, tensors, signs, norms)

    folder = out_dir / hook_name_for_layer(layer)
    sae.save_model(folder)
    reloaded = SAE.load_from_disk(folder, device="cpu")
    if not isinstance(reloaded, StandardSAE):
        raise TypeError(f"reloaded SAE is {type(reloaded).__name__}")
    check_sae(reloaded, tensors, signs, norms)

    fitting = entry.get("fitting", {})
    sidecar = {
        "format": "neuronpedia-icalens-components",
        "format_version": 1,
        "layer": layer,
        "activation_site": ACTIVATION_SITE,
        "sign_source": SIGN_SOURCE_PROFILE,
        "activation_definition": "ReLU(sign * writing_norm * score); score = act / writing_norm",
        "components": [
            {
                "index": j,
                "sign": int(signs[j].item()),
                "writing_norm": float(norms[j].item()),
                "strength": (fitting.get("component_strengths") or [None] * d)[j],
                **stats[j],
            }
            for j in range(d)
        ],
    }
    with open(folder / SIDECAR_FILENAME, "w", encoding="utf-8") as f:
        json.dump(sidecar, f)
    n_neg = int((signs < 0).sum().item())
    print(
        f"layer {layer}: d={d}, negative-tail components={n_neg}, "
        f"writing norm min/median/max={norms.min():.3f}/{norms.median():.3f}/{norms.max():.3f}"
    )


def write_readme(
    out_dir: Path, args: argparse.Namespace, source_sha: str, manifest: dict[str, Any]
) -> None:
    """Write a short model card for the converted repository."""
    layers = sorted(int(k) for k in manifest["layers"])
    text = f"""---
library_name: saelens
tags:
- ica-lens
- sparse-autoencoder
---
# ICA Lens for {args.hf_model_id}, in SAELens format

Converted from [{args.source_repo}](https://huggingface.co/{args.source_repo}) at revision `{source_sha}`.
ICA Lens is by Liu & Han et al., [Non-Gaussianity Reveals Interpretable Directions in Language Models](https://arxiv.org/abs/2606.11722).

Each folder `blocks.L.hook_resid_post/` is a SAELens `StandardSAE` with `d_sae = d_in = {manifest["hidden_size"]}`.
Layers: {layers[0]}-{layers[-1]} ({len(layers)} layers), zero-based block outputs.

- `W_enc = (s * n * R).T`, `b_enc = 0`, `W_dec = (W * s / n).T`, `b_dec = center`.
- `R`, `W` and `center` are the lens reading matrix, writing matrix and center.
- `n` is the norm of each writing column, so `W_dec` rows are unit norm.
- `s` is +1 or -1 from the profile `tail_direction`, so ReLU keeps the interpretable tail.
- An activation is `|s * n * score|` on that tail. Divide by `n` to get the standardized ICA score.
- `{SIDECAR_FILENAME}` keeps `sign`, `writing_norm`, `strength`, `skewness` and `excess_kurtosis` per component.

Components are in the order of the source lens: component 0 has the largest non-Gaussianity.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def document_prefix(manifest: dict[str, Any], layers: list[int]) -> str | None:
    """Return the token the lens put before each fitting document, or None."""
    tokens = set()
    for layer in layers:
        fitting = manifest["layers"][str(layer)].get("fitting") or {}
        framing = (fitting.get("provenance") or {}).get("document_framing") or {}
        prepends = framing.get("strategy") in ("prepend-bos", "prepend-eos")
        tokens.add(framing["token"] if prepends else None)
    if len(tokens) != 1:
        raise ValueError(f"layers disagree on the document framing token: {tokens}")
    return tokens.pop()


def registry_bos_token(prefix: str | None, hf_model_id: str) -> str | None:
    """Return the `bos_token` inference must set so its prompts match the fit framing.

    Inference puts the tokenizer's BOS before every prompt. When the lens used that
    token, nothing is needed. When the tokenizer has no BOS (Qwen), the lens prefix
    becomes the BOS. A different BOS cannot reproduce the fit, so it is an error.
    """
    if prefix is None:
        return None
    tokenizer_bos = AutoTokenizer.from_pretrained(hf_model_id).bos_token
    if tokenizer_bos == prefix:
        return None
    if tokenizer_bos is not None:
        raise ValueError(
            f"lens prefix {prefix!r} differs from the {hf_model_id} BOS {tokenizer_bos!r}"
        )
    return prefix


def registry_entry(
    args: argparse.Namespace, layers: list[int], bos_token: str | None
) -> dict[str, Any]:
    """Return the `extra_saes.yaml` entry that lets inference load this repository."""
    entry: dict[str, Any] = {"model": args.hf_model_id}
    if bos_token is not None:
        entry["bos_token"] = bos_token
    entry["saes"] = [
        {
            "path": hook_name_for_layer(layer),
            "neuronpedia": f"{args.np_model_id}/{layer}-{args.np_set_name}",
        }
        for layer in layers
    ]
    return {args.upload_repo: entry}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--source-repo",
        required=True,
        help="ICA Lens repo id on the Hub, or a local directory",
    )
    p.add_argument("--source-revision", default=None)
    p.add_argument(
        "--np-model-id", required=True, help="Neuronpedia model id, e.g. gpt2-small"
    )
    p.add_argument(
        "--hf-model-id",
        required=True,
        help="Hugging Face model id, e.g. openai-community/gpt2",
    )
    p.add_argument(
        "--model-name",
        required=True,
        help="cfg.json model_name: TransformerLens name, or HF id",
    )
    p.add_argument(
        "--np-set-name",
        required=True,
        help="Neuronpedia source set, e.g. icalens-res",
    )
    p.add_argument(
        "--upload-repo",
        required=True,
        help="Target repo id, e.g. neuronpedia/icalens-gpt2-small",
    )
    p.add_argument("--layers", default="all", help="Comma-separated layers, or 'all'")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: ./exports/<upload repo name>",
    )
    p.add_argument("--context-size", type=int, default=1024)
    p.add_argument("--prepend-bos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--upload",
        action="store_true",
        help="Upload to a private repo after the conversion",
    )
    return p.parse_args()


def main() -> None:
    """Convert every requested layer, write the registry entry, and optionally upload."""
    args = parse_args()
    source = Path(args.source_repo)
    if source.is_dir():
        lens_dir, source_sha = source, "local"
    else:
        info = HfApi().model_info(args.source_repo, revision=args.source_revision)
        source_sha = info.sha or "unknown"
        lens_dir = Path(snapshot_download(args.source_repo, revision=source_sha))
    manifest = json.loads((lens_dir / "icalens.json").read_text(encoding="utf-8"))
    if manifest.get("activation_site") != ACTIVATION_SITE:
        raise ValueError(f"activation_site is {manifest.get('activation_site')!r}")
    # The SAE applies only `- center` before the reading matrix, so any other transform is unsupported.
    prep = manifest.get("input_preprocessing") or {}
    unsupported = {k: v for k, v in prep.items() if k != "norm_eps" and v != "none"}
    if unsupported:
        raise ValueError(f"input_preprocessing {unsupported!r} is not supported")
    available = sorted(int(k) for k in manifest["layers"])
    layers = (
        available if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    )

    out_dir = args.out_dir or Path("exports") / args.upload_repo.split("/")[-1]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for layer in layers:
        metadata = {
            "model_name": args.model_name,
            "hook_name": hook_name_for_layer(layer),
            "hook_head_index": None,
            "prepend_bos": args.prepend_bos,
            "context_size": args.context_size,
            "neuronpedia_id": f"{args.np_model_id}/{layer}-{args.np_set_name}",
            "model_from_pretrained_kwargs": {},
            "icalens_source_repo": args.source_repo,
            "icalens_source_revision": source_sha,
        }
        convert_layer(lens_dir, manifest, layer, out_dir, metadata)
    write_readme(out_dir, args, source_sha, manifest)
    bos_token = registry_bos_token(document_prefix(manifest, layers), args.hf_model_id)
    entry = registry_entry(args, layers, bos_token)
    with open(
        out_dir.parent / f"{out_dir.name}.registry.yaml", "w", encoding="utf-8"
    ) as f:
        yaml.safe_dump(entry, f, sort_keys=False)
    print(f"wrote {len(layers)} layers to {out_dir}")

    if args.upload:
        api = HfApi()
        api.create_repo(
            args.upload_repo, private=True, exist_ok=True, repo_type="model"
        )
        api.upload_folder(
            repo_id=args.upload_repo,
            folder_path=str(out_dir),
            commit_message=f"ICA Lens {args.source_repo}@{source_sha[:10]} as SAELens StandardSAE",
        )
        print(f"uploaded to https://huggingface.co/{args.upload_repo} (private)")


if __name__ == "__main__":
    main()
