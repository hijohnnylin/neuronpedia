"""Which tensor a Gemma Scope transcoder actually encodes, measured rather than read off a name.

Written to settle why the server disagrees with the shipping dashboards for
`gemma-2-2b/*-gemmascope-transcoder-16k`, and kept because it is the audit for any future source
whose SAELens `hook_name` is a `hook_normalized` -- the one hook no HuggingFace module outputs, so
the only one a reader has to reproduce rather than capture.

SAELens gives those sources `blocks.{i}.ln2.hook_normalized` and the server reproduces it: the
pre-MLP norm's input over its own RMS, the norm's gain excluded. This scores that candidate against
its two neighbours by the only things a transcoder can be scored by -- how well it reconstructs the
MLP output it was trained to predict, and at what L0 against the declared one -- and then asks
whether the difference between the served reading and the dashboards' is a rescale or a reordering.

The answer, on 32 pile documents at layer 19: the served reading is the trained input (L0 12.8
against a declared 12, FVU 0.50 against `mlp_out_post`), and the dashboards were built on the
unnormalized `resid_mid` (L0 664, FVU 6.8e4, and 2.8% density on feature 0 against the 2.194% the
dashboard publishes and the 0.024% the trained input gives). Nothing here is a rescale of anything:
the peak ratio spreads over 3x-107x and 39% of features keep their top-activating token.

Reads the model with plain `torch` hooks and the transcoder straight out of its `params.npz`, so the
answer depends on neither the engine, nor SAELens, nor TransformerLens.

Run with the inference venv, on a box that can hold gemma-2-2b at fp32 (~11 GB):

    uv run --no-sync python scripts/gemma_transcoder_hook_check.py            # one text
    uv run --no-sync python scripts/gemma_transcoder_hook_check.py --pile 32  # the table above
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

MODEL_ID = "google/gemma-2-2b"
TRANSCODER_REPO = "google/gemma-scope-2b-pt-transcoders"
LAYER = 19
PARAMS_PATH = f"layer_{LAYER}/width_16k/average_l0_12/params.npz"
DECLARED_L0 = 12
FEATURE = 0

# The dashboard's top-activating text for `19-gemmascope-transcoder-16k/0`, which the shipping
# dashboard scores at 409.74 and the server now answers with a far smaller number.
TEXT = (
    " the responsibility of the trial court, as well as the trial jury, both of whom are in a better"
    " position to determine which witnesses have testified truthfully and which falsely. We do not"
    " have such a situation in the instant case, in which there is little, if any, material conflict"
    " in the evidence.\nOn the question of the guilt or innocence of defendant of the crime charged,"
    " the evidence against him is entirely circumstantial; the evidence in his favor is both direct"
    " and circumstantial. It is a mistake, a common one, to denigrate the nature of circumstantial"
    " *938 evidence by considering it as inferior to direct evidence. That"
)


def capture(model: torch.nn.Module, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    """The four tensors around layer LAYER's MLP, by their engine point names.

    Each is taken from a module boundary by path -- the spellings `Gemma2DecoderLayer` uses -- so
    the reading is the model's own forward rather than anything reconstructed. `resid_mid` is the
    pre-MLP norm's INPUT and `mlp_in` that same norm's OUTPUT, which is the pair the whole question
    turns on.
    """
    caught: dict[str, torch.Tensor] = {}

    def keep(name: str, index: int):
        def hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor):
            caught[name] = (args[index] if index < 0 else output).detach().float()

        return hook

    block = f"model.layers.{LAYER}"
    wanted = {
        "resid_mid": (f"{block}.pre_feedforward_layernorm", -1),
        "mlp_in": (f"{block}.pre_feedforward_layernorm", 0),
        "mlp_out": (f"{block}.mlp", 0),
        "mlp_out_post": (f"{block}.post_feedforward_layernorm", 0),
    }
    handles = [
        model.get_submodule(path).register_forward_hook(keep(name, index)) for name, (path, index) in wanted.items()
    ]
    try:
        with torch.no_grad():
            model(tokens)
    finally:
        for handle in handles:
            handle.remove()
    return caught


def pre_gain_normalized(x: torch.Tensor, eps: float) -> torch.Tensor:
    """What the server computes for `ln2.hook_normalized`: the norm's input over its own RMS."""
    return x / (x.float().pow(2).mean(-1, keepdim=True) + eps).sqrt()


def encode(params: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    pre = x @ params["W_enc"] + params["b_enc"]
    return torch.relu(pre) * (pre > params["threshold"])


def fvu(target: torch.Tensor, prediction: torch.Tensor) -> float:
    """Fraction of variance unexplained. Above 1.0 is worse than predicting the mean."""
    residual = (target - prediction).pow(2).sum()
    variance = (target - target.mean(0, keepdim=True)).pow(2).sum()
    return float(residual / variance)


def pile_texts(count: int) -> list[str]:
    """The transcoder's own training distribution, so the declared L0 becomes a check."""
    from datasets import load_dataset

    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    return [row["text"] for row in dataset.take(count * 4) if len(row["text"]) > 2000][:count]


def candidate_inputs(caught: dict[str, torch.Tensor], eps: float) -> dict[str, torch.Tensor]:
    """The three tensors a reader of `blocks.{i}.ln2.hook_normalized` could plausibly have meant."""
    return {
        "pre_gain_normalized(resid_mid)": pre_gain_normalized(caught["resid_mid"][0], eps),
        "mlp_in (norm output, gain included)": caught["mlp_in"][0],
        "resid_mid (unnormalized)": caught["resid_mid"][0],
    }


def report(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    params: dict[str, torch.Tensor],
    texts: list[str],
    eps: float,
    device: str,
) -> None:
    """One row per candidate input, scored on L0 and on reconstruction of both MLP-output points.

    Then, between the winner and what the dashboards were built on, whether the difference is a
    rescale -- which a stored dashboard could be corrected by -- or a reordering, which it cannot.
    """
    totals: dict[str, dict[str, float]] = {}
    served: list[torch.Tensor] = []
    dashboard: list[torch.Tensor] = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).input_ids.to(device)
        caught = capture(model, tokens)
        for name, x in candidate_inputs(caught, eps).items():
            acts = encode(params, x)
            prediction = acts @ params["W_dec"] + params["b_dec"]
            row = totals.setdefault(name, {"n": 0.0, "rms": 0.0, "l0": 0.0, "max_f0": 0.0, "density_f0": 0.0})
            row["n"] += 1
            row["rms"] += float(x.pow(2).mean().sqrt())
            row["l0"] += float((acts > 0).sum(-1).float().mean())
            row["max_f0"] = max(row["max_f0"], float(acts[:, FEATURE].max()))
            row["density_f0"] += float((acts[:, FEATURE] > 0).float().mean())
            for target_name in ("mlp_out", "mlp_out_post"):
                row[target_name] = row.get(target_name, 0.0) + fvu(caught[target_name][0], prediction)
            if name == "pre_gain_normalized(resid_mid)":
                served.append(acts)
            elif name == "resid_mid (unnormalized)":
                dashboard.append(acts)

    header = (
        f"{'candidate input':<38} {'rms':>7} {'L0':>6} {'max f0':>8} {'f0 density':>11}"
        f"{'FVU mlp_out':>14}{'FVU mlp_out_post':>18}"
    )
    print(header)
    print("-" * len(header))
    for name, row in totals.items():
        n = row["n"]
        print(
            f"{name:<38} {row['rms'] / n:>7.3f} {row['l0'] / n:>6.1f} {row['max_f0']:>8.2f}"
            f"{row['density_f0'] / n:>10.3%}{row['mlp_out'] / n:>14.3f}{row['mlp_out_post'] / n:>18.3f}"
        )

    print("\nserved reading vs the dashboards' reading, per feature over the same tokens:")
    a, b = torch.cat(served), torch.cat(dashboard)
    peak_served, peak_dashboard = a.max(0).values, b.max(0).values
    both = (peak_served > 0) & (peak_dashboard > 0)
    ratio = peak_dashboard[both] / peak_served[both]
    quartiles = torch.quantile(ratio, torch.tensor([0.25, 0.5, 0.75], device=ratio.device))
    print(f"  features firing under the served reading:    {int((peak_served > 0).sum())}")
    print(f"  features firing under the dashboards':       {int((peak_dashboard > 0).sum())}")
    print(f"  firing only in the dashboards' reading:      {int(((peak_dashboard > 0) & (peak_served == 0)).sum())}")
    print(
        f"  peak-activation ratio over the {int(both.sum())} shared: median {quartiles[1]:.2f}, "
        f"quartiles {quartiles[0]:.2f}-{quartiles[2]:.2f}, full range {ratio.min():.2f}-{ratio.max():.2f}"
    )
    agree = (a[:, both].argmax(0) == b[:, both].argmax(0)).float().mean()
    print(f"  shared features whose top-activating token agrees: {agree:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text", default=TEXT)
    parser.add_argument(
        "--pile",
        type=int,
        default=0,
        help="score on this many pile documents instead of the dashboard's own text, which is what "
        "makes the declared L0 a check rather than a curiosity",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, attn_implementation="eager").to(
        args.device
    )
    model.eval()
    eps = model.config.rms_norm_eps

    npz = np.load(hf_hub_download(TRANSCODER_REPO, PARAMS_PATH))
    params = {k: torch.from_numpy(npz[k]).to(args.device) for k in npz}

    texts = pile_texts(args.pile) if args.pile else [args.text]
    print(f"layer {LAYER}, {len(texts)} text(s), eps={eps}\n")
    report(model, tokenizer, params, texts, eps, args.device)
    print(f"\ndeclared L0 {DECLARED_L0}; dashboard max for feature {FEATURE} on its own top text: 409.74")


if __name__ == "__main__":
    main()
