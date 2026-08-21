"""Validate vLLM off-kernel attention-probs recompute vs EagerModel.

Captures post-rope q/k/v at ``self_attn.attn`` on vLLM, recomputes softmax probs,
and compares to EagerModel's eager ``attn_probs`` (output_attentions) per head. Also
checks the captured value stream vs EagerModel per_head_value. Parity gate for
activation/attention + DFA on vLLM.

    IE_VLLM_GPU_UTIL=0.8 \
      .venv/bin/python scripts/vllm_attn_recompute_check.py --model Qwen/Qwen3-0.6B

**The default model and prompt cannot fail this check**, in the same way
``vllm_unembed_check.py``'s default cannot fail that one. Qwen3-0.6B has no sliding
window and no attention sinks, so it exercises only the plain causal softmax. The
recompute has to reapply three per-architecture terms the fused kernel owns
(softcap, sliding window, sinks), and each is silent when missed -- so run this on a
model that actually has the quirk:

    --model openai/gpt-oss-20b      # sinks + a 128-token window on even layers
    --model google/gemma-3-270m-it  # 512-token window, five banded layers per full one
    --model google/gemma-2-2b       # attn-logit softcap + a 4096-token window

The window is the one that also needs a long enough **prompt**: a band only shows up
once a query can reach past it, so the prompt below is grown automatically to exceed
the model's window rather than left at a fixed sentence. Layer selection is likewise
quirk-aware -- it picks a banded layer *and* a full-attention one, since applying the
window everywhere and applying it nowhere are both wrong and only differ on those two.
"""

from __future__ import annotations

import argparse
import asyncio
import os

FILLER = "The quick brown fox jumps over the lazy dog near the river bank. "


def _pick_layers(n_layers: int, layer_types: tuple[str, ...]) -> list[int]:
    """Layers worth checking: the ends, the middle, and one of each attention kind.

    A model whose ``layer_types`` alternates fails differently on banded and full layers,
    and a fixed ``{0, n//2, n-1}`` can easily miss one kind entirely.
    """
    picked = {0, n_layers // 2, n_layers - 1}
    for kind in ("sliding", "full"):
        match = next((i for i, t in enumerate(layer_types) if kind in t.lower()), None)
        if match is not None:
            picked.add(match)
    return sorted(i for i in picked if 0 <= i < n_layers)


def _grow_prompt(tokenizer, prompt: str, min_tokens: int) -> tuple[list[int], str]:
    """Token ids for ``prompt``, padded with filler until it is at least ``min_tokens``."""
    text = prompt
    while len(tokenizer(text, add_special_tokens=True)["input_ids"]) < min_tokens:
        text += FILLER
    return tokenizer(text, add_special_tokens=True)["input_ids"], text


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument(
        "--prompt",
        default="The capital of France is Paris, and the capital of Japan is",
    )
    ap.add_argument(
        "--min-tokens",
        type=int,
        default=0,
        help="Force a prompt of at least this many tokens (default: window + 64 when the "
        "model is banded, otherwise the prompt as given).",
    )
    args = ap.parse_args()

    import gc

    import torch
    from interp_engine import VLLMModel
    from interp_engine.vllm_backend import read_attn_dims, sliding_window_for_layer

    dims = read_attn_dims(args.model)
    window = dims.get("sliding_window")
    layer_types: tuple[str, ...] = dims.get("layer_types") or ()
    # A banded model needs a prompt that reaches past the band, or the window term is
    # inert and this check silently degrades to the plain-causal case.
    min_tokens = args.min_tokens or ((int(window) + 64) if window else 0)

    backend = VLLMModel(
        args.model,
        dtype="bfloat16",
        enforce_eager=True,
        enable_extraction=False,
        gpu_memory_utilization=float(os.environ.get("IE_VLLM_GPU_UTIL", "0.8")),
    )
    n_layers = backend.n_layers
    layers = _pick_layers(n_layers, layer_types)
    prompt_ids, _ = _grow_prompt(backend.tokenizer, args.prompt, min_tokens)
    got = await backend.capture_attention(prompt_ids, layers)

    # Release the EngineCore's VRAM before the eager model asks for its own.
    await backend.shutdown()
    del backend
    gc.collect()
    torch.cuda.empty_cache()

    from interp_engine import EagerModel, per_head_value, run_with_cache

    npm = EagerModel(args.model, device="cuda", dtype="bfloat16", attn_implementation="eager")
    ids = torch.tensor([prompt_ids], device="cuda")
    cache = run_with_cache(
        npm,
        ids,
        [("attn_probs", layer) for layer in layers] + [("value", layer) for layer in layers],
    )

    seq = len(prompt_ids)
    print(f"\nmodel={args.model} seq={seq} layers={layers}")
    print(f"sliding_window={window} banded_layers={sum('sliding' in t.lower() for t in layer_types)}/{n_layers}")
    if window and seq <= int(window):
        print(f"  WARNING: seq {seq} <= window {window} — the sliding-window term is NOT exercised")
    print(f"{'layer':>6}  {'kind':>8}  {'probs_cos':>10}  {'probs_maxabs':>12}  {'value_cos':>10}  {'rowsum_min':>10}")
    worst = 1.0
    for layer in layers:
        ref_probs = cache.get("attn_probs", layer)[0].float().cpu()  # [n_heads, dest, src]
        g_probs = got[layer]["probs"].float()[:, : ref_probs.shape[1], : ref_probs.shape[2]]
        a, b = g_probs.reshape(-1).double(), ref_probs.reshape(-1).double()
        pcos = float(torch.dot(a, b) / (a.norm() * b.norm()))
        pmax = float((g_probs - ref_probs).abs().max())
        ref_v = per_head_value(npm, cache, layer)[0].float().cpu()  # [src, n_kv, head_dim]
        g_v = got[layer]["value"].float()[: ref_v.shape[0]]
        va, vb = g_v.reshape(-1).double(), ref_v.reshape(-1).double()
        vcos = float(torch.dot(va, vb) / (va.norm() * vb.norm()))
        worst = min(worst, pcos, vcos)
        banded = sliding_window_for_layer(dims, layer)
        kind = f"win{banded}" if banded else "full"
        # Below 1 means sink mass, which is correct on gpt-oss and a bug anywhere else.
        # Renormalizing sinks away is exactly what this column is here to expose.
        rowsum_min = float(g_probs.sum(dim=-1).min())
        print(f"{layer:>6}  {kind:>8}  {pcos:>10.6f}  {pmax:>12.3e}  {vcos:>10.6f}  {rowsum_min:>10.4f}")
    print(f"\nworst cosine = {worst:.6f}  ({'PASS' if worst > 0.999 else 'CHECK'})")


if __name__ == "__main__":
    asyncio.run(main())
