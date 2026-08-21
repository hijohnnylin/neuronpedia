"""Validate the generic engine-owned vLLM worker-hook capture vs the EagerModel.

Captures the same ``(name, layer)`` points the interpretability endpoints read
(resid_post / resid_pre / mlp_out / mlp_in / z) on both backends for one fixed
prompt and reports per-point cosine + max-abs. This is the parity gate for wiring
the activation/* + jlens endpoints onto ``VLLMModel.capture``.

Run (5090, .venv with engine + vllm):

    IE_VLLM_GPU_UTIL=0.3 \
      .venv/bin/python scripts/vllm_capture_points_check.py --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import asyncio
import os


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument(
        "--prompt",
        default="The capital of France is Paris, and the capital of Japan is",
    )
    ap.add_argument(
        "--no-extract",
        action="store_true",
        help="build vLLM without native hidden-state extraction (isolates the speculative aux-forward)",
    )
    args = ap.parse_args()

    import torch
    from interp_engine import Address, EagerModel, VLLMModel, run_with_cache

    # --- 1) eager reference (EagerModel), first, then free the GPU for vLLM --------
    npm = EagerModel(args.model, device="cuda", dtype="bfloat16")
    n_layers = npm.n_layers
    layers = sorted({0, n_layers // 2, n_layers - 1})
    points = (
        [Address("resid_post", layer) for layer in layers]
        + [Address("resid_pre", layer) for layer in layers]
        + [
            Address("mlp_out", layers[len(layers) // 2]),
            Address("mlp_in", layers[len(layers) // 2]),
        ]
        + [Address("z", layers[len(layers) // 2])]
    )
    tokens = npm.to_tokens(args.prompt)
    token_ids = tokens[0].tolist()
    cache = run_with_cache(npm, tokens, points)
    ref = {address: cache[address][0].float().cpu() for address in points}
    del npm, cache
    torch.cuda.empty_cache()

    # --- 2) vLLM worker-hook capture -------------------------------------------
    backend = VLLMModel(
        args.model,
        dtype="bfloat16",
        enforce_eager=True,
        enable_extraction=not args.no_extract,
        gpu_memory_utilization=float(os.environ.get("IE_VLLM_GPU_UTIL", "0.8")),
    )
    got = await backend.capture(token_ids, points)

    # --- 3) compare -------------------------------------------------------------
    print(f"\nmodel={args.model} seq={len(token_ids)} layers={layers}")
    print(f"{'point':>16}  {'shape(vllm)':>16}  {'cos':>10}  {'max_abs':>10}")
    worst = 1.0
    for key in points:
        r = ref[key]
        g = got[key].float()
        if g.shape != r.shape:
            print(f"{str(key):>16}  SHAPE MISMATCH vllm={tuple(g.shape)} ref={tuple(r.shape)}")
            worst = 0.0
            continue
        a, b = g.reshape(-1).double(), r.reshape(-1).double()
        cos = float(torch.dot(a, b) / (a.norm() * b.norm()))
        mx = float((g - r).abs().max())
        worst = min(worst, cos)
        print(f"{str(key):>16}  {str(tuple(g.shape)):>16}  {cos:>10.6f}  {mx:>10.3e}")
        if cos < 0.99:
            print(f"      DEBUG {key}: vllm[0,:6]={g[0, :6].tolist()}")
            print(f"      DEBUG {key}:  ref[0,:6]={r[0, :6].tolist()}")
            print(
                f"      DEBUG {key}: vllm_norm={g.norm():.3f} ref_norm={r.norm():.3f} ratio_mean={(g / r.clamp_min(1e-6)).mean():.4f}"
            )
    print(f"\nworst cosine = {worst:.6f}  ({'PASS' if worst > 0.999 else 'CHECK'})")


if __name__ == "__main__":
    asyncio.run(main())
