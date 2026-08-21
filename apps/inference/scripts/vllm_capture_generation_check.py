"""Validate decode-time (generation) capture on vLLM vs the EagerModel.

Generates greedily on vLLM while accumulating resid_post at prompt AND generated
positions, then runs EagerModel over the SAME full token sequence and compares
position-by-position. This is the parity gate for jlens generation-time capture.

Run (5090, .venv with engine + vllm):

    IE_VLLM_GPU_UTIL=0.8 \
      .venv/bin/python scripts/vllm_capture_generation_check.py --model Qwen/Qwen3-0.6B --max-tokens 8
"""

from __future__ import annotations

import argparse
import asyncio
import os


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()

    import gc

    import torch
    from interp_engine import Address, EagerModel, VLLMModel, run_with_cache

    # --- 1) vLLM: greedy generate + decode-time capture --------------------------
    backend = VLLMModel(
        args.model,
        dtype="bfloat16",
        enforce_eager=True,
        enable_extraction=False,
        gpu_memory_utilization=float(os.environ.get("IE_VLLM_GPU_UTIL", "0.8")),
    )
    n_layers = backend.n_layers
    layers = sorted({0, n_layers // 2, n_layers - 1})
    points = [Address("resid_post", layer) for layer in layers]

    prompt_ids = backend.tokenizer(args.prompt, add_special_tokens=True)["input_ids"]
    completion, caps = await backend.capture_generation(prompt_ids, points, max_tokens=args.max_tokens, temperature=0.0)
    gen_ids = list(completion.token_ids)
    # Captured length == prompt + (generated - 1): the last sampled token is never processed.
    processed_ids = list(prompt_ids) + gen_ids[: max(len(gen_ids) - 1, 0)]
    cap_len = caps[points[0]].shape[0]
    print(f"\nmodel={args.model} prompt={len(prompt_ids)} generated={len(gen_ids)} captured_len={cap_len}")
    print(
        f"expected captured_len = prompt + generated-1 = {len(processed_ids)}  -> {'OK' if cap_len == len(processed_ids) else 'MISMATCH'}"
    )

    # Release the EngineCore's VRAM before the eager model asks for its own.
    await backend.shutdown()
    del backend
    gc.collect()
    torch.cuda.empty_cache()

    # --- 2) EagerModel reference over the SAME full sequence -------------------
    npm = EagerModel(args.model, device="cuda", dtype="bfloat16")
    ids = torch.tensor([processed_ids], device="cuda")
    cache = run_with_cache(npm, ids, points)

    print(f"\n{'point':>16}  {'cos(full)':>10}  {'cos(gen-only)':>13}  {'max_abs':>10}")
    worst = 1.0
    prompt_len = len(prompt_ids)
    for key in points:
        ref = cache[key][0].float().cpu()
        got = caps[key].float()[: ref.shape[0]]
        a, b = got.reshape(-1).double(), ref.reshape(-1).double()
        cos = float(torch.dot(a, b) / (a.norm() * b.norm()))
        # Generated-position-only agreement (the part decode-time capture adds).
        if ref.shape[0] > prompt_len:
            ga = got[prompt_len:].reshape(-1).double()
            gb = ref[prompt_len:].reshape(-1).double()
            gcos = float(torch.dot(ga, gb) / (ga.norm() * gb.norm()))
        else:
            gcos = float("nan")
        mx = float((got - ref).abs().max())
        worst = min(worst, cos)
        print(f"{str(key):>16}  {cos:>10.6f}  {gcos:>13.6f}  {mx:>10.3e}")
    print(f"\nworst cosine = {worst:.6f}  ({'PASS' if worst > 0.999 and cap_len == len(processed_ids) else 'CHECK'})")


if __name__ == "__main__":
    asyncio.run(main())
