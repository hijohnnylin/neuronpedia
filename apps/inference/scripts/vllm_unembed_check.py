"""Validate the vLLM worker-side unembed (compute_logits) vs the EagerModel decode.

Captures resid_post at a few layers on vLLM, decodes to logits via the uniform
worker compute_logits (reusing vLLM's own norm+lm_head), and compares to EagerModel's
decode_residuals (real transformers final_norm + lm_head) over the same prompt. This
is the parity gate for jlens logit/Jacobian-lens read-out on vLLM.

Run it on a softcapped model (Gemma-2) as well as an uncapped one. vLLM applies
``final_logit_softcapping`` inside ``compute_logits`` while the eager ``lm_head``
returns raw logits, so the eager side is given the same cap below to keep the two
comparable. An uncapped model cannot exercise that difference at all:

    IE_VLLM_GPU_UTIL=0.8 \
      .venv/bin/python scripts/vllm_unembed_check.py --model Qwen/Qwen3-0.6B
    IE_VLLM_GPU_UTIL=0.8 \
      .venv/bin/python scripts/vllm_unembed_check.py --model google/gemma-2-2b
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument(
        "--prompt",
        default="The capital of France is Paris, and the capital of Japan is",
    )
    args = ap.parse_args()

    import gc

    import torch
    from interp_engine import Address, VLLMModel

    backend = VLLMModel(
        args.model,
        dtype="bfloat16",
        enforce_eager=True,
        enable_extraction=False,
        gpu_memory_utilization=float(os.environ.get("IE_VLLM_GPU_UTIL", "0.8")),
    )
    n_layers = backend.n_layers
    layers = sorted({0, n_layers // 2, n_layers - 1})
    prompt_ids = backend.tokenizer(args.prompt, add_special_tokens=True)["input_ids"]
    caps = await backend.capture(prompt_ids, [Address("resid_post", layer) for layer in layers])
    # Decode each layer's residuals -> logits via the worker unembed.
    vllm_logits = {layer: await backend.decode_residuals(caps[Address("resid_post", layer)]) for layer in layers}

    # Release the EngineCore's VRAM before the eager model asks for its own.
    await backend.shutdown()
    del backend
    gc.collect()
    torch.cuda.empty_cache()

    from interp_engine import EagerModel, decode_residuals, run_with_cache

    npm = EagerModel(args.model, device="cuda", dtype="bfloat16")
    ids = torch.tensor([prompt_ids], device="cuda")
    cache = run_with_cache(npm, ids, [Address("resid_post", layer) for layer in layers])

    # vLLM already softcapped its logits; cap the eager reference to match.
    # HF configs are not typed as callables on ``nn.Module`` attributes; treat as Any.
    text_config: Any = npm.hf_model.config
    if hasattr(text_config, "get_text_config"):
        text_config = text_config.get_text_config()
    softcap = getattr(text_config, "final_logit_softcapping", None)
    softcap = float(softcap) if softcap is not None else None

    print(
        f"\nmodel={args.model} seq={len(prompt_ids)} layers={layers} "
        f"vocab={npm.arch.vocab_size} final_logit_softcapping={softcap}"
    )
    print(f"{'layer':>6}  {'logits_cos':>11}  {'top1_agree':>10}  {'max_abs':>10}")
    worst = 1.0
    for layer in layers:
        ref = decode_residuals(npm, cache.get("resid_post", layer)[0], softcap=softcap).float().cpu()
        got = vllm_logits[layer].float()[: ref.shape[0]]
        a, b = got.reshape(-1).double(), ref.reshape(-1).double()
        cos = float(torch.dot(a, b) / (a.norm() * b.norm()))
        top1 = float((got.argmax(-1) == ref.argmax(-1)).float().mean())
        mx = float((got - ref).abs().max())
        worst = min(worst, cos)
        print(f"{layer:>6}  {cos:>11.6f}  {top1:>10.4f}  {mx:>10.3e}")
    print(f"\nworst logits cosine = {worst:.6f}  ({'PASS' if worst > 0.999 else 'CHECK'})")


if __name__ == "__main__":
    asyncio.run(main())
