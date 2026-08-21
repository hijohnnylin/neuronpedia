"""Validate DFA-on-vLLM parity vs EagerModel end-to-end.

DFA = per-head value (GQA-expanded) x attention pattern x W_enc. The math
(engine_adapter.dfa_from_v_and_probs) is shared; the only backend difference is the
(value, attn_probs) inputs. This runs the FULL DFA with a fixed synthetic W_enc on
both backends' captured inputs and compares dfa_values (so no real -att- SAE needed).

Unlike its sibling checks this one imports ``neuronpedia_inference`` (for the shared DFA
math), which running a file under ``scripts/`` does not put on the path -- hence PYTHONPATH:

    PYTHONPATH=. IE_VLLM_GPU_UTIL=0.8 \
      .venv/bin/python scripts/vllm_dfa_check.py --model Qwen/Qwen3-0.6B
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
    args = ap.parse_args()

    import gc

    import torch
    from interp_engine import VLLMModel

    from neuronpedia_inference.engine_adapter import dfa_from_v_and_probs

    backend = VLLMModel(
        args.model,
        dtype="bfloat16",
        enforce_eager=True,
        enable_extraction=False,
        gpu_memory_utilization=float(os.environ.get("IE_VLLM_GPU_UTIL", "0.8")),
    )
    ad = backend._attn_dims
    dims = {
        "n_heads": ad["n_heads"],
        "n_kv_heads": ad["n_kv_heads"],
        "head_dim": ad["head_dim"],
    }
    layer = backend.n_layers // 2
    z_width = dims["n_heads"] * dims["head_dim"]
    torch.manual_seed(0)
    W_enc = torch.randn(z_width, 4, dtype=torch.float32)  # [d_model_z, n_features]

    prompt_ids = backend.tokenizer(args.prompt, add_special_tokens=True)["input_ids"]
    seq = len(prompt_ids)
    mvi = seq - 1  # DFA target = last position
    got = await backend.capture_attention(prompt_ids, [layer])
    v_vllm = got[layer]["value"].unsqueeze(0)
    probs_vllm = got[layer]["probs"].unsqueeze(0)
    dfa_vllm = dfa_from_v_and_probs(v_vllm, probs_vllm, W_enc, 0, mvi, **dims)

    # Release the EngineCore's VRAM before the eager model asks for its own.
    await backend.shutdown()
    del backend
    gc.collect()
    torch.cuda.empty_cache()

    from interp_engine import EagerModel, per_head_value, run_with_cache

    npm = EagerModel(args.model, device="cuda", dtype="bfloat16", attn_implementation="eager")
    ids = torch.tensor([prompt_ids], device="cuda")
    cache = run_with_cache(npm, ids, [("value", layer), ("attn_probs", layer)])
    v_eager = per_head_value(npm, cache, layer)
    probs_eager = cache.get("attn_probs", layer)
    dfa_eager = dfa_from_v_and_probs(
        v_eager,
        probs_eager,
        W_enc.to(v_eager.device),
        0,
        mvi,
        n_heads=npm.n_heads,
        n_kv_heads=npm.n_kv_heads,
        head_dim=npm.head_dim,
    )

    a = torch.tensor(dfa_vllm["dfa_values"]).double()
    b = torch.tensor(dfa_eager["dfa_values"]).double()
    cos = float(torch.dot(a, b) / (a.norm() * b.norm()))
    mx = float((a - b).abs().max())
    print(f"\nmodel={args.model} layer={layer} seq={seq} z_width={z_width}")
    print(f"dfa_values cos = {cos:.6f}   max_abs = {mx:.3e}")
    print(f"vllm  dfa_max_value = {dfa_vllm['dfa_max_value']:.4f}")
    print(f"eager dfa_max_value = {dfa_eager['dfa_max_value']:.4f}")
    print(f"\n{'PASS' if cos > 0.999 else 'CHECK'}")


if __name__ == "__main__":
    asyncio.run(main())
