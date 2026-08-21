"""Validate the engine-owned VLLMModel native capture vs HF.

Constructs interp_engine.VLLMModel with native extract_hidden_states enabled, captures
resid_post for a prompt, and compares to a HuggingFace forward. Proves the engine owns
vLLM construction + native capture.

Run: .venv/bin/python scripts/vllm_backend_check.py
"""

import asyncio
from typing import cast

import torch
from interp_engine import VLLMModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The Jedi in Star Wars wield lightsabers made of pure energy."
CHECK = [0, 12, 26]  # resid_post indices (0 .. N-2 available; N-1 excluded on 0.25.1)


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


async def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, add_special_tokens=True).input_ids

    # enable_extraction is off by default (worker-hook capture serves every point without
    # the speculative forwards it adds), but native resid_post is exactly what this script
    # is here to validate, so it has to opt in.
    backend = VLLMModel(
        MODEL,
        gpu_memory_utilization=0.35,
        max_model_len=512,
        enable_extraction=True,
    )
    print(f"backend layers={backend.n_layers}, prompt tokens={len(ids)}")
    resid = await backend.capture_resid_post(ids, layers=CHECK)
    assert resid, "native extraction returned nothing -- is enable_extraction on?"
    for k in sorted(resid):
        print(f"  resid_post[{k}] shape={tuple(resid[k].shape)}")

    await backend.shutdown()
    del backend
    torch.cuda.empty_cache()

    # `Module.to` is wrapped by a transformers decorator that hides the bound signature,
    # so load through the nn.Module view.
    hf = cast(
        torch.nn.Module,
        AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16),
    )
    hf = hf.to("cuda").eval()
    t = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        ref = hf(t, output_hidden_states=True, use_cache=False)
    hs = ref.hidden_states  # hs[L+1] = resid_post[L]

    print("\nVLLMModel resid_post parity vs HF:")
    for L in CHECK:
        cap = resid[L]
        r = hs[L + 1][0].to("cpu")
        print(f"  resid_post[{L}]: cos={cos(cap, r):.5f}  MAE={(cap.float() - r.float()).abs().mean().item():.5f}")


if __name__ == "__main__":
    asyncio.run(main())
