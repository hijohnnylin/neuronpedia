"""Validate the async engine-owned VLLMModel: generation + native capture.

Run: .venv/bin/python scripts/vllm_async_backend_check.py
"""

import asyncio

import torch
from interp_engine import (
    AddSpec,
    LayerSteeringSpec,
    SteeringSpec,
    VLLMModel,
)
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"


async def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, add_special_tokens=True).input_ids

    # enable_extraction (off by default) is what populates the native resid_post path
    # exercised below; without it capture_resid_post returns an empty dict.
    backend = VLLMModel(
        MODEL,
        gpu_memory_utilization=0.35,
        max_model_len=512,
        enable_extraction=True,
    )
    print(f"async backend layers={backend.n_layers}, prompt tokens={len(ids)}")

    text = await backend.generate_text(ids, max_tokens=16, temperature=0.0)
    print("GENERATED:", repr(text))

    # Streaming: collect deltas and confirm they reconstruct the full text.
    deltas = []
    async for d in backend.generate_stream(ids, max_tokens=16, temperature=0.0):
        deltas.append(d)
    print(f"STREAM: {len(deltas)} deltas, joined={''.join(deltas)!r}")

    # Logprobs plumbing.
    full = await backend.generate_full(ids, max_tokens=8, temperature=0.0, logprobs=3)
    lp = full.logprobs
    print(f"LOGPROBS: present={lp is not None} n_steps={len(lp) if lp else 0}")

    resid = await backend.capture_resid_post(ids, layers=[0, 13, 26])
    assert resid, "native extraction returned nothing -- is enable_extraction on?"
    for k in sorted(resid):
        t = resid[k]
        print(
            f"  resid_post[{k}] shape={tuple(t.shape)} finite={bool(torch.isfinite(t).all())} norm={float(t.float().norm()):.2f}"
        )

    # VLLMSteerModel-style generate_steered: a large steer should change the output.
    from vllm import SamplingParams

    torch.manual_seed(0)
    vec = torch.randn(backend.d_model)
    vec = vec / vec.norm()
    spec = SteeringSpec(layers={5: LayerSteeringSpec(operations=[AddSpec(vector=vec, scale=40.0)])})
    sp = SamplingParams(max_tokens=16, temperature=0.0)
    unsteered = await backend.generate_steered(ids, sp, steering_spec=None, stream=False)
    steered = await backend.generate_steered(ids, sp, steering_spec=spec, stream=False)
    print("UNSTEERED:", repr(unsteered))
    print("STEERED  :", repr(steered))
    print("differ:", unsteered != steered)


if __name__ == "__main__":
    asyncio.run(main())
