"""Validate engine-owned vLLM steering write-hooks.

Steering resid_post at layer L by coeff*v should shift the captured resid_post[L]
by exactly coeff*v (all positions). Confirms the collective_rpc write-hooks work
and compose with native capture.

Run: .venv/bin/python scripts/vllm_steer_check.py
"""

import asyncio

import torch
from interp_engine import VLLMModel
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"
LAYER = 13
COEFF = 4.0


async def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, add_special_tokens=True).input_ids
    # enable_extraction because capture_resid_post below reads vLLM's NATIVE hidden-state
    # extraction, which is what this script deliberately composes steering with; without it
    # the engine never writes hidden_states_path and every capture call fails.
    backend = VLLMModel(MODEL, gpu_memory_utilization=0.35, max_model_len=512, enable_extraction=True)

    base = (await backend.capture_resid_post(ids, layers=[LAYER]))[LAYER]  # [T, H]
    hidden = base.shape[-1]

    torch.manual_seed(0)
    v = torch.randn(hidden)
    v = v / v.norm()

    await backend.set_steering([{"layer": LAYER, "point": "resid_post", "vector": v.tolist(), "coeff": COEFF}])
    steered = (await backend.capture_resid_post(ids, layers=[LAYER]))[LAYER]
    await backend.clear_steering()
    cleared = (await backend.capture_resid_post(ids, layers=[LAYER]))[LAYER]

    expected = base.float() + COEFF * v  # broadcast over positions
    delta_err = (steered.float() - expected).abs().mean().item()
    base_shift = (steered.float() - base.float()).abs().mean().item()
    cleared_err = (cleared.float() - base.float()).abs().mean().item()

    print(f"[add] steered vs (base + coeff*v)  MAE={delta_err:.5f}  (should be ~0)")
    print(f"[add] steered vs base              MAE={base_shift:.5f}  (should be > 0)")
    print(f"[add] cleared vs base              MAE={cleared_err:.5f}  (should be ~0)")

    # projection-cap: cap the projection onto u to half the base projection.
    unit = v / v.norm()
    base_proj = (base.float() * unit).sum(-1)  # [T]
    cap = float(base_proj.mean().item()) * 0.5
    await backend.set_steering(
        [
            {
                "layer": LAYER,
                "point": "resid_post",
                "op": "projection_cap",
                "vector": v.tolist(),
                "max": cap,
            }
        ]
    )
    capped = (await backend.capture_resid_post(ids, layers=[LAYER]))[LAYER]
    await backend.clear_steering()
    capped_proj = (capped.float() * unit).sum(-1)  # [T]
    over = float((capped_proj - cap).clamp_min(0).max().item())
    print(
        f"[cap] max_over_cap={over:.5f} (should be ~0); base_proj_mean={base_proj.mean():.3f} cap={cap:.3f} capped_proj_max={capped_proj.max():.3f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
