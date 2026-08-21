"""Validate interp-engine vLLM worker capture vs an HF reference forward.

Boots a tiny model in vLLM (enforce_eager), captures resid_post/mlp_out/attn_out
via the engine's worker-side hooks, then compares resid_post to a HuggingFace
forward's hidden_states on the same tokens. Confirms the collective_rpc hook
mechanism + the fused-norm (hidden+residual) reconstruction.

Run with the inference venv:
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 .venv/bin/python scripts/vllm_capture_check.py

The env var is required HERE and nowhere else among these checks: this script builds a raw
``vllm.LLM`` and hands ``collective_rpc`` the worker functions themselves, which vLLM v1
will not serialize to its out-of-process engine core without it. Keep it -- exercising the
bare hook mechanism is the point of this script.

Everything else, including ``VLLMModel``, goes through the ``worker_extension_cls`` in
``interp_engine.vllm_plugin`` and invokes those same hooks by NAME, which needs no flag.
That is the path to copy for real use; see that module's docstring.
"""

from typing import cast

import torch
from interp_engine import Address, decode_capture_payload

# The worker functions themselves, rather than the by-name plugin path: handing a callable to
# collective_rpc is exactly the mechanism this script exists to exercise. They are not
# top-level exports for that reason -- see interp_engine.vllm_plugin for the supported route.
from interp_engine.vllm_capture import worker_collect_capture, worker_install_capture
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-0.6B"
# Qwen3-0.6B has 28 layers, and the LAST one is deliberately absent: HF appends its final
# hidden state only after applying the final norm, so hidden_states[28] is post-norm while
# our capture is the pre-norm residual. Comparing them reports cos~0.57 -- an artifact of the
# reference, not a capture bug (vllm_capture_generation_check reads resid_post.27 at 0.9994).
LAYERS = [0, 1, 13, 26]
PROMPT = "The Jedi in Star Wars wield lightsabers made of pure energy."
MIN_COS = 0.999


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    token_ids = tok(PROMPT, add_special_tokens=True).input_ids
    print(f"prompt tokens: {len(token_ids)}")

    llm = LLM(model=MODEL, enforce_eager=True, gpu_memory_utilization=0.35, max_model_len=512)

    points = []
    for layer in LAYERS:
        points += [("resid_post", layer), ("mlp_out", layer), ("attn_out", layer)]

    llm.collective_rpc(worker_install_capture, kwargs={"points": points})
    llm.generate(
        [{"prompt_token_ids": list(token_ids)}],
        SamplingParams(max_tokens=1, temperature=0.0),
    )
    payloads = llm.collective_rpc(worker_collect_capture)
    captured = decode_capture_payload(payloads[0])
    print("captured keys:", sorted(str(a) for a in captured))
    for address, t in sorted(captured.items(), key=lambda kv: str(kv[0])):
        print(f"  {address} shape={tuple(t.shape)} dtype={t.dtype}")

    # HF reference
    del llm
    torch.cuda.empty_cache()
    # `Module.to` is wrapped by a transformers decorator that hides the bound signature,
    # so load through the nn.Module view.
    hf = cast(
        torch.nn.Module,
        AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16),
    )
    hf = hf.to("cuda").eval()
    ids = torch.tensor([token_ids], device="cuda")
    with torch.no_grad():
        out = hf(ids, output_hidden_states=True, use_cache=False)
    # tuple len n_layers+1; hs[L+1] = resid_post of layer L, except for the final layer (see
    # the note on LAYERS).
    hs = out.hidden_states

    print("\nresid_post parity (vLLM capture vs HF hidden_states):")
    worst = 1.0
    for layer in LAYERS:
        cap = captured[Address("resid_post", layer)]  # [num_tokens, hidden]
        ref = hs[layer + 1][0].to("cpu")  # [num_tokens, hidden]
        assert cap.shape == ref.shape, f"layer {layer}: shape mismatch cap={tuple(cap.shape)} ref={tuple(ref.shape)}"
        similarity = cos(cap, ref)
        worst = min(worst, similarity)
        mae = (cap.float() - ref.float()).abs().mean().item()
        print(f"  layer {layer}: cos={similarity:.5f}  MAE={mae:.5f}")

    # Without this the script cannot fail: it printed a bad number and exited 0.
    print(f"\nworst cosine = {worst:.5f}  ({'PASS' if worst >= MIN_COS else 'FAIL'})")
    assert worst >= MIN_COS, f"resid_post parity broke: worst cosine {worst:.5f} < {MIN_COS}"


if __name__ == "__main__":
    main()
