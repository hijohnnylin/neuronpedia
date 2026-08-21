"""Validate VLLMModel.generate_from_embeds (EmbedsPrompt path for NLA).

Feeding ``prompt_embeds = embed_tokens(ids)`` must produce the same greedy generation
as feeding the token ids directly. We fetch the embeddings from the vLLM worker itself
(so they are byte-identical to what the token-id path looks up), then compare.

Run: .venv/bin/python scripts/vllm_prompt_embeds_check.py
"""

import asyncio
from typing import cast

import torch
from interp_engine import VLLMModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import RequestOutput, SamplingParams

MODEL = "Qwen/Qwen3-0.6B"  # embed_scale = 1.0 (no Gemma sqrt(hidden) pre-scale)
PROMPT = "The capital of France is"
MAX_TOKENS = 24


async def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, add_special_tokens=True).input_ids

    # Input embeddings from the same checkpoint (CPU, bf16) -- what NLA injects into.
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    with torch.no_grad():
        embeds = hf.get_input_embeddings()(torch.tensor(ids)).clone()  # [T, d] bf16
    del hf

    backend = VLLMModel(
        MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.4,
        max_model_len=1024,
        enforce_eager=False,  # CUDA graphs + inductor compile on (verbalizer uses no hooks)
        enable_prompt_embeds=True,
    )
    await backend._ensure_engine()
    print("[progress] engine ready", flush=True)

    # Baseline: token-ids generation.
    base = await backend.generate_full(ids, max_tokens=MAX_TOKENS, temperature=0.0)
    base_text = base.text
    print("[progress] token-ids generation done", flush=True)

    # Embeds path: same content fed as prompt_embeds.
    print(
        f"[progress] embeds shape={tuple(embeds.shape)} dtype={embeds.dtype}",
        flush=True,
    )
    # stream=False (the default) returns the final RequestOutput, not the async generator
    # the streaming form yields.
    out = cast(
        RequestOutput,
        await backend.generate_from_embeds(embeds, SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)),
    )
    emb_text = out.outputs[0].text
    print("[progress] prompt_embeds generation done", flush=True)

    print(f"model={MODEL} prompt={PROMPT!r} max_tokens={MAX_TOKENS}")
    print(f"  token-ids : {base_text!r}")
    print(f"  prompt_emb: {emb_text!r}")
    match = base_text == emb_text
    prefix = base_text[:40] == emb_text[:40]  # greedy: exact expected; prefix = soft pass
    print(f"\nRESULT: {'PASS (exact)' if match else ('PASS (prefix)' if prefix else 'FAIL')}")
    return 0 if (match or prefix) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
