"""Validate vLLM 0.25 NATIVE extract_hidden_states vs an HF reference.

Uses the in-tree `extract_hidden_states` speculative method + ExampleHiddenStatesConnector
(no monkeypatching, no hooks) to capture per-layer residuals, then compares to a
HuggingFace forward's hidden_states. Includes the FINAL layer (layer id = num_hidden_layers)
to confirm the earlier hook-based final-layer caveat is resolved by the native path.

Run: .venv/bin/python scripts/vllm_native_extract_check.py
"""

from typing import cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    cleanup_hidden_states,
    load_hidden_states,
)

MODEL = "Qwen/Qwen3-0.6B"
NUM_LAYERS = 28  # Qwen3-0.6B
# aux layer id L captures the residual ENTERING layer L (== HF hidden_states[L]);
# id == num_hidden_layers captures the final layer output (HF hidden_states[NUM_LAYERS]).
LAYER_IDS = [1, 13, 27, NUM_LAYERS]
PROMPT = "The Jedi in Star Wars wield lightsabers made of pure energy."


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, add_special_tokens=True).input_ids
    print("prompt tokens:", len(ids))

    llm = LLM(
        model=MODEL,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": LAYER_IDS}},
        },
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleHiddenStatesConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={"shared_storage_path": "/dev/shm/np_hs"},
        ),
        enforce_eager=True,
        gpu_memory_utilization=0.35,
        max_model_len=512,
        enable_chunked_prefill=False,
    )
    out = llm.generate([{"prompt_token_ids": list(ids)}], SamplingParams(max_tokens=1))
    kv_params = out[0].kv_transfer_params
    if kv_params is None:
        raise RuntimeError("vLLM returned no kv_transfer_params; the hidden-states connector did not run")
    path = kv_params["hidden_states_path"]
    print("hidden_states_path:", path)
    data = load_hidden_states(path)
    hs_vllm = data["hidden_states"]  # [num_tokens, num_layer_ids, hidden]
    tids = data["token_ids"]
    print("vllm hidden_states:", tuple(hs_vllm.shape), "token_ids:", tuple(tids.shape))

    del llm
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
    hs_ref = ref.hidden_states  # tuple len num_layers+1; hs_ref[L] = resid entering layer L

    print("\nnative extract parity (vLLM vs HF hidden_states[L]):")
    for i, L in enumerate(LAYER_IDS):
        cap = hs_vllm[:, i, :]  # [num_tokens, hidden]
        r = hs_ref[L][0].to("cpu")
        tag = " (FINAL)" if L == NUM_LAYERS else ""
        if cap.shape != r.shape:
            print(f"  layer_id {L}{tag}: SHAPE {tuple(cap.shape)} vs {tuple(r.shape)}")
            continue
        mae = (cap.float() - r.float()).abs().mean().item()
        print(f"  layer_id {L}{tag}: cos={cos(cap, r):.5f}  MAE={mae:.5f}")
        if L == NUM_LAYERS:
            # Characterize the divergence: is only the LAST position correct
            # (vLLM last-token/logits optimization) or all positions wrong?
            print(
                f"      last-pos cos={cos(cap[-1], r[-1]):.5f}  MAE={(cap[-1].float() - r[-1].float()).abs().mean().item():.5f}"
            )
            if cap.shape[0] > 1:
                print(
                    f"      non-last cos={cos(cap[:-1], r[:-1]):.5f}  MAE={(cap[:-1].float() - r[:-1].float()).abs().mean().item():.5f}"
                )

    cleanup_hidden_states(path)


if __name__ == "__main__":
    main()
