"""Validate the per-request demux: concurrent capture + steering stay isolated.

The worker hooks are GLOBAL (one set of PyTorch hooks for the whole batched forward).
The demux (``vllm_capture/_demux.py``) attributes each row of the flattened batch to its
request via ``model_runner.input_batch.req_ids`` + ``num_scheduled_tokens``. This test
fires several requests CONCURRENTLY (so vLLM batches them into shared forwards) and
checks each one's result matches its single-request baseline -- i.e. no request's
steering/capture leaks into another's rows.

Design: steering ``resid_post[L]`` by ``coeff*v`` shifts the captured ``resid_post[L]``
by exactly ``coeff*v`` at every position (steer-then-capture in one hook). So:
  - concurrent unsteered capture must equal the unsteered baseline (NOT shifted) ->
    proves no steering leaked in;
  - concurrent steered-X capture minus unsteered baseline must equal coeff*v_X ->
    proves X's steering hit X's rows and only those.

Run:
  .venv/bin/python scripts/vllm_concurrency_isolation_check.py
"""

import asyncio
from typing import cast

import torch
from interp_engine import Address, VLLMModel
from interp_engine.steer_specs import AddSpec, LayerSteeringSpec, SteeringSpec
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
LAYER = 14
COEFF = 8.0
PROMPTS = [
    "The capital of France is",
    "Once upon a time there was a",
    "The mitochondria is the powerhouse of the",
    "In the beginning the universe was",
]


def _spec(hidden: int, dim: int) -> SteeringSpec:
    """AddSpec at LAYER: a one-hot direction (+COEFF at `dim`) so the delta is obvious."""
    v = torch.zeros(hidden)
    v[dim] = 1.0
    return SteeringSpec(layers={LAYER: LayerSteeringSpec(operations=[AddSpec(vector=v, scale=COEFF)])})


Capture = dict[Address, torch.Tensor]


def _cap(cap: Capture) -> torch.Tensor:
    return cap[Address("resid_post", LAYER)].float()  # [T, H]


async def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = [tok(p, add_special_tokens=True).input_ids for p in PROMPTS]
    backend = VLLMModel(MODEL, dtype="bfloat16", gpu_memory_utilization=0.4, max_model_len=1024)
    await backend._ensure_engine()

    hidden = backend.d_model
    specX = _spec(hidden, dim=3)
    specY = _spec(hidden, dim=7)
    points = [Address("resid_post", LAYER)]
    P = ids[0]

    # ---- single-request baselines (serialized) ----
    baseC = _cap(await backend.capture(P, points))  # unsteered
    baseX = _cap(await backend.capture(P, points, steering_spec=specX))  # +COEFF at dim 3
    baseY = _cap(await backend.capture(P, points, steering_spec=specY))  # +COEFF at dim 7

    # sanity: steering shifts exactly coeff at the steered dim, ~0 elsewhere
    dX = (baseX - baseC).mean(0)  # [H]
    assert abs(dX[3].item() - COEFF) < 0.2, f"baseline X delta at dim3={dX[3].item()} != {COEFF}"
    assert dX[7].abs().item() < 0.2, f"baseline X leaked to dim7={dX[7].item()}"

    # ---- concurrent: X-steered, Y-steered, unsteered in the SAME batched forwards ----
    NREP = 3
    tasks = []
    kinds = []
    for _ in range(NREP):
        tasks += [
            backend.capture(P, points, steering_spec=specX),
            backend.capture(P, points, steering_spec=specY),
            backend.capture(P, points),
        ]
        kinds += ["X", "Y", "C"]
    results = await asyncio.gather(*tasks)

    ok = True
    for kind, res in zip(kinds, results):
        cur = _cap(res)
        if cur.shape != baseC.shape:
            print(f"  [{kind}] SHAPE MISMATCH {cur.shape} vs {baseC.shape}")
            ok = False
            continue
        if kind == "C":
            # Unsteered under concurrency must NOT pick up either concept's steering.
            # The leak signal is a systematic per-dim mean shift (a real leak shifts dim3
            # or dim7 by ~COEFF at every position); global max-abs is just bf16 batching
            # noise (batch-of-1 baseline vs batched run), largest at high-norm BOS tokens.
            d = (cur - baseC).mean(0)
            cos = torch.nn.functional.cosine_similarity(cur.flatten(), baseC.flatten(), dim=0).item()
            maxabs = (cur - baseC).abs().max().item()
            good = abs(d[3].item()) < 0.5 and abs(d[7].item()) < 0.5 and cos > 0.999
            print(
                f"  [C] leak@dim3={d[3].item():.3f} leak@dim7={d[7].item():.3f} (~0) "
                f"cos={cos:.6f} maxabs={maxabs:.3f}(noise)  {'OK' if good else 'FAIL (pollution!)'}"
            )
            ok = ok and good
        else:
            base = baseX if kind == "X" else baseY
            sdim = 3 if kind == "X" else 7
            odim = 7 if kind == "X" else 3
            d = (cur - baseC).mean(0)
            cos = torch.nn.functional.cosine_similarity(cur.flatten(), base.flatten(), dim=0).item()
            # steered dim shifted by ~COEFF, the OTHER concept's dim NOT shifted
            good = abs(d[sdim].item() - COEFF) < 0.5 and d[odim].abs().item() < 0.5 and cos > 0.9995
            print(
                f"  [{kind}] delta@dim{sdim}={d[sdim].item():.3f}(~{COEFF}) "
                f"leak@dim{odim}={d[odim].item():.3f}(~0) cos_vs_base={cos:.6f}  "
                f"{'OK' if good else 'FAIL'}"
            )
            ok = ok and good

    # ---- mixed prefill/decode: a steered generation batched with prompt-only captures ----
    print("mixed prefill/decode:")
    base_gen_out, base_gen_caps = await backend.capture_generation(
        P, points, max_tokens=10, temperature=0.0, steering_spec=specX
    )
    base_gen = _cap(base_gen_caps)  # [prompt+gen-1, H]

    gen_task = backend.capture_generation(P, points, max_tokens=10, temperature=0.0, steering_spec=specX)
    prompt_tasks = [backend.capture(ids[i % len(ids)], points) for i in range(6)]
    gathered = await asyncio.gather(gen_task, *prompt_tasks)
    # capture_generation yields (text, captures); the plain captures yield just the dict.
    _, gen_caps = cast(tuple[str, Capture], gathered[0])
    cap_res = cast(list[Capture], list(gathered[1:]))
    cur_gen = _cap(gen_caps)

    gen_cos = torch.nn.functional.cosine_similarity(cur_gen.flatten(), base_gen.flatten(), dim=0).item()
    gen_len_ok = cur_gen.shape == base_gen.shape
    print(
        f"  [gen] shape={tuple(cur_gen.shape)} match_base={gen_len_ok} cos_vs_base={gen_cos:.6f} "
        f"{'OK' if (gen_len_ok and gen_cos > 0.999) else 'FAIL'}"
    )
    ok = ok and gen_len_ok and gen_cos > 0.999

    # the concurrent prompt-only captures must be unsteered (gen steers dim3 -> must NOT
    # leak into a neighbor: neighbor's dim3 mean shift ~0). maxabs is bf16 noise only.
    for i, res in enumerate(cap_res):
        cur = _cap(res)
        base_i = _cap(await backend.capture(ids[i % len(ids)], points))
        if cur.shape != base_i.shape:
            print(f"  [gen-neighbor {i}] SHAPE MISMATCH {cur.shape} vs {base_i.shape}")
            ok = False
            continue
        d = (cur - base_i).mean(0)
        cos = torch.nn.functional.cosine_similarity(cur.flatten(), base_i.flatten(), dim=0).item()
        maxabs = (cur - base_i).abs().max().item()
        good = abs(d[3].item()) < 0.5 and cos > 0.999
        print(
            f"  [gen-neighbor {i}] leak@dim3={d[3].item():.3f}(~0) cos={cos:.6f} "
            f"maxabs={maxabs:.3f}(noise)  {'OK' if good else 'FAIL (pollution!)'}"
        )
        ok = ok and good

    print(
        "\nRESULT:",
        "PASS - per-request isolation holds" if ok else "FAIL - cross-request contamination",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
