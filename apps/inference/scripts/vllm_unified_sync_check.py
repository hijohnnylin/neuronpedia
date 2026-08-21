"""Validate the unified sync surface on vLLM: the same free functions the eager backend takes.

Every call here is the one from `docs/USAGE.md` with `backend="vllm"` -- no `await`, no backend
branch -- so what this checks is the part CPU CI cannot reach: that the loop bridge, the per-request
steering context and `generate_steps` really do serve those signatures against a live engine.

`interp-engine`'s own CPU suite covers the eager arm of each of these and the shapes both promise;
what only a GPU can answer is whether the vLLM arm agrees.

Run: PATH="$PWD/.venv/bin:$PATH" .venv/bin/python scripts/vllm_unified_sync_check.py

The venv's `bin` has to be on `PATH`, not just its interpreter: vLLM shells out to `ninja` to JIT
a flashinfer sampling kernel during warmup, and without it the engine core dies at startup with a
bare `FileNotFoundError: 'ninja'` several frames inside the sampler.
"""

import torch
from interp_engine import (
    AddSpec,
    LayerSteeringSpec,
    SteeringSpec,
    capture_attention,
    capture_generation,
    generate_stream,
    load_model,
    run_with_cache,
    steer,
    sync_model,
)

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"
LAYER = 5


def main() -> None:
    model = load_model(
        MODEL,
        backend="vllm",
        gpu_memory_utilization=0.35,
        max_model_len=512,
    )
    sync = sync_model(model)
    sync.warmup()
    ids = model.to_tokens(PROMPT)
    print(f"loaded {MODEL} on vllm: layers={model.n_layers} d_model={model.d_model} tokens={ids.shape}")

    # 1. run_with_cache, which on vLLM goes protocol -> facade -> loop thread and comes back with
    #    the batch axis the eager Cache has, so `cache[point][0]` reads the same on both.
    cache = run_with_cache(model, ids, [f"resid_post.{LAYER}"])
    resid = cache.get("resid_post", LAYER)
    print(f"run_with_cache: {tuple(resid.shape)} finite={bool(torch.isfinite(resid).all())}")
    assert resid.shape[0] == 1 and resid.shape[1] == ids.shape[1], "expected [1, seq, d_model]"

    # 2. capture_attention: one call, three tensors, no batch axis -- and `probs` must be the
    #    softmax of `scores`, which is what says the recompute is self-consistent.
    attn = capture_attention(model, ids, [LAYER])[LAYER]
    print("capture_attention:", {k: tuple(v.shape) for k, v in sorted(attn.items())})
    torch.testing.assert_close(
        torch.softmax(attn["scores"].float(), dim=-1), attn["probs"].float(), rtol=1e-3, atol=1e-4
    )
    print("  probs == softmax(scores): ok")

    # 3. generate_stream: per-token ids and logprobs. `logits` is None here by design, which is the
    #    one field the two backends do not agree on.
    steps = list(generate_stream(model, ids, max_tokens=8, temperature=0.0, n_logprobs=5))
    print(f"generate_stream: {len(steps)} steps, text={''.join(s.token_str for s in steps)!r}")
    assert all(s.logits is None for s in steps), "vLLM cannot ship logits out of the worker"
    assert all(s.logprobs and len(s.logprobs) == 5 for s in steps), "n_logprobs did not reach the sampler"
    first = steps[0].logprobs
    assert first is not None
    # Descending, like eager's `top_logprobs` -- vLLM's own mapping is insertion-ordered and
    # includes the sampled token even when it was outside the top n, so the ordering is converted.
    print(f"  step 0 top-5: {[round(float(e['logprob']), 3) for e in first]}")
    assert [float(e["logprob"]) for e in first] == sorted((float(e["logprob"]) for e in first), reverse=True), (
        "logprobs are not in descending order"
    )

    # 4. capture_generation: prompt + generated - 1 rows, on this backend captured during decode.
    completion, gen_cache = capture_generation(model, ids, [f"resid_post.{LAYER}"], max_tokens=8)
    rows = gen_cache.get("resid_post", LAYER).shape[1]
    print(f"capture_generation: {len(completion.token_ids)} tokens, {rows} captured rows")
    assert rows == ids.shape[1] + len(completion.token_ids) - 1, "captured length is off by more than the last token"

    # 5. The steer() context, which on vLLM registers against each request it opens rather than
    #    installing a global hook. Both the generation and the capture inside must see it.
    vector = torch.randn(model.d_model, generator=torch.Generator().manual_seed(0))
    vector /= vector.norm()
    spec = SteeringSpec(layers={LAYER: LayerSteeringSpec(operations=[AddSpec(vector=vector, scale=40.0)])})

    baseline = [s.token_id for s in generate_stream(model, ids, max_tokens=8, temperature=0.0)]
    with steer(model, spec):
        steered = [s.token_id for s in generate_stream(model, ids, max_tokens=8, temperature=0.0)]
        steered_cache = run_with_cache(model, ids, [f"resid_post.{LAYER}"])
    after = [s.token_id for s in generate_stream(model, ids, max_tokens=8, temperature=0.0)]

    print(f"steer: baseline={baseline}\n       steered ={steered}\n       after   ={after}")
    assert steered != baseline, "the steer did not reach the generation"
    assert after == baseline, "the steer outlived its block"
    assert not torch.allclose(steered_cache.get("resid_post", LAYER), resid), (
        "the capture inside the block was not of the steered forward"
    )
    print("  steered inside the block, clean outside it, and the capture saw it: ok")

    sync.shutdown()
    print("\nall unified sync calls served by vllm")


if __name__ == "__main__":
    main()
