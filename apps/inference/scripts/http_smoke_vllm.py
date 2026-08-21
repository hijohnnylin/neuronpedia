"""HTTP smoke for the vLLM-backed endpoints (wiring check, not numeric parity).

Hits the newly-wired endpoints on a running vLLM inference server with minimal
valid bodies (no-SAE paths) and reports status + a snippet. Numeric parity is
covered by the engine-level cos checks; this catches request/response wiring bugs
(async streaming, backend branches, shapes).

    python scripts/http_smoke_vllm.py --base http://127.0.0.1:5119 --model Qwen/Qwen3-0.6B --d-model 1024
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def _post(base: str, path: str, body: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read().decode(errors="replace")
            return resp.status, data
    except urllib.error.HTTPError as e:  # noqa
        return e.code, e.read().decode(errors="replace")
    except Exception as e:  # noqa: BLE001
        return -1, repr(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5119")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--layer", type=int, default=14)
    args = ap.parse_args()

    base = args.base.rstrip("/") + "/v1"
    hook = f"blocks.{args.layer}.hook_resid_post"
    vec = [0.01] * args.d_model
    results: list[tuple[str, int, bool]] = []

    def check(name: str, status: int, body: str, ok_if: str = "") -> None:
        ok = status == 200 and (ok_if in body if ok_if else True)
        results.append((name, status, ok))
        snippet = body.replace("\n", " ")[:200]
        print(f"[{'OK ' if ok else 'FAIL'}] {name}: HTTP {status}  {snippet}")

    # 1) tokenize
    s, b = _post(base, "/tokenize", {"text": "The capital of France is", "model": args.model})
    check("tokenize", s, b)

    # 2) lens read-out (LOGIT_LENS, no generation, no SAE)
    s, b = _post(
        base,
        "/lens/prompt",
        {
            "model": args.model,
            "prompt": "The capital of France is",
            "type": ["LOGIT_LENS"],
            "num_completion_tokens": 0,
            "temperature": 0.0,
            "top_n": 5,
            "stream": False,
        },
    )
    check("lens/prompt readout", s, b, ok_if="done")

    # 3) lens with generation (decode-time capture path)
    s, b = _post(
        base,
        "/lens/prompt",
        {
            "model": args.model,
            "prompt": "The capital of France is",
            "type": ["LOGIT_LENS"],
            "num_completion_tokens": 4,
            "temperature": 0.0,
            "top_n": 5,
            "stream": False,
        },
    )
    check("lens/prompt generation", s, b, ok_if="done")

    # 4) activation/single vector path (capture via worker hooks)
    s, b = _post(
        base,
        "/activation/single",
        {
            "model": args.model,
            "prompt": "The capital of France is Paris.",
            "vector": vec,
            "hook": hook,
        },
    )
    check("activation/single vector", s, b, ok_if="activation")

    # 5) steer/completion vector path (decode-time steering, non-stream)
    s, b = _post(
        base,
        "/steer/completion",
        {
            "model": args.model,
            "prompt": "I think that",
            "vectors": [{"steering_vector": vec, "strength": 4.0, "hook": hook}],
            "types": ["STEERED", "DEFAULT"],
            "steer_method": "SIMPLE_ADDITIVE",
            "strength_multiplier": 1.0,
            "temperature": 0.0,
            "n_completion_tokens": 6,
            "seed": 0,
            "stream": False,
            "normalize_steering": False,
            "freq_penalty": 0.0,
        },
    )
    check("steer/completion vector", s, b, ok_if="outputs")

    # 6) activation/attention (off-kernel probs recompute)
    s, b = _post(
        base,
        "/activation/attention",
        {
            "model": args.model,
            "prompt": "When Mary and John went to the store, John gave a drink to Mary.",
            "layer": args.layer,
            "head": 1,
        },
    )
    check("activation/attention", s, b, ok_if="attention_values")

    # 7) lens intervention: additive steer during generation (norm-scaled worker op)
    s, b = _post(
        base,
        "/lens/prompt",
        {
            "model": args.model,
            "prompt": "The capital of France is",
            "type": ["LOGIT_LENS"],
            "num_completion_tokens": 4,
            "temperature": 0.0,
            "top_n": 5,
            "steer_tokens": [{"token": " Paris", "type": "LOGIT_LENS"}],
            "steer_layers": [args.layer],
            "steer_strength": 8.0,
            "steer_generated_tokens": True,
            "stream": False,
        },
    )
    check("lens steer intervention", s, b, ok_if="done")

    # 8) lens intervention: swap source->target readout direction
    s, b = _post(
        base,
        "/lens/prompt",
        {
            "model": args.model,
            "prompt": "The capital of France is",
            "type": ["LOGIT_LENS"],
            "num_completion_tokens": 0,
            "temperature": 0.0,
            "top_n": 5,
            "steer_tokens": [{"token": " Paris", "type": "LOGIT_LENS"}],
            "swap_token": {"token": " London", "type": "LOGIT_LENS"},
            "steer_layers": [args.layer],
            "stream": False,
        },
    )
    check("lens swap intervention", s, b, ok_if="done")

    print("\n=== summary ===")
    for name, status, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name} (HTTP {status})")
    n_ok = sum(1 for _, _, ok in results if ok)
    print(f"\n{n_ok}/{len(results)} endpoints OK")


if __name__ == "__main__":
    main()
