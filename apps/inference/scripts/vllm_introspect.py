"""Introspect a vLLM worker's model to design the np-engine worker-side capture.

Boots a tiny model with enforce_eager and uses collective_rpc to dump, from
inside the worker: how to reach the model, the decoder-layer module layout,
submodule forward signatures, and the decoder-layer forward return structure
(tuple vs tensor, i.e. the Qwen fused-norm case). Run with the inference venv:

    VLLM_ALLOW_INSECURE_SERIALIZATION=1 .venv/bin/python scripts/vllm_introspect.py

The flag is needed because this dumps arbitrary worker internals through an ad-hoc
function object, which vLLM v1 will not msgpack to the engine core without it. The
capture path proper does not need it -- those hooks are worker methods invoked by name
(see interp_engine.vllm_plugin).
"""

import contextlib
import inspect

from vllm import LLM


def introspect(worker):  # runs inside each vLLM worker process
    import torch

    out: dict = {}
    # Find the nn.Module model on the worker.
    model = None
    for path in ("model_runner.model", "model_runner.model.model"):
        obj = worker
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, torch.nn.Module):
                out[f"found:{path}"] = type(obj).__name__
        except Exception as e:  # noqa: BLE001
            out[f"err:{path}"] = repr(e)
    model = worker.model_runner.model
    out["model_type"] = type(model).__name__
    out["model_children"] = [n for n, _ in model.named_children()]

    # Locate decoder layers (commonly model.model.layers).
    layers = None
    for path in ("model.layers", "layers", "model.model.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            layers = obj
            out["layers_path"] = path
            break
        except Exception:  # noqa: BLE001
            continue
    if layers is None:
        out["layers"] = "NOT FOUND"
        return out

    layer0 = layers[0]
    out["layer_type"] = type(layer0).__name__
    out["layer_forward_sig"] = str(inspect.signature(type(layer0).forward))
    out["layer_children"] = [n for n, _ in layer0.named_children()]

    for sub in ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm"):
        m = getattr(layer0, sub, None)
        if m is None:
            continue
        out[f"{sub}.type"] = type(m).__name__
        out[f"{sub}.children"] = [n for n, _ in m.named_children()]
        with contextlib.suppress(Exception):
            out[f"{sub}.forward_sig"] = str(inspect.signature(type(m).forward))
    return out


def main():
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=512,
    )
    results = llm.collective_rpc(introspect)
    import pprint

    for i, r in enumerate(results):
        print(f"===== worker {i} =====")
        pprint.pprint(r, width=140, sort_dicts=True)


if __name__ == "__main__":
    main()
