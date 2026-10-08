"""Write an HF copy of a base model with the MLX affine-quantized weights put back as bf16."""

import json
import os
import shutil
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

base_id, mlx_id, out = sys.argv[1:4]
base = snapshot_download(
    base_id, allow_patterns=["*.json", "*.safetensors*", "*.txt", "*.jinja"]
)
mlx = snapshot_download(mlx_id)
with open(f"{mlx}/config.json") as cfg:
    q = json.load(cfg)["quantization"]
group, bits = q["group_size"], q["bits"]
assert q.get("mode", "affine") == "affine", q
per_word = 32 // bits


def dequant(w, scales, biases):
    shifts = torch.arange(per_word, dtype=torch.int64) * bits
    vals = (w.to(torch.int64).unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    vals = vals.reshape(w.shape[0], -1, group).float()
    return (vals * scales.float().unsqueeze(-1) + biases.float().unsqueeze(-1)).reshape(
        w.shape[0], -1
    )


with safe_open(f"{mlx}/model.safetensors", "pt") as m:
    keys = set(m.keys())
    quant = {k[: -len(".scales")] for k in keys if k.endswith(".scales")}
    os.makedirs(out, exist_ok=True)
    replaced, worst = 0, 0.0
    for f in os.listdir(base):
        src = f"{base}/{f}"
        if not f.endswith(".safetensors"):
            shutil.copy(src, f"{out}/{f}")
            continue
        tensors = {}
        with safe_open(src, "pt") as b:
            for k in b.keys():  # noqa: SIM118 (safe_open is not iterable)
                t = b.get_tensor(k)
                if k.startswith("model.language_model."):
                    stem = (
                        "language_model.model."
                        + k[len("model.language_model.") : -len(".weight")]
                    )
                    if stem in quant and k.endswith(".weight"):
                        d = dequant(
                            m.get_tensor(stem + ".weight"),
                            m.get_tensor(stem + ".scales"),
                            m.get_tensor(stem + ".biases"),
                        )
                        assert d.shape == t.shape, (k, d.shape, t.shape)
                        rel = ((d - t.float()).norm() / t.float().norm()).item()
                        worst = max(worst, rel)
                        t = d.to(t.dtype)
                        replaced += 1
                tensors[k] = t.contiguous()
        save_file(tensors, f"{out}/{f}", metadata={"format": "pt"})
print(
    f"replaced {replaced} of {len(quant)} quantized tensors; worst relative error {worst:.4f}"
)
assert replaced == len(quant), "some quantized tensors have no HF match"
