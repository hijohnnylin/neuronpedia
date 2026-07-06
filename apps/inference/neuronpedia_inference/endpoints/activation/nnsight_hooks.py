from __future__ import annotations

from typing import Any

import nnsight


def is_nnsight_mlp_input_hook(hook_name: str) -> bool:
    return "hook_mlp_in" in hook_name or "ln2.hook_normalized" in hook_name


def save_nnsight_hook_outputs(model: Any, hook_name: str, layer_num: int):
    if "resid_post" in hook_name:
        return nnsight.save(model.layers_output[layer_num])
    if "resid_pre" in hook_name:
        if layer_num == 0:
            return nnsight.save(model.embeddings_output)
        return nnsight.save(model.layers_output[layer_num - 1])
    if is_nnsight_mlp_input_hook(hook_name):
        return nnsight.save(model.mlps_input[layer_num])
    raise ValueError(f"Unsupported hook name for nnsight: {hook_name}")
