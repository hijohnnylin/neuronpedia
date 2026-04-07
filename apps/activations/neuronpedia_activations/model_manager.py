import logging
import os
from dataclasses import dataclass
from threading import Lock
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from neuronpedia_activations.model_patterns import get_residual_stream_layers_for_model

logger = logging.getLogger(__name__)

STR_TO_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class LoadedModel:
    model_id: str
    dtype_request: str
    dtype_name: str
    device: str
    model: Any
    tokenizer: PreTrainedTokenizerBase


@dataclass
class RawLayerActivation:
    layer: int
    token_indices: list[int]
    values: list[list[float]]


@dataclass
class RawActivationResult:
    token_strings: list[str]
    token_ids: list[int]
    activations: list[RawLayerActivation]


@dataclass
class RawBatchActivationResult:
    hook_point: str
    type: str
    dtype: str
    device: str
    results: list[RawActivationResult]


class ModelManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._loaded: LoadedModel | None = None
        self.max_batch_size = int(os.environ.get("MAX_BATCH_SIZE", "16"))
        self.max_prompt_tokens = int(os.environ.get("MAX_PROMPT_TOKENS", "2048"))

    def _resolve_device(self, override_device: str | None) -> str:
        if override_device:
            return override_device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _serialize_vector(self, vector: torch.Tensor, dtype_name: str) -> list[float]:
        values = vector.tolist()
        if dtype_name == "bfloat16":
            return [round(float(v), 4) for v in values]
        return [float(v) for v in values]

    def _resolve_dtype_name(self, override_dtype: str | None, model: Any) -> str:
        if override_dtype is not None:
            if override_dtype not in STR_TO_DTYPE:
                raise ValueError(
                    f"Unsupported dtype '{override_dtype}'. "
                    "Allowed values: float32, float16, bfloat16."
                )
            return override_dtype

        model_dtype = model.dtype
        if model_dtype == torch.float16:
            return "float16"
        if model_dtype == torch.bfloat16:
            return "bfloat16"
        return "float32"

    def _clear_model(self) -> None:
        if self._loaded is None:
            return
        self._loaded = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _load_model(
        self, model_id: str, dtype_override: str | None, device_override: str | None
    ) -> LoadedModel:
        device = self._resolve_device(device_override)
        torch_dtype = STR_TO_DTYPE[dtype_override] if dtype_override else None

        logger.info(
            "Loading model '%s' with dtype=%s on device=%s",
            model_id,
            dtype_override if dtype_override else "default",
            device,
        )
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if torch_dtype is not None:
            model_kwargs["dtype"] = torch_dtype

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        resolved_dtype = self._resolve_dtype_name(dtype_override, model)
        first_param = next(model.parameters(), None)
        resolved_device = first_param.device.type if first_param is not None else device

        if dtype_override is not None and resolved_dtype != dtype_override:
            raise RuntimeError(
                f"Requested dtype '{dtype_override}' but resolved dtype is '{resolved_dtype}' "
                f"for model '{model_id}'."
            )
        if device_override is not None and resolved_device != device_override:
            raise RuntimeError(
                f"Requested device '{device_override}' but resolved device is '{resolved_device}' "
                f"for model '{model_id}'."
            )

        logger.info(
            "Loaded model '%s' on device=%s (dtype=%s, torch_dtype=%s)",
            model_id,
            resolved_device,
            resolved_dtype,
            str(model.dtype),
        )

        return LoadedModel(
            model_id=model_id,
            dtype_request=dtype_override if dtype_override else "default",
            dtype_name=resolved_dtype,
            device=resolved_device,
            model=model,
            tokenizer=tokenizer,
        )

    def _get_or_load_model(
        self, model_id: str, dtype_override: str | None, device_override: str | None
    ) -> LoadedModel:
        with self._lock:
            requested_device = self._resolve_device(device_override)
            requested_dtype = dtype_override if dtype_override else "default"

            if (
                self._loaded is not None
                and self._loaded.model_id == model_id
                and self._loaded.device == requested_device
                and self._loaded.dtype_request == requested_dtype
            ):
                return self._loaded

            self._clear_model()
            self._loaded = self._load_model(model_id, dtype_override, device_override)
            return self._loaded

    def preload_model(
        self, model_id: str, dtype_override: str | None, device_override: str | None
    ) -> LoadedModel:
        loaded = self._get_or_load_model(model_id, dtype_override, device_override)
        # Validate that this model has a known residual stream mapping so startup
        # fails fast instead of waiting for the first request.
        _ = get_residual_stream_layers_for_model(model_id, loaded.model)
        return loaded

    def capture_raw_residual_stream_batch(
        self,
        *,
        model_id: str,
        prompts: list[str],
        hook_point: str,
        extraction_type: str,
        dtype_override: str | None,
        device_override: str | None,
    ) -> RawBatchActivationResult:
        if hook_point != "residual_stream":
            raise ValueError(
                "Unsupported hook_point. Only 'residual_stream' is supported right now."
            )
        if extraction_type != "final_output_token":
            raise ValueError(
                "Unsupported type. Only 'final_output_token' is supported right now."
            )
        if len(prompts) == 0:
            raise ValueError("prompts must contain at least one string.")
        if len(prompts) > self.max_batch_size:
            raise ValueError(
                f"Batch too large: got {len(prompts)} prompts, "
                f"max is {self.max_batch_size}."
            )

        loaded = self._get_or_load_model(model_id, dtype_override, device_override)
        model = loaded.model
        tokenizer = loaded.tokenizer

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation=False,
        )
        input_ids = encoded["input_ids"].to(loaded.device)
        attention_mask = encoded["attention_mask"].to(loaded.device)
        prompt_lengths = attention_mask.sum(dim=1)
        longest_prompt = int(prompt_lengths.max().item())
        if longest_prompt > self.max_prompt_tokens:
            raise ValueError(
                f"Prompt too long: longest prompt has {longest_prompt} tokens, "
                f"max is {self.max_prompt_tokens}."
            )

        final_token_indices = attention_mask.sum(dim=1) - 1
        batch_size = int(input_ids.shape[0])
        batch_positions = torch.arange(batch_size, device=loaded.device)

        layer_modules = get_residual_stream_layers_for_model(model_id, model)
        captured: list[torch.Tensor | None] = [None] * len(layer_modules)
        handles: list[Any] = []

        def _build_hook(layer_idx: int):
            def _hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                tensor = output[0] if isinstance(output, tuple | list) else output
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(
                        f"Unexpected non-tensor layer output at layer {layer_idx}."
                    )
                # Store only each prompt's final-token residual state to keep
                # activation capture memory linear in batch size and hidden dim.
                captured[layer_idx] = tensor[
                    batch_positions, final_token_indices, :
                ].detach()

            return _hook

        for idx, layer_module in enumerate(layer_modules):
            handles.append(layer_module.register_forward_hook(_build_hook(idx)))

        try:
            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
        finally:
            for handle in handles:
                handle.remove()

        per_prompt_activations: list[list[RawLayerActivation]] = [[] for _ in range(batch_size)]
        for layer_idx, layer_tensor in enumerate(captured):
            if layer_tensor is None:
                raise RuntimeError(f"Missing captured activation for layer {layer_idx}.")
            layer_tensor_cpu = layer_tensor.to("cpu")
            for prompt_idx in range(batch_size):
                token_index = int(final_token_indices[prompt_idx].item())
                per_prompt_activations[prompt_idx].append(
                    RawLayerActivation(
                        layer=layer_idx,
                        token_indices=[token_index],
                        values=[
                            self._serialize_vector(
                                layer_tensor_cpu[prompt_idx], loaded.dtype_name
                            )
                        ],
                    )
                )

        results: list[RawActivationResult] = []
        for prompt_idx in range(batch_size):
            prompt_len = int(attention_mask[prompt_idx].sum().item())
            token_id_list = input_ids[prompt_idx, :prompt_len].detach().cpu().tolist()
            token_strings = tokenizer.convert_ids_to_tokens(token_id_list)
            results.append(
                RawActivationResult(
                    token_strings=token_strings,
                    token_ids=token_id_list,
                    activations=per_prompt_activations[prompt_idx],
                )
            )

        return RawBatchActivationResult(
            hook_point=hook_point,
            type=extraction_type,
            dtype=loaded.dtype_name,
            device=loaded.device,
            results=results,
        )
