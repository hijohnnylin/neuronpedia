"""HuggingFace steering model implementation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from steerllm.core.capture import CaptureHandle, MessageBoundary
from steerllm.core.exceptions import BackendError
from steerllm.core.protocols import SyncWrapperMixin
from steerllm.core.specs import LayerSteeringSpec, SteeringSpec

from steerllm.backends.huggingface.hooks import (
    ResidualHook,
    apply_steering_ops,
    create_capture_hook,
    create_steering_hook,
)

logger = logging.getLogger(__name__)


def _get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Extract decoder layers from a HuggingFace model.

    Supports Qwen, Llama, Gemma, and similar architectures.
    """
    # Try common paths
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers

    raise BackendError(
        f"Could not find decoder layers in model {type(model).__name__}. "
        "Supported architectures: Qwen, Llama, Gemma, GPT-NeoX."
    )


class HFSteeringModel(nn.Module, SyncWrapperMixin):
    """HuggingFace steering backend for training and inference.

    Supports trainable steering vectors via PyTorch forward hooks.
    Can be used for:
    - Training steering vectors with gradient descent
    - Inference with trained or fixed steering vectors
    - Activation capture for interpretability

    Parameters
    ----------
    model_name :
        HuggingFace model identifier or path.
    target_layers :
        Layer indices where trainable steering will be added.
        Empty tuple for inference-only mode.
    init_scale :
        Initialization scale for trainable steering vectors.
        Zero for deterministic start.
    **model_kwargs :
        Additional arguments passed to AutoModelForCausalLM.from_pretrained().

    Example
    -------
    Training::

        model = HFSteeringModel("Qwen/Qwen3-0.6B", target_layers=[5])
        optimizer = Adam(model.get_trainable_parameters(), lr=1e-4)

        for batch in dataloader:
            outputs = model(**batch)
            loss = compute_loss(outputs)
            loss.backward()
            optimizer.step()

    Inference with SteeringSpec::

        model = HFSteeringModel("Qwen/Qwen3-0.6B")
        spec = SteeringSpec.simple_add(layer=5, vector=v, scale=1.0)
        texts, handles = await model.generate(prompts, steering_spec=spec)
    """

    def __init__(
        self,
        model_name: str,
        *,
        target_layers: Sequence[int] = (),
        init_scale: float = 0.0,
        **model_kwargs: Any,
    ) -> None:
        super().__init__()

        # Lazy import to allow optional dependency
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise BackendError(
                "HuggingFace backend requires transformers. "
                "Install with: pip install steerllm[huggingface]"
            ) from e

        self._model_name = model_name
        self._target_layers = tuple(target_layers)
        self._init_scale = init_scale

        # Load model and tokenizer
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Expose config for compatibility
        self.config = self.base_model.config

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad_(False)

        # Get decoder layers
        self._layers = _get_decoder_layers(self.base_model)

        # Create trainable steering modules
        self._steering_modules: dict[int, ResidualHook] = {}
        self._hook_handles: list[Any] = []

        for layer_idx in target_layers:
            self._add_trainable_layer(layer_idx)

    @property
    def hidden_size(self) -> int:
        """Model's hidden dimension."""
        return self.config.hidden_size

    @property
    def layer_count(self) -> int:
        """Number of transformer layers."""
        return len(self._layers)

    @property
    def model_name(self) -> str:
        """Model identifier."""
        return self._model_name

    def _add_trainable_layer(self, layer_idx: int) -> None:
        """Add a trainable steering hook at a layer."""
        if layer_idx in self._steering_modules:
            return

        hook = ResidualHook(self.hidden_size, self._init_scale)
        self._steering_modules[layer_idx] = hook

        # Register as submodule for proper parameter tracking
        self.add_module(f"steering_{layer_idx}", hook)

        # Install forward hook
        def make_hook(steering_module: ResidualHook):
            def hook_fn(module, args, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    # Move steering to same device/dtype
                    if (
                        steering_module.vector.device != hidden.device
                        or steering_module.vector.dtype != hidden.dtype
                    ):
                        with torch.no_grad():
                            steering_module.vector.data = steering_module.vector.data.to(
                                device=hidden.device, dtype=hidden.dtype
                            )
                    steered = steering_module(hidden)
                    return (steered,) + output[1:]
                else:
                    if (
                        steering_module.vector.device != output.device
                        or steering_module.vector.dtype != output.dtype
                    ):
                        with torch.no_grad():
                            steering_module.vector.data = steering_module.vector.data.to(
                                device=output.device, dtype=output.dtype
                            )
                    return steering_module(output)
            return hook_fn

        layer = self._layers[layer_idx]
        handle = layer.register_forward_hook(make_hook(hook))
        self._hook_handles.append(handle)

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return all trainable steering parameters."""
        return [m.vector for m in self._steering_modules.values()]

    def set_steering_vector(self, layer: int, vector: torch.Tensor) -> None:
        """Set a steering vector at a layer.

        Creates a trainable module if not already present.
        """
        if layer not in self._steering_modules:
            self._add_trainable_layer(layer)

        with torch.no_grad():
            module = self._steering_modules[layer]
            vec = vector.view(-1).to(
                device=module.vector.device, dtype=module.vector.dtype
            )
            module.vector.data.copy_(vec)

    def get_steering_vector(self, layer: int) -> torch.Tensor | None:
        """Get current steering vector at a layer."""
        module = self._steering_modules.get(layer)
        if module is None:
            return None
        return module.vector.detach().clone()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass with steering applied via hooks."""
        return self.base_model(*args, **kwargs)

    async def generate(
        self,
        prompts: list[str],
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        steering_spec: SteeringSpec | None = None,
        capture_layers: Sequence[int] | None = None,
        **sampling_kwargs: Any,
    ) -> tuple[list[str], list[CaptureHandle] | None]:
        """Generate text with optional steering and capture.

        For steering_spec, temporary hooks are installed during generation
        and removed afterward. This does not affect trainable steering hooks.
        """
        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else None,
            "do_sample": temperature > 0,
            **sampling_kwargs,
        }

        # Remove None values
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        # Install temporary steering hooks if needed
        temp_handles: list[Any] = []
        if steering_spec is not None and not steering_spec.is_empty():
            for layer_idx, layer_spec in steering_spec.layers.items():
                if layer_spec.is_empty():
                    continue
                layer = self._layers[layer_idx]
                hook = create_steering_hook(layer_spec)
                handle = layer.register_forward_hook(hook)
                temp_handles.append(handle)

        # Install capture hooks if needed
        capture_data: dict[int, dict[str, torch.Tensor]] = {}
        if capture_layers:
            for layer_idx in capture_layers:
                captures: dict[str, torch.Tensor] = {}
                capture_data[layer_idx] = captures
                layer = self._layers[layer_idx]
                hook = create_capture_hook(captures)
                handle = layer.register_forward_hook(hook)
                temp_handles.append(handle)

        try:
            # Tokenize
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
            ).to(self.base_model.device)

            # Generate
            with torch.no_grad():
                outputs = self.base_model.generate(
                    **inputs,
                    **gen_kwargs,
                )

            # Decode
            texts = self.tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

        finally:
            # Remove temporary hooks
            for handle in temp_handles:
                handle.remove()

        # Create capture handles if needed
        handles: list[CaptureHandle] | None = None
        if capture_layers:
            handles = []
            for i, prompt in enumerate(prompts):
                # Format captures for this request
                request_captures: dict[int, list[dict[str, Any]]] = {}
                for layer_idx in capture_layers:
                    hidden = capture_data[layer_idx]["hidden"]
                    # For batch, extract this request's data
                    if hidden.dim() == 3:
                        h = hidden[i]
                    else:
                        h = hidden
                    request_captures[layer_idx] = [{"hidden": h}]

                async def make_fetch(caps: dict):
                    async def fetch():
                        return caps
                    return fetch

                handle = CaptureHandle(
                    request_id=f"hf_{i}",
                    layer_indices=tuple(capture_layers),
                    fetch_fn=await make_fetch(request_captures),
                )
                # Pre-populate captures (already fetched)
                handle._captures = request_captures
                handles.append(handle)

        return texts, handles

    async def chat(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        steering_spec: SteeringSpec | None = None,
        capture_layers: Sequence[int] | None = None,
        **sampling_kwargs: Any,
    ) -> tuple[list[str], list[CaptureHandle] | None]:
        """Chat-style generation."""
        # Normalize to list of conversations
        if messages and isinstance(messages[0], dict):
            conversations = [messages]
        else:
            conversations = messages

        # Apply chat template
        prompts = []
        for conv in conversations:
            prompt = self.tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

        return await self.generate(
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            steering_spec=steering_spec,
            capture_layers=capture_layers,
            **sampling_kwargs,
        )

    def save_steering(self, path: str) -> None:
        """Save steering vectors to disk."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save vectors
        vectors = {}
        for layer_idx, module in self._steering_modules.items():
            vectors[str(layer_idx)] = module.vector.detach().cpu()

        torch.save(vectors, save_path / "steering_vectors.pt")

        # Save config
        config = {
            "model_name": self._model_name,
            "target_layers": list(self._steering_modules.keys()),
            "hidden_size": self.hidden_size,
        }
        with open(save_path / "steering_config.json", "w") as f:
            json.dump(config, f, indent=2)

    def load_steering(self, path: str) -> None:
        """Load steering vectors from disk."""
        load_path = Path(path)

        vectors = torch.load(load_path / "steering_vectors.pt", map_location="cpu")

        for layer_str, vector in vectors.items():
            layer_idx = int(layer_str)
            self.set_steering_vector(layer_idx, vector)

    # Proxy methods for compatibility
    def gradient_checkpointing_enable(self, *args: Any, **kwargs: Any) -> Any:
        """Proxy to base model."""
        return self.base_model.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self, *args: Any, **kwargs: Any) -> Any:
        """Proxy to base model."""
        return self.base_model.gradient_checkpointing_disable(*args, **kwargs)
