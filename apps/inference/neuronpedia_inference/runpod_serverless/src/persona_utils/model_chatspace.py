"""ProbingModelChatSpace - Wraps ChatSpace/vLLM model for activation extraction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch


class ProbingModelChatSpace:
    """
    Wraps a ChatSpace VLLMSteerModel with helper methods for generation
    and activation extraction.
    """

    @classmethod
    def from_existing(
        cls,
        model: Any,
        tokenizer: Any = None,
        model_name: Optional[str] = None,
    ) -> "ProbingModelChatSpace":
        """
        Create a ProbingModelChatSpace from an already-loaded VLLMSteerModel.

        Args:
            model: Already-loaded VLLMSteerModel instance
            tokenizer: Ignored (tokenizer comes from the model)
            model_name: Optional model name override

        Returns:
            ProbingModelChatSpace wrapping the provided model
        """
        instance = cls.__new__(cls)
        instance._model = model
        
        # Try to detect model name from various possible attributes
        if model_name:
            instance.model_name = model_name
        elif hasattr(model, "model_name"):
            instance.model_name = model.model_name
        elif hasattr(model, "_model_name"):
            instance.model_name = model._model_name
        elif hasattr(model, "cfg") and hasattr(model.cfg, "model_name"):
            instance.model_name = model.cfg.model_name
        else:
            instance.model_name = "Unknown"
            
        instance._bootstrap_layers = ()
        instance._model_type = None
        return instance

    @property
    def model(self) -> Any:
        """Access the underlying VLLMSteerModel."""
        return self._model

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer from the underlying model."""
        return self._model.tokenizer

    @property
    def hidden_size(self) -> int:
        """Get the hidden size of the model."""
        return self._model.hidden_size

    @property
    def layer_count(self) -> int:
        """Get the number of transformer layers."""
        return self._model.layer_count

    def detect_type(self) -> str:
        """Detect the model family (qwen, llama, gemma, etc)."""
        if self._model_type is not None:
            return self._model_type

        model_name_lower = self.model_name.lower()

        if "qwen" in model_name_lower:
            self._model_type = "qwen"
        elif "llama" in model_name_lower or "meta-llama" in model_name_lower:
            self._model_type = "llama"
        elif "gemma" in model_name_lower:
            self._model_type = "gemma"
        else:
            self._model_type = "unknown"

        return self._model_type

    @property
    def is_qwen(self) -> bool:
        """Check if this is a Qwen model."""
        return self.detect_type() == "qwen"

    @property
    def is_gemma(self) -> bool:
        """Check if this is a Gemma model."""
        return self.detect_type() == "gemma"

    @property
    def is_llama(self) -> bool:
        """Check if this is a Llama model."""
        return self.detect_type() == "llama"

    async def capture_conversation_with_boundaries_async(
        self,
        conversation: List[Dict[str, str]],
        layers: Union[int, List[int]],
        **chat_kwargs,
    ) -> tuple[Dict[int, torch.Tensor], Optional[tuple]]:
        """
        Capture hidden states for a full conversation with message boundaries (async).

        Args:
            conversation: List of {"role", "content"} dicts
            layers: Single layer index or list of layer indices
            **chat_kwargs: Additional arguments for chat template

        Returns:
            Tuple of:
            - Dict mapping layer index to hidden state tensor of shape (seq_len, hidden_size)
            - Message boundaries tuple (or None if not available)
        """
        from vllm import SamplingParams

        # Ensure layers is a list
        if isinstance(layers, int):
            layer_list = [layers]
        else:
            layer_list = list(layers)

        # Use minimal generation to just capture the input activations
        sampling = SamplingParams(max_tokens=1, temperature=0)

        # Convert conversation format if needed
        messages = []
        for turn in conversation:
            if hasattr(turn, "role") and hasattr(turn, "content"):
                messages.append({"role": turn.role, "content": turn.content})
            else:
                messages.append(turn)

        _, handles = await self._model.chat(
            messages,
            sampling_params=sampling,
            capture_layers=layer_list,
            chat_options=chat_kwargs,
        )

        result: Dict[int, torch.Tensor] = {}
        message_boundaries = None
        
        if handles:
            handle = handles[0]
            await handle.fetch()
            captures = handle.captures
            message_boundaries = handle.message_boundaries

            for layer_idx in layer_list:
                if layer_idx in captures:
                    hidden = captures[layer_idx][0]["hidden"]
                    result[layer_idx] = hidden
                else:
                    print(f"WARNING: Layer {layer_idx} not in captures!")

            await handle.close()
        else:
            print("WARNING: No handles returned from chat()!")

        return result, message_boundaries

