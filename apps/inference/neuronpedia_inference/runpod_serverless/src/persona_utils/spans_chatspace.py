"""SpanMapperChatSpace - Map token spans to activations for ChatSpace models."""

from __future__ import annotations

from typing import Any, List, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .conversation import ConversationEncoder
    from .model_chatspace import ProbingModelChatSpace


class SpanMapperChatSpace:
    """
    Maps token span indices to activations for ChatSpace models.

    Handles:
    - Mapping spans to activation tensors from ChatSpace capture
    - Computing mean activations per turn
    """

    def __init__(self, tokenizer):
        """
        Initialize the span mapper.

        Args:
            tokenizer: HuggingFace tokenizer for span calculations
        """
        self.tokenizer = tokenizer

    async def mean_all_turn_activations_async(
        self,
        probing_model: "ProbingModelChatSpace",
        encoder: "ConversationEncoder",
        conversation: List[Any],
        layer: int = 15,
        **chat_kwargs,
    ) -> torch.Tensor:
        """
        Get mean activations for all turns in a conversation (async).

        Uses ChatSpace's capture_layers to extract hidden states, then
        computes mean activation for each turn based on message boundaries.

        Args:
            probing_model: ProbingModelChatSpace instance
            encoder: ConversationEncoder instance
            conversation: List of conversation turns
            layer: Layer index to extract activations from
            **chat_kwargs: Additional arguments for chat template

        Returns:
            torch.Tensor: Mean activations of shape (num_turns, hidden_size)
        """
        # Convert conversation to dict format if needed
        messages = []
        for turn in conversation:
            if hasattr(turn, "role") and hasattr(turn, "content"):
                messages.append({"role": turn.role, "content": turn.content})
            else:
                messages.append(turn)

        # Capture hidden states
        captures, message_boundaries = await probing_model.capture_conversation_with_boundaries_async(
            messages, layers=layer, **chat_kwargs
        )

        if layer not in captures:
            print(f"WARNING: Layer {layer} not found in captures.")
            return torch.empty(0, probing_model.hidden_size)

        activations = captures[layer]
        
        # Compute mean activation for each turn using message boundaries
        turn_mean_activations = []
        skipped_count = 0
        
        if message_boundaries is None:
            print("WARNING: No message boundaries available, falling back to span-based approach")
            full_ids, spans = encoder.build_turn_spans(conversation, **chat_kwargs)
            
            for span in spans:
                start_idx = span["start"]
                end_idx = span["end"]
                if start_idx < end_idx and end_idx <= activations.shape[0]:
                    turn_activations = activations[start_idx:end_idx, :]
                    mean_activation = turn_activations.mean(dim=0)
                    turn_mean_activations.append(mean_activation)
        else:
            # Check if conversation was truncated
            if message_boundaries:
                last_boundary = message_boundaries[-1]
                total_tokens_needed = last_boundary.end_token
                if total_tokens_needed > activations.shape[0]:
                    print(f"WARNING: Conversation truncated! Needs {total_tokens_needed} tokens but only {activations.shape[0]} captured.")
            
            # Use message boundaries for accurate alignment
            for i, boundary in enumerate(message_boundaries):
                start_idx = boundary.start_token
                end_idx = boundary.end_token
                
                if start_idx < end_idx and end_idx <= activations.shape[0]:
                    turn_activations = activations[start_idx:end_idx, :]
                    mean_activation = turn_activations.mean(dim=0)
                    turn_mean_activations.append(mean_activation)
                elif start_idx < activations.shape[0] < end_idx:
                    # Partial overlap
                    print(f"WARNING: Message {i} ({boundary.role}) partially truncated")
                    turn_activations = activations[start_idx:activations.shape[0], :]
                    mean_activation = turn_activations.mean(dim=0)
                    turn_mean_activations.append(mean_activation)
                else:
                    skipped_count += 1
            
            if skipped_count > 0:
                print(f"WARNING: Skipped {skipped_count} messages due to context length truncation.")

        if not turn_mean_activations:
            return torch.empty(0, probing_model.hidden_size)

        return torch.stack(turn_mean_activations)

