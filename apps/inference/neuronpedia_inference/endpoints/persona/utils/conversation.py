"""ConversationEncoder - Handles chat formatting and token indexing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from transformers import AutoTokenizer


class ConversationEncoder:
    """
    Handles conversation formatting, tokenization, and response index extraction.

    This class knows about model-specific quirks (Qwen vs LLaMA vs Gemma) and
    provides a unified interface for working with chat templates.
    """

    def __init__(self, tokenizer: AutoTokenizer, model_name: Optional[str] = None):
        """
        Initialize the conversation encoder.

        Args:
            tokenizer: HuggingFace tokenizer with chat template support
            model_name: Optional model name for detecting model-specific behavior
        """
        self.tokenizer = tokenizer
        self.model_name = (model_name or getattr(tokenizer, "name_or_path", "")).lower()

    def _is_qwen(self) -> bool:
        """Check if this is a Qwen model."""
        return 'qwen' in self.model_name

    def _is_llama(self) -> bool:
        """Check if this is a Llama model."""
        return 'llama' in self.model_name or 'meta-llama' in self.model_name

    def _is_gemma(self) -> bool:
        """Check if this is a Gemma model."""
        return 'gemma' in self.model_name

    def build_turn_spans(
        self,
        conversation: List[Dict[str, str]],
        **chat_kwargs,
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        """
        Build token spans for each turn in a conversation.

        Args:
            conversation: List of {"role", "content"} dicts
            **chat_kwargs: Additional arguments for apply_chat_template

        Returns:
            Tuple of (full_ids, spans) where:
            - full_ids: tokenized ids of the whole conversation
            - spans: list of dicts with absolute [start, end) token spans for content per turn
        """
        # Tokenize the full conversation first
        full_ids = self.tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=False, **chat_kwargs
        )

        spans = []
        msgs_before = []
        turn_idx = 0

        for msg in conversation:
            role = msg.role if hasattr(msg, 'role') else msg['role']
            text = msg.content if hasattr(msg, 'content') else msg['content']

            if role == "system":
                msgs_before.append(msg)
                continue

            content_ids, start_in_delta = self._content_only_ids_and_offset(
                msgs_before, role, text, **chat_kwargs
            )

            # For Qwen models, use a different approach to find absolute position
            if self._is_qwen():
                # Find where the content appears in the full conversation
                abs_start = self._find_subsequence(full_ids, content_ids)
                if abs_start == -1:
                    # Fallback: skip this span
                    msgs_before.append(msg)
                    continue
                abs_end = abs_start + len(content_ids)
            else:
                # Standard approach for non-Qwen models
                # Calculate absolute start based on the empty message template
                msgs_empty_for_this = msgs_before + [{"role": role, "content": ""}]
                ids_empty_full = self.tokenizer.apply_chat_template(
                    msgs_empty_for_this, tokenize=True, add_generation_prompt=False, **chat_kwargs
                )

                # Find where the content appears in the full sequence
                ids_full_for_this = self.tokenizer.apply_chat_template(
                    msgs_before + [{"role": role, "content": text}], tokenize=True, add_generation_prompt=False, **chat_kwargs
                )

                pref_len = self._longest_common_prefix_len(ids_full_for_this, ids_empty_full)
                abs_start = pref_len + start_in_delta
                abs_end = abs_start + len(content_ids)

            spans.append({
                "turn": turn_idx,
                "role": role,
                "start": abs_start,
                "end": abs_end,   # exclusive
                "n_tokens": len(content_ids),
                "text": text,
            })
            msgs_before.append(msg)
            turn_idx += 1

        return full_ids, spans

    def _content_only_ids_and_offset(
        self,
        messages_before: List[Dict[str, str]],
        role: str,
        content: str,
        **chat_kwargs,
    ) -> Tuple[List[int], int]:
        """
        Extract content-only token IDs and their offset within the turn's delta.

        Returns:
            (content_ids, start_in_delta) where content_ids are ONLY the message content
            tokens as they appear inside the chat template, and start_in_delta is their offset
            within the new suffix added by this message.
        """
        # Dispatch to model-specific implementation if needed
        if self._is_qwen() and role == "assistant":
            return self._content_only_ids_and_offset_qwen(messages_before, role, content, **chat_kwargs)
        else:
            return self._content_only_ids_and_offset_standard(messages_before, role, content, **chat_kwargs)

    def _content_only_ids_and_offset_qwen(
        self,
        messages_before: List[Dict[str, str]],
        role: str,
        content: str,
        **chat_kwargs,
    ) -> Tuple[List[int], int]:
        """Qwen-specific version that handles thinking tokens properly."""
        # For Qwen assistant turns, thinking tokens interfere even when disabled
        if role == "assistant":
            # Find where content appears in the full tokenized conversation
            msgs_full = messages_before + [{"role": role, "content": content}]
            ids_full = self.tokenizer.apply_chat_template(
                msgs_full, tokenize=True, add_generation_prompt=False, **chat_kwargs
            )

            # Find the content tokens in the full sequence
            plain = self.tokenizer(content, add_special_tokens=False).input_ids
            content_start = self._find_subsequence(ids_full, plain)

            if content_start != -1:
                # Calculate offset from the beginning of the conversation
                if messages_before:
                    ids_before = self.tokenizer.apply_chat_template(
                        messages_before, tokenize=True, add_generation_prompt=False, **chat_kwargs
                    )
                    prefix_len = len(ids_before)
                else:
                    prefix_len = 0

                start_in_delta = content_start - prefix_len
                return plain, max(0, start_in_delta)

        # Fall back to standard approach for user turns or if assistant approach fails
        return self._content_only_ids_and_offset_standard(messages_before, role, content, **chat_kwargs)

    def _content_only_ids_and_offset_standard(
        self,
        messages_before: List[Dict[str, str]],
        role: str,
        content: str,
        **chat_kwargs,
    ) -> Tuple[List[int], int]:
        """Standard implementation for most models."""
        msgs_empty = messages_before + [{"role": role, "content": ""}]
        msgs_full  = messages_before + [{"role": role, "content": content}]

        # Handle empty messages_before case
        if messages_before:
            ids_before = self.tokenizer.apply_chat_template(
                messages_before, tokenize=True, add_generation_prompt=False, **chat_kwargs
            )
        else:
            ids_before = []
        ids_empty = self.tokenizer.apply_chat_template(
            msgs_empty, tokenize=True, add_generation_prompt=False, **chat_kwargs
        )
        ids_full  = self.tokenizer.apply_chat_template(
            msgs_full,  tokenize=True, add_generation_prompt=False, **chat_kwargs
        )

        # Suffix introduced by adding this message (template + content)
        pref = self._longest_common_prefix_len(ids_full, ids_empty)
        delta = ids_full[pref:]
        delta = self._strip_trailing_special(delta, set(self.tokenizer.all_special_ids))

        # Try to locate the raw content (with/without a leading space) inside delta
        plain = self.tokenizer(content, add_special_tokens=False).input_ids
        sp    = self.tokenizer(" " + content, add_special_tokens=False).input_ids

        start = self._find_subsequence(delta, plain)
        use = plain
        if start == -1:
            start = self._find_subsequence(delta, sp)
            use = sp if start != -1 else plain

        if start == -1:
            # Fallback: keep the whole delta (may include a fused leading-space token)
            return delta, 0
        else:
            return delta[start:start+len(use)], start

    @staticmethod
    def _longest_common_prefix_len(a: List[int], b: List[int]) -> int:
        """Find the length of the longest common prefix between two sequences."""
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    @staticmethod
    def _strip_trailing_special(ids: List[int], special_ids: set) -> List[int]:
        """Strip trailing special tokens from a sequence."""
        i = len(ids)
        while i > 0 and ids[i-1] in special_ids:
            i -= 1
        return ids[:i]

    @staticmethod
    def _find_subsequence(hay: List[int], needle: List[int]) -> int:
        """Find the starting index of needle in hay, or -1 if not found."""
        if not needle or len(needle) > len(hay):
            return -1
        for i in range(len(hay) - len(needle) + 1):
            if hay[i:i+len(needle)] == needle:
                return i
        return -1

