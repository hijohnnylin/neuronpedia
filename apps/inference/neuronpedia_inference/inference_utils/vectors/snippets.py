"""Conversation-text helpers for readout responses."""

from __future__ import annotations

SNIPPET_MAX_LENGTH = 120


def truncate_content(content: str, max_length: int = SNIPPET_MAX_LENGTH) -> str:
    """Shorten a turn's text to a length worth putting in a response payload."""
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."
