"""Filesystem and path helpers shared across Chatspace modules."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def iso_now() -> str:
    """Return the current UTC timestamp in ISO 8601 form."""
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed, returning the resolved Path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def sanitize_component(value: str, *, lowercase: bool = True) -> str:
    """Sanitize a string so it can be safely used as part of a filesystem path."""
    text = value.strip()
    if lowercase:
        text = text.lower()
    if not text:
        return "unnamed"

    safe: list[str] = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        elif char == "/":
            safe.append("__")
        else:
            safe.append("-")

    sanitized = "".join(safe)
    while "--" in sanitized:
        sanitized = sanitized.replace("--", "-")
    sanitized = sanitized.strip("-")
    return sanitized or "unnamed"
