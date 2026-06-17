"""Clipboard image detection (stub on unsupported platforms)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClipboardImage:
    bytes: bytes
    mime_type: str


def read_clipboard_image() -> ClipboardImage | None:
    """Return clipboard image bytes if available, else None.

    Full platform support is not implemented in the Python port yet.
    """
    return None
