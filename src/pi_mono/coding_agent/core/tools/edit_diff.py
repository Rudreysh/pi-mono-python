"""Shared diff computation utilities for the edit tool."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import TypedDict


class Edit(TypedDict):
    oldText: str
    newText: str


@dataclass
class AppliedEditsResult:
    base_content: str
    new_content: str


def detect_line_ending(content: str) -> str:
    crlf_idx = content.find("\r\n")
    lf_idx = content.find("\n")
    if lf_idx == -1:
        return "\n"
    if crlf_idx == -1:
        return "\n"
    return "\r\n" if crlf_idx < lf_idx else "\n"


def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


def strip_bom(content: str) -> str:
    return content.removeprefix("\ufeff")


def apply_edits_to_normalized_content(content: str, edits: list[Edit]) -> AppliedEditsResult:
    working = content
    for edit in edits:
        old_text = edit["oldText"]
        new_text = edit["newText"]
        index = working.find(old_text)
        if index == -1:
            preview = old_text[:80].replace("\n", "\\n")
            raise ValueError(f"Could not find oldText in file: {preview!r}")
        count = working.count(old_text)
        if count > 1:
            raise ValueError("oldText must be unique in the file")
        working = working[:index] + new_text + working[index + len(old_text) :]
    return AppliedEditsResult(base_content=content, new_content=working)


def generate_diff_string(old_content: str, new_content: str, path: str) -> str:
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile=path, tofile=path)
    return "".join(diff)
