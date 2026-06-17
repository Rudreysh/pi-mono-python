"""Read tool."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from pi_mono.agent.types import AgentTool, AgentToolResult
from pi_mono.coding_agent.core.tools.path_utils import resolve_read_path_async
from pi_mono.coding_agent.core.tools.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    formatSize,
    truncateHead,
)


@dataclass
class ReadToolDetails:
    truncation: TruncationResult | None = None


class ReadOperations(Protocol):
    async def read_file(self, absolute_path: str) -> bytes: ...
    async def access(self, absolute_path: str) -> None: ...


class DefaultReadOperations:
    async def read_file(self, absolute_path: str) -> bytes:
        with open(absolute_path, "rb") as handle:
            return handle.read()

    async def access(self, absolute_path: str) -> None:
        if not os.access(absolute_path, os.R_OK):
            raise PermissionError(f"Cannot read: {absolute_path}")


@dataclass
class ReadToolOptions:
    auto_resize_images: bool = True
    operations: ReadOperations | None = None


READ_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to read (relative or absolute)",
        },
        "offset": {
            "type": "number",
            "description": "Line number to start reading from (1-indexed)",
        },
        "limit": {"type": "number", "description": "Maximum number of lines to read"},
    },
    "required": ["path"],
}


async def execute_read(
    cwd: str,
    path: str,
    offset: int | None = None,
    limit: int | None = None,
    *,
    options: ReadToolOptions | None = None,
) -> AgentToolResult:
    opts = options or ReadToolOptions()
    ops = opts.operations or DefaultReadOperations()
    absolute_path = await resolve_read_path_async(path, cwd)
    await ops.access(absolute_path)
    buffer = await ops.read_file(absolute_path)
    text_content = buffer.decode("utf-8")
    all_lines = text_content.split("\n")
    total_file_lines = len(all_lines)
    start_line = max(0, (offset or 1) - 1)
    start_line_display = start_line + 1
    if start_line >= len(all_lines):
        raise ValueError(f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)")

    details: ReadToolDetails | None = None
    if limit is not None:
        end_line = min(start_line + limit, len(all_lines))
        selected_content = "\n".join(all_lines[start_line:end_line])
        user_limited_lines = end_line - start_line
    else:
        selected_content = "\n".join(all_lines[start_line:])
        user_limited_lines = None

    truncation = truncateHead(selected_content)
    if truncation["firstLineExceedsLimit"]:
        first_line_size = formatSize(len(all_lines[start_line].encode("utf-8")))
        output_text = (
            f"[Line {start_line_display} is {first_line_size}, exceeds "
            f"{formatSize(DEFAULT_MAX_BYTES)} limit.]"
        )
        details = ReadToolDetails(truncation=truncation)
    elif truncation["truncated"]:
        end_line_display = start_line_display + truncation["outputLines"] - 1
        next_offset = end_line_display + 1
        output_text = truncation["content"]
        if truncation["truncatedBy"] == "lines":
            output_text += (
                f"\n\n[Showing lines {start_line_display}-{end_line_display} of "
                f"{total_file_lines}. Use offset={next_offset} to continue.]"
            )
        else:
            output_text += (
                f"\n\n[Showing lines {start_line_display}-{end_line_display} of "
                f"{total_file_lines} ({formatSize(DEFAULT_MAX_BYTES)} limit). "
                f"Use offset={next_offset} to continue.]"
            )
        details = ReadToolDetails(truncation=truncation)
    elif user_limited_lines is not None and start_line + user_limited_lines < len(all_lines):
        remaining = len(all_lines) - (start_line + user_limited_lines)
        next_offset = start_line + user_limited_lines + 1
        output_text = (
            f"{truncation['content']}\n\n[{remaining} more lines in file. "
            f"Use offset={next_offset} to continue.]"
        )
    else:
        output_text = truncation["content"]

    return {
        "content": [{"type": "text", "text": output_text}],
        "details": details.__dict__ if details else None,
    }


def create_read_tool(cwd: str, options: ReadToolOptions | None = None) -> AgentTool:
    opts = options or ReadToolOptions()

    class ReadTool:
        name = "read"
        label = "read"
        description = (
            f"Read the contents of a file. Output is truncated to {DEFAULT_MAX_LINES} lines "
            f"or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first)."
        )
        parameters = READ_PARAMETERS
        executionMode = None

        async def execute(
            self,
            tool_call_id: str,
            params: dict[str, Any],
            signal: Any = None,
            on_update: Any = None,
        ) -> AgentToolResult:
            if signal is not None and getattr(signal, "aborted", False):
                raise RuntimeError("Operation aborted")
            return await execute_read(
                cwd,
                params["path"],
                params.get("offset"),
                params.get("limit"),
                options=opts,
            )

    return ReadTool()  # type: ignore[return-value]
