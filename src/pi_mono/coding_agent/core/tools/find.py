"""Find tool."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from typing import Any, Protocol

from pi_mono.agent.types import AgentTool, AgentToolResult
from pi_mono.coding_agent.core.tools.path_utils import path_exists, resolve_to_cwd, to_posix_path
from pi_mono.coding_agent.core.tools.truncate import DEFAULT_MAX_BYTES, formatSize, truncateHead

DEFAULT_LIMIT = 1000


class FindOperations(Protocol):
    async def exists(self, absolute_path: str) -> bool: ...
    async def glob(
        self,
        pattern: str,
        search_cwd: str,
        *,
        ignore: list[str],
        limit: int,
    ) -> list[str]: ...


class DefaultFindOperations:
    async def exists(self, absolute_path: str) -> bool:
        return await path_exists(absolute_path)

    async def glob(
        self,
        pattern: str,
        search_cwd: str,
        *,
        ignore: list[str],
        limit: int,
    ) -> list[str]:
        results: list[str] = []
        for root, dirs, files in os.walk(search_cwd):
            rel_root = os.path.relpath(root, search_cwd)
            if rel_root == ".":
                rel_root = ""
            dirs[:] = [d for d in dirs if d != ".git" and d != "node_modules"]
            for name in files:
                rel_path = to_posix_path(os.path.join(rel_root, name)) if rel_root else name
                if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(name, pattern):
                    results.append(rel_path)
                    if len(results) >= limit:
                        return results
        return results


@dataclass
class FindToolOptions:
    operations: FindOperations | None = None


FIND_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Glob pattern to match files"},
        "path": {"type": "string", "description": "Directory to search in"},
        "limit": {"type": "number", "description": "Maximum number of results"},
    },
    "required": ["pattern"],
}


async def execute_find(
    cwd: str,
    pattern: str,
    path: str | None = None,
    *,
    limit: int | None = None,
    options: FindToolOptions | None = None,
) -> AgentToolResult:
    opts = options or FindToolOptions()
    ops = opts.operations or DefaultFindOperations()
    search_path = resolve_to_cwd(path or ".", cwd)
    if not await ops.exists(search_path):
        raise FileNotFoundError(f"Path not found: {search_path}")
    effective_limit = limit or DEFAULT_LIMIT
    results = await ops.glob(
        pattern,
        search_path,
        ignore=["**/node_modules/**", "**/.git/**"],
        limit=effective_limit,
    )
    if not results:
        return {
            "content": [{"type": "text", "text": "No files found matching pattern"}],
            "details": None,
        }
    raw_output = "\n".join(results)
    truncation = truncateHead(raw_output, {"maxLines": 10**9})
    details: dict[str, Any] = {}
    notices: list[str] = []
    if len(results) >= effective_limit:
        notices.append(f"{effective_limit} results limit reached")
        details["resultLimitReached"] = effective_limit
    if truncation["truncated"]:
        notices.append(f"{formatSize(DEFAULT_MAX_BYTES)} limit reached")
        details["truncation"] = truncation
    output = truncation["content"]
    if notices:
        output += f"\n\n[{'. '.join(notices)}]"
    return {
        "content": [{"type": "text", "text": output}],
        "details": details or None,
    }


def create_find_tool(cwd: str, options: FindToolOptions | None = None) -> AgentTool:
    opts = options or FindToolOptions()

    class FindTool:
        name = "find"
        label = "find"
        description = "Search for files by glob pattern."
        parameters = FIND_PARAMETERS
        executionMode = None

        async def execute(
            self,
            tool_call_id: str,
            params: dict[str, Any],
            signal: Any = None,
            on_update: Any = None,
        ) -> AgentToolResult:
            return await execute_find(
                cwd,
                params["pattern"],
                params.get("path"),
                limit=params.get("limit"),
                options=opts,
            )

    return FindTool()  # type: ignore[return-value]
