"""Bash tool."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from pi_mono.agent.types import AgentTool, AgentToolResult
from pi_mono.coding_agent.core.tools.truncate import DEFAULT_MAX_BYTES, truncateTail


class BashOperations(Protocol):
    async def exec(
        self,
        command: str,
        cwd: str,
        *,
        on_data: Callable[[bytes], None],
        signal: Any = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, int | None]: ...


class LocalBashOperations:
    async def exec(
        self,
        command: str,
        cwd: str,
        *,
        on_data: Callable[[bytes], None],
        signal: Any = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, int | None]:
        if not os.path.isdir(cwd):
            raise RuntimeError(
                f"Working directory does not exist: {cwd}\nCannot execute bash commands."
            )
        if signal is not None and getattr(signal, "aborted", False):
            raise RuntimeError("aborted")

        shell = os.environ.get("SHELL") or ("/bin/zsh" if sys.platform != "win32" else "cmd.exe")
        if sys.platform == "win32":
            args = ["/c", command]
        else:
            args = ["-c", command]

        process = await asyncio.create_subprocess_exec(
            shell,
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env or os.environ.copy(),
        )

        try:
            if process.stdout is None:
                raise RuntimeError("Failed to capture command output")
            while True:
                if signal is not None and getattr(signal, "aborted", False):
                    process.kill()
                    raise RuntimeError("aborted")
                chunk = await asyncio.wait_for(process.stdout.read(4096), timeout=timeout or None)
                if not chunk:
                    break
                on_data(chunk)
            exit_code = await process.wait()
            if signal is not None and getattr(signal, "aborted", False):
                raise RuntimeError("aborted")
            return {"exitCode": exit_code}
        except asyncio.TimeoutError:
            process.kill()
            raise RuntimeError(f"timeout:{timeout}") from None


@dataclass
class BashToolOptions:
    operations: BashOperations | None = None
    command_prefix: str | None = None


BASH_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Bash command to execute"},
        "timeout": {"type": "number", "description": "Timeout in seconds (optional)"},
    },
    "required": ["command"],
}


async def execute_bash(
    cwd: str,
    command: str,
    *,
    timeout: float | None = None,
    options: BashToolOptions | None = None,
    signal: Any = None,
) -> AgentToolResult:
    opts = options or BashToolOptions()
    ops = opts.operations or LocalBashOperations()
    resolved_command = f"{opts.command_prefix}\n{command}" if opts.command_prefix else command
    chunks: list[bytes] = []

    def on_data(data: bytes) -> None:
        chunks.append(data)

    try:
        result = await ops.exec(
            resolved_command,
            cwd,
            on_data=on_data,
            signal=signal,
            timeout=timeout,
        )
    except RuntimeError as error:
        output = b"".join(chunks).decode("utf-8", errors="replace")
        truncation = truncateTail(output)
        text = truncation["content"] or ""
        if str(error) == "aborted":
            raise RuntimeError(
                f"{text}\n\nCommand aborted" if text else "Command aborted"
            ) from error
        if str(error).startswith("timeout:"):
            timeout_secs = str(error).split(":", 1)[1]
            raise RuntimeError(
                f"{text}\n\nCommand timed out after {timeout_secs} seconds"
                if text
                else f"Command timed out after {timeout_secs} seconds"
            ) from error
        raise

    output = b"".join(chunks).decode("utf-8", errors="replace")
    truncation = truncateTail(output)
    text = truncation["content"] or "(no output)"
    details = {"truncation": truncation} if truncation["truncated"] else None
    exit_code = result.get("exitCode")
    if exit_code not in (0, None):
        raise RuntimeError(f"{text}\n\nCommand exited with code {exit_code}")
    return {"content": [{"type": "text", "text": text}], "details": details}


def create_bash_tool(cwd: str, options: BashToolOptions | None = None) -> AgentTool:
    opts = options or BashToolOptions()

    class BashTool:
        name = "bash"
        label = "bash"
        description = (
            f"Execute a bash command in the current working directory. "
            f"Output is truncated to last lines or {DEFAULT_MAX_BYTES // 1024}KB."
        )
        parameters = BASH_PARAMETERS
        executionMode = None

        async def execute(
            self,
            tool_call_id: str,
            params: dict[str, Any],
            signal: Any = None,
            on_update: Any = None,
        ) -> AgentToolResult:
            return await execute_bash(
                cwd,
                params["command"],
                timeout=params.get("timeout"),
                options=opts,
                signal=signal,
            )

    return BashTool()  # type: ignore[return-value]
