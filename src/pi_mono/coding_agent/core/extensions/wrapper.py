"""Wrap extension ToolDefinitions as AgentTools for the agent runtime."""

from __future__ import annotations

from typing import Any, Callable

from pi_mono.agent.types import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi_mono.coding_agent.core.extensions.runner import ExtensionRunner
from pi_mono.coding_agent.core.extensions.types import RegisteredTool, ToolDefinition
from pi_mono.utils.abort_signals import AbortSignal


def wrap_tool_definition(
    definition: ToolDefinition,
    ctx_factory: Callable[[], Any] | None = None,
) -> AgentTool:
    class _WrappedTool:
        name = definition.name
        label = definition.label
        description = definition.description
        parameters = definition.parameters
        executionMode = None

        async def execute(
            self,
            tool_call_id: str,
            params: Any,
            signal: AbortSignal | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            ctx = ctx_factory() if ctx_factory else None
            return await definition.execute(tool_call_id, params, signal, on_update, ctx)

    return _WrappedTool()  # type: ignore[return-value]


def wrap_registered_tool(registered_tool: RegisteredTool, runner: ExtensionRunner) -> AgentTool:
    return wrap_tool_definition(registered_tool.definition, runner.create_context)


def wrap_registered_tools(
    registered_tools: list[RegisteredTool], runner: ExtensionRunner
) -> list[AgentTool]:
    return [wrap_registered_tool(tool, runner) for tool in registered_tools]
