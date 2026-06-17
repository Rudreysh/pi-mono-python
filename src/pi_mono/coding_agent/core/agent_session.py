"""AgentSession - core abstraction for agent lifecycle and session management."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, TypedDict, cast

from pi_mono.agent.agent import Agent
from pi_mono.agent.harness.compaction.compaction import compact, prepare_compaction
from pi_mono.agent.harness.messages import create_user_message
from pi_mono.agent.types import AgentEvent, AgentMessage, AgentTool, ThinkingLevel
from pi_mono.ai.models import clamp_thinking_level, get_supported_thinking_levels, models_are_equal
from pi_mono.ai.types import AssistantMessage, ImageContent, Model
from pi_mono.ai.utils.overflow import is_context_overflow
from pi_mono.core.defaults import DEFAULT_THINKING_LEVEL
from pi_mono.coding_agent.core.bash_executor import (
    BashExecutorOptions,
    BashResult,
    execute_bash_with_operations,
)
from pi_mono.coding_agent.core.tools.bash import LocalBashOperations
from pi_mono.core.event_bus import EventBusController, create_event_bus
from pi_mono.core.model_registry import ModelRegistry
from pi_mono.core.session_manager import SessionManager
from pi_mono.core.settings_manager import SettingsManager
from pi_mono.coding_agent.core.auth_guidance import (
    format_no_api_key_found_message,
    format_no_model_selected_message,
)
from pi_mono.coding_agent.core.resource_loader import ResourceLoader
from pi_mono.coding_agent.core.system_prompt import build_system_prompt
from pi_mono.coding_agent.core.extensions import (
    ExtensionActions,
    ExtensionCommandContextActions,
    ExtensionContextActions,
    ExtensionRunner,
    LoadExtensionsResult,
    discover_and_load_extensions,
)
from pi_mono.coding_agent.core.extensions.loader import create_extension_runtime
from pi_mono.coding_agent.core.extensions.types import ExtensionError
from pi_mono.coding_agent.core.extensions.wrapper import wrap_registered_tools
from pi_mono.coding_agent.core.tools import ToolName, create_tool
from pi_mono.config import get_agent_dir  # used by AgentSessionRuntime
from pi_mono.utils.abort_signals import AbortController, AbortSignal

CompactionReason = Literal["manual", "threshold", "overflow"]
SteeringMode = Literal["all", "one-at-a-time"]
FollowUpMode = Literal["all", "one-at-a-time"]


class ModelCycleResult(TypedDict):
    model: Model[Any]
    thinkingLevel: ThinkingLevel
    isScoped: bool


class SessionStats(TypedDict, total=False):
    sessionFile: str | None
    sessionId: str
    userMessages: int
    assistantMessages: int
    toolCalls: int
    toolResults: int
    totalMessages: int
    tokens: dict[str, int]
    cost: float


_DEFAULT_TOOL_SNIPPETS: dict[str, str] = {
    "read": "Read file contents",
    "bash": "Execute shell commands",
    "edit": "Edit files with search/replace",
    "write": "Write or overwrite files",
    "grep": "Search file contents",
    "find": "Find files by pattern",
    "ls": "List directory contents",
}


class AgentSessionEventQueueUpdate(TypedDict):
    type: Literal["queue_update"]
    steering: list[str]
    followUp: list[str]


class AgentSessionEventSessionInfoChanged(TypedDict):
    type: Literal["session_info_changed"]
    name: str | None


class AgentSessionEventThinkingLevelChanged(TypedDict):
    type: Literal["thinking_level_changed"]
    level: ThinkingLevel


class AgentSessionEventCompactionStart(TypedDict):
    type: Literal["compaction_start"]
    reason: CompactionReason


class AgentSessionEventCompactionEnd(TypedDict):
    type: Literal["compaction_end"]
    reason: CompactionReason
    result: Any | None
    aborted: bool
    willRetry: bool
    errorMessage: str | None


class AgentSessionEventAutoRetryStart(TypedDict):
    type: Literal["auto_retry_start"]
    attempt: int
    maxAttempts: int
    delayMs: int
    errorMessage: str


class AgentSessionEventAutoRetryEnd(TypedDict):
    type: Literal["auto_retry_end"]
    success: bool
    attempt: int
    finalError: str | None


AgentSessionEvent = (
    AgentEvent
    | AgentSessionEventQueueUpdate
    | AgentSessionEventSessionInfoChanged
    | AgentSessionEventThinkingLevelChanged
    | AgentSessionEventCompactionStart
    | AgentSessionEventCompactionEnd
    | AgentSessionEventAutoRetryStart
    | AgentSessionEventAutoRetryEnd
)

AgentSessionEventListener = Callable[[AgentSessionEvent], None]


@dataclass
class PromptOptions:
    expand_templates: bool = True
    images: list[ImageContent] | None = None
    streaming_behavior: Literal["steer", "followUp"] | None = None


@dataclass
class AgentSessionConfig:
    agent: Agent
    session_manager: SessionManager
    settings_manager: SettingsManager
    cwd: str
    model_registry: ModelRegistry
    resource_loader: ResourceLoader
    scoped_models: list[dict[str, Any]] | None = None
    initial_active_tool_names: list[str] | None = None
    allowed_tool_names: list[str] | None = None
    excluded_tool_names: list[str] | None = None
    system_prompt: str | None = None
    extension_paths: list[str] | None = None
    no_extensions: bool = False


def _default_active_tools() -> list[ToolName]:
    return ["read", "bash", "edit", "write"]


def _resolve_tools(
    cwd: str,
    *,
    initial_active_tool_names: list[str] | None = None,
    allowed_tool_names: list[str] | None = None,
    excluded_tool_names: list[str] | None = None,
) -> list[AgentTool]:
    active = initial_active_tool_names or _default_active_tools()
    if allowed_tool_names is not None:
        active = [name for name in active if name in allowed_tool_names]
    if excluded_tool_names:
        active = [name for name in active if name not in excluded_tool_names]
    return [create_tool(name, cwd) for name in active]  # type: ignore[arg-type]


class AgentSession:
    """Shared session abstraction for interactive, print, and rpc modes."""

    def __init__(self, config: AgentSessionConfig) -> None:
        self._config = config
        self._event_bus: EventBusController = create_event_bus()
        self._listeners: list[AgentSessionEventListener] = []
        self._disposed = False
        self._agent_listener: Callable[[], None] | None = None
        self._scoped_models = list(config.scoped_models or [])
        self._resource_loader = config.resource_loader
        self._steering_messages: list[str] = []
        self._follow_up_messages: list[str] = []
        self._compaction_abort_controller: AbortController | None = None
        self._bash_abort_controller: AbortController | None = None
        self.agent = config.agent
        self.session_manager = config.session_manager
        self.settings_manager = config.settings_manager
        self.cwd = config.cwd
        self.model_registry = config.model_registry
        self._extension_paths = list(config.extension_paths or [])
        self._no_extensions = config.no_extensions
        self._extension_load_result: LoadExtensionsResult | None = None
        self._extension_runner: ExtensionRunner | None = None
        self._extension_error_unsubscribe: Callable[[], None] | None = None
        self._extension_mode: str = "print"
        self._session_start_reason: str = "startup"
        self._last_assistant_message: AssistantMessage | None = None
        self._retry_attempt = 0
        self._retry_abort_controller: AbortController | None = None

        tools = _resolve_tools(
            config.cwd,
            initial_active_tool_names=config.initial_active_tool_names,
            allowed_tool_names=config.allowed_tool_names,
            excluded_tool_names=config.excluded_tool_names,
        )
        self.agent.state.tools = tools
        self._refresh_system_prompt(custom_prompt=config.system_prompt)

        self._agent_listener = self.agent.subscribe(self._handle_agent_event_with_signal)

    @property
    def state(self) -> Any:
        return self.agent.state

    @property
    def model(self) -> Model[Any] | None:
        return self.agent.state.model

    @property
    def thinking_level(self) -> ThinkingLevel:
        return self.agent.state.thinkingLevel  # type: ignore[return-value]

    @property
    def is_streaming(self) -> bool:
        return self.agent.state.isStreaming

    @property
    def system_prompt(self) -> str:
        return self.agent.state.systemPrompt

    @property
    def messages(self) -> list[AgentMessage]:
        return self.agent.state.messages

    @property
    def resource_loader(self) -> ResourceLoader:
        return self._resource_loader

    @property
    def session_file(self) -> str | None:
        path = getattr(self.session_manager, "session_file", None)
        return str(path) if path else None

    @property
    def session_id(self) -> str:
        return self.session_manager.get_session_id()

    @property
    def steering_mode(self) -> str:
        return self.agent.steeringMode

    @property
    def follow_up_mode(self) -> str:
        return self.agent.followUpMode

    @property
    def pending_message_count(self) -> int:
        return len(self._steering_messages) + len(self._follow_up_messages)

    @property
    def is_compacting(self) -> bool:
        return self._compaction_abort_controller is not None

    @property
    def session_name(self) -> str | None:
        return self.session_manager.get_session_name()

    @property
    def auto_compaction_enabled(self) -> bool:
        return self.settings_manager.get_compaction_enabled()

    @property
    def is_retrying(self) -> bool:
        return self._retry_abort_controller is not None

    @property
    def retry_attempt(self) -> int:
        return self._retry_attempt

    def subscribe(self, listener: AgentSessionEventListener) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def _emit(self, event: AgentSessionEvent) -> None:
        for listener in list(self._listeners):
            listener(event)
        self._event_bus.emit("session", event)

    def _emit_queue_update(self) -> None:
        self._emit(
            {
                "type": "queue_update",
                "steering": list(self._steering_messages),
                "followUp": list(self._follow_up_messages),
            }
        )

    def _handle_agent_event_with_signal(self, event: AgentEvent, _signal: AbortSignal) -> None:
        self._handle_agent_event(event)

    def _handle_agent_event(self, event: AgentEvent) -> None:
        event_type = event.get("type")
        if event_type == "agent_end":
            payload = cast(AgentEvent, dict(event))
            payload["willRetry"] = self._will_retry_after_agent_end(event)  # type: ignore[typeddict-unknown-key]
            self._emit(payload)  # type: ignore[arg-type]
        else:
            self._emit(event)

        if self._extension_runner is not None:
            asyncio.create_task(self._emit_extension_event(event))
        if event_type == "message_end":
            message = event.get("message")
            if message and message.get("role") == "assistant":
                self._last_assistant_message = cast(AssistantMessage, message)
                stop_reason = message.get("stopReason")
                if stop_reason != "error" and self._retry_attempt > 0:
                    self._emit(
                        {
                            "type": "auto_retry_end",
                            "success": True,
                            "attempt": self._retry_attempt,
                            "finalError": None,
                        }
                    )
                    self._retry_attempt = 0
            if message and message.get("role") in ("user", "assistant"):
                self._persist_message(message)
                self._remove_delivered_queue_message(message)

    def _remove_delivered_queue_message(self, message: AgentMessage) -> None:
        content = message.get("content")
        if not isinstance(content, list):
            return
        text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
        message_text = "".join(text_parts)
        if message_text in self._steering_messages:
            self._steering_messages.remove(message_text)
            self._emit_queue_update()
        elif message_text in self._follow_up_messages:
            self._follow_up_messages.remove(message_text)
            self._emit_queue_update()

    def _persist_message(self, message: AgentMessage) -> None:
        append = getattr(self.session_manager, "append_message", None)
        if callable(append):
            append(message)

    def _refresh_system_prompt(self, *, custom_prompt: str | None = None) -> None:
        loader = self._resource_loader
        agents_files = loader.get_agents_files().get("agentsFiles", [])
        skills = loader.get_skills().get("skills", [])
        append_parts = loader.get_append_system_prompt()
        selected_tool_names = [tool.name for tool in self.agent.state.tools]
        tool_snippets = {
            name: snippet
            for name, snippet in _DEFAULT_TOOL_SNIPPETS.items()
            if name in selected_tool_names
        }
        prompt = build_system_prompt(
            custom_prompt=custom_prompt or loader.get_system_prompt(),
            selected_tools=selected_tool_names,
            tool_snippets=tool_snippets,
            append_system_prompt="\n\n".join(append_parts) if append_parts else None,
            cwd=self.cwd,
            context_files=agents_files,
            skills=skills,
        )
        self.agent.state.systemPrompt = prompt

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        effective_level = (
            clamp_thinking_level(self.model, level) if self.model else "off"  # type: ignore[assignment]
        )
        previous_level = self.agent.state.thinkingLevel
        self.agent.state.thinkingLevel = effective_level
        if effective_level != previous_level:
            self.session_manager.append_thinking_level_change(effective_level)
            self.settings_manager.set_default_thinking_level(effective_level)
            self._emit({"type": "thinking_level_changed", "level": effective_level})

    async def set_model(self, model: Model[Any]) -> None:
        if not self.model_registry.has_configured_auth(model):
            raise RuntimeError(f"No API key for {model['provider']}/{model['id']}")
        thinking_level = self.thinking_level
        self.agent.state.model = model
        self.session_manager.append_model_change(model["provider"], model["id"])
        self.settings_manager.set_default_model_and_provider(model["provider"], model["id"])
        self.set_thinking_level(thinking_level)

    async def cycle_model(
        self, direction: Literal["forward", "backward"] = "forward"
    ) -> ModelCycleResult | None:
        if self._scoped_models:
            return await self._cycle_scoped_model(direction)
        return await self._cycle_available_model(direction)

    async def _cycle_scoped_model(
        self, direction: Literal["forward", "backward"]
    ) -> ModelCycleResult | None:
        scoped_models = [
            item
            for item in self._scoped_models
            if self.model_registry.has_configured_auth(item["model"])
        ]
        if len(scoped_models) <= 1:
            return None

        current_model = self.model
        current_index = next(
            (
                index
                for index, item in enumerate(scoped_models)
                if models_are_equal(item["model"], current_model)
            ),
            0,
        )
        length = len(scoped_models)
        next_index = (
            (current_index + 1) % length
            if direction == "forward"
            else (current_index - 1 + length) % length
        )
        next_item = scoped_models[next_index]
        thinking_level = self._get_thinking_level_for_model_switch(next_item.get("thinkingLevel"))
        self.agent.state.model = next_item["model"]
        self.session_manager.append_model_change(
            next_item["model"]["provider"], next_item["model"]["id"]
        )
        self.settings_manager.set_default_model_and_provider(
            next_item["model"]["provider"], next_item["model"]["id"]
        )
        self.set_thinking_level(thinking_level)
        return {
            "model": next_item["model"],
            "thinkingLevel": self.thinking_level,
            "isScoped": True,
        }

    async def _cycle_available_model(
        self, direction: Literal["forward", "backward"]
    ) -> ModelCycleResult | None:
        available_models = self.model_registry.get_available()
        if len(available_models) <= 1:
            return None

        current_model = self.model
        current_index = next(
            (
                index
                for index, model in enumerate(available_models)
                if models_are_equal(model, current_model)
            ),
            0,
        )
        length = len(available_models)
        next_index = (
            (current_index + 1) % length
            if direction == "forward"
            else (current_index - 1 + length) % length
        )
        next_model = available_models[next_index]
        thinking_level = self._get_thinking_level_for_model_switch()
        self.agent.state.model = next_model
        self.session_manager.append_model_change(next_model["provider"], next_model["id"])
        self.settings_manager.set_default_model_and_provider(
            next_model["provider"], next_model["id"]
        )
        self.set_thinking_level(thinking_level)
        return {
            "model": next_model,
            "thinkingLevel": self.thinking_level,
            "isScoped": False,
        }

    def supports_thinking(self) -> bool:
        return bool(self.model and self.model.get("reasoning"))

    def get_available_thinking_levels(self) -> list[ThinkingLevel]:
        if not self.model:
            return ["off", "minimal", "low", "medium", "high", "xhigh"]  # type: ignore[list-item]
        return get_supported_thinking_levels(self.model)  # type: ignore[return-value]

    def _get_thinking_level_for_model_switch(
        self, explicit_level: ThinkingLevel | None = None
    ) -> ThinkingLevel:
        if explicit_level is not None:
            return explicit_level
        if not self.supports_thinking():
            return self.settings_manager.get_default_thinking_level() or DEFAULT_THINKING_LEVEL  # type: ignore[return-value]
        return self.thinking_level

    def cycle_thinking_level(self) -> ThinkingLevel | None:
        if not self.supports_thinking():
            return None
        levels = self.get_available_thinking_levels()
        current_index = levels.index(self.thinking_level) if self.thinking_level in levels else 0
        next_index = (current_index + 1) % len(levels)
        next_level = levels[next_index]
        self.set_thinking_level(next_level)
        return next_level

    def set_steering_mode(self, mode: SteeringMode) -> None:
        self.agent.steeringMode = mode
        self.settings_manager.set_steering_mode(mode)

    def set_follow_up_mode(self, mode: FollowUpMode) -> None:
        self.agent.followUpMode = mode
        self.settings_manager.set_follow_up_mode(mode)

    def set_auto_compaction(self, enabled: bool) -> None:
        self.settings_manager.set_compaction_enabled(enabled)

    def set_auto_retry(self, enabled: bool) -> None:
        self.settings_manager.set_retry_enabled(enabled)

    def get_user_messages_for_forking(self) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        for entry in self.session_manager.get_entries():
            if entry.get("type") != "message":
                continue
            message = entry.get("message", {})
            if message.get("role") != "user":
                continue
            text = _extract_user_message_text(message.get("content"))
            if text:
                result.append({"entryId": str(entry["id"]), "text": text})
        return result

    def get_session_stats(self) -> SessionStats:
        user_messages = sum(1 for message in self.messages if message.get("role") == "user")
        assistant_messages = sum(
            1 for message in self.messages if message.get("role") == "assistant"
        )
        tool_results = sum(1 for message in self.messages if message.get("role") == "toolResult")
        tool_calls = 0
        total_input = 0
        total_output = 0
        total_cache_read = 0
        total_cache_write = 0
        total_cost = 0.0

        for message in self.messages:
            if message.get("role") != "assistant":
                continue
            content = message.get("content", [])
            if isinstance(content, list):
                tool_calls += sum(1 for part in content if part.get("type") == "toolCall")
            usage = message.get("usage") or {}
            total_input += int(usage.get("input", 0) or 0)
            total_output += int(usage.get("output", 0) or 0)
            total_cache_read += int(usage.get("cacheRead", 0) or 0)
            total_cache_write += int(usage.get("cacheWrite", 0) or 0)
            cost = usage.get("cost") or {}
            total_cost += float(cost.get("total", 0) or 0)

        return {
            "sessionFile": self.session_file,
            "sessionId": self.session_id,
            "userMessages": user_messages,
            "assistantMessages": assistant_messages,
            "toolCalls": tool_calls,
            "toolResults": tool_results,
            "totalMessages": len(self.messages),
            "tokens": {
                "input": total_input,
                "output": total_output,
                "cacheRead": total_cache_read,
                "cacheWrite": total_cache_write,
                "total": total_input + total_output + total_cache_read + total_cache_write,
            },
            "cost": total_cost,
        }

    def get_last_assistant_text(self) -> str | None:
        for message in reversed(self.messages):
            if message.get("role") != "assistant":
                continue
            if message.get("stopReason") == "aborted" and not message.get("content"):
                continue
            text_parts: list[str] = []
            content = message.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(str(part.get("text", "")))
            text = "".join(text_parts).strip()
            if text:
                return text
        return None

    async def execute_bash(
        self,
        command: str,
        on_chunk: Callable[[str], None] | None = None,
        *,
        exclude_from_context: bool = False,
        operations: LocalBashOperations | None = None,
    ) -> BashResult:
        self._bash_abort_controller = AbortController()
        prefix = self.settings_manager.get_shell_command_prefix()
        resolved_command = f"{prefix}\n{command}" if prefix else command
        bash_operations = operations or LocalBashOperations()

        try:
            result = await execute_bash_with_operations(
                resolved_command,
                self.session_manager.get_cwd(),
                bash_operations,
                BashExecutorOptions(
                    on_chunk=on_chunk,
                    signal=self._bash_abort_controller.signal,
                ),
            )
            self.record_bash_result(command, result, exclude_from_context=exclude_from_context)
            return result
        finally:
            self._bash_abort_controller = None

    def record_bash_result(
        self,
        command: str,
        result: BashResult,
        *,
        exclude_from_context: bool = False,
    ) -> None:
        bash_message: AgentMessage = {
            "role": "bashExecution",
            "command": command,
            "output": result.output,
            "exitCode": result.exit_code,
            "cancelled": result.cancelled,
            "truncated": result.truncated,
            "fullOutputPath": result.full_output_path,
            "timestamp": int(time.time() * 1000),
            "excludeFromContext": exclude_from_context,
        }
        self.agent.state.messages.append(bash_message)
        append = getattr(self.session_manager, "append_message", None)
        if callable(append):
            append(bash_message)

    def abort_bash(self) -> None:
        if self._bash_abort_controller is not None:
            self._bash_abort_controller.abort()

    async def wait_for_idle(self) -> None:
        await self.agent.waitForIdle()

    async def abort(self) -> None:
        self.agent.abort()
        await self.wait_for_idle()

    async def steer(self, text: str, images: list[ImageContent] | None = None) -> None:
        self._steering_messages.append(text)
        self._emit_queue_update()
        self.agent.steer(create_user_message(text, images))

    async def follow_up(self, text: str, images: list[ImageContent] | None = None) -> None:
        self._follow_up_messages.append(text)
        self._emit_queue_update()
        self.agent.followUp(create_user_message(text, images))

    def get_steering_messages(self) -> list[str]:
        return list(self._steering_messages)

    def get_follow_up_messages(self) -> list[str]:
        return list(self._follow_up_messages)

    def clear_queue(self) -> dict[str, list[str]]:
        steering = list(self._steering_messages)
        follow_up = list(self._follow_up_messages)
        self._steering_messages.clear()
        self._follow_up_messages.clear()
        self.agent.clearAllQueues()
        self._emit_queue_update()
        return {"steering": steering, "followUp": follow_up}

    def set_session_name(self, name: str) -> None:
        self.session_manager.append_session_info(name)
        self._emit(
            {
                "type": "session_info_changed",
                "name": self.session_manager.get_session_name(),
            }
        )

    async def navigate_tree(
        self,
        target_id: str,
        *,
        summarize: bool = False,
    ) -> dict[str, Any]:
        old_leaf_id = self.session_manager.get_leaf_id()
        if target_id == old_leaf_id:
            return {"cancelled": False}

        target_entry = self.session_manager.get_entry(target_id)
        if not target_entry:
            raise RuntimeError(f"Entry {target_id} not found")

        if summarize:
            raise RuntimeError("Tree summarization is not implemented in the Python scaffold yet")

        if (
            target_entry.get("type") == "message"
            and target_entry.get("message", {}).get("role") == "user"
        ):
            new_leaf_id = target_entry.get("parentId")
            editor_text = _extract_user_message_text(target_entry.get("message", {}).get("content"))
        else:
            new_leaf_id = target_id
            editor_text = None

        if new_leaf_id:
            self.session_manager.branch(new_leaf_id)
        else:
            self.session_manager.reset_leaf()

        context = self.session_manager.build_session_context()
        self.agent.state.messages = list(context.get("messages", []))
        result: dict[str, Any] = {"cancelled": False}
        if editor_text is not None:
            result["editorText"] = editor_text
        return result

    async def compact(self, custom_instructions: str | None = None) -> dict[str, Any]:
        await self.abort()
        self._compaction_abort_controller = AbortController()
        self._emit({"type": "compaction_start", "reason": "manual"})
        aborted = False
        error_message: str | None = None
        compaction_result: dict[str, Any] | None = None

        try:
            if not self.model:
                raise RuntimeError(format_no_model_selected_message())

            auth = await self.model_registry.get_api_key_and_headers(self.model)
            if not auth.get("ok"):
                raise RuntimeError(
                    format_no_api_key_found_message(self.model.get("provider", "unknown"))
                )

            path_entries = self.session_manager.get_branch()
            settings = self.settings_manager.get_compaction_settings()
            preparation_result = prepare_compaction(path_entries, settings)
            if not preparation_result.ok:
                raise preparation_result.error
            preparation = preparation_result.value
            if not preparation:
                last_entry = path_entries[-1] if path_entries else None
                if last_entry and last_entry.get("type") == "compaction":
                    raise RuntimeError("Already compacted")
                raise RuntimeError("Nothing to compact (session too small)")

            if self._compaction_abort_controller.signal.aborted:
                aborted = True
                raise RuntimeError("Compaction cancelled")

            compact_result = await compact(
                preparation,
                self.model,
                auth.get("apiKey") or "",
                auth.get("headers"),
                custom_instructions,
                self._compaction_abort_controller.signal,
                self.thinking_level,
            )
            if not compact_result.ok:
                raise compact_result.error

            result = compact_result.value
            self.session_manager.append_compaction(
                result["summary"],
                result["firstKeptEntryId"],
                result["tokensBefore"],
                result.get("details"),
                False,
            )
            session_context = self.session_manager.build_session_context()
            self.agent.state.messages = list(session_context.get("messages", []))
            compaction_result = result
            return compaction_result
        except Exception as error:
            error_message = str(error)
            if "cancelled" in error_message.lower():
                aborted = True
            raise
        finally:
            self._emit(
                {
                    "type": "compaction_end",
                    "reason": "manual",
                    "result": compaction_result,
                    "aborted": aborted,
                    "willRetry": False,
                    "errorMessage": error_message,
                }
            )
            self._compaction_abort_controller = None

    async def try_execute_extension_command(self, text: str) -> bool:
        if not text.startswith("/"):
            return False
        runner = self._extension_runner
        if runner is None:
            return False
        space_index = text.find(" ")
        command_name = text[1:space_index] if space_index != -1 else text[1:]
        args = text[space_index + 1 :] if space_index != -1 else ""
        command = runner.get_command(command_name)
        if command is None:
            return False
        ctx = runner.create_command_context()
        try:
            await command.handler(args, ctx)
        except Exception as error:
            runner.emit_error(
                ExtensionError(
                    extension_path=f"command:{command_name}",
                    event="command",
                    error=str(error),
                )
            )
        return True

    async def _emit_extension_event(self, event: AgentEvent) -> None:
        runner = self._extension_runner
        if runner is None:
            return
        event_type = event.get("type")
        if event_type == "message_end":
            message = event.get("message")
            if message is not None:
                await runner.emit_message_end({"type": "message_end", "message": message})
        elif event_type in ("agent_start", "agent_end", "turn_start", "turn_end"):
            await runner.emit(event)

    def _refresh_tool_registry(self) -> None:
        runner = self._extension_runner
        if runner is None:
            return
        active_names = [tool.name for tool in self.agent.state.tools]
        extension_by_name = {
            tool.name: tool
            for tool in wrap_registered_tools(runner.get_all_registered_tools(), runner)
        }
        next_tools: list[AgentTool] = []
        for name in active_names:
            if name in extension_by_name:
                next_tools.append(extension_by_name[name])
                continue
            try:
                next_tools.append(create_tool(name, self.cwd))  # type: ignore[arg-type]
            except (KeyError, ValueError):
                continue
        self.agent.state.tools = next_tools
        self._refresh_system_prompt()

    async def export_to_html(
        self, output_path: str | None = None, *, theme_name: str | None = None
    ) -> str:
        from pi_mono.coding_agent.core.export_html.export_html import export_session_to_html

        return export_session_to_html(self, output_path=output_path, theme_name=theme_name)

    def abort_retry(self) -> None:
        if self._retry_abort_controller is not None:
            self._retry_abort_controller.abort()

    def _is_non_retryable_provider_limit_error(self, error_message: str) -> bool:
        return bool(
            re.search(
                r"GoUsageLimitError|FreeUsageLimitError|Monthly usage limit reached|available balance|insufficient.?credits|insufficient_quota|out of budget|quota exceeded|billing|error code: 402|'code': 402",
                error_message,
                re.IGNORECASE,
            )
        )

    def _is_retryable_error(self, message: AssistantMessage) -> bool:
        if message.get("stopReason") != "error" or not message.get("errorMessage"):
            return False
        context_window = (self.model or {}).get("contextWindow", 0)
        if is_context_overflow(message, context_window):
            return False
        error_message = str(message.get("errorMessage", ""))
        if self._is_non_retryable_provider_limit_error(error_message):
            return False
        return bool(
            re.search(
                r"overloaded|provider.?returned.?error|rate.?limit|too many requests|429|500|502|503|504|service.?unavailable|server.?error|internal.?error|network.?error|connection.?error|connection.?refused|connection.?lost|websocket.?closed|websocket.?error|other side closed|fetch failed|upstream.?connect|reset before headers|socket hang up|ended without|stream ended before message_stop|http2 request did not get a response|timed? out|timeout|terminated|retry delay",
                error_message,
                re.IGNORECASE,
            )
        )

    def _will_retry_after_agent_end(self, event: AgentEvent) -> bool:
        settings = self.settings_manager.get_retry_settings()
        if not settings.get("enabled") or self._retry_attempt >= int(settings.get("maxRetries", 3)):
            return False
        messages = event.get("messages", [])
        for message in reversed(messages):
            if message.get("role") == "assistant":
                return self._is_retryable_error(cast(AssistantMessage, message))
        return False

    async def _abortable_sleep(self, delay_ms: int, signal: AbortSignal) -> None:
        if signal.aborted:
            raise RuntimeError("Retry cancelled")
        done = asyncio.Event()

        def on_abort() -> None:
            done.set()

        signal.add_event_listener("abort", on_abort)
        try:
            try:
                await asyncio.wait_for(done.wait(), timeout=delay_ms / 1000)
            except TimeoutError:
                return
            if signal.aborted:
                raise RuntimeError("Retry cancelled")
        finally:
            signal.remove_event_listener("abort", on_abort)

    async def _prepare_retry(self, message: AssistantMessage) -> bool:
        settings = self.settings_manager.get_retry_settings()
        if not settings.get("enabled"):
            return False

        self._retry_attempt += 1
        max_retries = int(settings.get("maxRetries", 3))
        if self._retry_attempt > max_retries:
            self._retry_attempt -= 1
            return False

        delay_ms = int(settings.get("baseDelayMs", 2000)) * (2 ** (self._retry_attempt - 1))
        self._emit(
            {
                "type": "auto_retry_start",
                "attempt": self._retry_attempt,
                "maxAttempts": max_retries,
                "delayMs": delay_ms,
                "errorMessage": str(message.get("errorMessage") or "Unknown error"),
            }
        )

        messages = self.agent.state.messages
        if messages and messages[-1].get("role") == "assistant":
            self.agent.state.messages = messages[:-1]

        self._retry_abort_controller = AbortController()
        try:
            await self._abortable_sleep(delay_ms, self._retry_abort_controller.signal)
        except RuntimeError:
            attempt = self._retry_attempt
            self._retry_attempt = 0
            self._emit(
                {
                    "type": "auto_retry_end",
                    "success": False,
                    "attempt": attempt,
                    "finalError": "Retry cancelled",
                }
            )
            return False
        finally:
            self._retry_abort_controller = None

        return True

    async def _handle_post_agent_run(self) -> bool:
        message = self._last_assistant_message
        self._last_assistant_message = None
        if message is None:
            return False

        if self._is_retryable_error(message) and await self._prepare_retry(message):
            return True

        if message.get("stopReason") == "error" and self._retry_attempt > 0:
            self._emit(
                {
                    "type": "auto_retry_end",
                    "success": False,
                    "attempt": self._retry_attempt,
                    "finalError": str(message.get("errorMessage") or "Unknown error"),
                }
            )
            self._retry_attempt = 0

        return self.agent.hasQueuedMessages()

    async def _run_agent_prompt(
        self,
        text: str,
        *,
        images: list[ImageContent] | None = None,
    ) -> None:
        await self.agent.prompt(text, images=images)
        while await self._handle_post_agent_run():
            await self.agent.continue_run()

    async def prompt(self, text: str, options: PromptOptions | None = None) -> None:
        opts = options or PromptOptions()
        if text.startswith("/") and await self.try_execute_extension_command(text):
            return
        if not self.model or self.model.get("id") in (None, "unknown"):
            raise RuntimeError(format_no_model_selected_message())
        if self.is_streaming:
            if not opts.streaming_behavior:
                raise RuntimeError(
                    "Agent is already processing. Specify streaming_behavior "
                    "('steer' or 'followUp') to queue the message."
                )
            if opts.streaming_behavior == "followUp":
                await self.follow_up(text, opts.images)
            else:
                await self.steer(text, opts.images)
            return
        await self._run_agent_prompt(text, images=opts.images)

    @property
    def extension_runner(self) -> ExtensionRunner | None:
        return self._extension_runner

    async def _ensure_extension_runner(self) -> ExtensionRunner:
        if self._extension_runner is not None:
            return self._extension_runner

        if self._no_extensions:
            runtime = create_extension_runtime()
            self._extension_load_result = LoadExtensionsResult(
                extensions=[], errors=[], runtime=runtime
            )
        else:
            self._extension_load_result = await discover_and_load_extensions(
                self._extension_paths,
                self.cwd,
                str(get_agent_dir()),
            )

        load_result = self._extension_load_result
        self._extension_runner = ExtensionRunner(
            load_result.extensions,
            load_result.runtime,
            self.cwd,
            self.session_manager,
            self.model_registry,
        )
        return self._extension_runner

    def _build_extension_actions(self) -> ExtensionActions:
        def get_active_tools() -> list[str]:
            return [tool.name for tool in self.agent.state.tools]

        def set_active_tools(tool_names: list[str]) -> None:
            self.agent.state.tools = _resolve_tools(self.cwd, initial_active_tool_names=tool_names)

        def refresh_tools() -> None:
            self._refresh_tool_registry()

        def get_commands() -> list[dict[str, Any]]:
            runner = self._extension_runner
            if runner is None:
                return []
            return [
                {
                    "name": command.invocation_name,
                    "description": command.description,
                    "source": "extension",
                }
                for command in runner.get_registered_commands()
            ]

        return ExtensionActions(
            send_message=lambda *_a, **_k: None,
            send_user_message=lambda *_a, **_k: None,
            append_entry=lambda *_a, **_k: None,
            set_session_name=self.set_session_name,
            get_session_name=lambda: self.session_manager.get_session_name(),
            set_label=lambda *_a, **_k: None,
            get_active_tools=get_active_tools,
            get_all_tools=lambda: [tool.name for tool in self.agent.state.tools],
            set_active_tools=set_active_tools,
            refresh_tools=refresh_tools,
            get_commands=get_commands,
            set_model=self._set_model_from_extension,
            get_thinking_level=lambda: self.thinking_level,
            set_thinking_level=self.set_thinking_level,
        )

    async def _set_model_from_extension(self, model: Model[Any]) -> bool:
        await self.set_model(model)
        return True

    def _build_extension_context_actions(self) -> ExtensionContextActions:
        return ExtensionContextActions(
            get_model=lambda: self.model,
            is_idle=lambda: not self.is_streaming,
            get_signal=lambda: None,
            abort=lambda: None,
            has_pending_messages=lambda: self.pending_message_count > 0,
            shutdown=lambda: None,
            get_context_usage=lambda: None,
            compact=lambda *_a: None,
            get_system_prompt=lambda: self.system_prompt,
            get_system_prompt_options=lambda: {"cwd": self.cwd},
        )

    async def bind_extensions(self, **kwargs: Any) -> None:
        mode = kwargs.get("mode", "print")
        ui_context = kwargs.get("ui_context")
        command_context_actions = kwargs.get("command_context_actions")
        on_error = kwargs.get("on_error")
        extension_paths = kwargs.get("extension_paths")
        no_extensions = kwargs.get("no_extensions")

        if extension_paths is not None:
            self._extension_paths = list(extension_paths)
        if no_extensions is not None:
            self._no_extensions = bool(no_extensions)

        runner = await self._ensure_extension_runner()
        runner.bind_core(self._build_extension_actions(), self._build_extension_context_actions())
        runner.set_ui_context(ui_context, mode)
        self._extension_mode = mode

        if command_context_actions is not None:
            if isinstance(command_context_actions, ExtensionCommandContextActions):
                runner.bind_command_context(command_context_actions)
            else:
                runner.bind_command_context(
                    ExtensionCommandContextActions(
                        wait_for_idle=command_context_actions.get("waitForIdle", _async_noop),
                        new_session=command_context_actions.get(
                            "newSession", _async_cancelled_false
                        ),
                        fork=command_context_actions.get("fork", _async_cancelled_false_entry),
                        navigate_tree=command_context_actions.get(
                            "navigateTree", _async_cancelled_false_entry
                        ),
                        switch_session=command_context_actions.get(
                            "switchSession", _async_cancelled_false_entry
                        ),
                        reload=command_context_actions.get("reload", _async_noop),
                    )
                )
        else:
            runner.bind_command_context(None)

        if self._extension_error_unsubscribe is not None:
            self._extension_error_unsubscribe()
            self._extension_error_unsubscribe = None
        if on_error is not None:
            self._extension_error_unsubscribe = runner.on_error(on_error)

        self._refresh_tool_registry()
        await runner.emit(
            {
                "type": "session_start",
                "reason": self._session_start_reason,
            }
        )
        await runner.emit_resources_discover(self.cwd, "startup")

    async def reload(self) -> None:
        await self._resource_loader.reload()
        self._refresh_system_prompt()
        context = self.session_manager.build_session_context()
        self.agent.state.messages = list(context.get("messages", []))

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        if self._extension_error_unsubscribe is not None:
            self._extension_error_unsubscribe()
            self._extension_error_unsubscribe = None
        if self._agent_listener is not None:
            self._agent_listener()
            self._agent_listener = None
        self._listeners.clear()
        self._event_bus.clear()


async def _async_noop(*_args: Any, **_kwargs: Any) -> None:
    return None


async def _async_cancelled_false(*_args: Any, **_kwargs: Any) -> dict[str, bool]:
    return {"cancelled": False}


async def _async_cancelled_false_entry(_entry: str, *_args: Any, **_kwargs: Any) -> dict[str, bool]:
    return {"cancelled": False}


def _extract_user_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


CreateAgentSessionRuntimeResult = tuple[AgentSession, Any, list[dict[str, str]], str | None]
CreateAgentSessionRuntimeFactory = Callable[
    ...,
    Awaitable[CreateAgentSessionRuntimeResult],
]


@dataclass
class AgentSessionRuntime:
    session: AgentSession
    services: Any
    diagnostics: list[dict[str, str]]
    model_fallback_message: str | None = None
    _create_runtime: CreateAgentSessionRuntimeFactory | None = None
    _rebind_session: Callable[[], Awaitable[None]] | None = None

    def set_rebind_session(self, handler: Callable[[], Awaitable[None]]) -> None:
        self._rebind_session = handler

    async def dispose(self) -> None:
        self.session.dispose()

    async def _finish_replacement(self) -> None:
        if self._rebind_session is not None:
            await self._rebind_session()

    async def _replace_session(
        self,
        session_manager: SessionManager,
        *,
        reason: str,
        previous_session_file: str | None = None,
    ) -> None:
        if self._create_runtime is not None:
            services = self.services
            agent_dir = getattr(services, "agent_dir", None) or str(get_agent_dir())
            self.session.dispose()
            session, services, diagnostics, model_fallback_message = await self._create_runtime(
                cwd=session_manager.get_cwd(),
                agent_dir=agent_dir,
                session_manager=session_manager,
                session_start_event={
                    "type": "session_start",
                    "reason": reason,
                    "previousSessionFile": previous_session_file,
                },
            )
            self.session = session
            self.services = services
            self.diagnostics = diagnostics
            self.model_fallback_message = model_fallback_message
            self.session._session_start_reason = reason
        else:
            self.session.session_manager = session_manager
            self.session._session_start_reason = reason
            await self.session.reload()

    async def new_session(self, options: dict[str, Any] | None = None) -> dict[str, bool]:
        previous_session_file = self.session.session_file
        session_dir = self.session.session_manager.get_session_dir()
        if session_dir:
            session_manager = SessionManager.create(self.session.cwd, session_dir)
        else:
            session_manager = SessionManager.in_memory(self.session.cwd)
        if options and options.get("parentSession"):
            session_manager.new_session({"parentSession": options["parentSession"]})
        await self._replace_session(
            session_manager,
            reason="new",
            previous_session_file=previous_session_file,
        )
        await self._finish_replacement()
        return {"cancelled": False}

    async def fork(
        self,
        entry_id: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        position = (options or {}).get("position", "before")
        selected_entry = self.session.session_manager.get_entry(entry_id)
        if not selected_entry:
            raise RuntimeError("Invalid entry ID for forking")

        selected_text: str | None = None
        if position == "at":
            target_leaf_id = selected_entry.get("id")
        else:
            if (
                selected_entry.get("type") != "message"
                or selected_entry.get("message", {}).get("role") != "user"
            ):
                raise RuntimeError("Invalid entry ID for forking")
            target_leaf_id = selected_entry.get("parentId")
            selected_text = _extract_user_message_text(
                selected_entry.get("message", {}).get("content")
            )

        previous_session_file = self.session.session_file
        session_manager = self.session.session_manager

        if session_manager.is_persisted():
            current_session_file = session_manager.get_session_file()
            if not current_session_file:
                raise RuntimeError("Persisted session is missing a session file")
            session_dir = session_manager.get_session_dir()
            if not target_leaf_id:
                new_session_manager = SessionManager.create(self.session.cwd, session_dir)
                new_session_manager.new_session({"parentSession": current_session_file})
                await self._replace_session(
                    new_session_manager,
                    reason="fork",
                    previous_session_file=previous_session_file,
                )
            else:
                branched_manager = SessionManager.open(current_session_file, session_dir)
                forked_session_path = branched_manager.create_branched_session(str(target_leaf_id))
                if not forked_session_path:
                    raise RuntimeError("Failed to create forked session")
                await self._replace_session(
                    branched_manager,
                    reason="fork",
                    previous_session_file=previous_session_file,
                )
        else:
            if not target_leaf_id:
                session_manager.new_session({"parentSession": self.session.session_file})
            else:
                session_manager.create_branched_session(str(target_leaf_id))
            await self._replace_session(
                session_manager,
                reason="fork",
                previous_session_file=previous_session_file,
            )

        await self._finish_replacement()
        return {"cancelled": False, "selectedText": selected_text}

    async def switch_session(
        self,
        session_path: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, bool]:
        previous_session_file = self.session.session_file
        cwd_override = (options or {}).get("cwdOverride") if options else None
        session_manager = SessionManager.open(session_path, None, cwd_override)
        await self._replace_session(
            session_manager,
            reason="resume",
            previous_session_file=previous_session_file,
        )
        await self._finish_replacement()
        return {"cancelled": False}
