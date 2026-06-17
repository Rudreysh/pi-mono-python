"""Interactive TUI mode for the coding agent.

Minimal port of packages/coding-agent/src/modes/interactive/interactive-mode.ts.
"""

from __future__ import annotations

import asyncio
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pi_mono.agent.types import AgentMessage
from pi_mono.ai.models import get_providers
from pi_mono.ai.oauth import OAuthLoginCallbacks
from pi_mono.ai.types import ImageContent, Model
from pi_mono.ai.utils.oauth import OAuthAuthInfo, OAuthDeviceCodeInfo, OAuthSelectPrompt
from pi_mono.coding_agent.core.keybindings import CodingAgentKeybindingsManager
from pi_mono.coding_agent.core.model_resolver import default_model_per_provider
from pi_mono.coding_agent.core.slash_commands import (
    BUILTIN_SLASH_COMMANDS as CANONICAL_SLASH_COMMANDS,
)
from pi_mono.coding_agent.modes.interactive.components.keybinding_hints import key_display_text
from pi_mono.coding_agent.modes.interactive.interactive_autocomplete import (
    build_interactive_autocomplete_provider,
)
from pi_mono.config import APP_NAME, VERSION, get_agent_dir, get_auth_path, get_docs_path
from pi_mono.coding_agent.core.agent_session import AgentSessionEvent, AgentSessionRuntime
from pi_mono.coding_agent.modes.interactive.components.assistant_message import (
    AssistantMessageComponent,
)
from pi_mono.coding_agent.modes.interactive.components.extension_selector import (
    ExtensionSelectorComponent,
)
from pi_mono.coding_agent.modes.interactive.components.login_dialog import LoginDialogComponent
from pi_mono.coding_agent.modes.interactive.components.oauth_selector import (
    AuthSelectorProvider,
    OAuthSelectorComponent,
)
from pi_mono.coding_agent.core.footer_data_provider import FooterDataProvider
from pi_mono.coding_agent.modes.interactive.components.footer import (
    FooterComponent,
    FooterRenderComponent,
)
from pi_mono.coding_agent.modes.interactive.components.model_selector import ModelSelectorComponent
from pi_mono.coding_agent.modes.interactive.components.session_selector import (
    SessionSelectorComponent,
)
from pi_mono.coding_agent.modes.interactive.components.settings_selector import (
    SettingsSelectorComponent,
    build_settings_config_from_session,
)
from pi_mono.coding_agent.modes.interactive.components.tree_selector import TreeSelectorComponent
from pi_mono.coding_agent.modes.interactive.components.tool_execution import ToolExecutionComponent
from pi_mono.coding_agent.modes.interactive.theme.theme import get_editor_theme, init_theme, theme
from pi_mono.core.provider_display_names import BUILT_IN_PROVIDER_DISPLAY_NAMES
from pi_mono.core.session_manager import SessionManager
from pi_mono.tui.components.editor import Editor, EditorOptions
from pi_mono.tui.components.loader import Loader
from pi_mono.tui.components.spacer import Spacer
from pi_mono.tui.components.text import Text
from pi_mono.tui.keybindings import get_keybindings, set_keybindings
from pi_mono.tui.keys import matches_key
from pi_mono.tui.terminal import ProcessTerminal
from pi_mono.tui.tui import Container, OverlayOptions, TUI

BEDROCK_PROVIDER_ID = "amazon-bedrock"
BUILT_IN_MODEL_PROVIDERS = frozenset(get_providers())


def is_api_key_login_provider(
    provider_id: str,
    oauth_provider_ids: set[str],
    built_in_provider_ids: set[str] | frozenset[str] = BUILT_IN_MODEL_PROVIDERS,
) -> bool:
    if provider_id in BUILT_IN_PROVIDER_DISPLAY_NAMES:
        return True
    if provider_id in built_in_provider_ids:
        return False
    return provider_id not in oauth_provider_ids


def _is_unknown_model(model: Model[Any] | None) -> bool:
    return bool(
        model
        and model.get("provider") == "unknown"
        and model.get("id") == "unknown"
        and model.get("api") == "unknown"
    )


HELP_TEXT = "\n".join(f"  /{cmd.name} - {cmd.description}" for cmd in CANONICAL_SLASH_COMMANDS)


@dataclass
class InteractiveModeOptions:
    initial_message: str | None = None
    initial_messages: list[str] | None = None
    initial_images: list[ImageContent] | None = None
    theme_name: str = "dark"
    verbose: bool = False


def _message_text(message: AgentMessage | dict[str, Any]) -> str:
    parts: list[str] = []
    for block in message.get("content", []):
        if block.get("type") == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(parts)


class InteractiveMode:
    """Interactive terminal UI wired to AgentSession."""

    def __init__(
        self, runtime_host: AgentSessionRuntime, options: InteractiveModeOptions | None = None
    ) -> None:
        self._runtime_host = runtime_host
        self._options = options or InteractiveModeOptions()
        self._session = runtime_host.session
        self._unsubscribe: Callable[[], None] | None = None
        self._input_waiter: asyncio.Future[str] | None = None
        self._is_initialized = False
        self._is_shutting_down = False
        self._last_sigint_time = 0.0
        self._signal_handlers: list[tuple[int, Any]] = []

        self._theme_name = self._options.theme_name
        self._ui: TUI | None = None
        self._chat_container: Container | None = None
        self._status_container: Container | None = None
        self._footer_container: Container | None = None
        self._footer: FooterComponent | None = None
        self._footer_render: FooterRenderComponent | None = None
        self._editor_container: Container | None = None
        self._streaming_component: AssistantMessageComponent | None = None
        self._loader: Loader | None = None
        self._editor: Editor | None = None
        self._pending_tools: dict[str, ToolExecutionComponent] = {}
        self._model_overlay: Any | None = None
        self._settings_overlay: Any | None = None
        self._sessions_overlay: Any | None = None
        self._tree_overlay: Any | None = None
        self._footer_data: FooterDataProvider | None = None
        self._keybindings_manager: CodingAgentKeybindingsManager | None = None
        self._last_escape_time = 0.0
        self._retry_loader: Loader | None = None

        self._runtime_host.set_rebind_session(self._rebind_session)

    @property
    def session(self):
        return self._session

    @property
    def ui(self) -> TUI:
        if self._ui is None:
            raise RuntimeError("InteractiveMode UI is not initialized; call init() first")
        return self._ui

    def _ensure_ui(self) -> None:
        if self._ui is not None:
            return
        init_theme(self._theme_name)
        self._ui = TUI(ProcessTerminal(), show_hardware_cursor=True)
        self._chat_container = Container()
        self._status_container = Container()
        self._footer_container = Container()
        self._editor_container = Container()
        self._editor = Editor(self._ui, get_editor_theme(), EditorOptions(padding_x=1))

    async def init(self) -> None:
        if self._is_initialized:
            return

        self._ensure_ui()
        self._register_signal_handlers()
        self._setup_layout()
        self._setup_keybindings()
        self._setup_editor()
        self._setup_input_handlers()
        self._ui.start()
        self._is_initialized = True
        await self._rebind_session()

    async def run(self) -> None:
        await self.init()

        if self._options.initial_message:
            await self._handle_prompt(
                self._options.initial_message, images=self._options.initial_images
            )

        for message in self._options.initial_messages or []:
            await self._handle_prompt(message)

        while not self._is_shutting_down:
            user_input = await self._wait_for_user_input()
            if self._is_shutting_down:
                break
            await self._handle_user_input(user_input)

    async def stop(self) -> None:
        if not self._is_initialized:
            return
        self._unregister_signal_handlers()
        if self._loader is not None:
            self._loader.stop()
        if self._model_overlay is not None:
            self._model_overlay.hide()
            self._model_overlay = None
        if self._settings_overlay is not None:
            self._settings_overlay.hide()
            self._settings_overlay = None
        if self._sessions_overlay is not None:
            self._sessions_overlay.hide()
            self._sessions_overlay = None
        if self._tree_overlay is not None:
            self._tree_overlay.hide()
            self._tree_overlay = None
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        if self._ui is not None:
            self._ui.stop()
        self._is_initialized = False

    def _setup_layout(self) -> None:
        assert self._ui is not None
        assert self._chat_container is not None
        assert self._status_container is not None
        assert self._footer_container is not None
        assert self._editor_container is not None
        assert self._editor is not None
        header = Text(
            f"{theme.bold(theme.fg('accent', APP_NAME))}{theme.fg('dim', f' v{VERSION}')}\n"
            f"{theme.fg('dim', 'Type a prompt, /help for commands, Ctrl+C to interrupt')}",
            padding_x=1,
            padding_y=0,
        )
        self._ui.add_child(header)
        self._ui.add_child(Spacer(1))
        self._ui.add_child(self._chat_container)
        self._ui.add_child(self._status_container)
        self._ui.add_child(self._footer_container)
        self._ui.add_child(self._editor_container)
        self._editor_container.add_child(self._editor)
        self._ui.set_focus(self._editor)

    def _setup_editor(self) -> None:
        assert self._editor is not None

        def on_submit(text: str) -> None:
            asyncio.create_task(self._submit_editor_text(text))

        self._editor.on_submit = on_submit

    async def _wait_for_idle(self) -> None:
        while self._session.is_streaming:
            await asyncio.sleep(0.05)

    async def _navigate_tree_from_extension(
        self, target_id: str, options: dict[str, Any] | None = None
    ) -> dict[str, bool]:
        del options
        result = await self._session.navigate_tree(target_id)
        if self._chat_container is not None:
            self._chat_container.clear()
        return {"cancelled": bool(result.get("cancelled"))}

    async def _reload_from_extension(self) -> None:
        if self._keybindings_manager is not None:
            self._keybindings_manager.reload()
        await self._session.reload()
        await self._rebind_session()
        self._show_status(theme.fg("success", "Reloaded extensions, skills, prompts, and themes"))

    def _setup_input_handlers(self) -> None:
        assert self._ui is not None
        assert self._editor is not None

        def handle_global_input(data: str) -> dict[str, object] | None:
            kb = get_keybindings()
            if kb.matches(data, "app.interrupt"):
                if self._session.is_retrying:
                    self._session.abort_retry()
                    self._show_status(theme.fg("warning", "Retry cancelled"))
                    return {"consume": True}
                if self._session.is_streaming:
                    self._session.agent.abort()
                    self._show_status(theme.fg("warning", "Interrupted"))
                    return {"consume": True}
                return None
            if kb.matches(data, "app.thinking.cycle"):
                asyncio.create_task(self._cycle_thinking_level())
                return {"consume": True}
            if kb.matches(data, "app.model.cycleForward"):
                asyncio.create_task(self._cycle_model("forward"))
                return {"consume": True}
            if kb.matches(data, "app.model.cycleBackward"):
                asyncio.create_task(self._cycle_model("backward"))
                return {"consume": True}
            if kb.matches(data, "app.model.select"):
                self._show_model_selector()
                return {"consume": True}
            if kb.matches(data, "app.session.tree"):
                self._show_tree_selector()
                return {"consume": True}
            if kb.matches(data, "app.session.resume"):
                self._show_session_selector()
                return {"consume": True}
            if matches_key(data, "ctrl+c"):
                if self._session.is_streaming:
                    self._session.agent.abort()
                    self._show_status(theme.fg("warning", "Interrupted"))
                    return {"consume": True}
                now = time.monotonic()
                if now - self._last_sigint_time < 0.5:
                    asyncio.create_task(self._shutdown())
                    return {"consume": True}
                self._last_sigint_time = now
                self._editor.set_text("")
                return {"consume": True}
            if matches_key(data, "ctrl+d") and not self._editor.get_text().strip():
                asyncio.create_task(self._shutdown())
                return {"consume": True}
            return None

        self._ui.add_input_listener(handle_global_input)

    def _register_signal_handlers(self) -> None:
        for signum in (signal.SIGTERM, signal.SIGHUP):
            try:
                previous = signal.getsignal(signum)

                def handler(_signum: int, _frame: object | None, _previous: Any = previous) -> None:
                    asyncio.create_task(self._shutdown(from_signal=True, signum=_signum))
                    if callable(_previous) and _previous not in (signal.SIG_DFL, signal.SIG_IGN):
                        _previous(_signum, _frame)  # type: ignore[misc]

                signal.signal(signum, handler)
                self._signal_handlers.append((signum, previous))
            except (AttributeError, ValueError, OSError):
                pass

    def _unregister_signal_handlers(self) -> None:
        for signum, previous in self._signal_handlers:
            try:
                signal.signal(signum, previous)
            except (AttributeError, ValueError, OSError):
                pass
        self._signal_handlers.clear()

    def _setup_keybindings(self) -> None:
        self._keybindings_manager = CodingAgentKeybindingsManager.create(str(get_agent_dir()))
        set_keybindings(self._keybindings_manager)

    async def _rebind_session(self) -> None:
        self._session = self._runtime_host.session
        await self._session.bind_extensions(
            mode="interactive",
            command_context_actions={
                "waitForIdle": self._wait_for_idle,
                "newSession": self._runtime_host.new_session,
                "fork": self._runtime_host.fork,
                "navigateTree": self._navigate_tree_from_extension,
                "switchSession": self._runtime_host.switch_session,
                "reload": self._reload_from_extension,
            },
        )
        if self._editor is not None:
            self._editor.set_autocomplete_provider(
                build_interactive_autocomplete_provider(self._session)
            )
        if self._footer is not None:
            self._footer.set_session(self._session)
        elif self._footer_container is not None:
            if self._footer_data is None:
                self._footer_data = FooterDataProvider(self._session.session_manager.get_cwd())
            else:
                self._footer_data.set_cwd(self._session.session_manager.get_cwd())
            self._footer = FooterComponent(self._session, self._footer_data)
            self._footer_render = FooterRenderComponent(self._footer)
            self._footer_container.add_child(self._footer_render)
        if self._unsubscribe is not None:
            self._unsubscribe()
        self._unsubscribe = self._session.subscribe(self._handle_session_event)

    def _handle_session_event(self, event: AgentSessionEvent) -> None:
        event_type = event.get("type")
        if event_type == "message_start":
            message = event.get("message")
            if message and message.get("role") == "assistant":
                self._start_streaming_assistant(message)
            elif message and message.get("role") == "user":
                self._append_user_message(message)
        elif event_type == "message_update":
            message = event.get("message")
            if message and message.get("role") == "assistant":
                self._update_streaming_assistant(message)
                for block in message.get("content", []):
                    if block.get("type") != "toolCall":
                        continue
                    tool_call_id = str(block.get("id", ""))
                    if not tool_call_id or tool_call_id in self._pending_tools:
                        continue
                    component = ToolExecutionComponent(
                        str(block.get("name", "tool")),
                        tool_call_id,
                        block.get("arguments"),
                    )
                    if self._chat_container is not None:
                        self._chat_container.add_child(component)
                    self._pending_tools[tool_call_id] = component
        elif event_type == "message_end":
            message = event.get("message")
            if message and message.get("role") == "assistant":
                self._finish_streaming_assistant(message)
                stop_reason = message.get("stopReason")
                if stop_reason in ("aborted", "error"):
                    error_message = str(message.get("errorMessage") or "Error")
                    if stop_reason == "aborted":
                        retry_attempt = self._session.retry_attempt
                        error_message = (
                            f"Aborted after {retry_attempt} retry attempt"
                            + ("s" if retry_attempt > 1 else "")
                            if retry_attempt > 0
                            else "Operation aborted"
                        )
                    for component in list(self._pending_tools.values()):
                        component.update_result(
                            {"content": [{"type": "text", "text": error_message}], "isError": True},
                            is_partial=False,
                        )
                    self._pending_tools.clear()
                else:
                    for component in self._pending_tools.values():
                        component.set_args_complete()
        elif event_type == "tool_execution_start":
            self._handle_tool_execution_start(event)
        elif event_type == "tool_execution_update":
            self._handle_tool_execution_update(event)
        elif event_type == "tool_execution_end":
            self._handle_tool_execution_end(event)
        elif event_type == "agent_start":
            self._stop_retry_loader()
            self._set_working(True)
        elif event_type == "agent_end":
            self._set_working(False)
            self._stop_retry_loader()
            if self._streaming_component is not None and self._chat_container is not None:
                self._chat_container.remove_child(self._streaming_component)
                self._streaming_component = None
            self._pending_tools.clear()
        elif event_type == "auto_retry_start":
            self._handle_auto_retry_start(event)
        elif event_type == "auto_retry_end":
            self._handle_auto_retry_end(event)
        if self._footer is not None:
            self._footer.invalidate()
        if self._ui is not None:
            self._ui.request_render()

    def _handle_tool_execution_start(self, event: AgentSessionEvent) -> None:
        if self._chat_container is None:
            return
        tool_call_id = str(event.get("toolCallId", ""))
        if not tool_call_id:
            return
        component = self._pending_tools.get(tool_call_id)
        if component is None:
            component = ToolExecutionComponent(
                str(event.get("toolName", "tool")),
                tool_call_id,
                event.get("args"),
            )
            self._chat_container.add_child(component)
            self._pending_tools[tool_call_id] = component
        component.mark_execution_started()

    def _handle_tool_execution_update(self, event: AgentSessionEvent) -> None:
        tool_call_id = str(event.get("toolCallId", ""))
        component = self._pending_tools.get(tool_call_id)
        if component is None:
            return
        partial_result = event.get("partialResult")
        if isinstance(partial_result, dict):
            component.update_result({**partial_result, "isError": False}, is_partial=True)

    def _handle_tool_execution_end(self, event: AgentSessionEvent) -> None:
        tool_call_id = str(event.get("toolCallId", ""))
        component = self._pending_tools.get(tool_call_id)
        if component is None:
            return
        result = event.get("result")
        if isinstance(result, dict):
            component.update_result(
                {**result, "isError": bool(event.get("isError"))}, is_partial=False
            )
        self._pending_tools.pop(tool_call_id, None)

    def _append_user_message(self, message: AgentMessage | dict[str, Any]) -> None:
        if self._chat_container is None:
            return
        text = _message_text(message)
        if not text.strip():
            return
        self._chat_container.add_child(
            Text(
                theme.fg("text", f"> {text}"),
                padding_x=1,
                padding_y=0,
                custom_bg_fn=theme.bg_fn("userMessageBg"),
            )
        )

    def _start_streaming_assistant(self, message: AgentMessage | dict[str, Any]) -> None:
        if self._chat_container is None:
            return
        self._finish_streaming_assistant(message, finalize=False)
        self._streaming_component = AssistantMessageComponent(message)
        self._chat_container.add_child(self._streaming_component)

    def _update_streaming_assistant(self, message: AgentMessage | dict[str, Any]) -> None:
        if self._streaming_component is None:
            self._start_streaming_assistant(message)
            return
        self._streaming_component.update_content(message)

    def _finish_streaming_assistant(
        self,
        message: AgentMessage | dict[str, Any],
        *,
        finalize: bool = True,
    ) -> None:
        if self._streaming_component is None:
            if finalize and self._chat_container is not None:
                text = _message_text(message)
                if text:
                    self._chat_container.add_child(AssistantMessageComponent(message))
            return

        self._streaming_component.update_content(message)
        self._streaming_component = None

    def _set_working(self, active: bool) -> None:
        if self._ui is None or self._status_container is None:
            return
        if active and self._session.is_streaming:
            if self._retry_loader is not None:
                return
            if self._loader is None:
                self._loader = Loader(
                    self._ui,
                    theme.fg_fn("accent"),
                    theme.fg_fn("muted"),
                    "Working...",
                )
                self._status_container.add_child(self._loader)
                self._loader.start()
            return
        if self._loader is not None:
            self._loader.stop()
            self._status_container.remove_child(self._loader)
            self._loader = None

    def _stop_retry_loader(self) -> None:
        if self._retry_loader is not None:
            self._retry_loader.stop()
            if self._status_container is not None:
                self._status_container.remove_child(self._retry_loader)
            self._retry_loader = None

    def _handle_auto_retry_start(self, event: AgentSessionEvent) -> None:
        if self._ui is None or self._status_container is None:
            return
        self._set_working(False)
        delay_seconds = max(1, int(event.get("delayMs", 0) / 1000))
        attempt = int(event.get("attempt", 0))
        max_attempts = int(event.get("maxAttempts", 0))
        message = f"Retrying ({attempt}/{max_attempts}) in {delay_seconds}s... " f"(Esc to cancel)"
        self._retry_loader = Loader(
            self._ui,
            theme.fg_fn("warning"),
            theme.fg_fn("muted"),
            message,
        )
        self._status_container.clear()
        self._status_container.add_child(self._retry_loader)
        self._retry_loader.start()

    def _handle_auto_retry_end(self, event: AgentSessionEvent) -> None:
        self._stop_retry_loader()
        if not event.get("success") and event.get("finalError"):
            self._show_status(theme.fg("error", str(event.get("finalError"))))

    def _show_status(self, text: str) -> None:
        if self._status_container is None:
            return
        self._status_container.clear()
        self._status_container.add_child(Text(text, padding_x=1, padding_y=0))

    async def _wait_for_user_input(self) -> str:
        loop = asyncio.get_running_loop()
        self._input_waiter = loop.create_future()
        try:
            return await self._input_waiter
        finally:
            self._input_waiter = None

    async def _submit_editor_text(self, text: str) -> None:
        if self._editor is None:
            return
        text = text.strip()
        self._editor.set_text("")
        if not text:
            return
        if self._input_waiter is not None and not self._input_waiter.done():
            self._input_waiter.set_result(text)
            return
        await self._handle_user_input(text)

    async def _handle_user_input(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if text.startswith("/"):
            await self._handle_slash_command(text)
            return
        await self._handle_prompt(text)

    async def _handle_slash_command(self, text: str) -> None:
        command, _, argument = text[1:].partition(" ")
        command = command.lower()
        argument = argument.strip()

        if command in ("help", "?"):
            self._append_system_message(HELP_TEXT)
            return
        if command in ("exit", "quit"):
            await self._shutdown()
            return
        if command in ("clear", "new"):
            await self._runtime_host.new_session()
            if self._chat_container is not None:
                self._chat_container.clear()
            self._show_status(theme.fg("success", "Started new session"))
            return
        if command == "compact":
            await self._session.compact(argument or None)
            self._show_status(theme.fg("success", "Session compacted"))
            return
        if command == "reload":
            await self._reload_from_extension()
            return
        if command == "session":
            stats = self._session.get_session_stats()
            self._append_system_message(
                "\n".join(f"  {key}: {value}" for key, value in stats.items())
            )
            return
        if command == "name":
            if not argument:
                current = self._session.session_name
                self._append_system_message(f"Session name: {current or '(unset)'}")
                return
            self._session.set_session_name(argument)
            self._show_status(theme.fg("success", f"Session name set to {argument}"))
            return
        if command == "copy":
            text = self._session.get_last_assistant_text()
            if not text:
                self._show_status(theme.fg("warning", "No assistant message to copy"))
                return
            try:
                from pi_mono.utils.clipboard import write_clipboard_text

                write_clipboard_text(text)
                self._show_status(theme.fg("success", "Copied last assistant message"))
            except Exception as error:
                self._show_status(theme.fg("error", f"Copy failed: {error}"))
            return
        if command == "export":
            output_path = argument or None
            path = await self._session.export_to_html(output_path)
            self._show_status(theme.fg("success", f"Exported to {path}"))
            return
        if command == "hotkeys":
            hints = [
                f"  {key_display_text('app.model.cycleForward')} next model",
                f"  {key_display_text('app.thinking.cycle')} cycle thinking",
                f"  {key_display_text('app.model.select')} model selector",
                f"  {key_display_text('app.session.tree')} session tree",
                f"  {key_display_text('app.session.resume')} resume session",
            ]
            self._append_system_message("\n".join(hints))
            return
        if command == "fork":
            self._show_status(
                theme.fg("warning", "Use /tree to pick a branch, or pass an entry id")
            )
            return
        if command == "clone":
            result = await self._runtime_host.fork(
                self._session.session_manager.leafId or "", {"position": "at"}
            )
            if not result.get("cancelled") and self._chat_container is not None:
                self._chat_container.clear()
            self._show_status(theme.fg("success", "Session cloned"))
            return
        if command == "model":
            await self._handle_model_command(argument or None)
            return
        if command in ("sessions", "resume"):
            self._show_session_selector()
            return
        if command == "settings":
            self._show_settings_selector()
            return
        if command == "tree":
            self._show_tree_selector(argument or None)
            return
        if command == "login":
            self._show_oauth_selector("login")
            return
        if command == "logout":
            self._show_oauth_selector("logout")
            return

        if await self._session.try_execute_extension_command(text):
            return

        self._show_status(theme.fg("error", f"Unknown command: /{command}"))

    async def _cycle_thinking_level(self) -> None:
        level = self._session.cycle_thinking_level()
        if level is None:
            self._show_status(theme.fg("warning", "No thinking levels available"))
            return
        self._show_status(theme.fg("success", f"Thinking level: {level}"))

    async def _cycle_model(self, direction: Literal["forward", "backward"]) -> None:
        result = await self._session.cycle_model(direction)
        if result is None:
            self._show_status(theme.fg("warning", "No models available to cycle"))
            return
        model = result["model"]
        self._show_status(theme.fg("success", f"Model: {model.get('provider')}/{model.get('id')}"))

    async def _handle_model_command(self, search: str | None) -> None:
        models = self._session.model_registry.get_available()
        if not models:
            self._show_status(theme.fg("warning", "No models available"))
            return

        if search:
            needle = search.lower()
            matches = [
                model
                for model in models
                if needle in f"{model.get('provider', '')}/{model.get('id', '')}".lower()
            ]
            if len(matches) == 1:
                self._set_model(matches[0])
                return
            if not matches:
                self._show_status(theme.fg("error", f"No model matched: {search}"))
                return

        self._show_model_selector(search)

    def _show_tree_selector(self, initial_search: str | None = None) -> None:
        if self._ui is None:
            return

        branch = self._session.session_manager.get_branch()
        if not branch:
            self._show_status(theme.fg("warning", "No session entries to navigate"))
            return

        def close_overlay() -> None:
            if self._tree_overlay is not None:
                self._tree_overlay.hide()
                self._tree_overlay = None
            if self._editor is not None:
                self._ui.set_focus(self._editor)

        async def on_select(entry_id: str) -> None:
            close_overlay()
            try:
                result = await self._session.navigate_tree(entry_id)
            except Exception as error:
                self._show_status(theme.fg("error", str(error)))
                return
            if result.get("cancelled"):
                return
            editor_text = result.get("editorText")
            if isinstance(editor_text, str) and self._editor is not None:
                self._editor.set_text(editor_text)
            if self._chat_container is not None:
                self._chat_container.clear()
            self._show_status(theme.fg("success", "Navigated session branch"))

        def handle_select(entry_id: str) -> None:
            asyncio.create_task(on_select(entry_id))

        selector = TreeSelectorComponent(
            self._ui,
            self._session.session_manager,
            handle_select,
            close_overlay,
            initial_search=initial_search,
        )
        self._tree_overlay = self._ui.show_overlay(
            selector, OverlayOptions(anchor="center", max_height="80%")
        )
        self._ui.set_focus(selector)

    def _show_model_selector(self, initial_search: str | None = None) -> None:
        if self._ui is None:
            return

        def close_overlay() -> None:
            if self._model_overlay is not None:
                self._model_overlay.hide()
                self._model_overlay = None
            if self._editor is not None:
                self._ui.set_focus(self._editor)

        def on_select(model: Model[Any]) -> None:
            self._set_model(model)
            close_overlay()

        selector = ModelSelectorComponent(
            self._ui,
            self._session.model,
            self._session.settings_manager,
            self._session.model_registry,
            on_select,
            close_overlay,
            initial_search=initial_search,
        )
        self._model_overlay = self._ui.show_overlay(
            selector, OverlayOptions(anchor="center", max_height="80%")
        )
        self._ui.set_focus(selector)

    def _set_model(self, model: Model[Any]) -> None:
        self._session.agent.state.model = model
        self._show_status(
            theme.fg("success", f"Model set to {model.get('provider')}/{model.get('id')}")
        )

    def _close_overlay(self, attr: str) -> None:
        overlay = getattr(self, attr, None)
        if overlay is not None:
            overlay.hide()
            setattr(self, attr, None)
        if self._editor is not None and self._ui is not None:
            self._ui.set_focus(self._editor)

    def _show_settings_selector(self) -> None:
        if self._ui is None:
            return

        def close_overlay() -> None:
            self._close_overlay("_settings_overlay")

        class _Callbacks:
            def on_auto_compact_change(self, enabled: bool) -> None:
                self_outer._session.set_auto_compaction(enabled)
                if self_outer._footer is not None:
                    self_outer._footer.set_auto_compact_enabled(enabled)

            def on_show_images_change(self, enabled: bool) -> None:
                self_outer._session.settings_manager.set_show_images(enabled)

            def on_steering_mode_change(self, mode: str) -> None:
                self_outer._session.set_steering_mode(mode)  # type: ignore[arg-type]

            def on_follow_up_mode_change(self, mode: str) -> None:
                self_outer._session.set_follow_up_mode(mode)  # type: ignore[arg-type]

            def on_thinking_level_change(self, level: str) -> None:
                self_outer._session.set_thinking_level(level)  # type: ignore[arg-type]
                if self_outer._footer is not None:
                    self_outer._footer.invalidate()

            def on_theme_change(self, theme_name: str) -> None:
                self_outer._theme_name = theme_name
                init_theme(theme_name)
                self_outer._session.settings_manager.set_theme(theme_name)
                if self_outer._ui is not None:
                    self_outer._ui.invalidate()

            def on_theme_preview(self, theme_name: str) -> None:
                init_theme(theme_name)
                if self_outer._ui is not None:
                    self_outer._ui.invalidate()
                    self_outer._ui.request_render()

            def on_cancel(self) -> None:
                close_overlay()

        self_outer = self
        callbacks = _Callbacks()
        config = build_settings_config_from_session(self._session)
        selector = SettingsSelectorComponent(config, callbacks)  # type: ignore[arg-type]
        self._settings_overlay = self._ui.show_overlay(
            selector, OverlayOptions(anchor="center", max_height="80%")
        )
        self._ui.set_focus(selector)

    def _show_session_selector(self) -> None:
        if self._ui is None:
            return

        session_manager = self._session.session_manager
        cwd = session_manager.get_cwd()
        session_dir = session_manager.get_session_dir()

        def close_overlay() -> None:
            self._close_overlay("_sessions_overlay")

        async def on_select(session_path: str) -> None:
            close_overlay()
            try:
                result = await self._runtime_host.switch_session(session_path)
                if result.get("cancelled"):
                    self._show_status(theme.fg("warning", "Resume cancelled"))
                    return
                if self._chat_container is not None:
                    self._chat_container.clear()
                self._show_status(theme.fg("success", "Resumed session"))
            except Exception as error:
                self._show_status(theme.fg("error", str(error)))

        def handle_select(session_path: str) -> None:
            asyncio.create_task(on_select(session_path))

        selector = SessionSelectorComponent(
            self._ui,
            lambda on_progress=None: SessionManager.list(cwd, session_dir, on_progress),
            lambda on_progress=None: (
                SessionManager.list_all(on_progress)
                if session_manager.uses_default_session_dir()
                else SessionManager.list_all(session_dir, on_progress)
            ),
            handle_select,
            close_overlay,
            current_session_path=self._session.session_file,
        )
        self._sessions_overlay = self._ui.show_overlay(
            selector, OverlayOptions(anchor="center", max_height="80%")
        )
        self._ui.set_focus(selector)

    def _show_error(self, text: str) -> None:
        self._show_status(theme.fg("error", text))

    def _restore_editor(self) -> None:
        if self._editor_container is None or self._editor is None or self._ui is None:
            return
        self._editor_container.clear()
        self._editor_container.add_child(self._editor)
        self._ui.set_focus(self._editor)
        self._ui.request_render()

    def _show_selector(
        self,
        create: Callable[[Callable[[], None]], tuple[Container, Container]],
    ) -> None:
        if self._editor_container is None or self._editor is None or self._ui is None:
            return

        def done() -> None:
            self._restore_editor()

        component, focus = create(done)
        self._editor_container.clear()
        self._editor_container.add_child(component)
        self._ui.set_focus(focus)
        self._ui.request_render()

    def _get_login_provider_options(
        self, auth_type: Literal["oauth", "api_key"] | None = None
    ) -> list[AuthSelectorProvider]:
        auth_storage = self._session.model_registry.auth_storage
        oauth_providers = auth_storage.get_oauth_providers()
        oauth_provider_ids = {provider.id for provider in oauth_providers}
        options: list[AuthSelectorProvider] = [
            AuthSelectorProvider(id=provider.id, name=provider.name, auth_type="oauth")
            for provider in oauth_providers
        ]

        model_providers = {
            model.get("provider", "") for model in self._session.model_registry.get_all()
        }
        for provider_id in model_providers:
            if not provider_id or not is_api_key_login_provider(provider_id, oauth_provider_ids):
                continue
            options.append(
                AuthSelectorProvider(
                    id=provider_id,
                    name=self._session.model_registry.get_provider_display_name(provider_id),
                    auth_type="api_key",
                )
            )

        if auth_type is not None:
            options = [option for option in options if option.auth_type == auth_type]
        return sorted(options, key=lambda option: option.name.lower())

    def _get_logout_provider_options(self) -> list[AuthSelectorProvider]:
        auth_storage = self._session.model_registry.auth_storage
        options: list[AuthSelectorProvider] = []
        for provider_id in auth_storage.list():
            credential = auth_storage.get(provider_id)
            if not credential:
                continue
            options.append(
                AuthSelectorProvider(
                    id=provider_id,
                    name=self._session.model_registry.get_provider_display_name(provider_id),
                    auth_type=credential.get("type", "api_key"),
                )
            )
        return sorted(options, key=lambda option: option.name.lower())

    def _show_login_auth_type_selector(self) -> None:
        subscription_label = "Use a subscription"
        api_key_label = "Use an API key"

        def create(done: Callable[[], None]) -> tuple[Container, Container]:
            def handle_select(option: str) -> None:
                done()
                auth_type: Literal["oauth", "api_key"] = (
                    "oauth" if option == subscription_label else "api_key"
                )
                self._show_login_provider_selector(auth_type)

            selector = ExtensionSelectorComponent(
                "Select authentication method:",
                [subscription_label, api_key_label],
                handle_select,
                done,
            )
            return selector, selector

        self._show_selector(create)

    def _show_login_provider_selector(self, auth_type: Literal["oauth", "api_key"]) -> None:
        provider_options = self._get_login_provider_options(auth_type)
        if not provider_options:
            self._show_status(
                theme.fg(
                    "warning",
                    (
                        "No subscription providers available."
                        if auth_type == "oauth"
                        else "No API key providers available."
                    ),
                )
            )
            return

        auth_storage = self._session.model_registry.auth_storage

        def create(done: Callable[[], None]) -> tuple[Container, Container]:
            def handle_select(provider_id: str) -> None:
                done()
                provider_option = next(
                    (provider for provider in provider_options if provider.id == provider_id),
                    None,
                )
                if provider_option is None:
                    return
                if provider_option.auth_type == "oauth":
                    asyncio.create_task(
                        self._show_login_dialog(provider_option.id, provider_option.name)
                    )
                elif provider_option.id == BEDROCK_PROVIDER_ID:
                    self._show_bedrock_setup_dialog(provider_option.id, provider_option.name)
                else:
                    asyncio.create_task(
                        self._show_api_key_login_dialog(provider_option.id, provider_option.name)
                    )

            def handle_cancel() -> None:
                done()
                self._show_login_auth_type_selector()

            selector = OAuthSelectorComponent(
                "login",
                auth_storage,
                provider_options,
                handle_select,
                handle_cancel,
                self._session.model_registry.get_provider_auth_status,
            )
            return selector, selector

        self._show_selector(create)

    def _show_oauth_selector(self, mode: Literal["login", "logout"]) -> None:
        if mode == "login":
            self._show_login_auth_type_selector()
            return

        provider_options = self._get_logout_provider_options()
        if not provider_options:
            self._show_status(
                theme.fg(
                    "warning",
                    "No stored credentials to remove. /logout only removes credentials saved by /login; "
                    "environment variables and models.json config are unchanged.",
                )
            )
            return

        auth_storage = self._session.model_registry.auth_storage

        def create(done: Callable[[], None]) -> tuple[Container, Container]:
            async def handle_logout(provider_id: str) -> None:
                done()
                provider_option = next(
                    (provider for provider in provider_options if provider.id == provider_id),
                    None,
                )
                if provider_option is None:
                    return
                try:
                    auth_storage.logout(provider_option.id)
                    self._session.model_registry.refresh()
                    if provider_option.auth_type == "oauth":
                        message = f"Logged out of {provider_option.name}"
                    else:
                        message = (
                            f"Removed stored API key for {provider_option.name}. "
                            "Environment variables and models.json config are unchanged."
                        )
                    self._show_status(theme.fg("success", message))
                    if self._footer is not None:
                        self._footer.invalidate()
                except Exception as error:
                    self._show_error(f"Logout failed: {error}")

            def handle_select(provider_id: str) -> None:
                asyncio.create_task(handle_logout(provider_id))

            selector = OAuthSelectorComponent(
                "logout",
                auth_storage,
                provider_options,
                handle_select,
                done,
            )
            return selector, selector

        self._show_selector(create)

    async def _complete_provider_authentication(
        self,
        provider_id: str,
        provider_name: str,
        auth_type: Literal["oauth", "api_key"],
        previous_model: Model[Any] | None,
    ) -> None:
        self._session.model_registry.refresh()
        action_label = (
            f"Logged in to {provider_name}"
            if auth_type == "oauth"
            else f"Saved API key for {provider_name}"
        )

        selected_model: Model[Any] | None = None
        selection_error: str | None = None
        if _is_unknown_model(previous_model):
            available_models = self._session.model_registry.get_available()
            provider_models = [
                model for model in available_models if model.get("provider") == provider_id
            ]
            if provider_id not in default_model_per_provider:
                selection_error = (
                    f'{action_label}, but no default model is configured for provider "{provider_id}". '
                    "Use /model to select a model."
                )
            elif not provider_models:
                selection_error = f"{action_label}, but no models are available for that provider. Use /model to select a model."
            else:
                default_model_id = default_model_per_provider[provider_id]
                selected_model = next(
                    (model for model in provider_models if model.get("id") == default_model_id),
                    None,
                )
                if selected_model is None:
                    selection_error = (
                        f'{action_label}, but its default model "{default_model_id}" is not available. '
                        "Use /model to select a model."
                    )
                else:
                    try:
                        await self._session.set_model(selected_model)
                    except Exception as error:
                        selected_model = None
                        selection_error = (
                            f"{action_label}, but selecting its default model failed: {error}. "
                            "Use /model to select a model."
                        )

        if self._footer is not None:
            self._footer.invalidate()
        if selected_model is not None:
            self._show_status(
                theme.fg(
                    "success",
                    f"{action_label}. Selected {selected_model.get('id')}. Credentials saved to {get_auth_path()}",
                )
            )
        else:
            self._show_status(
                theme.fg("success", f"{action_label}. Credentials saved to {get_auth_path()}")
            )
            if selection_error:
                self._show_error(selection_error)

    def _show_bedrock_setup_dialog(self, provider_id: str, provider_name: str) -> None:
        if self._editor_container is None or self._ui is None:
            return

        dialog = LoginDialogComponent(
            self._ui,
            provider_id,
            lambda _success, _message: self._restore_editor(),
            provider_name=provider_name,
            title="Amazon Bedrock setup",
        )
        dialog.show_info(
            [
                theme.fg(
                    "text", "Amazon Bedrock uses AWS credentials instead of a single API key."
                ),
                theme.fg(
                    "text",
                    "Configure an AWS profile, IAM keys, bearer token, or role-based credentials.",
                ),
                theme.fg("muted", "See:"),
                theme.fg("accent", f"  {Path(get_docs_path()) / 'providers.md'}"),
            ]
        )
        self._editor_container.clear()
        self._editor_container.add_child(dialog)
        self._ui.set_focus(dialog)
        self._ui.request_render()

    async def _show_api_key_login_dialog(self, provider_id: str, provider_name: str) -> None:
        if self._ui is None or self._editor_container is None:
            return

        previous_model = self._session.model
        dialog = LoginDialogComponent(
            self._ui,
            provider_id,
            lambda _success, _message: None,
            provider_name=provider_name,
        )
        self._editor_container.clear()
        self._editor_container.add_child(dialog)
        self._ui.set_focus(dialog)
        self._ui.request_render()

        try:
            api_key = (await dialog.show_prompt("Enter API key:")).strip()
            if not api_key:
                raise ValueError("API key cannot be empty.")
            self._session.model_registry.auth_storage.set(
                provider_id, {"type": "api_key", "key": api_key}
            )
            self._restore_editor()
            await self._complete_provider_authentication(
                provider_id, provider_name, "api_key", previous_model
            )
        except Exception as error:
            self._restore_editor()
            error_msg = str(error)
            if error_msg != "Login cancelled":
                self._show_error(f"Failed to save API key for {provider_name}: {error_msg}")

    def _show_oauth_login_select(
        self,
        dialog: LoginDialogComponent,
        prompt: OAuthSelectPrompt,
    ) -> asyncio.Future[str | None]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str | None] = loop.create_future()

        def restore_dialog() -> None:
            if self._editor_container is None or self._ui is None:
                return
            self._editor_container.clear()
            self._editor_container.add_child(dialog)
            self._ui.set_focus(dialog)
            self._ui.request_render()

        labels = [option["label"] for option in prompt["options"]]

        def handle_select(option_label: str) -> None:
            restore_dialog()
            selected = next(
                (option["id"] for option in prompt["options"] if option["label"] == option_label),
                None,
            )
            if not future.done():
                future.set_result(selected)

        def handle_cancel() -> None:
            restore_dialog()
            if not future.done():
                future.set_result(None)

        selector = ExtensionSelectorComponent(
            prompt["message"], labels, handle_select, handle_cancel
        )
        if self._editor_container is None or self._ui is None:
            if not future.done():
                future.set_result(None)
            return future
        self._editor_container.clear()
        self._editor_container.add_child(selector)
        self._ui.set_focus(selector)
        self._ui.request_render()
        return future

    async def _show_login_dialog(self, provider_id: str, provider_name: str) -> None:
        if self._ui is None or self._editor_container is None:
            return

        provider_info = next(
            (
                provider
                for provider in self._session.model_registry.auth_storage.get_oauth_providers()
                if provider.id == provider_id
            ),
            None,
        )
        previous_model = self._session.model
        uses_callback_server = bool(getattr(provider_info, "uses_callback_server", False))

        dialog = LoginDialogComponent(
            self._ui,
            provider_id,
            lambda _success, _message: None,
            provider_name=provider_name,
        )
        self._editor_container.clear()
        self._editor_container.add_child(dialog)
        self._ui.set_focus(dialog)
        self._ui.request_render()

        loop = asyncio.get_running_loop()
        manual_code_future: asyncio.Future[str] = loop.create_future()

        def resolve_manual(value: str) -> None:
            if not manual_code_future.done():
                manual_code_future.set_result(value)

        def reject_manual(error: BaseException) -> None:
            if not manual_code_future.done():
                manual_code_future.set_exception(error)

        class _LoginCallbacks(OAuthLoginCallbacks):
            def on_auth(self, info: OAuthAuthInfo) -> None:
                dialog.show_auth(info.get("url", ""), info.get("instructions"))
                if uses_callback_server:

                    async def wait_for_manual() -> None:
                        try:
                            value = await dialog.show_manual_input(
                                "Paste redirect URL below, or complete login in browser:"
                            )
                            if value:
                                resolve_manual(value)
                        except Exception as error:
                            reject_manual(
                                error
                                if isinstance(error, BaseException)
                                else RuntimeError(str(error))
                            )

                    asyncio.create_task(wait_for_manual())

            def on_device_code(self, info: OAuthDeviceCodeInfo) -> None:
                dialog.show_device_code(info)
                dialog.show_waiting("Waiting for authentication...")

            async def on_prompt(self, prompt: dict[str, str]) -> str:
                return await dialog.show_prompt(prompt["message"], prompt.get("placeholder"))

            def on_progress(self, message: str) -> None:
                dialog.show_progress(message)

            async def on_select(self, prompt: OAuthSelectPrompt) -> str | None:
                return await self_outer._show_oauth_login_select(dialog, prompt)

            async def on_manual_code_input(self) -> str:
                return await manual_code_future

            @property
            def signal(self) -> Any:
                return dialog.signal

        self_outer = self
        callbacks = _LoginCallbacks()

        try:
            await self._session.model_registry.auth_storage.login(provider_id, callbacks)
            self._restore_editor()
            await self._complete_provider_authentication(
                provider_id, provider_name, "oauth", previous_model
            )
        except Exception as error:
            self._restore_editor()
            error_msg = str(error)
            if error_msg != "Login cancelled":
                self._show_error(f"Failed to login to {provider_name}: {error_msg}")

    def _append_system_message(self, text: str) -> None:
        if self._chat_container is None:
            return
        self._chat_container.add_child(Text(theme.fg("muted", text), padding_x=1, padding_y=0))

    async def _handle_prompt(self, text: str, images: list[ImageContent] | None = None) -> None:
        try:
            from pi_mono.coding_agent.core.agent_session import PromptOptions

            await self._session.prompt(text, PromptOptions(images=images))
        except Exception as error:
            self._show_status(theme.fg("error", str(error)))

    async def _shutdown(self, *, from_signal: bool = False, signum: int | None = None) -> None:
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        await self.stop()
        await self._runtime_host.dispose()
        if from_signal and signum is not None:
            raise SystemExit(129 if signum == signal.SIGHUP else 143)


async def run_interactive_mode(
    runtime_host: AgentSessionRuntime,
    options: InteractiveModeOptions | None = None,
) -> None:
    mode = InteractiveMode(runtime_host, options)
    try:
        await mode.run()
    finally:
        if not mode._is_shutting_down:
            await mode.stop()
            await runtime_host.dispose()
