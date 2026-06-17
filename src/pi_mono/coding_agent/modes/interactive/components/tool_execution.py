"""Simplified tool execution display for interactive mode."""

from __future__ import annotations

import json
from typing import Any

from pi_mono.coding_agent.modes.interactive.theme.theme import theme
from pi_mono.tui.components.box import Box
from pi_mono.tui.components.spacer import Spacer
from pi_mono.tui.components.text import Text
from pi_mono.tui.tui import Container


def _truncate_output(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


class ToolExecutionComponent(Container):
    """Show tool name and truncated output."""

    def __init__(
        self,
        tool_name: str,
        tool_call_id: str,
        args: Any,
    ) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._tool_call_id = tool_call_id
        self._args = args
        self._expanded = False
        self._is_partial = True
        self._result: dict[str, Any] | None = None
        self._content_box = Box(1, 1, theme.bg_fn("toolPendingBg"))
        self.add_child(Spacer(1))
        self.add_child(self._content_box)
        self._update_display()

    @property
    def tool_call_id(self) -> str:
        return self._tool_call_id

    def update_args(self, args: Any) -> None:
        self._args = args
        self._update_display()

    def mark_execution_started(self) -> None:
        self._update_display()

    def set_args_complete(self) -> None:
        self._update_display()

    def update_result(self, result: dict[str, Any], is_partial: bool = False) -> None:
        self._result = result
        self._is_partial = is_partial
        self._update_display()

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._update_display()

    def invalidate(self) -> None:
        super().invalidate()
        self._update_display()

    def _get_text_output(self) -> str:
        if not self._result:
            return ""
        parts: list[str] = []
        for block in self._result.get("content", []):
            if block.get("type") == "text" and block.get("text"):
                parts.append(str(block["text"]))
        return "\n".join(parts)

    def _update_display(self) -> None:
        if self._is_partial:
            bg_fn = theme.bg_fn("toolPendingBg")
        elif self._result and self._result.get("isError"):
            bg_fn = theme.bg_fn("toolErrorBg")
        else:
            bg_fn = theme.bg_fn("toolSuccessBg")

        self._content_box.set_bg_fn(bg_fn)
        self._content_box.clear()

        title = theme.fg("toolTitle", theme.bold(self._tool_name))
        self._content_box.add_child(Text(title, padding_x=0, padding_y=0))

        args_text = json.dumps(self._args, indent=2) if self._args else ""
        if args_text and self._expanded:
            self._content_box.add_child(
                Text(theme.fg("muted", args_text), padding_x=0, padding_y=0)
            )

        output = self._get_text_output()
        if output:
            limit = 2000 if self._expanded else 500
            self._content_box.add_child(
                Text(
                    theme.fg("toolOutput", _truncate_output(output, limit)),
                    padding_x=0,
                    padding_y=0,
                )
            )
