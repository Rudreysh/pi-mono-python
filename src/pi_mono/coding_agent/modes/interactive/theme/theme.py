"""Simplified theme support for the interactive TUI."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pi_mono.tui.components.editor import EditorTheme
from pi_mono.tui.components.markdown import MarkdownTheme
from pi_mono.tui.components.select_list import SelectListTheme
from pi_mono.tui.components.settings_list import SettingsListTheme

THEMES_DIR = Path(__file__).resolve().parent

ThemeColorFn = Callable[[str], str]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    cleaned = hex_color.lstrip("#")
    if len(cleaned) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16)


def _fg_ansi(color: str | int) -> str:
    if color == "":
        return "\x1b[39m"
    if isinstance(color, int):
        return f"\x1b[38;5;{color}m"
    if isinstance(color, str) and color.startswith("#"):
        r, g, b = _hex_to_rgb(color)
        return f"\x1b[38;2;{r};{g};{b}m"
    raise ValueError(f"Invalid color value: {color}")


def _bg_ansi(color: str | int) -> str:
    if color == "":
        return "\x1b[49m"
    if isinstance(color, int):
        return f"\x1b[48;5;{color}m"
    if isinstance(color, str) and color.startswith("#"):
        r, g, b = _hex_to_rgb(color)
        return f"\x1b[48;2;{r};{g};{b}m"
    raise ValueError(f"Invalid color value: {color}")


def _resolve_var_refs(
    value: str | int,
    vars_map: dict[str, str | int],
    visited: set[str] | None = None,
) -> str | int:
    if isinstance(value, int) or value == "" or (isinstance(value, str) and value.startswith("#")):
        return value
    seen = visited or set()
    if value in seen:
        raise ValueError(f"Circular variable reference detected: {value}")
    if value not in vars_map:
        raise ValueError(f"Variable reference not found: {value}")
    seen.add(value)
    return _resolve_var_refs(vars_map[value], vars_map, seen)


def _resolve_theme_colors(
    colors: dict[str, str | int],
    vars_map: dict[str, str | int],
) -> dict[str, str | int]:
    return {key: _resolve_var_refs(value, vars_map) for key, value in colors.items()}


class Theme:
    """Terminal color theme."""

    def __init__(
        self,
        fg_colors: dict[str, str | int],
        bg_colors: dict[str, str | int],
        *,
        name: str | None = None,
    ) -> None:
        self.name = name
        self._fg_ansi = {key: _fg_ansi(value) for key, value in fg_colors.items()}
        self._bg_ansi = {key: _bg_ansi(value) for key, value in bg_colors.items()}

    def fg(self, color: str, text: str) -> str:
        ansi = self._fg_ansi.get(color)
        if ansi is None:
            raise KeyError(f"Unknown theme color: {color}")
        return f"{ansi}{text}\x1b[39m"

    def fg_fn(self, color: str) -> ThemeColorFn:
        return lambda text: self.fg(color, text)

    def bg(self, color: str, text: str) -> str:
        ansi = self._bg_ansi.get(color)
        if ansi is None:
            raise KeyError(f"Unknown theme background color: {color}")
        return f"{ansi}{text}\x1b[49m"

    def bg_fn(self, color: str) -> ThemeColorFn:
        return lambda text: self.bg(color, text)

    def bold(self, text: str) -> str:
        return f"\x1b[1m{text}\x1b[22m"

    def italic(self, text: str) -> str:
        return f"\x1b[3m{text}\x1b[23m"

    def underline(self, text: str) -> str:
        return f"\x1b[4m{text}\x1b[24m"


def _load_theme_json(path: Path) -> Theme:
    with path.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    vars_map = data.get("vars", {})
    colors = _resolve_theme_colors(data["colors"], vars_map)
    fg_colors = {key: value for key, value in colors.items() if not key.endswith("Bg")}
    bg_keys = (
        "selectedBg",
        "userMessageBg",
        "customMessageBg",
        "toolPendingBg",
        "toolSuccessBg",
        "toolErrorBg",
    )
    bg_colors = {key: colors[key] for key in bg_keys if key in colors}
    return Theme(fg_colors, bg_colors, name=data.get("name"))


def get_theme_by_name(name: str) -> Theme:
    path = THEMES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Theme not found: {name}")
    return _load_theme_json(path)


def init_theme(name: str = "dark") -> Theme:
    global theme
    theme = get_theme_by_name(name)
    return theme


def get_available_themes() -> list[str]:
    return sorted(path.stem for path in THEMES_DIR.glob("*.json"))


def get_markdown_theme() -> MarkdownTheme:
    return MarkdownTheme(
        heading=theme.fg_fn("mdHeading"),
        link=theme.fg_fn("mdLink"),
        link_url=theme.fg_fn("mdLinkUrl"),
        code=theme.fg_fn("mdCode"),
        code_block=theme.fg_fn("mdCodeBlock"),
        code_block_border=theme.fg_fn("mdCodeBlockBorder"),
        quote=theme.fg_fn("mdQuote"),
        quote_border=theme.fg_fn("mdQuoteBorder"),
        hr=theme.fg_fn("mdHr"),
        list_bullet=theme.fg_fn("mdListBullet"),
        bold=theme.bold,
        italic=theme.italic,
        strikethrough=lambda text: f"\x1b[9m{text}\x1b[29m",
        underline=theme.underline,
    )


def get_select_list_theme() -> SelectListTheme:
    return SelectListTheme(
        selected_prefix=theme.fg_fn("accent"),
        selected_text=theme.fg_fn("text"),
        description=theme.fg_fn("muted"),
        scroll_info=theme.fg_fn("dim"),
        no_match=theme.fg_fn("error"),
    )


def get_settings_list_theme() -> SettingsListTheme:
    return SettingsListTheme(
        label=lambda text, selected: theme.fg("accent", text) if selected else text,
        value=lambda text, selected: (
            theme.fg("accent", text) if selected else theme.fg("muted", text)
        ),
        description=lambda text: theme.fg("dim", text),
        cursor=theme.fg("accent", "→ "),
        hint=lambda text: theme.fg("dim", text),
    )


def get_editor_theme() -> EditorTheme:
    return EditorTheme(
        border_color=theme.fg_fn("border"),
        select_list=get_select_list_theme(),
    )


_default_theme_name = "dark"
if os.environ.get("COLORFGBG", "").endswith(";15") or os.environ.get("PI_THEME") == "light":
    _default_theme_name = "light"

theme = init_theme(_default_theme_name)
