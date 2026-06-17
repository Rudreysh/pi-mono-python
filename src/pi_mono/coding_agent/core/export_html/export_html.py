"""HTML export from session files using the shared template bundle."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

from pi_mono.config import APP_NAME, get_export_template_dir
from pi_mono.core.session_manager import SessionManager
from pi_mono.utils.paths import normalize_path, resolve_path


def _escape(text: str) -> str:
    return html.escape(text, quote=True)


def _message_text(message: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in message.get("content", []):
        if block.get("type") == "text" and block.get("text"):
            parts.append(str(block["text"]))
        elif block.get("type") == "thinking" and block.get("thinking"):
            parts.append(f"[thinking] {block['thinking']}")
    return "\n".join(parts)


def _render_message_html(message: dict[str, Any]) -> str:
    role = message.get("role", "unknown")
    text = _message_text(message)
    css_class = {
        "user": "user-message",
        "assistant": "assistant-message",
        "toolResult": "tool-message",
    }.get(role, "message")
    title = role
    if role == "toolResult":
        title = f"tool: {message.get('toolName', 'unknown')}"
    return (
        f'<section class="{css_class}">'
        f"<h3>{_escape(title)}</h3>"
        f"<pre>{_escape(text)}</pre>"
        f"</section>"
    )


def _generate_theme_vars(theme_name: str | None = None) -> str:
    from pi_mono.coding_agent.modes.interactive.theme.theme import THEMES_DIR

    name = theme_name or "dark"
    theme_path = THEMES_DIR / f"{name}.json"
    if not theme_path.exists():
        theme_path = THEMES_DIR / "dark.json"
    with theme_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    vars_map = data.get("vars", {})
    colors = data.get("colors", {})
    lines: list[str] = []
    for key, ref in colors.items():
        value = vars_map.get(ref, ref) if isinstance(ref, str) else ref
        if isinstance(value, str):
            lines.append(f"--{key}: {value};")
    lines.append("--exportPageBg: rgb(24, 24, 30);")
    lines.append("--exportCardBg: rgb(30, 30, 36);")
    lines.append("--exportInfoBg: rgb(60, 55, 40);")
    return "\n      ".join(lines)


def _template_assets_available() -> bool:
    template_dir = get_export_template_dir()
    return (template_dir / "template.html").exists()


def _generate_template_html(session_data: dict[str, Any], theme_name: str | None = None) -> str:
    template_dir = get_export_template_dir()
    template = (template_dir / "template.html").read_text(encoding="utf-8")
    template_css = (template_dir / "template.css").read_text(encoding="utf-8")
    template_js = (template_dir / "template.js").read_text(encoding="utf-8")
    marked_js = (template_dir / "vendor" / "marked.min.js").read_text(encoding="utf-8")
    hljs_js = (template_dir / "vendor" / "highlight.min.js").read_text(encoding="utf-8")

    theme_vars = _generate_theme_vars(theme_name)
    session_data_base64 = base64.b64encode(json.dumps(session_data).encode("utf-8")).decode("ascii")

    css = (
        template_css.replace("{{THEME_VARS}}", theme_vars)
        .replace("{{BODY_BG}}", "rgb(24, 24, 30)")
        .replace("{{CONTAINER_BG}}", "rgb(30, 30, 36)")
        .replace("{{INFO_BG}}", "rgb(60, 55, 40)")
    )

    return (
        template.replace("{{CSS}}", css)
        .replace("{{JS}}", template_js)
        .replace("{{SESSION_DATA}}", session_data_base64)
        .replace("{{MARKED_JS}}", marked_js)
        .replace("{{HIGHLIGHT_JS}}", hljs_js)
    )


def _generate_minimal_html(session_data: dict[str, Any]) -> str:
    header = session_data.get("header") or {}
    entries = session_data.get("entries") or []
    title = header.get("id") or "Session"
    cwd = header.get("cwd") or ""

    message_sections: list[str] = []
    for entry in entries:
        if entry.get("type") != "message":
            continue
        message = entry.get("message")
        if message:
            message_sections.append(_render_message_html(message))

    body = "\n".join(message_sections) if message_sections else "<p>(no messages)</p>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_escape(str(title))}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #1e1e24; color: #d4d4d4; }}
    .user-message pre {{ background: #343541; padding: 1rem; border-radius: 8px; white-space: pre-wrap; }}
    .assistant-message pre {{ background: #2a2a32; padding: 1rem; border-radius: 8px; white-space: pre-wrap; }}
    .tool-message pre {{ background: #283228; padding: 1rem; border-radius: 8px; white-space: pre-wrap; }}
    h1, h3 {{ color: #8abeb7; }}
    .meta {{ color: #808080; margin-bottom: 2rem; }}
  </style>
</head>
<body>
  <h1>{_escape(APP_NAME)} session export</h1>
  <p class="meta">cwd: {_escape(str(cwd))}</p>
  {body}
</body>
</html>
"""


def _build_session_data_from_manager(session_manager: SessionManager) -> dict[str, Any]:
    entries = session_manager.get_entries()
    header = next((entry for entry in entries if entry.get("type") == "session"), None)
    return {
        "header": header,
        "entries": entries,
        "leafId": session_manager.leafId,
    }


def generate_html(session_data: dict[str, Any], theme_name: str | None = None) -> str:
    if _template_assets_available():
        return _generate_template_html(session_data, theme_name)
    return _generate_minimal_html(session_data)


def export_session_to_html(
    session: Any,
    output_path: str | None = None,
    *,
    theme_name: str | None = None,
) -> str:
    session_data = _build_session_data_from_manager(session.session_manager)
    session_data["systemPrompt"] = session.system_prompt
    session_data["tools"] = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in session.agent.state.tools
    ]
    html_content = generate_html(session_data, theme_name)

    if output_path:
        resolved_output = normalize_path(output_path)
    else:
        session_id = session.session_id or "session"
        resolved_output = f"{APP_NAME}-session-{session_id}.html"

    Path(resolved_output).write_text(html_content, encoding="utf-8")
    return resolved_output


def export_from_file(
    session_path: str, output_path: str | None = None, *, theme_name: str | None = None
) -> str:
    """Export a JSONL session file to HTML."""
    resolved_input = resolve_path(session_path)
    input_file = Path(resolved_input)
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {resolved_input}")

    session_manager = SessionManager.open(resolved_input)
    session_data = _build_session_data_from_manager(session_manager)
    html_content = generate_html(session_data, theme_name)

    if output_path:
        resolved_output = normalize_path(output_path)
    else:
        resolved_output = f"{APP_NAME}-session-{input_file.stem}.html"

    Path(resolved_output).write_text(html_content, encoding="utf-8")
    return resolved_output
