"""Regression: Editor must accept printable input."""

from __future__ import annotations

from pi_mono.tui.components.editor import Editor, EditorOptions


class _FakeTheme:
    border_color = staticmethod(lambda text: text)


class _FakeTui:
    class terminal:
        rows = 24

    def request_render(self) -> None:
        return None


def test_editor_accepts_printable_input() -> None:
    editor = Editor(_FakeTui(), _FakeTheme(), EditorOptions())
    editor.focused = True
    for char in "hello":
        editor.handle_input(char)
    assert editor.get_text() == "hello"
