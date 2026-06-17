"""Model selector overlay for interactive mode."""

from __future__ import annotations

from typing import Any, Callable

from pi_mono.ai.models import models_are_equal
from pi_mono.ai.types import Model
from pi_mono.coding_agent.modes.interactive.theme.theme import get_editor_theme, theme
from pi_mono.core.model_registry import ModelRegistry
from pi_mono.core.settings_manager import SettingsManager
from pi_mono.tui.components.input import Input
from pi_mono.tui.components.select_list import SelectItem, SelectList
from pi_mono.tui.components.spacer import Spacer
from pi_mono.tui.components.text import Text
from pi_mono.tui.fuzzy import fuzzy_filter
from pi_mono.tui.keybindings import get_keybindings
from pi_mono.tui.tui import Container, TUI


class ModelSelectorComponent(Container):
    """SelectList-based model selector overlay."""

    def __init__(
        self,
        ui: TUI,
        current_model: Model[Any] | None,
        settings_manager: SettingsManager,
        model_registry: ModelRegistry,
        on_select: Callable[[Model[Any]], None],
        on_cancel: Callable[[], None],
        *,
        initial_search: str | None = None,
    ) -> None:
        super().__init__()
        self._ui = ui
        self._current_model = current_model
        self._settings_manager = settings_manager
        self._model_registry = model_registry
        self._on_select = on_select
        self._on_cancel = on_cancel
        self._all_models: list[Model[Any]] = []
        self._filtered_models: list[Model[Any]] = []
        self._error_message: str | None = None

        self.add_child(Text(theme.fg("accent", "Select model"), padding_x=1, padding_y=0))
        self.add_child(Spacer(1))

        self._search_input = Input()
        if initial_search:
            self._search_input.set_value(initial_search)
        self._search_input.on_submit = self._select_first_filtered
        self.add_child(self._search_input)
        self.add_child(Spacer(1))

        editor_theme = get_editor_theme()
        self._select_list = SelectList([], 12, editor_theme.select_list)
        self._select_list.on_select = self._handle_select_item
        self._select_list.on_cancel = on_cancel
        self.add_child(self._select_list)

        self._load_models()
        if initial_search:
            self._filter_models(initial_search)
        else:
            self._update_list()

    @property
    def focused(self) -> bool:
        return self._search_input.focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._search_input.focused = value

    def _load_models(self) -> None:
        self._model_registry.refresh()
        self._error_message = self._model_registry.get_error()
        try:
            self._all_models = self._model_registry.get_available()
        except Exception as error:
            self._all_models = []
            self._error_message = str(error)
        self._all_models.sort(
            key=lambda model: (
                0 if models_are_equal(self._current_model, model) else 1,
                str(model.get("provider", "")),
                str(model.get("id", "")),
            )
        )
        self._filtered_models = list(self._all_models)
        current_index = next(
            (
                index
                for index, model in enumerate(self._filtered_models)
                if models_are_equal(self._current_model, model)
            ),
            0,
        )
        self._select_list.set_selected_index(current_index)
        self._update_list()

    def _filter_models(self, query: str) -> None:
        if query.strip():
            self._filtered_models = fuzzy_filter(
                self._all_models,
                query,
                lambda model: (
                    f"{model.get('id', '')} {model.get('provider', '')} "
                    f"{model.get('provider', '')}/{model.get('id', '')}"
                ),
            )
        else:
            self._filtered_models = list(self._all_models)
        self._update_list()

    def _update_list(self) -> None:
        items: list[SelectItem] = []
        for model in self._filtered_models:
            provider = str(model.get("provider", ""))
            model_id = str(model.get("id", ""))
            label = model_id
            if models_are_equal(self._current_model, model):
                label = f"{model_id} ✓"
            description = f"[{provider}]"
            items.append(
                SelectItem(value=f"{provider}/{model_id}", label=label, description=description)
            )
        self._select_list._items = items  # noqa: SLF001
        self._select_list._filtered_items = list(items)  # noqa: SLF001
        self._select_list.set_selected_index(0)
        self._ui.request_render()

    def _select_first_filtered(self) -> None:
        selected = self._select_list.get_selected_item()
        if selected:
            provider, model_id = selected.value.split("/", 1)
            model = self._model_registry.find(provider, model_id)
            if model:
                self._handle_select(model)

    def _handle_select_item(self, item: SelectItem) -> None:
        provider, model_id = item.value.split("/", 1)
        model = self._model_registry.find(provider, model_id)
        if model:
            self._handle_select(model)

    def _handle_select(self, model: Model[Any]) -> None:
        self._settings_manager.set_default_model_and_provider(
            str(model.get("provider", "")),
            str(model.get("id", "")),
        )
        self._on_select(model)

    def handle_input(self, data: str) -> None:
        kb = get_keybindings()
        if kb.matches(data, "tui.select.up") or kb.matches(data, "tui.select.down"):
            self._select_list.handle_input(data)
            return
        if kb.matches(data, "tui.select.confirm"):
            self._select_list.handle_input(data)
            return
        if kb.matches(data, "tui.select.cancel"):
            self._on_cancel()
            return
        self._search_input.handle_input(data)
        self._filter_models(self._search_input.get_value())
