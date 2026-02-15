"""Modal screen for switching models."""

from dataclasses import dataclass

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Label, Static

from vox.config import LLM_MODELS, STT_MODELS, TTS_MODELS
from vox.themes import PALETTE_2, PALETTE_4, PALETTE_8


@dataclass(slots=True)
class ModelSelection:
    """Result of model selection dialog."""

    stt_model: str
    llm_model: str
    tts_model: str


class ModelItem(Static, can_focus=True):
    """Single model item in the list."""

    def __init__(
        self,
        model_name: str,
        is_selected: bool,
        is_loaded: bool,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.is_selected = is_selected
        self.is_loaded = is_loaded

    def render(self) -> Text:
        """Render model item with appropriate styling."""
        text = Text()

        if self.has_focus:
            text.append("▸ ", style=PALETTE_4)
        else:
            text.append("  ")

        style = f"bold {PALETTE_4}" if self.has_focus else PALETTE_8
        text.append(self.model_name, style=style)

        if self.is_loaded:
            text.append(" ✓", style=PALETTE_2)

        return text

    def update_state(self, is_selected: bool, is_loaded: bool) -> None:
        """Update item state and re-render."""
        self.is_selected = is_selected
        self.is_loaded = is_loaded
        self.refresh()

    def on_focus(self) -> None:
        """Called when item gains focus - trigger re-render."""
        self.refresh()

    def on_blur(self) -> None:
        """Called when item loses focus - trigger re-render."""
        self.refresh()


class ModelSelector(ModalScreen[ModelSelection | None]):
    """Modal screen for selecting models across STT, LLM, and TTS."""

    class Changed(Message):
        """Posted when model selection changes."""

        def __init__(self, selection: ModelSelection) -> None:
            super().__init__()
            self.selection = selection

    BINDINGS = [
        ("escape", "cancel", "Close"),
        ("enter", "select", "Select"),
        ("up,k", "cursor_up", "Up"),
        ("down,j", "cursor_down", "Down"),
    ]

    def __init__(
        self,
        current_stt: str,
        current_llm: str,
        current_tts: str,
    ) -> None:
        """Initialize with current model selections."""
        super().__init__()
        self._current_stt = current_stt
        self._current_llm = current_llm
        self._current_tts = current_tts
        self._selected_stt = current_stt
        self._selected_llm = current_llm
        self._selected_tts = current_tts
        self._all_models: list[tuple[str, str]] = []
        self._cursor = 0

    def compose(self) -> ComposeResult:
        """Compose the modal layout."""
        with Vertical(id="model-dialog"), VerticalScroll(id="model-list"):
            yield Label("STT", classes="model-header")
            for model in STT_MODELS:
                self._all_models.append(("stt", model))
                yield ModelItem(
                    model,
                    is_selected=model == self._selected_stt,
                    is_loaded=model == self._current_stt,
                )

            yield Label("LLM", classes="model-header")
            for model in LLM_MODELS:
                self._all_models.append(("llm", model))
                yield ModelItem(
                    model,
                    is_selected=model == self._selected_llm,
                    is_loaded=model == self._current_llm,
                )

            yield Label("TTS", classes="model-header")
            for model in TTS_MODELS:
                self._all_models.append(("tts", model))
                yield ModelItem(
                    model,
                    is_selected=model == self._selected_tts,
                    is_loaded=model == self._current_tts,
                )

    def on_mount(self) -> None:
        """Set initial focus on first model item."""
        items = list(self.query(ModelItem))
        if items:
            items[0].focus()

    def _get_focused_item_index(self) -> int:
        """Get index of currently focused ModelItem."""
        items = list(self.query(ModelItem))
        for i, item in enumerate(items):
            if item.has_focus:
                return i
        return 0

    def action_cursor_up(self) -> None:
        """Move focus to previous item."""
        items = list(self.query(ModelItem))
        if not items:
            return

        current = self._get_focused_item_index()
        previous = (current - 1) % len(items)
        items[previous].focus()

    def action_cursor_down(self) -> None:
        """Move focus to next item."""
        items = list(self.query(ModelItem))
        if not items:
            return

        current = self._get_focused_item_index()
        next_idx = (current + 1) % len(items)
        items[next_idx].focus()

    def action_select(self) -> None:
        """Select currently focused model and apply changes."""
        current = self._get_focused_item_index()

        if current >= len(self._all_models):
            return

        model_type, model_name = self._all_models[current]

        if model_type == "stt":
            self._selected_stt = model_name
        elif model_type == "llm":
            self._selected_llm = model_name
        elif model_type == "tts":
            self._selected_tts = model_name

        self._update_all_items()

        selection = ModelSelection(
            stt_model=self._selected_stt,
            llm_model=self._selected_llm,
            tts_model=self._selected_tts,
        )
        self.post_message(self.Changed(selection))

    def _update_all_items(self) -> None:
        """Update all model items to reflect current selection."""
        items = list(self.query(ModelItem))
        for i, item in enumerate(items):
            if i >= len(self._all_models):
                continue

            model_type, model_name = self._all_models[i]

            is_selected = False
            is_loaded = False

            if model_type == "stt":
                is_selected = model_name == self._selected_stt
                is_loaded = model_name == self._current_stt
            elif model_type == "llm":
                is_selected = model_name == self._selected_llm
                is_loaded = model_name == self._current_llm
            elif model_type == "tts":
                is_selected = model_name == self._selected_tts
                is_loaded = model_name == self._current_tts

            item.update_state(is_selected, is_loaded)

    def update_loaded_models(self, stt: str, llm: str, tts: str) -> None:
        """Update which models are marked as loaded."""
        self._current_stt = stt
        self._current_llm = llm
        self._current_tts = tts
        self._update_all_items()

    def action_cancel(self) -> None:
        """Cancel and close the modal."""
        self.dismiss(None)
