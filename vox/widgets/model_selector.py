"""Modal screen for switching models."""

from dataclasses import dataclass

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import RadioButton, RadioSet, TabbedContent, TabPane

from vox.config import LLM_MODELS, STT_MODELS, TTS_MODELS


@dataclass(slots=True)
class ModelSelection:
    """Result of model selection dialog."""

    stt_model: str
    llm_model: str
    tts_model: str


class ModelSelector(ModalScreen[ModelSelection | None]):
    """Modal screen for selecting models across STT, LLM, and TTS."""

    class Changed(Message):
        """Posted when model selection changes."""

        def __init__(self, selection: ModelSelection) -> None:
            super().__init__()
            self.selection = selection

    BINDINGS = [("escape", "cancel", "Close")]

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

    def compose(self) -> ComposeResult:
        """Compose the modal layout."""
        with Vertical(id="model-dialog"), TabbedContent(id="model-tabs"):
            with TabPane("STT", id="tab-stt"), RadioSet(id="stt-radio"):
                for model in STT_MODELS:
                    is_current = model == self._current_stt
                    yield RadioButton(
                        model,
                        value=is_current,
                        classes="model-loaded" if is_current else "",
                    )

            with TabPane("LLM", id="tab-llm"), RadioSet(id="llm-radio"):
                for model in LLM_MODELS:
                    is_current = model == self._current_llm
                    yield RadioButton(
                        model,
                        value=is_current,
                        classes="model-loaded" if is_current else "",
                    )

            with TabPane("TTS", id="tab-tts"), RadioSet(id="tts-radio"):
                for model in TTS_MODELS:
                    is_current = model == self._current_tts
                    yield RadioButton(
                        model,
                        value=is_current,
                        classes="model-loaded" if is_current else "",
                    )

    def _get_selected_model(self, radio_set_id: str, models: tuple[str, ...]) -> str:
        """Return the selected model from a radio set."""
        radio_set = self.query_one(f"#{radio_set_id}", RadioSet)
        index = radio_set.pressed_index
        if index < 0 or index >= len(models):
            return models[0]
        return models[index]

    @on(RadioSet.Changed)
    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Apply selection when any radio changes."""
        event.stop()
        selection = ModelSelection(
            stt_model=self._get_selected_model("stt-radio", STT_MODELS),
            llm_model=self._get_selected_model("llm-radio", LLM_MODELS),
            tts_model=self._get_selected_model("tts-radio", TTS_MODELS),
        )
        self.post_message(self.Changed(selection))

    def update_loaded_models(self, stt: str, llm: str, tts: str) -> None:
        """Update which models are marked as loaded."""
        self._current_stt = stt
        self._current_llm = llm
        self._current_tts = tts

        for radio_set_id, models, current in [
            ("stt-radio", STT_MODELS, stt),
            ("llm-radio", LLM_MODELS, llm),
            ("tts-radio", TTS_MODELS, tts),
        ]:
            radio_set = self.query_one(f"#{radio_set_id}", RadioSet)
            for i, button in enumerate(radio_set.query(RadioButton)):
                if models[i] == current:
                    button.add_class("model-loaded")
                else:
                    button.remove_class("model-loaded")

    def action_cancel(self) -> None:
        """Cancel and close the modal."""
        self.dismiss(None)
