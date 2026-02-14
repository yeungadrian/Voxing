"""Main Vox TUI application."""

import asyncio
import contextlib
import time

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.events import Key
from textual.reactive import reactive
from textual.widgets import Footer, Label, TextArea

from vox.models import Models, load_llm, load_stt, load_tts
from vox.models import audio as audio_mod
from vox.models import llm as llm_mod
from vox.models import stt as stt_mod
from vox.models import tts as tts_mod
from vox.state import AppState, InteractionStats
from vox.themes import TOKYO_NIGHT
from vox.widgets import ConversationLog, MetricsPanel, StatusPanel

COMMAND_DESCRIPTIONS: dict[str, str] = {
    "/record": "Record and process voice",
    "/transcribe": "Transcribe audio to text",
    "/tts": "Toggle text-to-speech",
    "/clear": "Clear conversation",
    "/exit": "Exit application",
}
COMMANDS = list(COMMAND_DESCRIPTIONS)

WELCOME_MESSAGE = "Welcome to Vox! Type a message, /record, or /transcribe."


def _longest_common_prefix(strings: list[str]) -> str:
    """Return the longest common prefix of a list of strings."""
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


class VoxApp(App):
    """Vox Voice Assistant TUI Application."""

    CSS_PATH = "styles.css"
    TITLE = "Vox Voice Assistant"

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("ctrl+c", "clear_conversation", "Clear"),
    ]

    state: reactive[AppState] = reactive(AppState.LOADING)
    current_metrics: reactive[InteractionStats | None] = reactive(None)

    def __init__(self) -> None:
        """Initialize the Vox TUI app."""
        super().__init__()
        self.models: Models = None
        self.is_processing = False
        self.tts_enabled = False
        self.chat_history: list[dict[str, str]] = []

    def compose(self) -> ComposeResult:
        """Compose the app layout."""
        yield ConversationLog(id="conversation-log", wrap=True)

        with Vertical(id="bottom-section"):
            with Horizontal(id="status-bar"):
                yield StatusPanel(id="status-panel")
                yield MetricsPanel(id="metrics-panel")

            with Container(id="input-container"):
                yield Label(id="command-hint", classes="hidden")
                yield TextArea(id="user-input")

        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.design = TOKYO_NIGHT

        text_area = self.query_one("#user-input", TextArea)
        text_area.focus()
        text_area.show_line_numbers = False
        text_area.disabled = True

        status_panel = self.query_one("#status-panel", StatusPanel)
        status_panel.current_state = self.state
        status_panel.tts_enabled = self.tts_enabled

        self.run_worker(self._load_models())

    async def _load_models(self) -> None:
        """Load all models in the background."""
        loop = asyncio.get_running_loop()
        status_panel = self.query_one("#status-panel", StatusPanel)

        status_panel.show_ephemeral_message("Loading STT...")
        stt = await loop.run_in_executor(None, load_stt)

        status_panel.show_ephemeral_message("Loading LLM...")
        llm, tokenizer = await loop.run_in_executor(None, load_llm)

        status_panel.show_ephemeral_message("Loading TTS...")
        tts = await loop.run_in_executor(None, load_tts)

        self.models = Models(stt=stt, llm=llm, tts=tts, tokenizer=tokenizer)

        text_area = self.query_one("#user-input", TextArea)
        text_area.disabled = False
        text_area.focus()
        self.state = AppState.READY

        conv_log = self.query_one("#conversation-log", ConversationLog)
        conv_log.add_system_message(WELCOME_MESSAGE, style="bold cyan")

    def watch_state(self, new_state: AppState) -> None:
        """Called when state changes."""
        with contextlib.suppress(NoMatches):
            status_panel = self.query_one("#status-panel", StatusPanel)
            status_panel.current_state = new_state

    def on_key(self, event: Key) -> None:
        """Handle Tab key for command autocomplete."""
        if event.key != "tab":
            return

        text_area = self.query_one("#user-input", TextArea)
        text = text_area.text.strip()

        if not text.startswith("/"):
            return

        matches = [cmd for cmd in COMMANDS if cmd.startswith(text.lower())]
        if not matches:
            return

        event.prevent_default()
        event.stop()

        if len(matches) == 1:
            replacement = matches[0]
        else:
            replacement = _longest_common_prefix(matches)

        text_area.clear()
        text_area.insert(replacement)

    @on(TextArea.Changed)
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text changes and command hints."""
        text_area = event.text_area
        text = text_area.text

        self._update_command_hints(text)

        if text.endswith("\n") and not self.is_processing:
            user_input = text[:-1].strip()
            if user_input:
                text_area.clear()
                self._hide_command_hints()
                self.run_worker(self._process_input(user_input))

    def _update_command_hints(self, text: str) -> None:
        """Update command hints based on current input."""
        hint_label = self.query_one("#command-hint", Label)

        if text.startswith("/") and not text.endswith("\n"):
            typed = text.strip().lower()
            matches = [cmd for cmd in COMMANDS if cmd.startswith(typed)]

            if matches:
                hint_text = Text()
                for i, cmd in enumerate(matches):
                    if i > 0:
                        hint_text.append("\n")
                    hint_text.append(cmd, style="bold #7aa2f7")
                    hint_text.append(f" {COMMAND_DESCRIPTIONS[cmd]}", style="#565f89")
                hint_label.update(hint_text)
                hint_label.remove_class("hidden")
            else:
                self._hide_command_hints()
        else:
            self._hide_command_hints()

    def _hide_command_hints(self) -> None:
        """Hide command hints."""
        with contextlib.suppress(Exception):
            self.query_one("#command-hint", Label).add_class("hidden")

    async def _process_input(self, text: str) -> None:
        """Route user input to the appropriate handler."""
        if self.state == AppState.LOADING:
            return
        self.is_processing = True
        conv_log = self.query_one("#conversation-log", ConversationLog)

        try:
            if text.startswith("/"):
                command = text.lower().strip()
                if command == "/record":
                    await self._run_record_pipeline()
                elif command == "/transcribe":
                    await self._run_transcribe()
                elif command == "/tts":
                    self._toggle_tts()
                elif command == "/clear":
                    self.action_clear_conversation()
                elif command == "/exit":
                    self.exit()
                else:
                    conv_log.add_system_message(
                        f"Unknown command: {command}. Available: {', '.join(COMMANDS)}",
                        style="red",
                    )
            else:
                conv_log.add_user_message(text)
                await self._run_llm_pipeline(text)
        except Exception as e:
            conv_log.add_error(str(e))
            self.state = AppState.READY
        finally:
            self.is_processing = False

    async def _run_llm_pipeline(self, text: str) -> None:
        """Run LLM generation on text input."""
        conv_log = self.query_one("#conversation-log", ConversationLog)
        metrics_panel = self.query_one("#metrics-panel", MetricsPanel)

        start_time = time.time()
        stats = InteractionStats()

        self.state = AppState.PROCESSING

        llm_start = time.time()
        first_token = True
        token_count = 0
        full_response = ""

        conv_log.start_streaming_response()

        async for token in llm_mod.generate_streaming(
            self.models.llm, self.models.tokenizer, text, history=self.chat_history
        ):
            if first_token:
                stats.ttft = time.time() - llm_start
                first_token = False

            full_response += token
            conv_log.update_streaming_response(token)
            token_count += 1

        conv_log.finish_streaming_response()

        self.chat_history.append({"role": "user", "content": text})
        self.chat_history.append({"role": "assistant", "content": full_response})

        stats.llm_time = time.time() - llm_start
        stats.tokens = token_count

        if self.tts_enabled:
            self.state = AppState.SPEAKING
            tts_start = time.time()
            await tts_mod.speak(self.models.tts, full_response)
            stats.tts_time = time.time() - tts_start

        stats.total_time = time.time() - start_time
        if stats.llm_time > 0 and stats.tokens > 0:
            stats.tokens_per_sec = stats.tokens / stats.llm_time

        self.current_metrics = stats
        metrics_panel.update_metrics(stats)
        self.state = AppState.READY

    def _show_status(self, message: str, timeout: float = 3.0) -> None:
        """Show an ephemeral message in the status bar."""
        status_panel = self.query_one("#status-panel", StatusPanel)
        status_panel.show_ephemeral_message(message, timeout)

    async def _run_record_pipeline(self) -> None:
        """Record audio, transcribe, then run LLM pipeline."""
        self.state = AppState.RECORDING
        self._show_status("Recording... speak now.")

        audio_data = await audio_mod.record()

        if audio_data is None:
            self._show_status("No audio detected.")
            self.state = AppState.READY
            return

        self.state = AppState.TRANSCRIBING
        transcribed = await stt_mod.transcribe(self.models.stt, audio_data)

        if not transcribed:
            self._show_status("Could not transcribe audio.")
            self.state = AppState.READY
            return

        conv_log = self.query_one("#conversation-log", ConversationLog)
        conv_log.add_user_message(transcribed)
        await self._run_llm_pipeline(transcribed)

    async def _run_transcribe(self) -> None:
        """Record extended audio and stream transcription."""
        conv_log = self.query_one("#conversation-log", ConversationLog)

        self.state = AppState.RECORDING
        self._show_status("Transcribe mode: recording up to 3 min...")

        audio_data = await audio_mod.record_long()

        if audio_data is None:
            self._show_status("No audio detected.")
            self.state = AppState.READY
            return

        self.state = AppState.TRANSCRIBING
        full_text = ""
        conv_log.start_streaming_response()

        async for chunk in stt_mod.transcribe_streaming(self.models.stt, audio_data):
            full_text += chunk
            conv_log.update_streaming_response(chunk)

        conv_log.finish_streaming_response()

        if full_text.strip():
            self.copy_to_clipboard(full_text.strip())
            self._show_status("Copied to clipboard.")
        else:
            self._show_status("Could not transcribe audio.")

        self.state = AppState.READY

    def _toggle_tts(self) -> None:
        """Toggle TTS on/off."""
        status_panel = self.query_one("#status-panel", StatusPanel)
        self.tts_enabled = not self.tts_enabled
        status_panel.tts_enabled = self.tts_enabled
        status = "enabled" if self.tts_enabled else "disabled"
        self._show_status(f"TTS {status}.")

    def action_clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.chat_history.clear()
        conv_log = self.query_one("#conversation-log", ConversationLog)
        conv_log.clear()
        conv_log.add_system_message(WELCOME_MESSAGE, style="bold cyan")

        metrics_panel = self.query_one("#metrics-panel", MetricsPanel)
        metrics_panel.clear_metrics()
