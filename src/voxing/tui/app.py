import mlx.nn as nn
from mlx_lm.tokenizer_utils import TokenizerWrapper
from textual import work
from textual.app import App, ComposeResult
from textual.events import Key
from textual.screen import Screen
from textual.widgets import Input

from voxing.config import Settings
from voxing.llm import LocalAgent, TextChunk, ToolCallInput, ToolCallOutput
from voxing.llm import load_model as load_llm
from voxing.parakeet import ParakeetTDT
from voxing.parakeet import load_model as load_stt
from voxing.stt import RealtimeTranscriber
from voxing.tui.screens import SettingsScreen
from voxing.tui.theme import CATPPUCCIN_MOCHA
from voxing.tui.widgets import (
    SLASH_COMMANDS,
    AssistantMessage,
    ChatInput,
    CommandHints,
    FooterBar,
    GenerationComplete,
    MessageList,
    ModelsReady,
    TokenReceived,
    ToolCallFinished,
    ToolCallStarted,
    TranscriptionFinal,
    TranscriptionUpdate,
    WelcomeMessage,
)

_DEFAULT_SYSTEM_PROMPT = "You are a helpful voice assistant."


class ChatScreen(Screen[None]):
    CSS_PATH = "styles.tcss"

    def __init__(self) -> None:
        super().__init__()
        self._settings = Settings()
        self._stt_model: ParakeetTDT | None = None
        self._llm_model: nn.Module | None = None
        self._llm_tokenizer: TokenizerWrapper | None = None
        self._agent: LocalAgent | None = None
        self._tools_enabled = True
        self._system_prompt = _DEFAULT_SYSTEM_PROMPT
        self._current_assistant_msg: AssistantMessage | None = None
        self._active_transcriber: RealtimeTranscriber | None = None
        self._current_tool_code = ""

    @property
    def footer_bar(self) -> FooterBar:
        return self.query_one(FooterBar)

    @property
    def message_list(self) -> MessageList:
        return self.query_one(MessageList)

    @property
    def chat_input(self) -> ChatInput:
        return self.query_one(ChatInput)

    @property
    def command_hints(self) -> CommandHints:
        return self.query_one(CommandHints)

    def compose(self) -> ComposeResult:
        yield MessageList()
        yield ChatInput()
        yield CommandHints()
        yield FooterBar()

    def on_mount(self) -> None:
        self.message_list.mount(WelcomeMessage())
        self.chat_input.disabled = True
        self.footer_bar.set_status("Loading models...")
        self._load_models()

    def _remove_welcome(self) -> None:
        """Remove the welcome message if present."""
        try:
            welcome = self.message_list.query_one(WelcomeMessage)
            welcome.remove()
        except Exception:
            pass

    def on_input_changed(self, event: Input.Changed) -> None:
        self.command_hints.update_hints(event.value)
        self.footer_bar.display = not self.command_hints.display

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        self.chat_input.value = ""
        self.command_hints.display = False
        self.footer_bar.display = True

        if text == "/clear":
            self._handle_clear()
        elif text == "/settings":
            self._handle_settings()
        elif text == "/transcribe":
            self._handle_transcribe()
        elif text == "/help":
            self._handle_help()
        elif text.startswith("/"):
            self.footer_bar.set_status(f"Unknown command: {text}")
        else:
            self._send_message(text)

    def _handle_clear(self) -> None:
        self.message_list.clear_messages()
        self.message_list.mount(WelcomeMessage())
        if self._agent is not None:
            self._agent = self._create_agent()
        self.footer_bar.set_status("Chat cleared")

    def _handle_settings(self) -> None:
        self.app.push_screen(
            SettingsScreen(self._tools_enabled, self._system_prompt),
            callback=self._on_settings_dismissed,
        )

    def _on_settings_dismissed(self, result: tuple[bool, str] | None) -> None:
        if result is None:
            return
        self._tools_enabled, self._system_prompt = result
        if self._llm_model is not None:
            self._agent = self._create_agent()
        self.message_list.clear_messages()
        self.message_list.mount(WelcomeMessage())
        self.footer_bar.set_status("Settings saved, chat reset")

    def _handle_transcribe(self) -> None:
        if self._stt_model is None:
            self.footer_bar.set_status("STT model not loaded yet")
            return
        self.chat_input.disabled = True
        self.footer_bar.set_status("Recording... (Escape to stop)")
        self._transcribe()

    def _handle_help(self) -> None:
        self.footer_bar.set_status("Commands: " + ", ".join(SLASH_COMMANDS))

    def _send_message(self, text: str) -> None:
        if self._agent is None:
            self.footer_bar.set_status("Models not loaded yet")
            return
        self._remove_welcome()
        self.chat_input.disabled = True
        self.message_list.add_user_message(text)
        self._current_assistant_msg = self.message_list.add_assistant_message()
        self.footer_bar.set_status("Generating...")
        self._generate(text)

    def on_key(self, event: Key) -> None:
        if event.key == "escape" and self._active_transcriber is not None:
            self._active_transcriber.stop()

    # --- Workers ---

    @work(thread=True, exclusive=True, group="models")
    def _load_models(self) -> None:
        """Load STT and LLM models in background."""

        def on_progress(desc: str, advance: int, total: int | None) -> None:
            self.app.call_from_thread(
                self.footer_bar.set_status, f"Downloading: {desc}"
            )

        stt_model = load_stt(self._settings.model_id, on_progress=on_progress)
        self.app.call_from_thread(
            self.footer_bar.set_status, "STT loaded, loading LLM..."
        )

        llm_model, tokenizer = load_llm(
            self._settings.llm_model_id, on_progress=on_progress
        )
        self._stt_model = stt_model
        self._llm_model = llm_model
        self._llm_tokenizer = tokenizer
        self.post_message(ModelsReady())

    @work(thread=True, exclusive=True, group="generate")
    def _generate(self, user_message: str) -> None:
        """Stream LLM generation."""
        assert self._agent is not None
        for event in self._agent.generate(user_message):
            match event:
                case TextChunk(content=token):
                    self.post_message(TokenReceived(token))
                case ToolCallInput(code=code):
                    self.post_message(ToolCallStarted(code))
                case ToolCallOutput(result=result):
                    self.post_message(ToolCallFinished(self._current_tool_code, result))
        self.post_message(GenerationComplete())

    @work(thread=True, exclusive=True, group="transcribe")
    def _transcribe(self) -> None:
        """Run real-time transcription."""
        assert self._stt_model is not None
        transcriber = RealtimeTranscriber(self._stt_model, self._settings)
        self._active_transcriber = transcriber
        with transcriber:
            last_text = ""
            for last_text in transcriber:
                self.post_message(TranscriptionUpdate(last_text))
            if last_text:
                self.post_message(TranscriptionFinal(last_text))
        self._active_transcriber = None

    # --- Message handlers ---

    def on_token_received(self, message: TokenReceived) -> None:
        if self._current_assistant_msg is not None:
            self._current_assistant_msg.append_token(message.token)
            self.message_list.scroll_end(animate=False)

    def on_tool_call_started(self, message: ToolCallStarted) -> None:
        self._current_tool_code = message.code
        if self._current_assistant_msg is not None:
            self._current_assistant_msg.finalize()
        self.footer_bar.set_status("Executing tool...")

    def on_tool_call_finished(self, message: ToolCallFinished) -> None:
        self.message_list.add_tool_call(message.code, message.result)
        self._current_assistant_msg = self.message_list.add_assistant_message()
        self.footer_bar.set_status("Generating...")

    def on_generation_complete(self, message: GenerationComplete) -> None:
        if self._current_assistant_msg is not None:
            self._current_assistant_msg.finalize()
            self._current_assistant_msg = None
        self.chat_input.disabled = False
        self.chat_input.focus()
        self.footer_bar.set_status("")

    def on_transcription_update(self, message: TranscriptionUpdate) -> None:
        self.footer_bar.set_status(f"Recording: {message.text[:50]}")

    def on_transcription_final(self, message: TranscriptionFinal) -> None:
        self.chat_input.value = message.text
        self.chat_input.disabled = False
        self.chat_input.focus()
        self.footer_bar.set_status("")

    def on_models_ready(self, message: ModelsReady) -> None:
        self._agent = self._create_agent()
        self.chat_input.disabled = False
        self.chat_input.focus()
        self.footer_bar.set_status("Ready")

    def _create_agent(self) -> LocalAgent:
        """Create a new LocalAgent with current settings."""
        assert self._llm_model is not None
        assert self._llm_tokenizer is not None
        return LocalAgent(
            self._llm_model,
            self._llm_tokenizer,
            self._settings,
            tools_enabled=self._tools_enabled,
            system_prompt=self._system_prompt,
        )


class VoxingApp(App[None]):
    TITLE = "Voxing"

    def on_mount(self) -> None:
        self.register_theme(CATPPUCCIN_MOCHA)
        self.theme = "catppuccin-mocha"
        self.push_screen(ChatScreen())
