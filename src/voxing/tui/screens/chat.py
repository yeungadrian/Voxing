import threading

import mlx.core as mx
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.events import Key
from textual.screen import Screen

from voxing.config import Settings
from voxing.llm import LocalAgent, Message, TextChunk, ToolCallInput, ToolCallOutput
from voxing.llm import load_model as load_llm
from voxing.parakeet import load_model as load_stt
from voxing.stt import RealtimeTranscriber
from voxing.tui.messages import (
    GenerationComplete,
    TokenReceived,
    ToolCallFinished,
    ToolCallStarted,
    TranscriptionFinal,
    TranscriptionUpdate,
)
from voxing.tui.screens.settings import SettingsResult, SettingsScreen
from voxing.tui.screens.tool_detail import ToolCallData, ToolDetailScreen
from voxing.tui.widgets import (
    SLASH_COMMANDS,
    AssistantMessage,
    ChatInput,
    CommandHints,
    FooterBar,
    MessageList,
    ToolCallWidget,
    TranscriptionDisplay,
    WelcomeMessage,
)

_DEFAULT_SYSTEM_PROMPT = "You are a helpful voice assistant."


class ChatScreen(Screen[None]):
    CSS_PATH = "../styles.tcss"
    BINDINGS = [
        Binding("ctrl+e", "show_tools", "Tools", priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._settings = Settings()
        self._tools_enabled = True
        self._system_prompt = _DEFAULT_SYSTEM_PROMPT
        self._messages: list[Message] = [
            {"role": "system", "content": _DEFAULT_SYSTEM_PROMPT}
        ]
        self._current_assistant_msg: AssistantMessage | None = None
        self._active_transcriber: RealtimeTranscriber | None = None
        self._transcription_display: TranscriptionDisplay | None = None
        self._transcribe_cancel: threading.Event = threading.Event()

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
        self.footer_bar.set_status("Ready")
        self.chat_input.focus()

    def on_screen_resume(self) -> None:
        self.chat_input.focus()

    def _remove_welcome(self) -> None:
        """Remove the welcome message if present."""
        try:
            welcome = self.message_list.query_one(WelcomeMessage)
            welcome.remove()
        except Exception:
            pass

    def on_text_area_changed(self, event: ChatInput.Changed) -> None:
        self.command_hints.update_hints(self.chat_input.text)
        self.footer_bar.display = not self.command_hints.display

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        text = event.value
        self.chat_input.clear()
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
        self._messages = [{"role": "system", "content": self._system_prompt}]
        self.footer_bar.set_status("Chat cleared")

    def _handle_settings(self) -> None:
        self.app.push_screen(
            SettingsScreen(self._settings, self._tools_enabled, self._system_prompt),
            callback=self._on_settings_dismissed,
        )

    def _on_settings_dismissed(self, result: SettingsResult | None) -> None:
        if result is None:
            return
        self._tools_enabled = result.tools_enabled
        self._system_prompt = result.system_prompt
        if result.config_overrides:
            merged = {**self._settings.model_dump(), **result.config_overrides}
            self._settings = Settings.model_validate(merged)
        self._messages[0] = {"role": "system", "content": self._system_prompt}
        self.footer_bar.set_status("Settings saved")

    def _handle_transcribe(self) -> None:
        self._remove_welcome()
        self._transcribe_cancel = threading.Event()
        self.chat_input.disabled = True
        self.footer_bar.set_status("Loading STT model...")
        self._transcription_display = TranscriptionDisplay()
        self.message_list.mount(self._transcription_display)
        self.message_list.scroll_end(animate=False)
        self._transcribe()

    def _handle_help(self) -> None:
        self.footer_bar.set_status("Commands: " + ", ".join(SLASH_COMMANDS))

    def _send_message(self, text: str) -> None:
        self._remove_welcome()
        self.chat_input.disabled = True
        self.message_list.add_user_message(text)
        self._current_assistant_msg = None
        self.footer_bar.set_status("Loading LLM...")
        self._generate(text)

    def on_key(self, event: Key) -> None:
        if event.key == "escape" and not self._transcribe_cancel.is_set():
            self._transcribe_cancel.set()
            if self._active_transcriber is not None:
                self._active_transcriber.stop()

    def action_show_tools(self) -> None:
        """Open the tool detail screen."""
        widgets = self.message_list.query(ToolCallWidget)
        if not widgets:
            self.footer_bar.set_status("No tool calls yet")
            return
        tool_calls = [
            ToolCallData(name=w.tool_name, code=w.code, result=w.result)
            for w in widgets
        ]
        self.app.push_screen(ToolDetailScreen(tool_calls))

    # --- Workers ---

    @work(thread=True, exclusive=True, group="generate")
    def _generate(self, user_message: str) -> None:
        """Load LLM, stream generation, then unload."""
        self.app.call_from_thread(self.footer_bar.set_status, "Loading LLM...")
        model, tokenizer = load_llm(self._settings.llm_model_id)
        self.app.call_from_thread(self.footer_bar.set_status, "Generating...")

        agent = LocalAgent(
            model,
            tokenizer,
            self._settings,
            self._messages,
            tools_enabled=self._tools_enabled,
        )
        last_tool_code = ""
        last_tool_name = ""
        for event in agent.generate(user_message):
            match event:
                case TextChunk(content=token):
                    self.post_message(TokenReceived(token))
                case ToolCallInput(code=code, name=name):
                    last_tool_code = code
                    last_tool_name = name
                    self.post_message(ToolCallStarted(code, name))
                case ToolCallOutput(result=result):
                    self.post_message(
                        ToolCallFinished(last_tool_code, result, last_tool_name)
                    )

        del agent, model, tokenizer
        mx.clear_cache()
        self.post_message(GenerationComplete())

    @work(thread=True, exclusive=True, group="transcribe")
    def _transcribe(self) -> None:
        """Load STT model, run real-time transcription, then unload."""
        self.app.call_from_thread(self.footer_bar.set_status, "Loading STT model...")
        model = load_stt(self._settings.model_id)
        self.app.call_from_thread(self.footer_bar.set_status, "")

        if self._transcribe_cancel.is_set():
            del model
            mx.clear_cache()
            self.post_message(TranscriptionFinal(""))
            return

        transcriber = RealtimeTranscriber(model, self._settings)
        self._active_transcriber = transcriber
        with transcriber:
            last_text = ""
            for last_text in transcriber:
                self.post_message(TranscriptionUpdate(last_text))
        self._active_transcriber = None
        self.post_message(TranscriptionFinal(last_text))

        del transcriber, model
        mx.clear_cache()

    # --- Message handlers ---

    def _ensure_assistant_msg(self) -> AssistantMessage:
        """Lazily create an assistant message widget."""
        if self._current_assistant_msg is None:
            self._current_assistant_msg = self.message_list.add_assistant_message()
        return self._current_assistant_msg

    def on_token_received(self, message: TokenReceived) -> None:
        msg = self._ensure_assistant_msg()
        msg.append_token(message.token)
        self.message_list.scroll_end(animate=False)

    def on_tool_call_started(self, message: ToolCallStarted) -> None:
        if self._current_assistant_msg is not None:
            self._current_assistant_msg.finalize()
            if self._current_assistant_msg.is_empty:
                self._current_assistant_msg.remove()
            self._current_assistant_msg = None
        self.footer_bar.set_status("Executing tool...")

    def on_tool_call_finished(self, message: ToolCallFinished) -> None:
        self.message_list.add_tool_call(message.code, message.result, message.name)
        self._current_assistant_msg = None
        self.footer_bar.set_status("Generating...")

    def on_generation_complete(self, message: GenerationComplete) -> None:
        if self._current_assistant_msg is not None:
            self._current_assistant_msg.finalize()
            if self._current_assistant_msg.is_empty:
                self._current_assistant_msg.remove()
            self._current_assistant_msg = None
        self.chat_input.disabled = False
        self.chat_input.focus()
        self.footer_bar.set_status("")

    def on_transcription_update(self, message: TranscriptionUpdate) -> None:
        if self._transcription_display is not None:
            self._transcription_display.update_text(message.text)
            self.message_list.scroll_end(animate=False)

    def on_transcription_final(self, message: TranscriptionFinal) -> None:
        if self._transcription_display is not None:
            self._transcription_display.remove()
            self._transcription_display = None
        if message.text:
            self.chat_input.clear()
            self.chat_input.insert(message.text)
        self.chat_input.disabled = False
        self.chat_input.focus()
        self.footer_bar.set_status("")
