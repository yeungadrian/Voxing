import threading
from collections.abc import Callable
from typing import ClassVar

import mlx.core as mx
from textual import work
from textual.app import ComposeResult
from textual.events import Key
from textual.screen import Screen

from voxing.config import Settings
from voxing.llm import LocalAgent, Message, TextChunk
from voxing.llm import load_model as load_llm
from voxing.parakeet import load_model as load_stt
from voxing.stt import RealtimeTranscriber
from voxing.tui.messages import (
    AudioChunk,
    GenerationComplete,
    TokenReceived,
    TranscriptionFinal,
    TranscriptionUpdate,
)
from voxing.tui.screens.settings import SettingsResult, SettingsScreen
from voxing.tui.widgets import (
    SLASH_COMMANDS,
    AssistantMessage,
    ChatInput,
    CommandHints,
    FooterBar,
    MessageList,
    TranscriptionDisplay,
    WelcomeMessage,
)

_IDLE_STATUS = "[dim]type / for commands[/]"


class ChatScreen(Screen[None]):
    CSS_PATH = "../styles.tcss"

    def __init__(self) -> None:
        super().__init__()
        self._settings = Settings()
        self._messages: list[Message] = [
            {"role": "system", "content": self._settings.llm_system_prompt}
        ]
        self._current_assistant_msg: AssistantMessage | None = None
        self._active_transcriber: RealtimeTranscriber | None = None
        self._transcriber_lock = threading.Lock()
        self._transcription_display: TranscriptionDisplay | None = None
        self._transcribe_cancel = threading.Event()

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
        self.footer_bar.set_status(_IDLE_STATUS)
        self.chat_input.focus()

    def on_screen_resume(self) -> None:
        self.chat_input.focus()

    def _remove_welcome(self) -> None:
        """Remove the welcome message if present."""
        for welcome in self.message_list.query(WelcomeMessage):
            welcome.remove()

    def on_text_area_changed(self, event: ChatInput.Changed) -> None:
        self.command_hints.update_hints(self.chat_input.text)
        self.footer_bar.display = not self.command_hints.display

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        text = event.value
        self.chat_input.clear()
        self.command_hints.display = False
        self.footer_bar.display = True

        handler = self._COMMANDS.get(text)
        if handler is not None:
            handler(self)
        elif text.startswith("/"):
            self.footer_bar.set_status(f"[$error]Unknown command: {text}[/]")
        else:
            self._send_message(text)

    def _handle_clear(self) -> None:
        self.message_list.clear_messages()
        self.message_list.mount(WelcomeMessage())
        self._messages = [
            {"role": "system", "content": self._settings.llm_system_prompt}
        ]
        self.footer_bar.set_status("[$success]Chat cleared[/]")

    def _handle_settings(self) -> None:
        self.app.push_screen(
            SettingsScreen(self._settings),
            callback=self._on_settings_dismissed,
        )

    def _on_settings_dismissed(self, result: SettingsResult | None) -> None:
        if result is None:
            return
        if result.config_overrides:
            merged = {**self._settings.model_dump(), **result.config_overrides}
            self._settings = Settings.model_validate(merged)
        self._messages[0] = {
            "role": "system",
            "content": self._settings.llm_system_prompt,
        }
        self.footer_bar.set_status("[$success]Settings saved[/]")

    def _handle_transcribe(self) -> None:
        self._remove_welcome()
        self._transcribe_cancel.clear()
        self.chat_input.disabled = True
        self.footer_bar.set_status("[$warning]Loading STT model...[/]")
        self._transcription_display = TranscriptionDisplay(
            audio_visual=self._settings.audio_visual,
            sample_rate=self._settings.sample_rate,
            chunk_duration=self._settings.chunk_duration,
        )
        self.message_list.mount(self._transcription_display)
        self.message_list.scroll_end(animate=False)
        self._transcribe()

    def _handle_help(self) -> None:
        self.footer_bar.set_status(
            "[dim]Commands: " + ", ".join(SLASH_COMMANDS) + "[/]"
        )

    def _handle_exit(self) -> None:
        """Shut down workers and exit the application."""
        self.shutdown_workers()
        self.app.exit()

    _COMMANDS: ClassVar[dict[str, Callable[["ChatScreen"], None]]] = {
        "/clear": _handle_clear,
        "/settings": _handle_settings,
        "/transcribe": _handle_transcribe,
        "/help": _handle_help,
        "/exit": _handle_exit,
    }

    def _send_message(self, text: str) -> None:
        self._remove_welcome()
        self.chat_input.disabled = True
        self.message_list.add_user_message(text)
        self._current_assistant_msg = None
        self._generate(text)

    def _stop_transcriber(self) -> None:
        """Signal active transcriber to stop."""
        self._transcribe_cancel.set()
        with self._transcriber_lock:
            transcriber = self._active_transcriber
        if transcriber is not None:
            transcriber.stop()

    def shutdown_workers(self) -> None:
        """Signal all background workers to stop."""
        self._stop_transcriber()

    def on_unmount(self) -> None:
        self.shutdown_workers()

    def on_key(self, event: Key) -> None:
        if event.key == "escape" and not self._transcribe_cancel.is_set():
            self._stop_transcriber()

    # --- Workers ---

    @work(thread=True, exclusive=True, group="generate")
    def _generate(self, user_message: str) -> None:
        """Load LLM, stream generation, then unload."""
        model = None
        tokenizer = None
        agent = None
        error: str | None = None
        try:
            self.app.call_from_thread(
                self.footer_bar.set_status, "[$warning]Loading LLM...[/]"
            )
            model, tokenizer = load_llm(self._settings.llm_model_id)
            self.app.call_from_thread(
                self.footer_bar.set_status, "[$primary]Generating...[/]"
            )

            agent = LocalAgent(
                model,
                tokenizer,
                self._settings,
                list(self._messages),
            )
            for event in agent.generate(user_message):
                match event:
                    case TextChunk(content=token):
                        self.post_message(TokenReceived(token))

            self._messages[:] = agent.messages
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        finally:
            del agent
            del tokenizer
            del model
            mx.clear_cache()
            self.post_message(GenerationComplete(error=error))

    @work(thread=True, exclusive=True, group="transcribe")
    def _transcribe(self) -> None:
        """Load STT model, run real-time transcription, then unload."""
        model = None
        transcriber = None
        last_text = ""
        error: str | None = None
        try:
            self.app.call_from_thread(
                self.footer_bar.set_status, "[$warning]Loading STT model...[/]"
            )
            model = load_stt(self._settings.model_id)
            self.app.call_from_thread(
                self.footer_bar.set_status,
                "[dim]esc to stop recording[/]",
            )

            if self._transcribe_cancel.is_set():
                return

            transcriber = RealtimeTranscriber(
                model,
                self._settings,
                on_chunk=lambda chunk: self.post_message(AudioChunk(chunk)),
            )
            with self._transcriber_lock:
                self._active_transcriber = transcriber
            try:
                with transcriber:
                    for last_text in transcriber:
                        self.post_message(TranscriptionUpdate(last_text))
            finally:
                with self._transcriber_lock:
                    self._active_transcriber = None
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        finally:
            del transcriber
            del model
            mx.clear_cache()
            self.post_message(TranscriptionFinal(last_text, error=error))

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

    def on_generation_complete(self, message: GenerationComplete) -> None:
        if self._current_assistant_msg is not None:
            self._current_assistant_msg.finalize()
            if self._current_assistant_msg.is_empty:
                self._current_assistant_msg.remove()
            self._current_assistant_msg = None
        self.chat_input.disabled = False
        self.chat_input.focus()
        if message.error:
            self.footer_bar.set_status(f"[$error]Generation error: {message.error}[/]")
        else:
            self.footer_bar.set_status(_IDLE_STATUS)

    def on_transcription_update(self, message: TranscriptionUpdate) -> None:
        if self._transcription_display is not None:
            self._transcription_display.update_text(message.text)
            self.message_list.scroll_end(animate=False)

    def on_audio_chunk(self, message: AudioChunk) -> None:
        if self._transcription_display is not None:
            self._transcription_display.push_chunk(message.chunk)

    def on_transcription_final(self, message: TranscriptionFinal) -> None:
        if self._transcription_display is not None:
            self._transcription_display.remove()
            self._transcription_display = None
        if message.text:
            self.chat_input.clear()
            self.chat_input.insert(message.text)
        self.chat_input.disabled = False
        self.chat_input.focus()
        if message.error:
            self.footer_bar.set_status(
                f"[$error]Transcription error: {message.error}[/]"
            )
        else:
            self.footer_bar.set_status("[dim]enter to send  ·  type / for commands[/]")
