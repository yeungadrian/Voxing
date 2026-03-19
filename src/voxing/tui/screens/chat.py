from __future__ import annotations

import threading
from collections.abc import Callable
from typing import ClassVar

import mlx.core as mx
from textual import work
from textual.app import ComposeResult
from textual.events import Key
from textual.screen import Screen

from voxing.config import Settings
from voxing.llm import LocalAgent, Message, TextChunk, load_llm
from voxing.parakeet import load_stt
from voxing.stt import RealtimeTranscriber
from voxing.tts import load_tts, synthesize_and_play
from voxing.tui.messages import (
    AudioChunk,
    GenerationComplete,
    Status,
    StatusChanged,
    SynthesisChunk,
    SynthesisComplete,
    TokenReceived,
    TranscriptionFinal,
    TranscriptionUpdate,
    status_markup,
)
from voxing.tui.screens.settings import SettingsResult, SettingsScreen
from voxing.tui.widgets import (
    SLASH_COMMANDS,
    AssistantMessage,
    ChatInput,
    CommandHints,
    FooterBar,
    MessageList,
    SynthesisDisplay,
    TranscriptionDisplay,
    WelcomeMessage,
)


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
        self._synthesis_display: SynthesisDisplay | None = None
        self._synthesis_stop_event: threading.Event | None = None

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
        self._set_status(Status.IDLE)
        self.chat_input.focus()

    def on_screen_resume(self) -> None:
        self.chat_input.focus()

    def _set_status(
        self, status: Status, error: str | None = None, detail: str | None = None
    ) -> None:
        """Update the footer status from the main thread."""
        self.footer_bar.set_status(status_markup(status, error, detail))

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

        if text.startswith("/"):
            cmd_word, _, args = text.partition(" ")
            handler = self._COMMANDS.get(cmd_word)
            if handler is not None:
                handler(self, args.strip())
            else:
                self._set_status(Status.ERROR, f"Unknown command: {cmd_word}")
        else:
            self._send_message(text)

    def _handle_clear(self, args: str) -> None:
        self.message_list.clear_messages()
        self.message_list.mount(WelcomeMessage())
        self._messages = [
            {"role": "system", "content": self._settings.llm_system_prompt}
        ]
        self._set_status(Status.CHAT_CLEARED)

    def _handle_settings(self, args: str) -> None:
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
        self._set_status(Status.SETTINGS_SAVED)

    def _handle_transcribe(self, args: str) -> None:
        self._remove_welcome()
        self.chat_input.disabled = True
        self._set_status(Status.LOADING_STT)
        self._transcription_display = TranscriptionDisplay(
            audio_visual=self._settings.audio_visual,
            sample_rate=self._settings.sample_rate,
            chunk_duration=self._settings.chunk_duration,
        )
        self.message_list.mount(self._transcription_display)
        self.message_list.scroll_end(animate=False)
        self._transcribe()

    def _handle_speak(self, args: str) -> None:
        """Synthesize and play the given text."""
        if not args:
            self._set_status(Status.ERROR, "Usage: /speak <text>")
            return
        self._remove_welcome()
        self.chat_input.disabled = True
        self._synthesis_display = SynthesisDisplay(
            text=args,
            audio_visual=self._settings.audio_visual,
        )
        self.message_list.mount(self._synthesis_display)
        self.message_list.scroll_end(animate=False)
        self._synthesize(args)

    def _handle_help(self, args: str) -> None:
        self.footer_bar.set_status(
            "[dim]Commands: " + ", ".join(SLASH_COMMANDS) + "[/]"
        )

    def _handle_exit(self, args: str) -> None:
        """Shut down workers and exit the application."""
        self.shutdown_workers()
        self.app.exit()

    _COMMANDS: ClassVar[dict[str, Callable[[ChatScreen, str], None]]] = {
        "/clear": _handle_clear,
        "/settings": _handle_settings,
        "/transcribe": _handle_transcribe,
        "/speak": _handle_speak,
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
        with self._transcriber_lock:
            transcriber = self._active_transcriber
        if transcriber is not None:
            transcriber.stop()

    def _stop_synthesis(self) -> None:
        """Signal active synthesis to stop."""
        if self._synthesis_stop_event is not None:
            self._synthesis_stop_event.set()

    def shutdown_workers(self) -> None:
        """Signal all background workers to stop."""
        self._stop_transcriber()
        self._stop_synthesis()

    def on_unmount(self) -> None:
        self.shutdown_workers()

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self._stop_transcriber()
            self._stop_synthesis()
        elif event.key == "enter" and self._active_transcriber is not None:
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
            self.post_message(
                StatusChanged(Status.LOADING_LLM, detail=self._settings.llm_model_id)
            )
            model, tokenizer = load_llm(self._settings.llm_model_id)
            self.post_message(StatusChanged(Status.GENERATING))

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
            self.post_message(
                StatusChanged(Status.LOADING_STT, detail=self._settings.model_id)
            )
            model = load_stt(self._settings.model_id)
            self.post_message(StatusChanged(Status.RECORDING))

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

    @work(thread=True, exclusive=True, group="synthesize")
    def _synthesize(self, text: str) -> None:
        """Load TTS model, stream synthesis, then unload."""
        model = None
        error: str | None = None
        stop_event = threading.Event()
        self._synthesis_stop_event = stop_event
        try:
            self.post_message(
                StatusChanged(Status.LOADING_TTS, detail=self._settings.tts_model_id)
            )
            model = load_tts(self._settings.tts_model_id)
            self.post_message(StatusChanged(Status.GENERATING_AUDIO))
            synthesize_and_play(
                model,
                text,
                on_chunk=lambda chunk: self.post_message(SynthesisChunk(chunk)),
                on_first_chunk=lambda: self.post_message(
                    StatusChanged(Status.SPEAKING)
                ),
                stop_event=stop_event,
            )
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        finally:
            self._synthesis_stop_event = None
            del model
            mx.clear_cache()
            self.post_message(SynthesisComplete(error=error))

    # --- Message handlers ---

    def on_status_changed(self, message: StatusChanged) -> None:
        self._set_status(message.status, message.error, message.detail)

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
        if message.error:
            self.chat_input.disabled = False
            self.chat_input.focus()
            self._set_status(Status.ERROR, f"Generation error: {message.error}")
        elif self._settings.tts_enabled and len(self._messages) >= 2:
            last = self._messages[-1]
            if last["role"] == "assistant" and last.get("content"):
                full_text = str(last["content"])
                self._synthesis_display = SynthesisDisplay(
                    text=full_text,
                    audio_visual=self._settings.audio_visual,
                )
                self.message_list.mount(self._synthesis_display)
                self.message_list.scroll_end(animate=False)
                self._synthesize(full_text)
                return
            self.chat_input.disabled = False
            self.chat_input.focus()
            self._set_status(Status.IDLE)
        else:
            self.chat_input.disabled = False
            self.chat_input.focus()
            self._set_status(Status.IDLE)

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
            self._set_status(Status.ERROR, f"Transcription error: {message.error}")
        else:
            self._set_status(Status.TRANSCRIPTION_READY)

    def on_synthesis_chunk(self, message: SynthesisChunk) -> None:
        if self._synthesis_display is not None:
            self._synthesis_display.push_chunk(message.chunk)

    def on_synthesis_complete(self, message: SynthesisComplete) -> None:
        if self._synthesis_display is not None:
            self._synthesis_display.remove()
            self._synthesis_display = None
        self.chat_input.disabled = False
        self.chat_input.focus()
        if message.error:
            self._set_status(Status.ERROR, f"Synthesis error: {message.error}")
        else:
            self._set_status(Status.IDLE)
