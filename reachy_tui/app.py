"""Main Reachy TUI application."""

import time

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Label, TextArea

from reachy_tui.mocks.audio_mock import MockAudioRecorder
from reachy_tui.mocks.llm_mock import MockLLM
from reachy_tui.mocks.stt_mock import MockSTT
from reachy_tui.mocks.tts_mock import MockTTS
from reachy_tui.state import AppState, InputMode, InteractionStats
from reachy_tui.themes import TOKYO_NIGHT
from reachy_tui.widgets import ConversationLog, MetricsPanel, StatusPanel


class ReachyTuiApp(App):
    """Reachy Voice Assistant TUI Application."""

    CSS_PATH = "styles.css"
    TITLE = "Reachy Voice Assistant"

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("ctrl+c", "clear_conversation", "Clear"),
        Binding("ctrl+r", "reset", "Reset"),
    ]

    # Reactive attributes
    state: reactive[AppState] = reactive(AppState.STANDBY)
    conversation_history: reactive[list] = reactive([])
    current_metrics: reactive[InteractionStats | None] = reactive(None)

    def __init__(self, *args, **kwargs):
        """Initialize the Reachy TUI app."""
        super().__init__(*args, **kwargs)

        # Initialize mock components
        self.audio_recorder = MockAudioRecorder()
        self.stt = MockSTT()
        self.llm = MockLLM()
        self.tts = MockTTS()

        # Processing flag
        self.is_processing = False

        # Input mode settings - start in text mode with TTS off
        self.input_mode = InputMode.TEXT
        self.tts_enabled = False

    def compose(self) -> ComposeResult:
        """Compose the app layout.

        Returns:
            ComposeResult with all widgets.
        """
        # Main conversation area
        yield ConversationLog(id="conversation-log", wrap=True)

        # Bottom section (status + input)
        with Vertical(id="bottom-section"):
            # Status bar
            with Horizontal(id="status-bar"):
                yield StatusPanel(id="status-panel")
                yield MetricsPanel(id="metrics-panel")

            # Input
            with Container(id="input-container"):
                yield Label(id="command-hint", classes="hidden")
                yield TextArea(
                    id="user-input",
                )

        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        # Apply Tokyo Night theme
        self.design = TOKYO_NIGHT

        text_area = self.query_one("#user-input", TextArea)
        text_area.focus()
        text_area.show_line_numbers = False

        # Display welcome message
        conv_log = self.query_one("#conversation-log", ConversationLog)
        conv_log.add_system_message(
            "Welcome to Reachy! Starting in text mode with TTS off. "
            "Type /wake to start.",
            style="bold cyan",
        )

        # Initialize status panel
        status_panel = self.query_one("#status-panel", StatusPanel)
        status_panel.current_state = self.state
        status_panel.input_mode = self.input_mode
        status_panel.tts_enabled = self.tts_enabled

    def watch_state(self, new_state: AppState) -> None:
        """Called when state changes.

        Args:
            new_state: The new application state.
        """
        # Update status panel (only if mounted and widgets exist)
        try:
            status_panel = self.query_one("#status-panel", StatusPanel)
            status_panel.current_state = new_state
        except Exception:
            # App not yet mounted or widgets not yet created
            pass

    @on(TextArea.Changed)
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text changes and command hints."""
        text_area = event.text_area
        text = text_area.text

        # Show command hints when "/" is typed
        self._update_command_hints(text)

        # Check if just pressed Enter (not Shift+Enter)
        if text.endswith("\n") and not self.is_processing:
            # Remove the trailing newline
            user_input = text[:-1].strip()

            if user_input:
                # Clear input and hints
                text_area.clear()
                self._hide_command_hints()
                # Process the input
                self.run_worker(self.process_user_input(user_input))

    def _update_command_hints(self, text: str) -> None:
        """Update command hints based on current input."""
        hint_label = self.query_one("#command-hint", Label)

        if text.startswith("/") and not text.endswith("\n"):
            # Show available commands
            commands = ["/wake", "/sleep", "/exit", "/text", "/audio", "/tts"]
            typed = text.strip().lower()

            # Filter matching commands
            matches = [cmd for cmd in commands if cmd.startswith(typed)]

            if matches:
                hint_text = "  ".join(matches)
                hint_label.update(hint_text)
                hint_label.remove_class("hidden")
            else:
                self._hide_command_hints()
        else:
            self._hide_command_hints()

    def _hide_command_hints(self) -> None:
        """Hide command hints."""
        try:
            hint_label = self.query_one("#command-hint", Label)
            hint_label.add_class("hidden")
        except Exception:
            pass

    async def process_user_input(self, text: str) -> None:
        """Process user input through the mock pipeline.

        Args:
            text: User's input text.
        """
        self.is_processing = True
        conv_log = self.query_one("#conversation-log", ConversationLog)

        # Handle commands
        if text.startswith("/"):
            command = text.lower().strip()
            if command == "/wake":
                self._handle_wake_command()
                self.is_processing = False
                return
            if command == "/sleep":
                self._handle_sleep_command()
                self.is_processing = False
                return
            if command == "/exit":
                self.exit()
                return
            if command == "/text":
                self._handle_text_mode_command()
                self.is_processing = False
                return
            if command == "/audio":
                self._handle_audio_mode_command()
                self.is_processing = False
                return
            if command == "/tts":
                self._handle_tts_toggle_command()
                self.is_processing = False
                return
            conv_log.add_system_message(
                f"Unknown command: {command}. "
                "Available: /wake, /sleep, /exit, /text, /audio, /tts",
                style="red",
            )
            self.is_processing = False
            return

        # Check if in standby mode
        if self.state == AppState.STANDBY:
            conv_log.add_system_message(
                "In standby mode. Type /wake to activate.", style="dim yellow"
            )
            self.is_processing = False
            return

        # Add user message to log
        conv_log.add_user_message(text)

        # Run the full pipeline
        try:
            await self._run_pipeline(text)
        except Exception as e:
            conv_log.add_error(str(e))
        finally:
            self.is_processing = False

    async def _run_pipeline(self, text: str) -> None:
        """Run the full mock pipeline for processing user input."""
        conv_log = self.query_one("#conversation-log", ConversationLog)
        metrics_panel = self.query_one("#metrics-panel", MetricsPanel)

        # Initialize timing
        start_time = time.time()
        stats = InteractionStats()

        # Phase 1: Audio + STT (only in audio mode)
        transcribed_text = text
        if self.input_mode == InputMode.AUDIO:
            self.state = AppState.PROCESSING

            # 1. Simulate audio recording
            audio_start = time.time()
            _, audio_duration = await self.audio_recorder.record(text)
            stats.audio_duration = time.time() - audio_start

            # 2. Simulate STT transcription
            stt_start = time.time()
            transcribed_text = await self.stt.transcribe(text)
            stats.transcribe_time = time.time() - stt_start

            # Show transcribed text if different
            if transcribed_text.lower() != text.lower():
                conv_log.add_system_message(
                    f"Heard: {transcribed_text}", style="dim italic"
                )

        # Phase 2: LLM generation (always runs)
        self.state = AppState.PROCESSING

        llm_start = time.time()
        first_token = True
        token_count = 0

        conv_log.start_streaming_response()

        async for token in self.llm.generate_streaming(transcribed_text):
            if first_token:
                stats.ttft = time.time() - llm_start
                first_token = False

            conv_log.update_streaming_response(token)
            token_count += 1

        conv_log.finish_streaming_response()

        stats.llm_time = time.time() - llm_start
        stats.tokens = token_count

        # Get the full response text
        full_response = " ".join(self.llm.conversation_history[-1]["content"].split())

        # Phase 3: TTS (only if enabled)
        if self.tts_enabled:
            self.state = AppState.SPEAKING

            tts_start = time.time()
            audio_duration = await self.tts.generate_audio(full_response)
            await self.tts.playback_audio(audio_duration)
            stats.tts_time = time.time() - tts_start

        # Calculate final metrics
        stats.total_time = time.time() - start_time
        if stats.llm_time > 0 and stats.tokens > 0:
            stats.tokens_per_sec = stats.tokens / stats.llm_time

        # Update metrics panel
        self.current_metrics = stats
        metrics_panel.update_metrics(stats)

        # Return to awake state
        self.state = AppState.AWAKE

    def _handle_wake_command(self) -> None:
        """Handle /wake command."""
        conv_log = self.query_one("#conversation-log", ConversationLog)

        if self.state == AppState.AWAKE:
            conv_log.add_system_message("Already awake.", style="yellow")
        else:
            self.state = AppState.AWAKE
            conv_log.add_system_message("Ready to chat!", style="green")

    def _handle_sleep_command(self) -> None:
        """Handle /sleep command."""
        conv_log = self.query_one("#conversation-log", ConversationLog)

        if self.state == AppState.STANDBY:
            conv_log.add_system_message("Already in standby.", style="yellow")
        else:
            self.state = AppState.STANDBY
            conv_log.add_system_message("Standby mode activated.", style="dim")

    def _handle_text_mode_command(self) -> None:
        """Switch to text input mode."""
        conv_log = self.query_one("#conversation-log", ConversationLog)
        status_panel = self.query_one("#status-panel", StatusPanel)

        if self.input_mode == InputMode.TEXT:
            conv_log.add_system_message("Already in text mode.", style="yellow")
        else:
            self.input_mode = InputMode.TEXT
            status_panel.input_mode = InputMode.TEXT
            conv_log.add_system_message("Switched to text input mode.", style="cyan")

    def _handle_audio_mode_command(self) -> None:
        """Switch to audio input mode."""
        conv_log = self.query_one("#conversation-log", ConversationLog)
        status_panel = self.query_one("#status-panel", StatusPanel)

        if self.input_mode == InputMode.AUDIO:
            conv_log.add_system_message("Already in audio mode.", style="yellow")
        else:
            self.input_mode = InputMode.AUDIO
            status_panel.input_mode = InputMode.AUDIO
            conv_log.add_system_message("Switched to audio input mode.", style="cyan")

    def _handle_tts_toggle_command(self) -> None:
        """Toggle TTS on/off."""
        conv_log = self.query_one("#conversation-log", ConversationLog)
        status_panel = self.query_one("#status-panel", StatusPanel)

        self.tts_enabled = not self.tts_enabled
        status_panel.tts_enabled = self.tts_enabled

        status = "enabled" if self.tts_enabled else "disabled"
        conv_log.add_system_message(f"TTS {status}.", style="cyan")

    def action_clear_conversation(self) -> None:
        """Clear the conversation history."""
        conv_log = self.query_one("#conversation-log", ConversationLog)
        conv_log.clear()
        conv_log.add_system_message("Conversation cleared.", style="dim")

        # Clear LLM history
        self.llm.clear_history()

        # Clear metrics
        metrics_panel = self.query_one("#metrics-panel", MetricsPanel)
        metrics_panel.clear_metrics()

    def action_reset(self) -> None:
        """Reset the app to initial state."""
        self.state = AppState.STANDBY
        self.action_clear_conversation()

        conv_log = self.query_one("#conversation-log", ConversationLog)
        conv_log.add_system_message(
            "App reset. Type /wake to start.", style="bold cyan"
        )
