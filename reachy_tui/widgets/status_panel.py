"""Status panel widget for displaying current app state."""

from datetime import datetime

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

from reachy_tui.state import AppState, InputMode


class StatusPanel(Static):
    """Widget for displaying current status and state."""

    DEFAULT_CLASSES = "status-panel"
    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    # Reactive attributes
    current_state: reactive[AppState] = reactive(AppState.STANDBY)
    input_mode: reactive[InputMode] = reactive(InputMode.TEXT)
    tts_enabled: reactive[bool] = reactive(False)
    last_update: reactive[datetime] = reactive(datetime.now())
    _frame_index: int = 0

    def __init__(self, *args, **kwargs):
        """Initialize the status panel."""
        super().__init__(*args, **kwargs)
        self._animation_timer = None

    def watch_current_state(self, new_state: AppState) -> None:
        """Called when current_state changes.

        Args:
            new_state: The new application state.
        """
        self.last_update = datetime.now()
        self.update_display()

    def watch_input_mode(self, new_mode: InputMode) -> None:
        """Called when input_mode changes."""
        self.update_display()

    def watch_tts_enabled(self, new_value: bool) -> None:
        """Called when tts_enabled changes."""
        self.update_display()

    def update_display(self) -> None:
        """Update the panel display with current state."""
        content = Text()

        # Compact status: spinner + state
        spinner = self.SPINNER_FRAMES[self._frame_index]
        content.append(f"{spinner} ", style=self._get_state_color())
        content.append(str(self.current_state), style=self._get_state_color())

        # Add input mode
        mode_label = "[TXT]" if self.input_mode == InputMode.TEXT else "[AUD]"
        content.append(f"  •  {mode_label}", style="cyan")

        # Add TTS status
        tts_label = "TTS:ON" if self.tts_enabled else "TTS:OFF"
        tts_style = "magenta" if self.tts_enabled else "dim"
        content.append(f"  •  {tts_label}", style=tts_style)

        # Add time
        time_str = self.last_update.strftime("%H:%M:%S")
        content.append(f"  •  {time_str}", style="dim")

        self.update(content)

    def _advance_animation(self) -> None:
        """Advance to the next animation frame."""
        self._frame_index = (self._frame_index + 1) % len(self.SPINNER_FRAMES)
        self.update_display()

    def _get_state_color(self) -> str:
        """Get the color for the current state."""
        color_map = {
            AppState.STANDBY: "dim blue",
            AppState.AWAKE: "green",
            AppState.PROCESSING: "yellow",
            AppState.SPEAKING: "magenta",
        }
        return color_map.get(self.current_state, "white")

    def on_mount(self) -> None:
        """Called when widget is mounted."""
        self.update_display()
        self._animation_timer = self.set_interval(0.1, self._advance_animation)
