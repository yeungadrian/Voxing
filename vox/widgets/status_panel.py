"""Status panel widget for displaying current app state."""

from datetime import datetime

from rich.text import Text
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Static

from vox.state import AppState


class StatusPanel(Static):
    """Widget for displaying current status and state."""

    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    current_state: reactive[AppState] = reactive(AppState.LOADING)
    tts_enabled: reactive[bool] = reactive(False)
    status_message: reactive[str | None] = reactive(None)
    last_update: reactive[datetime] = reactive(datetime.now())
    _frame_index: int = 0
    _animation_timer: Timer | None = None
    _ephemeral_timer: Timer | None = None

    def show_status_message(self, message: str) -> None:
        """Show a status message that persists until explicitly cleared."""
        if self._ephemeral_timer is not None:
            self._ephemeral_timer.stop()
            self._ephemeral_timer = None
        self.status_message = message

    def show_ephemeral_message(self, message: str, timeout: float = 3.0) -> None:
        """Show a temporary status message that auto-clears."""
        if self._ephemeral_timer is not None:
            self._ephemeral_timer.stop()
        self.status_message = message
        self._ephemeral_timer = self.set_timer(timeout, self._clear_ephemeral)

    def clear_status_message(self) -> None:
        """Clear any active status message."""
        self.status_message = None
        if self._ephemeral_timer is not None:
            self._ephemeral_timer.stop()
            self._ephemeral_timer = None

    def _clear_ephemeral(self) -> None:
        """Clear the ephemeral status message."""
        self.status_message = None
        self._ephemeral_timer = None

    def watch_status_message(self) -> None:
        """Called when status_message changes."""
        self.update_display()

    def watch_current_state(self, new_state: AppState) -> None:
        """Called when current_state changes."""
        self.last_update = datetime.now()
        if self._animation_timer is not None:
            if new_state == AppState.READY:
                self._animation_timer.pause()
            else:
                self._animation_timer.resume()
        self.update_display()

    def watch_tts_enabled(self, new_value: bool) -> None:
        """Called when tts_enabled changes."""
        self.update_display()

    def update_display(self) -> None:
        """Update the panel display with current state."""
        content = Text()

        if self.current_state == AppState.READY:
            content.append("● ", style=self._get_state_color())
        else:
            spinner = self.SPINNER_FRAMES[self._frame_index]
            content.append(f"{spinner} ", style=self._get_state_color())
        content.append(str(self.current_state), style=self._get_state_color())

        if self.status_message:
            content.append(f"  •  {self.status_message}", style="dim yellow")

        tts_label = "TTS:ON" if self.tts_enabled else "TTS:OFF"
        tts_style = "magenta" if self.tts_enabled else "dim"
        content.append(f"  •  {tts_label}", style=tts_style)

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
            AppState.LOADING: "yellow",
            AppState.READY: "green",
            AppState.RECORDING: "yellow",
            AppState.TRANSCRIBING: "cyan",
            AppState.PROCESSING: "yellow",
            AppState.SPEAKING: "magenta",
        }
        return color_map.get(self.current_state, "white")

    def on_mount(self) -> None:
        """Called when widget is mounted."""
        self.update_display()
        pause = self.current_state == AppState.READY
        self._animation_timer = self.set_interval(
            0.1, self._advance_animation, pause=pause
        )
