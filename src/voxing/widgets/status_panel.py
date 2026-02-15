"""Status panel widget for displaying current app state."""

from rich.text import Text
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Static

from voxing.state import AppState
from voxing.themes import (
    FOREGROUND,
    PALETTE_1,
    PALETTE_3,
    PALETTE_4,
    PALETTE_5,
    PALETTE_6,
)


class StatusPanel(Static):
    """Widget for displaying current status and state."""

    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    current_state: reactive[AppState] = reactive(AppState.LOADING)
    tts_enabled: reactive[bool] = reactive(False)
    status_message: reactive[str | None] = reactive(None)
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

        content.append("Voxing", style=PALETTE_4)
        content.append("  •  ")

        if self.current_state != AppState.READY:
            spinner = self.SPINNER_FRAMES[self._frame_index]
            content.append(f"{spinner} ", style=self._get_state_color())
        content.append(str(self.current_state), style=self._get_state_color())

        if self.status_message:
            content.append(f"  •  {self.status_message}", style=f"dim {PALETTE_3}")

        if self.tts_enabled:
            content.append("  •  TTS", style=PALETTE_6)

        self.update(content)

    def _advance_animation(self) -> None:
        """Advance to the next animation frame."""
        self._frame_index = (self._frame_index + 1) % len(self.SPINNER_FRAMES)
        self.update_display()

    def _get_state_color(self) -> str:
        """Get the color for the current state."""
        color_map = {
            AppState.LOADING: PALETTE_3,  # #e0af68 (yellow) - processing
            AppState.READY: PALETTE_5,  # #bb9af7 (purple) - idle
            AppState.RECORDING: PALETTE_1,  # #f7768e (red) - active I/O
            AppState.TRANSCRIBING: PALETTE_3,  # #e0af68 (yellow) - processing
            AppState.THINKING: PALETTE_3,  # #e0af68 (yellow) - processing
            AppState.SYNTHESIZING: PALETTE_3,  # #e0af68 (yellow) - processing
            AppState.SPEAKING: PALETTE_1,  # #f7768e (red) - active I/O
        }
        return color_map.get(self.current_state, FOREGROUND)

    def on_mount(self) -> None:
        """Called when widget is mounted."""
        self.update_display()
        pause = self.current_state == AppState.READY
        self._animation_timer = self.set_interval(
            0.1, self._advance_animation, pause=pause
        )
