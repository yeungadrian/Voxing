"""Status panel widget for displaying current app state."""

from datetime import datetime

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

from reachy_tui.state import AppState


class StatusPanel(Static):
    """Widget for displaying current status and state."""

    DEFAULT_CLASSES = "status-panel"

    # Reactive attributes
    current_state: reactive[AppState] = reactive(AppState.STANDBY)
    last_update: reactive[datetime] = reactive(datetime.now())

    def __init__(self, *args, **kwargs):
        """Initialize the status panel."""
        super().__init__(*args, **kwargs)

    def watch_current_state(self, new_state: AppState) -> None:
        """Called when current_state changes.

        Args:
            new_state: The new application state.
        """
        self.last_update = datetime.now()
        self.update_display()

    def update_display(self) -> None:
        """Update the panel display with current state."""
        content = Text()

        # Compact status: emoji + state
        content.append(f"{self.current_state.emoji} ", style=self._get_state_color())
        content.append(str(self.current_state), style=self._get_state_color())

        # Add time
        time_str = self.last_update.strftime("%H:%M:%S")
        content.append(f"  •  {time_str}", style="dim")

        self.update(content)

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
