"""Metrics panel widget for displaying memory usage."""

import psutil
from rich.text import Text
from textual.widgets import Static

from voxing.themes import FOREGROUND

_process = psutil.Process()


def _get_memory_mb() -> int:
    """Return current RSS in megabytes."""
    return _process.memory_info().rss // (1024 * 1024)


class MetricsPanel(Static):
    """Widget for displaying memory usage in the status bar."""

    def update_display(self) -> None:
        """Update the panel display with current memory usage."""
        content = Text()
        content.append("Mem: ", style="dim")
        content.append(f"{_get_memory_mb()} MB", style=FOREGROUND)
        self.update(content)

    def on_mount(self) -> None:
        """Called when widget is mounted."""
        self.update_display()
        self.set_interval(3.0, self.update_display)
