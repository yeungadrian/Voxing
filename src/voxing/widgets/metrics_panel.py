"""Metrics panel widget for displaying performance statistics."""

import psutil
from rich.text import Text
from textual.widgets import Static

from voxing.state import InteractionStats
from voxing.themes import FOREGROUND

_process = psutil.Process()


def _get_memory_mb() -> int:
    """Return current RSS in megabytes."""
    return _process.memory_info().rss // (1024 * 1024)


class MetricsPanel(Static):
    """Widget for displaying performance metrics."""

    current_stats: InteractionStats | None = None

    def update_metrics(self, stats: InteractionStats) -> None:
        """Update the displayed metrics."""
        self.current_stats = stats
        self.update_display()

    def update_display(self) -> None:
        """Update the panel display with current metrics."""
        content = Text()

        if self.current_stats is None:
            content.append("Mem: ", style="dim")
            content.append(f"{_get_memory_mb()} MB", style=FOREGROUND)
            self.update(content)
            return
        stats = self.current_stats

        if stats.transcribe_time is not None:
            content.append("STT: ", style="dim")
            content.append(f"{stats.transcribe_time:.2f}s", style=FOREGROUND)
            content.append("  ")

        content.append("LLM: ", style="dim")
        content.append(f"{stats.llm_time:.1f}s", style=FOREGROUND)
        content.append("  Tokens: ", style="dim")
        content.append(f"{stats.tokens_per_sec:.0f}/s", style=FOREGROUND)

        if stats.tts_time is not None:
            content.append("  TTS: ", style="dim")
            content.append(f"{stats.tts_time:.2f}s", style=FOREGROUND)

        if stats.max_tokens > 0:
            content.append("  Tokens: ", style="dim")
            content.append(
                f"{stats.total_conversation_tokens}/{stats.max_tokens}",
                style=FOREGROUND,
            )

        content.append("  Mem: ", style="dim")
        content.append(f"{_get_memory_mb()} MB", style=FOREGROUND)
        self.update(content)

    def clear_metrics(self) -> None:
        """Clear the current metrics."""
        self.current_stats = None
        self.update_display()

    def on_mount(self) -> None:
        """Called when widget is mounted."""
        self.update_display()
        self.set_interval(3.0, self.update_display)
