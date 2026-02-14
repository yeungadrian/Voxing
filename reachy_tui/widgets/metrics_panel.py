"""Metrics panel widget for displaying performance statistics."""

from rich.text import Text
from textual.widgets import Static

from reachy_tui.state import InteractionStats


class MetricsPanel(Static):
    """Widget for displaying performance metrics."""

    DEFAULT_CLASSES = "metrics-panel"

    def __init__(self, *args, **kwargs):
        """Initialize the metrics panel."""
        super().__init__(*args, **kwargs)
        self.current_stats = None
        self.update_display()

    def update_metrics(self, stats: InteractionStats) -> None:
        """Update the displayed metrics.

        Args:
            stats: New performance statistics.
        """
        self.current_stats = stats
        self.update_display()

    def update_display(self) -> None:
        """Update the panel display with current metrics."""
        content = Text()

        if self.current_stats is None:
            content.append("No metrics", style="dim")
        else:
            stats = self.current_stats

            # Show audio/STT only if available
            if stats.audio_duration is not None:
                content.append("Audio: ", style="dim")
                content.append(f"{stats.audio_duration:.2f}s", style="cyan")
                content.append("  ")
            if stats.transcribe_time is not None:
                content.append("STT: ", style="dim")
                content.append(f"{stats.transcribe_time:.2f}s", style="cyan")
                content.append("  ")

            # Always show LLM metrics
            content.append("TTFT: ", style="dim")
            content.append(f"{stats.ttft:.2f}s", style="cyan")
            content.append("  LLM: ", style="dim")
            content.append(f"{stats.llm_time:.1f}s", style="cyan")
            content.append("  Tokens: ", style="dim")
            content.append(f"{stats.tokens_per_sec:.0f}/s", style="cyan")

            # Show TTS only if available
            if stats.tts_time is not None:
                content.append("  TTS: ", style="dim")
                content.append(f"{stats.tts_time:.2f}s", style="cyan")

            # Always show total
            content.append("  Total: ", style="dim")
            content.append(f"{stats.total_time:.1f}s", style="yellow")

        self.update(content)

    def clear_metrics(self) -> None:
        """Clear the current metrics."""
        self.current_stats = None
        self.update_display()

    def on_mount(self) -> None:
        """Called when widget is mounted."""
        self.update_display()
