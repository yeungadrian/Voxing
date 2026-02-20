"""Performance stats widget displayed above the input area."""

from rich.text import Text
from textual.widgets import Static

from voxing.state import InteractionStats


class PerfStats(Static):
    """Widget for displaying performance stats from the last interaction."""

    current_stats: InteractionStats | None = None

    def update_stats(self, stats: InteractionStats) -> None:
        """Update the displayed performance stats."""
        self.current_stats = stats
        content = Text()

        if stats.transcribe_time is not None:
            content.append("STT: ", style="dim")
            content.append(f"{stats.transcribe_time:.2f}s", style="dim")
            content.append("  ", style="dim")

        content.append("LLM: ", style="dim")
        content.append(f"{stats.llm_time:.1f}s", style="dim")
        content.append("  Tokens: ", style="dim")
        content.append(f"{stats.tokens_per_sec:.0f}/s", style="dim")

        if stats.tts_time is not None:
            content.append("  TTS: ", style="dim")
            content.append(f"{stats.tts_time:.2f}s", style="dim")

        if stats.max_tokens > 0:
            content.append("  Tokens: ", style="dim")
            content.append(
                f"{stats.total_conversation_tokens}/{stats.max_tokens}",
                style="dim",
            )

        self.update(content)

    def clear_stats(self) -> None:
        """Clear the displayed stats."""
        self.update("")
