"""State management data structures for Vox TUI."""

from dataclasses import dataclass
from enum import Enum


class AppState(Enum):
    """Application state machine states."""

    LOADING = "loading"
    READY = "ready"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SYNTHESIZING = "synthesizing"
    SPEAKING = "speaking"

    def __str__(self) -> str:
        """Return human-readable state name."""
        return self.value.capitalize()


@dataclass
class InteractionStats:
    """Performance metrics for a single interaction."""

    transcribe_time: float | None = None
    llm_time: float = 0.0
    tts_time: float | None = None
    tokens: int = 0
    tokens_per_sec: float = 0.0

    def __post_init__(self) -> None:
        """Calculate derived metrics."""
        if self.llm_time > 0 and self.tokens > 0:
            self.tokens_per_sec = self.tokens / self.llm_time
