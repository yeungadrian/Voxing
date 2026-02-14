"""State management data structures for Reachy TUI."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AppState(Enum):
    """Application state machine states."""

    STANDBY = "standby"  # On standby, not processing
    AWAKE = "awake"  # Awake and ready for commands
    PROCESSING = "processing"  # Processing user request
    SPEAKING = "speaking"  # Generating/playing TTS response

    def __str__(self) -> str:
        """Return human-readable state name."""
        return self.value.capitalize()


class InputMode(Enum):
    """Input mode for user interaction."""

    TEXT = "text"
    AUDIO = "audio"

    def __str__(self) -> str:
        """Return human-readable mode name."""
        return self.value.capitalize()


@dataclass
class ConversationMessage:
    """A single message in the conversation history."""

    role: str  # "user", "assistant", or "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        """Return formatted message string."""
        time_str = self.timestamp.strftime("%H:%M:%S")
        prefix = {
            "user": "You",
            "assistant": "Assistant",
            "system": "System",
        }.get(self.role, "Unknown")

        return f"[{time_str}] {prefix}: {self.content}"


@dataclass
class InteractionStats:
    """Performance metrics for a single interaction."""

    audio_duration: float | None = None  # None in text mode
    transcribe_time: float | None = None  # None in text mode
    ttft: float = 0.0  # Time to first token from LLM (seconds)
    llm_time: float = 0.0  # Total LLM generation time (seconds)
    tts_time: float | None = None  # None when TTS disabled
    total_time: float = 0.0  # End-to-end latency (seconds)
    tokens: int = 0  # Number of tokens generated
    tokens_per_sec: float = 0.0  # Generation speed

    def __post_init__(self):
        """Calculate derived metrics."""
        if self.llm_time > 0 and self.tokens > 0:
            self.tokens_per_sec = self.tokens / self.llm_time
        if self.total_time == 0.0:
            self.total_time = (
                (self.audio_duration or 0.0)
                + (self.transcribe_time or 0.0)
                + self.llm_time
                + (self.tts_time or 0.0)
            )

    def format_summary(self) -> str:
        """Return formatted summary string."""
        lines = []
        if self.audio_duration is not None:
            lines.append(f"Audio: {self.audio_duration:.2f}s")
        if self.transcribe_time is not None:
            lines.append(f"Transcribe: {self.transcribe_time:.2f}s")
        lines.append(f"TTFT: {self.ttft:.2f}s")
        lines.append(
            f"LLM: {self.llm_time:.2f}s "
            f"({self.tokens} tokens @ {self.tokens_per_sec:.1f}t/s)"
        )
        if self.tts_time is not None:
            lines.append(f"TTS: {self.tts_time:.2f}s")
        lines.append(f"Total: {self.total_time:.2f}s")
        return "\n".join(lines)
