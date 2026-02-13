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

    @property
    def emoji(self) -> str:
        """Return emoji representing the state."""
        return {
            AppState.STANDBY: "💤",
            AppState.AWAKE: "👂",
            AppState.PROCESSING: "🤔",
            AppState.SPEAKING: "💬",
        }[self]


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

    audio_duration: float = 0.0  # Simulated recording duration (seconds)
    transcribe_time: float = 0.0  # STT processing time (seconds)
    ttft: float = 0.0  # Time to first token from LLM (seconds)
    llm_time: float = 0.0  # Total LLM generation time (seconds)
    tts_time: float = 0.0  # TTS generation + playback time (seconds)
    total_time: float = 0.0  # End-to-end latency (seconds)
    tokens: int = 0  # Number of tokens generated
    tokens_per_sec: float = 0.0  # Generation speed

    def __post_init__(self):
        """Calculate derived metrics."""
        if self.llm_time > 0 and self.tokens > 0:
            self.tokens_per_sec = self.tokens / self.llm_time
        if self.total_time == 0.0:
            self.total_time = (
                self.audio_duration
                + self.transcribe_time
                + self.llm_time
                + self.tts_time
            )

    def format_summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            f"Audio: {self.audio_duration:.2f}s",
            f"Transcribe: {self.transcribe_time:.2f}s",
            f"TTFT: {self.ttft:.2f}s",
            (
                f"LLM: {self.llm_time:.2f}s "
                f"({self.tokens} tokens @ {self.tokens_per_sec:.1f}t/s)"
            ),
            f"TTS: {self.tts_time:.2f}s",
            f"Total: {self.total_time:.2f}s",
        ]
        return "\n".join(lines)
