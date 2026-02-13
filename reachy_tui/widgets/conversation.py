"""Conversation log widget for displaying chat history."""

from datetime import datetime

from rich.text import Text
from textual.widgets import RichLog


class ConversationLog(RichLog):
    """Widget for displaying conversation history with rich formatting."""

    DEFAULT_CLASSES = "conversation-log"

    def __init__(self, *args, **kwargs):
        """Initialize the conversation log."""
        super().__init__(*args, **kwargs)
        self.current_streaming_message: datetime | None = None
        self.streaming_tokens: list[str] = []

    def add_user_message(self, text: str, metadata: dict | None = None) -> None:
        """Add a user message to the conversation log.

        Args:
            text: The message text.
            metadata: Optional metadata (timestamp, etc.).
        """
        timestamp = (
            metadata.get("timestamp", datetime.now())
            if metadata
            else datetime.now()
        )
        time_str = timestamp.strftime("%H:%M:%S")

        message = Text()
        message.append(f"[{time_str}] ", style="dim")
        message.append("You: ", style="bold cyan")
        message.append(text, style="white")

        self.write(message)

    def add_assistant_message(self, text: str, metadata: dict | None = None) -> None:
        """Add an assistant message to the conversation log.

        Args:
            text: The message text.
            metadata: Optional metadata (timestamp, etc.).
        """
        timestamp = (
            metadata.get("timestamp", datetime.now())
            if metadata
            else datetime.now()
        )
        time_str = timestamp.strftime("%H:%M:%S")

        message = Text()
        message.append(f"[{time_str}] ", style="dim")
        message.append("Assistant: ", style="bold magenta")
        message.append(text, style="italic white")

        self.write(message)

    def add_system_message(self, text: str, style: str = "yellow") -> None:
        """Add a system message to the conversation log.

        Args:
            text: The message text.
            style: Rich style for the message.
        """
        timestamp = datetime.now()
        time_str = timestamp.strftime("%H:%M:%S")

        message = Text()
        message.append(f"[{time_str}] ", style="dim")
        message.append(f"[{text}]", style=style)

        self.write(message)

    def start_streaming_response(self) -> None:
        """Start a new streaming assistant response."""
        self.streaming_tokens = []
        self.current_streaming_message = datetime.now()

    def update_streaming_response(self, token: str) -> None:
        """Update the streaming response with a new token."""
        if self.current_streaming_message is None:
            self.start_streaming_response()

        self.streaming_tokens.append(token)

    def finish_streaming_response(self) -> None:
        """Finish the streaming response."""
        if not self.streaming_tokens or self.current_streaming_message is None:
            return

        # Build complete message
        time_str = self.current_streaming_message.strftime("%H:%M:%S")
        full_text = " ".join(self.streaming_tokens)

        message = Text()
        message.append(f"[{time_str}] ", style="dim")
        message.append("Assistant: ", style="bold magenta")
        message.append(full_text, style="italic white")

        self.write(message)

        # Clear streaming state
        self.current_streaming_message = None
        self.streaming_tokens = []

    def add_error(self, error_text: str) -> None:
        """Add an error message.

        Args:
            error_text: The error message.
        """
        self.add_system_message(f"Error: {error_text}", style="bold red")
