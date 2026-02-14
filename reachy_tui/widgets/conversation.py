"""Conversation log widget for displaying chat history."""

from datetime import datetime

from rich.text import Text
from textual.widgets import RichLog


class ConversationLog(RichLog):
    """Widget for displaying conversation history with rich formatting."""

    DEFAULT_CLASSES = "conversation-log"

    def __init__(self, **kwargs: object) -> None:
        """Initialize the conversation log."""
        super().__init__(**kwargs)  # ty: ignore[invalid-argument-type]
        self._streaming_start: datetime | None = None
        self._streaming_text: str = ""
        self._streaming_line_count: int = 0

    def add_user_message(self, text: str) -> None:
        """Add a user message to the conversation log."""
        time_str = datetime.now().strftime("%H:%M:%S")
        message = Text()
        message.append(f"[{time_str}] ", style="dim")
        message.append("You: ", style="bold cyan")
        message.append(text, style="white")
        self.write(message)

    def add_assistant_message(self, text: str) -> None:
        """Add an assistant message to the conversation log."""
        time_str = datetime.now().strftime("%H:%M:%S")
        message = Text()
        message.append(f"[{time_str}] ", style="dim")
        message.append("Assistant: ", style="bold magenta")
        message.append(text, style="italic white")
        self.write(message)

    def add_system_message(self, text: str, style: str = "yellow") -> None:
        """Add a system message to the conversation log."""
        time_str = datetime.now().strftime("%H:%M:%S")
        message = Text()
        message.append(f"[{time_str}] ", style="dim")
        message.append(f"[{text}]", style=style)
        self.write(message)

    def start_streaming_response(self) -> None:
        """Start a new streaming assistant response."""
        self._streaming_text = ""
        self._streaming_start = datetime.now()
        self._streaming_line_count = 0

    def _pop_streaming_lines(self) -> None:
        """Remove all lines belonging to the current streaming message."""
        for _ in range(self._streaming_line_count):
            if self.lines:
                self.lines.pop()
        self._line_cache.clear()

    def update_streaming_response(self, token: str) -> None:
        """Update the streaming response with a new token."""
        if self._streaming_start is None:
            self.start_streaming_response()

        self._streaming_text += token
        assert self._streaming_start is not None
        time_str = self._streaming_start.strftime("%H:%M:%S")

        message = Text()
        message.append(f"[{time_str}] ", style="dim")
        message.append("Assistant: ", style="bold magenta")
        message.append(self._streaming_text + " \u2588", style="italic white")

        self._pop_streaming_lines()
        lines_before = len(self.lines)
        self.write(message)
        self._streaming_line_count = len(self.lines) - lines_before

    def finish_streaming_response(self) -> None:
        """Finish the streaming response."""
        if not self._streaming_text or self._streaming_start is None:
            return

        time_str = self._streaming_start.strftime("%H:%M:%S")

        self._pop_streaming_lines()

        message = Text()
        message.append(f"[{time_str}] ", style="dim")
        message.append("Assistant: ", style="bold magenta")
        message.append(self._streaming_text, style="italic white")
        self.write(message)

        self._streaming_start = None
        self._streaming_text = ""
        self._streaming_line_count = 0

    def add_error(self, error_text: str) -> None:
        """Add an error message."""
        self.add_system_message(f"Error: {error_text}", style="bold red")
