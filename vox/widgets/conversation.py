"""Conversation log widget for displaying chat history."""

from rich.text import Text
from textual.geometry import Size
from textual.widgets import RichLog

from vox.themes import FOREGROUND, PALETTE_1, PALETTE_3, PALETTE_5, PALETTE_6


class ConversationLog(RichLog):
    """Widget for displaying conversation history with rich formatting."""

    _is_streaming: bool = False
    _streaming_text: str = ""
    _streaming_line_count: int = 0

    def add_user_message(self, text: str) -> None:
        """Add a user message to the conversation log."""
        message = Text()
        message.append("You: ", style=f"bold {PALETTE_6}")
        message.append(text, style=FOREGROUND)
        self.write(message)

    def add_system_message(self, text: str, style: str = PALETTE_3) -> None:
        """Add a system message to the conversation log."""
        message = Text()
        message.append(f"[{text}]", style=style)
        self.write(message)

    def start_streaming_response(self) -> None:
        """Start a new streaming assistant response."""
        self._streaming_text = ""
        self._is_streaming = True
        self._streaming_line_count = 0

    def _pop_streaming_lines(self) -> None:
        """Remove all lines belonging to the current streaming message."""
        for _ in range(self._streaming_line_count):
            if self.lines:
                self.lines.pop()
        self._line_cache.clear()
        self.virtual_size = Size(self.virtual_size.width, len(self.lines))

    def update_streaming_response(self, token: str) -> None:
        """Update the streaming response with a new token."""
        if not self._is_streaming:
            self.start_streaming_response()

        self._streaming_text += token

        message = Text()
        message.append("Assistant: ", style=f"bold {PALETTE_5}")
        message.append(self._streaming_text + " \u2588", style=f"italic {FOREGROUND}")

        self._pop_streaming_lines()
        lines_before = len(self.lines)
        self.write(message)
        self._streaming_line_count = len(self.lines) - lines_before

    def finish_streaming_response(self) -> None:
        """Finish the streaming response."""
        if not self._streaming_text or not self._is_streaming:
            return

        self._pop_streaming_lines()

        message = Text()
        message.append("Assistant: ", style=f"bold {PALETTE_5}")
        message.append(self._streaming_text, style=f"italic {FOREGROUND}")
        self.write(message)

        self._is_streaming = False
        self._streaming_text = ""
        self._streaming_line_count = 0

    def add_error(self, error_text: str) -> None:
        """Add an error message."""
        self.add_system_message(f"Error: {error_text}", style=f"bold {PALETTE_1}")
