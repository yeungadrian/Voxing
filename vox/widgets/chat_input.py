from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea


class ChatInput(TextArea):
    """TextArea subclass where Enter submits and Ctrl+J inserts a newline."""

    BINDINGS = [
        Binding("enter", "submit", "Submit", priority=True),
        Binding("Shift+enter,ctrl+j", "newline", "Newline"),
    ]

    class Submitted(Message):
        """Posted when the user presses Enter to submit input."""

        def __init__(self, chat_input: "ChatInput", value: str) -> None:
            super().__init__()
            self.chat_input = chat_input
            self.value = value

    def action_submit(self) -> None:
        """Submit the current text."""
        self.post_message(self.Submitted(self, self.text))

    def action_newline(self) -> None:
        """Insert a newline character."""
        self.insert("\n")
