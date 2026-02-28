from textual.containers import VerticalScroll
from textual.message import Message
from textual.suggester import SuggestFromList
from textual.widget import Widget
from textual.widgets import Input, Static

SLASH_COMMANDS: dict[str, str] = {
    "/transcribe": "Start voice transcription",
    "/settings": "Open settings panel",
    "/clear": "Clear chat history",
    "/help": "Show available commands",
}

CURSOR = "\u258c"


class TokenReceived(Message):
    def __init__(self, token: str) -> None:
        self.token = token
        super().__init__()


class GenerationComplete(Message):
    pass


class ToolCallStarted(Message):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__()


class ToolCallFinished(Message):
    def __init__(self, code: str, result: str) -> None:
        self.code = code
        self.result = result
        super().__init__()


class TranscriptionUpdate(Message):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class TranscriptionFinal(Message):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class ModelsReady(Message):
    pass


class WelcomeMessage(Static):
    DEFAULT_CSS = """
    WelcomeMessage {
        height: auto;
        padding: 2 2;
        color: $text-muted;
        content-align: center middle;
        text-align: center;
        width: 100%;
    }
    """

    def __init__(self) -> None:
        super().__init__("voxing\n\nlocal voice assistant")


class FooterBar(Static):
    DEFAULT_CSS = """
    FooterBar {
        height: auto;
        border: none;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__("")

    def set_status(self, status: str) -> None:
        """Update the footer status text."""
        self.update(status)


class UserMessage(Widget):
    DEFAULT_CSS = """
    UserMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0;
    }
    """

    def __init__(self, content: str) -> None:
        self._content = content
        super().__init__()

    def render(self) -> str:
        return f"> {self._content}"


class AssistantMessage(Widget):
    DEFAULT_CSS = """
    AssistantMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """

    def __init__(self) -> None:
        self._text = ""
        self._finalized = False
        super().__init__()

    def render(self) -> str:
        return self._text + ("" if self._finalized else CURSOR)

    def append_token(self, token: str) -> None:
        """Append a streaming token and refresh."""
        self._text += token
        self.refresh()

    def finalize(self) -> None:
        """Mark generation complete, remove cursor."""
        self._finalized = True
        self.refresh()


class ToolCallWidget(Widget):
    DEFAULT_CSS = """
    ToolCallWidget {
        height: auto;
        padding: 0 1;
        margin: 0 0;
    }
    ToolCallWidget.expanded {
        height: auto;
    }
    """

    def __init__(self, code: str, result: str) -> None:
        self._code = code
        self._result = result
        self._expanded = False
        super().__init__()

    def render(self) -> str:
        icon = "\u25bc" if self._expanded else "\u25b6"
        header = f"{icon} tool call"
        if not self._expanded:
            return header
        return f"{header}\n\n{self._code}\n\nOutput:\n{self._result}"

    def on_click(self) -> None:
        self._expanded = not self._expanded
        self.toggle_class("expanded")
        self.refresh()


class MessageList(VerticalScroll):
    DEFAULT_CSS = """
    MessageList {
        height: 1fr;
        scrollbar-size: 0 0;
    }
    """

    def add_user_message(self, content: str) -> None:
        """Add a user message and scroll to bottom."""
        self.mount(UserMessage(content))
        self.scroll_end(animate=False)

    def add_assistant_message(self) -> AssistantMessage:
        """Add a new assistant message widget and return it."""
        msg = AssistantMessage()
        self.mount(msg)
        self.scroll_end(animate=False)
        return msg

    def add_tool_call(self, code: str, result: str) -> None:
        """Add a tool call widget."""
        self.mount(ToolCallWidget(code, result))
        self.scroll_end(animate=False)

    def clear_messages(self) -> None:
        """Remove all child widgets."""
        self.remove_children()


class CommandHints(Static):
    DEFAULT_CSS = """
    CommandHints {
        height: auto;
        max-height: 6;
        padding: 0 2;
        background: $background;
        color: $text;
    }
    """

    def __init__(self) -> None:
        super().__init__("")
        self.display = False

    def update_hints(self, prefix: str) -> None:
        """Show matching commands for the given prefix."""
        if not prefix.startswith("/"):
            self.display = False
            return
        matches = [
            f"{cmd}  {desc}"
            for cmd, desc in SLASH_COMMANDS.items()
            if cmd.startswith(prefix)
        ]
        if matches:
            self.update("\n".join(matches))
            self.display = True
        else:
            self.display = False


class ChatInput(Input):
    DEFAULT_CSS = """
    ChatInput {
        height: auto;
        border: none;
        border-top: solid $panel;
        border-bottom: solid $panel;
        padding: 0 2;
        background: $background;
    }
    ChatInput:focus {
        border: none;
        border-top: solid $panel;
        border-bottom: solid $panel;
        background: $background;
        background-tint: transparent;
    }
    """

    def __init__(self) -> None:
        super().__init__(
            placeholder="Type a message or / for commands...",
            suggester=SuggestFromList(list(SLASH_COMMANDS.keys()), case_sensitive=True),
        )
