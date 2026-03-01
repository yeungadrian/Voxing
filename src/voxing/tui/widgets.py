import psutil
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.suggester import SuggestFromList
from textual.widget import Widget
from textual.widgets import Input, Markdown, Static

SLASH_COMMANDS: dict[str, str] = {
    "/transcribe": "Start voice transcription",
    "/settings": "Open settings panel",
    "/clear": "Clear chat history",
    "/help": "Show available commands",
}

CURSOR = "\u258c"


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


class MemoryDisplay(Static):
    DEFAULT_CSS = """
    MemoryDisplay {
        width: auto;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def on_mount(self) -> None:
        """Start polling memory usage every second."""
        self._process = psutil.Process()
        self.set_interval(1.0, self._update)

    def _update(self) -> None:
        """Refresh displayed memory from process RSS."""
        mb = self._process.memory_info().rss / 1024 / 1024
        self.update(f"{mb:.0f} MB")


class FooterBar(Widget):
    DEFAULT_CSS = """
    FooterBar {
        height: auto;
        border: none;
        layout: horizontal;
    }
    FooterBar > #status {
        width: 1fr;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose status label and memory display."""
        yield Static("", id="status")
        yield MemoryDisplay()

    def set_status(self, status: str) -> None:
        """Update the footer status text."""
        self.query_one("#status", Static).update(status)


class TranscriptionDisplay(Widget):
    DEFAULT_CSS = """
    TranscriptionDisplay {
        height: auto;
        padding: 0 1;
        margin: 0 0;
    }
    TranscriptionDisplay > #transcription-text {
        color: $text-muted;
    }
    TranscriptionDisplay > #transcription-text.active {
        color: $text;
    }
    TranscriptionDisplay > #transcription-hint {
        color: $text-muted;
    }
    """

    _HINT = "Press Escape to stop"

    def __init__(self) -> None:
        self._text = ""
        super().__init__()

    def compose(self) -> ComposeResult:
        """Compose the mic indicator, live text, and stop hint."""
        yield Static("⏺ Recording", id="recording-label")
        yield Static("Listening...", id="transcription-text")
        yield Static(self._HINT, id="transcription-hint")

    def update_text(self, text: str) -> None:
        """Update the live transcription text and switch to active colour."""
        self._text = text
        text_widget = self.query_one("#transcription-text", Static)
        text_widget.update(text)
        text_widget.add_class("active")


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
        layout: horizontal;
        height: auto;
        padding: 0 1;
        margin: 0;
    }
    AssistantMessage > Static {
        width: 2;
    }
    AssistantMessage > Markdown {
        width: 1fr;
        margin: 0;
        padding: 0;
    }
    """

    ICON = "\u25cf"

    def __init__(self) -> None:
        self._text = ""
        self._finalized = False
        super().__init__()

    def compose(self) -> ComposeResult:
        """Compose the icon and markdown widget."""
        yield Static(self.ICON)
        yield Markdown(CURSOR)

    @property
    def is_empty(self) -> bool:
        """Check if the message has no text content."""
        return not self._text.strip()

    def append_token(self, token: str) -> None:
        """Append a streaming token and refresh."""
        self._text += token
        self.query_one(Markdown).update(self._text + CURSOR)

    def finalize(self) -> None:
        """Mark generation complete, remove cursor."""
        self._finalized = True
        self.query_one(Markdown).update(self._text)


class ToolCallWidget(Static):
    DEFAULT_CSS = """
    ToolCallWidget {
        height: auto;
        margin: 0;
        padding: 0 1;
        color: $text-muted;
    }
    """

    can_focus = False

    def __init__(self, code: str, result: str, name: str) -> None:
        self.code = code
        self.result = result
        self.tool_name = name
        super().__init__(f"\u26a1 {name}")


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

    def add_tool_call(self, code: str, result: str, name: str) -> None:
        """Add a tool call widget."""
        self.mount(ToolCallWidget(code, result, name))
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
