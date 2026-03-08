import mlx.core as mx
import numpy as np
from rich.segment import Segment
from rich.style import Style
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.events import Key
from textual.message import Message
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Markdown, Static, TextArea

from voxing.palette import BLUE
from voxing.viz import (
    BRAILLE_BASE,
    OscilloscopeViz,
    Visualizer,
    VizFrame,
    WaveformViz,
)


class VizWidget(Widget):
    """Generic visualizer widget that renders any Visualizer implementation."""

    DEFAULT_CSS = """
    VizWidget {
        height: 1fr;
        width: 1fr;
    }
    """

    def __init__(self, viz: Visualizer, refresh_rate: float = 30.0) -> None:
        super().__init__()
        self._viz = viz
        self._refresh_rate = refresh_rate
        self._frame: VizFrame | None = None
        self._style = Style.parse(BLUE)
        self._color_cache: dict[str, Style] = {}

    def on_mount(self) -> None:
        """Start periodic refresh to animate the visualizer."""
        self.set_interval(1.0 / self._refresh_rate, self._tick)

    def _tick(self) -> None:
        """Recompute the frame then refresh."""
        self._frame = self._viz.render(self.size.width or 1, self.size.height or 3)
        self.refresh()

    def push_chunk(self, chunk: np.ndarray) -> None:
        """Forward an audio chunk to the underlying visualizer."""
        self._viz.push(chunk)

    @property
    def viz(self) -> Visualizer:
        """Expose wrapped visualizer instance."""
        return self._viz

    def _color_style(self, color: str) -> Style:
        """Get or create a cached Style for a hex color."""
        style = self._color_cache.get(color)
        if style is None:
            style = Style.parse(color)
            self._color_cache[color] = style
        return style

    def render_line(self, y: int) -> Strip:
        """Render one terminal row from the current frame."""
        frame = self._frame
        if frame is None or y >= len(frame.grid):
            return Strip([Segment(" " * (self.size.width or 1))])

        row = frame.grid[y]
        colors = frame.colors
        color_row = colors[y] if colors is not None and y < len(colors) else None
        segments: list[Segment] = []

        for ci, bits in enumerate(row):
            if bits:
                style = (
                    self._color_style(color_row[ci])
                    if color_row is not None
                    else self._style
                )
                segments.append(Segment(chr(BRAILLE_BASE + bits), style))
            else:
                segments.append(Segment(" "))

        return Strip(segments)


SLASH_COMMANDS: dict[str, str] = {
    "/transcribe": "Start voice transcription",
    "/settings": "Open settings panel",
    "/clear": "Clear chat history",
    "/help": "Show available commands",
    "/exit": "Quit the application",
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
        super().__init__("[bold $primary]voxing[/]\n\n[dim]local voice assistant[/]")


class MemoryDisplay(Static):
    DEFAULT_CSS = """
    MemoryDisplay {
        width: auto;
        color: $text-muted;
        padding: 0 2;
    }
    """

    def on_mount(self) -> None:
        """Start polling memory usage every second."""
        self.set_interval(1.0, self._update)

    def _update(self) -> None:
        """Refresh displayed memory from MLX active memory."""
        mb = mx.get_active_memory() / 1024 / 1024
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
        padding: 0 2;
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
    TranscriptionDisplay > VizWidget {
        height: 3;
        width: 33%;
    }
    TranscriptionDisplay > #recording-label {
        color: $error;
    }
    TranscriptionDisplay > #transcription-text {
        color: $text-muted;
    }
    TranscriptionDisplay > #transcription-text.active {
        color: $text;
    }
    """

    def __init__(
        self,
        audio_visual: str = "waveform",
        sample_rate: int = 16_000,
        chunk_duration: float = 0.1,
    ) -> None:
        super().__init__()
        viz: Visualizer | None = None
        match audio_visual:
            case "waveform":
                viz = WaveformViz()
            case "oscilloscope":
                viz = OscilloscopeViz()
        self._visualizer = VizWidget(viz) if viz else None

    def compose(self) -> ComposeResult:
        """Compose the visualizer, mic indicator and live text."""
        if self._visualizer is not None:
            yield self._visualizer
        yield Static("\u23fa Recording", id="recording-label")
        yield Static("Listening...", id="transcription-text")

    def update_text(self, text: str) -> None:
        """Update the live transcription text and switch to active colour."""
        text_widget = self.query_one("#transcription-text", Static)
        text_widget.update(text)
        text_widget.add_class("active")

    def push_chunk(self, chunk: np.ndarray) -> None:
        """Forward an audio chunk to the visualizer."""
        if self._visualizer is not None:
            self._visualizer.push_chunk(chunk)


class UserMessage(Widget):
    DEFAULT_CSS = """
    UserMessage {
        layout: horizontal;
        height: auto;
        padding: 0 1;
        margin: 0 0;
    }
    UserMessage > .user-icon {
        width: 2;
        color: $secondary;
    }
    UserMessage > .user-text {
        width: 1fr;
    }
    """

    def __init__(self, content: str) -> None:
        self._content = content
        super().__init__()

    def compose(self) -> ComposeResult:
        """Compose the icon and text."""
        yield Static(">", classes="user-icon")
        yield Static(self._content, classes="user-text")


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
        color: $accent;
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
        self.query_one(Markdown).update(self._text)


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
        color: $text-muted;
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


class ChatInput(TextArea):
    DEFAULT_CSS = """
    ChatInput {
        height: auto;
        max-height: 6;
        min-height: 1;
        border: none;
        border-top: solid $panel;
        border-bottom: solid $panel;
        padding: 0 2;
        background: $background;
        background-tint: transparent;
        scrollbar-size: 0 0;
    }
    ChatInput:focus {
        border: none;
        border-top: solid $panel;
        border-bottom: solid $panel;
    }
    ChatInput .text-area--cursor-line {
        background: transparent;
    }
    """

    class Submitted(Message):
        """Posted when the user presses Enter."""

        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    def __init__(self) -> None:
        super().__init__(
            language=None,
            soft_wrap=True,
            show_line_numbers=False,
            tab_behavior="focus",
        )

    def update_suggestion(self) -> None:
        """Suggest slash command completion based on current input."""
        text = self.text
        if text.startswith("/") and "\n" not in text:
            for cmd in SLASH_COMMANDS:
                if cmd.startswith(text) and cmd != text:
                    self.suggestion = cmd[len(text) :]
                    return
        self.suggestion = ""

    def _on_text_area_changed(self) -> None:
        """Update slash command suggestion when text changes."""
        self.update_suggestion()

    async def _on_key(self, event: Key) -> None:
        """Intercept Enter to submit and Tab to accept suggestion or no-op."""
        if event.key == "tab":
            event.prevent_default()
            event.stop()
            if self.suggestion:
                self.insert(self.suggestion)
                self.suggestion = ""
            return
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            text = self.text.strip()
            if text:
                self.post_message(self.Submitted(text))
            return
