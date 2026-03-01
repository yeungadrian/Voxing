from dataclasses import dataclass

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Markdown, Static


@dataclass
class ToolCallData:
    name: str
    code: str
    result: str


class ToolEntry(Horizontal):
    DEFAULT_CSS = """
    ToolEntry {
        height: auto;
        padding: 0 1;
        margin: 0;
    }
    ToolEntry > .tool-icon {
        width: 2;
        color: $warning;
    }
    ToolEntry > .tool-content {
        width: 1fr;
    }
    ToolEntry > .tool-content Markdown {
        margin: 0;
        padding: 0;
    }
    """

    def __init__(self, tool_call: ToolCallData) -> None:
        self._tc = tool_call
        super().__init__()

    def compose(self) -> ComposeResult:
        """Compose icon and tool content."""
        yield Static("\u26a1", classes="tool-icon")
        parts = [f"**{self._tc.name}**\n```python\n{self._tc.code}\n```"]
        if self._tc.result.strip():
            parts.append(f"```\n{self._tc.result.strip()}\n```")
        yield Markdown("\n".join(parts), classes="tool-content")


class ToolDetailScreen(Screen[None]):
    BINDINGS = [
        Binding("escape", "dismiss", "Back"),
    ]

    DEFAULT_CSS = """
    ToolDetailScreen {
        background: $background;
    }
    ToolDetailScreen > VerticalScroll {
        scrollbar-size: 0 0;
    }
    ToolDetailScreen > #hint {
        color: $text-muted;
        padding: 0 1;
        dock: bottom;
    }
    """

    def __init__(self, tool_calls: list[ToolCallData]) -> None:
        self._tool_calls = tool_calls
        super().__init__()

    def compose(self) -> ComposeResult:
        """Compose the tool detail view."""
        with VerticalScroll():
            for tc in self._tool_calls:
                yield ToolEntry(tc)
        yield Static("[dim]esc to return[/]", id="hint")
