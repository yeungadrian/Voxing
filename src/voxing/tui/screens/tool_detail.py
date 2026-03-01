from dataclasses import dataclass

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Markdown, Static


@dataclass
class ToolCallData:
    name: str
    code: str
    result: str


class ToolDetailScreen(Screen[None]):
    BINDINGS = [
        Binding("escape", "dismiss", "Back"),
    ]

    DEFAULT_CSS = """
    ToolDetailScreen {
        background: $background;
    }
    ToolDetailScreen Static {
        color: $text-muted;
        padding: 0 1;
    }
    ToolDetailScreen .tool-header {
        color: $warning;
    }
    """

    def __init__(self, tool_calls: list[ToolCallData]) -> None:
        self._tool_calls = tool_calls
        super().__init__()

    def compose(self) -> ComposeResult:
        """Compose the tool detail view."""
        with VerticalScroll():
            for i, tc in enumerate(self._tool_calls, 1):
                yield Static(f"\u26a1 {i}. {tc.name}", classes="tool-header")
                parts = [f"**Input**\n```python\n{tc.code}\n```"]
                if tc.result.strip():
                    parts.append(f"**Output**\n```\n{tc.result.strip()}\n```")
                yield Markdown("\n\n".join(parts))
        yield Static("escape to return")
