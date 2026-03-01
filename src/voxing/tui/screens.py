from dataclasses import dataclass

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Label, Markdown, Static, Switch, TextArea


class SettingRow(Vertical):
    DEFAULT_CSS = """
    SettingRow {
        height: auto;
        padding: 1 2;
    }
    SettingRow:focus-within {
        background: $surface;
    }
    """


class SettingsScreen(Screen[tuple[bool, str] | None]):
    BINDINGS = [
        Binding("escape", "cancel", "Back"),
    ]

    DEFAULT_CSS = """
    SettingsScreen {
        background: $background;
    }
    """

    def __init__(self, tools_enabled: bool, system_prompt: str) -> None:
        self._tools_enabled = tools_enabled
        self._system_prompt = system_prompt
        super().__init__()

    def compose(self):
        yield Label("Settings", id="settings-title")
        with VerticalScroll(id="settings-list"):
            with SettingRow():
                yield Label("Tools enabled")
                yield Switch(value=self._tools_enabled, id="tools-switch")
            with SettingRow():
                yield Label("System prompt")
                yield TextArea(self._system_prompt, id="system-prompt")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_save(self) -> None:
        tools = self.query_one("#tools-switch", Switch).value
        prompt = self.query_one("#system-prompt", TextArea).text
        self.dismiss((tools, prompt))


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
    """

    def __init__(self, tool_calls: list[ToolCallData]) -> None:
        self._tool_calls = tool_calls
        super().__init__()

    def compose(self) -> ComposeResult:
        """Compose the tool detail view."""
        with VerticalScroll():
            for i, tc in enumerate(self._tool_calls, 1):
                yield Static(f"\u26a1 {i}. {tc.name}")
                parts = [f"**Input**\n```python\n{tc.code}\n```"]
                if tc.result.strip():
                    parts.append(f"**Output**\n```\n{tc.result.strip()}\n```")
                yield Markdown("\n\n".join(parts))
        yield Static("escape to return")
