from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Label, Switch, TextArea


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
