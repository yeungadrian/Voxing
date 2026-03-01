from dataclasses import dataclass, field
from typing import Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Input, Markdown, Static

from voxing.config import Settings

type SettingKind = Literal["bool", "int", "float", "str"]


@dataclass
class SettingEntry:
    key: str
    label: str
    kind: SettingKind
    value: bool | int | float | str


@dataclass
class SettingsResult:
    tools_enabled: bool
    system_prompt: str
    config_overrides: dict[str, bool | int | float | str] = field(default_factory=dict)


_EXCLUDED_SETTINGS = frozenset({"model_id", "llm_model_id", "sample_rate"})


def _build_entries(
    settings: Settings, tools_enabled: bool, system_prompt: str
) -> list[SettingEntry]:
    """Build setting entries from runtime state and config fields."""
    entries: list[SettingEntry] = [
        SettingEntry(
            key="tools_enabled", label="Tools enabled", kind="bool", value=tools_enabled
        ),
        SettingEntry(
            key="system_prompt",
            label="System prompt",
            kind="str",
            value=system_prompt,
        ),
    ]
    for name in settings.model_fields:
        if name in _EXCLUDED_SETTINGS:
            continue
        raw = getattr(settings, name)
        if isinstance(raw, bool):
            kind: SettingKind = "bool"
        elif isinstance(raw, int):
            kind = "int"
        elif isinstance(raw, float):
            kind = "float"
        else:
            kind = "str"
        label = name.replace("_", " ").title()
        entries.append(SettingEntry(key=name, label=label, kind=kind, value=raw))
    return entries


class SettingRow(Static):
    can_focus = True

    def __init__(self, entry: SettingEntry) -> None:
        self.entry = entry
        super().__init__()

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Static(self.entry.label, classes="setting-label")
            yield Static(self._format_value(), classes="setting-value")
            yield Input(value=str(self.entry.value), classes="setting-input")

    def _format_value(self) -> str:
        """Format the setting value for display."""
        if self.entry.kind == "bool":
            return "true" if self.entry.value else "false"
        return str(self.entry.value)

    def begin_edit(self) -> None:
        """Enter inline edit mode."""
        self.add_class("-editing")
        inp = self.query_one(Input)
        inp.value = str(self.entry.value)
        inp.focus()

    def commit_edit(self) -> bool:
        """Parse and commit the edited value. Returns False on parse error."""
        raw = self.query_one(Input).value
        try:
            match self.entry.kind:
                case "bool":
                    if raw.lower() in ("true", "1", "yes"):
                        parsed: bool | int | float | str = True
                    elif raw.lower() in ("false", "0", "no"):
                        parsed = False
                    else:
                        return False
                case "int":
                    parsed = int(raw)
                case "float":
                    parsed = float(raw)
                case "str":
                    parsed = raw
        except ValueError:
            return False
        self.entry.value = parsed
        self.query_one(".setting-value", Static).update(self._format_value())
        self.remove_class("-editing")
        return True

    def cancel_edit(self) -> None:
        """Cancel editing and restore display."""
        self.remove_class("-editing")

    def toggle_bool(self) -> None:
        """Toggle a boolean setting without entering edit mode."""
        if self.entry.kind != "bool":
            return
        self.entry.value = not self.entry.value
        self.query_one(".setting-value", Static).update(self._format_value())


class SettingsList(VerticalScroll):
    def __init__(self, entries: list[SettingEntry]) -> None:
        self._all_entries = entries
        self._highlight_index: int = 0
        super().__init__()

    def on_mount(self) -> None:
        for entry in self._all_entries:
            self.mount(SettingRow(entry))
        self._update_highlight()

    def filter(self, query: str) -> None:
        """Filter visible rows by query string."""
        for child in list(self.query(SettingRow)):
            child.remove()
        q = query.lower()
        for entry in self._all_entries:
            if q in entry.label.lower() or q in entry.key.lower():
                self.mount(SettingRow(entry))
        self._highlight_index = 0
        self._update_highlight()

    def _update_highlight(self) -> None:
        """Sync the -highlight class to the current index."""
        rows = list(self.query(SettingRow))
        for i, row in enumerate(rows):
            if i == self._highlight_index:
                row.add_class("-highlight")
            else:
                row.remove_class("-highlight")

    def move_up(self) -> None:
        """Move highlight up one row."""
        if self._highlight_index > 0:
            self._highlight_index -= 1
            self._update_highlight()
            row = self.highlighted_row
            if row is not None:
                row.scroll_visible()

    def move_down(self) -> None:
        """Move highlight down one row."""
        rows = list(self.query(SettingRow))
        if self._highlight_index < len(rows) - 1:
            self._highlight_index += 1
            self._update_highlight()
            row = self.highlighted_row
            if row is not None:
                row.scroll_visible()

    @property
    def highlighted_row(self) -> SettingRow | None:
        """Return the currently highlighted SettingRow."""
        rows = list(self.query(SettingRow))
        if 0 <= self._highlight_index < len(rows):
            return rows[self._highlight_index]
        return None


class SettingsScreen(Screen[SettingsResult | None]):
    BINDINGS = [
        Binding("escape", "save_and_dismiss", "Save & back", priority=True),
        Binding("enter", "activate", "Edit/Toggle", priority=True),
    ]

    def __init__(
        self, settings: Settings, tools_enabled: bool, system_prompt: str
    ) -> None:
        self._settings = settings
        self._tools_enabled = tools_enabled
        self._system_prompt = system_prompt
        self._entries = _build_entries(settings, tools_enabled, system_prompt)
        self._editing = False
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search settings...", id="search-bar")
        yield SettingsList(self._entries)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search-bar":
            self.query_one(SettingsList).filter(event.value)

    def action_save_and_dismiss(self) -> None:
        if self._editing:
            row = self.query_one(SettingsList).highlighted_row
            if row is not None:
                row.cancel_edit()
            self._editing = False
            self.query_one(SettingsList).focus()
            return
        overrides: dict[str, bool | int | float | str] = {}
        tools = self._tools_enabled
        prompt = self._system_prompt
        for entry in self._entries:
            if entry.key == "tools_enabled":
                tools = bool(entry.value)
            elif entry.key == "system_prompt":
                prompt = str(entry.value)
            else:
                original = getattr(self._settings, entry.key)
                if entry.value != original:
                    overrides[entry.key] = entry.value
        self.dismiss(SettingsResult(tools, prompt, overrides))

    def action_activate(self) -> None:
        if self._editing:
            row = self.query_one(SettingsList).highlighted_row
            if row is not None:
                row.commit_edit()
            self._editing = False
            self.query_one(SettingsList).focus()
            return
        row = self.query_one(SettingsList).highlighted_row
        if row is None:
            return
        if row.entry.kind == "bool":
            row.toggle_bool()
        else:
            row.begin_edit()
            self._editing = True

    def on_key(self, event) -> None:
        if self._editing:
            return
        settings_list = self.query_one(SettingsList)
        search = self.query_one("#search-bar", Input)
        if not search.has_focus:
            if event.key == "up":
                settings_list.move_up()
                event.prevent_default()
            elif event.key == "down":
                settings_list.move_down()
                event.prevent_default()
            elif event.key in ("left", "right"):
                row = settings_list.highlighted_row
                if row is not None and row.entry.kind == "bool":
                    row.toggle_bool()
                event.prevent_default()
            elif event.is_printable and event.character:
                search.focus()
                search.value += event.character
                event.prevent_default()


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
