from dataclasses import dataclass, field
from typing import Literal, get_args

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.events import Key
from textual.screen import Screen
from textual.widgets import Input, Static

from voxing.config import Settings
from voxing.tui.widgets import FooterBar

type SettingKind = Literal["bool", "int", "float", "str", "choice"]


@dataclass
class SettingEntry:
    key: str
    label: str
    kind: SettingKind
    value: bool | int | float | str
    choices: list[str] = field(default_factory=list)


@dataclass
class SettingsResult:
    config_overrides: dict[str, bool | int | float | str] = field(default_factory=dict)


_EXCLUDED_SETTINGS = frozenset(
    {"model_id", "llm_model_id", "tts_model_id", "sample_rate"}
)


def _build_entries(settings: Settings) -> list[SettingEntry]:
    """Build setting entries from config fields."""
    entries: list[SettingEntry] = []
    for name in Settings.model_fields:
        if name in _EXCLUDED_SETTINGS:
            continue
        raw = getattr(settings, name)
        annotation = Settings.model_fields[name].annotation
        args = get_args(annotation) if annotation is not None else ()
        choices: list[str] = []
        if args and all(isinstance(a, str) for a in args):
            kind: SettingKind = "choice"
            choices = list(args)
        elif isinstance(raw, bool):
            kind = "bool"
        elif isinstance(raw, int):
            kind = "int"
        elif isinstance(raw, float):
            kind = "float"
        else:
            kind = "str"
        label = name.replace("_", " ").title()
        entries.append(
            SettingEntry(key=name, label=label, kind=kind, value=raw, choices=choices)
        )
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
        inp.action_end()

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
                case "str" | "choice":
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

    def cycle_choice(self, direction: int) -> None:
        """Cycle through choices by +1 or -1, wrapping around."""
        if self.entry.kind != "choice" or not self.entry.choices:
            return
        current = str(self.entry.value)
        try:
            idx = self.entry.choices.index(current)
        except ValueError:
            idx = 0
        idx = (idx + direction) % len(self.entry.choices)
        self.entry.value = self.entry.choices[idx]
        self.query_one(".setting-value", Static).update(self._format_value())


class SettingsList(VerticalScroll):
    def __init__(self, entries: list[SettingEntry]) -> None:
        self._entries = entries
        self._highlight_index: int = 0
        super().__init__()

    def on_mount(self) -> None:
        for entry in self._entries:
            self.mount(SettingRow(entry))
        self._update_highlight()

    def filter(self, query: str) -> None:
        """Filter visible rows by query string."""
        q = query.lower()
        for row in self.query(SettingRow):
            row.display = (
                not q or q in row.entry.label.lower() or q in row.entry.key.lower()
            )
        self._highlight_index = 0
        self._update_highlight()

    def _visible_rows(self) -> list[SettingRow]:
        """Return only the currently visible (non-hidden) rows."""
        return [row for row in self.query(SettingRow) if row.display]

    def _update_highlight(self) -> None:
        """Sync the -highlight class to the current index."""
        rows = self._visible_rows()
        for i, row in enumerate(rows):
            if i == self._highlight_index:
                row.add_class("-highlight")
            else:
                row.remove_class("-highlight")

    def move_up(self) -> SettingRow | None:
        """Move highlight up one row, returning the new row."""
        if self._highlight_index > 0:
            self._highlight_index -= 1
            self._update_highlight()
            row = self.highlighted_row
            if row is not None:
                row.scroll_visible()
            return row
        return None

    def move_down(self) -> SettingRow | None:
        """Move highlight down one row, returning the new row."""
        rows = self._visible_rows()
        if self._highlight_index < len(rows) - 1:
            self._highlight_index += 1
            self._update_highlight()
            row = self.highlighted_row
            if row is not None:
                row.scroll_visible()
            return row
        return None

    @property
    def highlighted_row(self) -> SettingRow | None:
        """Return the currently highlighted SettingRow."""
        rows = self._visible_rows()
        if 0 <= self._highlight_index < len(rows):
            return rows[self._highlight_index]
        return None


class SettingsScreen(Screen[SettingsResult | None]):
    BINDINGS = [
        Binding("escape", "save_and_dismiss", "Save & back", priority=True),
        Binding("tab", "focus_search", "Search", priority=True),
    ]

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._entries = _build_entries(settings)
        self._editing = False
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search settings...", id="search-bar")
        yield SettingsList(self._entries)
        yield FooterBar()

    def on_mount(self) -> None:
        self.query_one(FooterBar).set_status(
            "[dim]esc to save & return  ·  tab to search  ·  ←→ toggle bools/choices[/]"
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search-bar":
            self.query_one(SettingsList).filter(event.value)

    def action_save_and_dismiss(self) -> None:
        if self._editing:
            row = self.query_one(SettingsList).highlighted_row
            if row is not None:
                row.commit_edit()
            self._editing = False
        overrides: dict[str, bool | int | float | str] = {}
        for entry in self._entries:
            original = getattr(self._settings, entry.key)
            if entry.value != original:
                overrides[entry.key] = entry.value
        self.dismiss(SettingsResult(overrides))

    def action_focus_search(self) -> None:
        search = self.query_one("#search-bar", Input)
        if search.has_focus:
            self.query_one(SettingsList).focus()
        else:
            self._commit_current_edit()
            search.focus()

    def _commit_current_edit(self) -> None:
        """Commit the current edit if one is active."""
        if not self._editing:
            return
        row = self.query_one(SettingsList).highlighted_row
        if row is not None:
            row.commit_edit()
        self._editing = False

    def _auto_edit_row(self, row: SettingRow | None) -> None:
        """Auto-enter edit mode if the row is non-bool and non-choice."""
        if row is not None and row.entry.kind not in ("bool", "choice"):
            row.begin_edit()
            self._editing = True

    def on_key(self, event: Key) -> None:
        settings_list = self.query_one(SettingsList)
        search = self.query_one("#search-bar", Input)

        if self._editing:
            if event.key in ("up", "down"):
                self._commit_current_edit()
                new_row = (
                    settings_list.move_up()
                    if event.key == "up"
                    else settings_list.move_down()
                )
                self._auto_edit_row(new_row)
                event.prevent_default()
            return

        if not search.has_focus:
            if event.key == "up":
                self._commit_current_edit()
                new_row = settings_list.move_up()
                self._auto_edit_row(new_row)
                event.prevent_default()
            elif event.key == "down":
                self._commit_current_edit()
                new_row = settings_list.move_down()
                self._auto_edit_row(new_row)
                event.prevent_default()
            elif event.key in ("left", "right"):
                row = settings_list.highlighted_row
                if row is not None:
                    if row.entry.kind == "bool":
                        row.toggle_bool()
                    elif row.entry.kind == "choice":
                        row.cycle_choice(1 if event.key == "right" else -1)
                event.prevent_default()
            elif event.is_printable and event.character:
                search.focus()
                search.value += event.character
                event.prevent_default()
