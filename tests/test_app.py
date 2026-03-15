from unittest.mock import patch

import pytest
from textual.widgets import Static

from voxing.tui.app import VoxingApp
from voxing.tui.widgets import (
    ChatInput,
    CommandHints,
    FooterBar,
    UserMessage,
    WelcomeMessage,
)


@pytest.fixture
def app() -> VoxingApp:
    return VoxingApp()


async def _type_and_submit(pilot, text: str) -> None:
    """Type text into the chat input and press enter."""
    for char in text:
        await pilot.press(char)
    await pilot.press("enter")


async def test_mount_shows_welcome_message(app: VoxingApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        assert len(app.screen.query(WelcomeMessage)) == 1


async def test_mount_has_chat_input(app: VoxingApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        assert len(app.screen.query(ChatInput)) == 1


async def test_mount_has_footer(app: VoxingApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        assert len(app.screen.query(FooterBar)) == 1


async def test_command_clear_resets_and_shows_welcome(app: VoxingApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type_and_submit(pilot, "/clear")
        await pilot.pause()
        assert len(app.screen.query(WelcomeMessage)) == 1


async def test_command_help_shows_commands_in_footer(app: VoxingApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type_and_submit(pilot, "/help")
        await pilot.pause()
        status_text = str(app.screen.query_one("#status", Static).content)
        assert "/clear" in status_text
        assert "/help" in status_text


async def test_command_unknown_shows_error(app: VoxingApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type_and_submit(pilot, "/foo")
        await pilot.pause()
        status_text = str(app.screen.query_one("#status", Static).content)
        assert "Unknown" in status_text


@patch("voxing.tui.screens.chat.load_llm")
async def test_user_message_adds_widget(mock_load_llm, app: VoxingApp) -> None:
    mock_load_llm.side_effect = Exception("no model")
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type_and_submit(pilot, "hello world")
        await pilot.pause()
        user_msgs = app.screen.query(UserMessage)
        assert len(user_msgs) == 1


async def test_hints_slash_shows_hints(app: VoxingApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("/")
        await pilot.pause()
        hints = app.screen.query_one(CommandHints)
        assert hints.display is True


async def test_hints_slash_cl_filters_to_clear(app: VoxingApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        for char in "/cl":
            await pilot.press(char)
        await pilot.pause()
        hints = app.screen.query_one(CommandHints)
        assert hints.display is True
        hints_text = str(hints.content)
        assert "/clear" in hints_text
        assert "/help" not in hints_text


async def test_hints_no_slash_hides_hints(app: VoxingApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("h")
        await pilot.pause()
        hints = app.screen.query_one(CommandHints)
        assert hints.display is False


async def test_settings_opens_and_dismisses(app: VoxingApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        await _type_and_submit(pilot, "/settings")
        await pilot.pause()
        assert app.screen.__class__.__name__ == "SettingsScreen"
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen.__class__.__name__ != "SettingsScreen"


def test_snap_initial(snap_compare):
    """Welcome screen on startup."""
    assert snap_compare(VoxingApp(), terminal_size=(80, 24))


def test_snap_command_hints(snap_compare):
    """Command hints appear when typing /."""
    assert snap_compare(VoxingApp(), press=["/"], terminal_size=(80, 24))


def test_snap_after_clear(snap_compare):
    """Screen after /clear command."""
    assert snap_compare(
        VoxingApp(),
        press=["/", "c", "l", "e", "a", "r", "enter"],
        terminal_size=(80, 24),
    )


def test_snap_settings_screen(snap_compare):
    """Settings screen opened via /settings."""
    assert snap_compare(
        VoxingApp(),
        press=["/", "s", "e", "t", "t", "i", "n", "g", "s", "enter"],
        terminal_size=(80, 24),
    )
