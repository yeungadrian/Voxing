"""Visual regression tests using Textual snapshots."""

from vox.app import VoxApp
from vox.widgets import ChatInput


def test_initial_app_screen(mock_model_loading, snap_compare):
    """Snapshot of initial app after models load."""

    async def run_before(pilot) -> None:
        await pilot.pause(0.5)

    assert snap_compare(VoxApp(), run_before=run_before)


def test_command_hints_display(mock_model_loading, snap_compare):
    """Snapshot showing command hints when typing '/'."""

    async def run_before(pilot) -> None:
        await pilot.pause(0.5)
        text_area = pilot.app.query_one("#user-input", ChatInput)
        text_area.insert("/rec")
        await pilot.pause(0.1)

    assert snap_compare(VoxApp(), run_before=run_before)
