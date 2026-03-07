from textual.app import App

from voxing.tui.screens import ChatScreen
from voxing.tui.theme import CATPPUCCIN_MOCHA


class VoxingApp(App[None]):
    TITLE = "Voxing"
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
    ]

    def on_mount(self) -> None:
        self.register_theme(CATPPUCCIN_MOCHA)
        self.theme = CATPPUCCIN_MOCHA.name
        self.push_screen(ChatScreen())

    async def action_quit(self) -> None:
        if isinstance(self.screen, ChatScreen):
            self.screen.shutdown_workers()
        await super().action_quit()
