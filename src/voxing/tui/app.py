from textual.app import App

from voxing.tui.screens import ChatScreen
from voxing.tui.theme import CATPPUCCIN_MOCHA


class VoxingApp(App[None]):
    TITLE = "Voxing"

    def on_mount(self) -> None:
        self.register_theme(CATPPUCCIN_MOCHA)
        self.theme = CATPPUCCIN_MOCHA.name
        self.push_screen(ChatScreen())
