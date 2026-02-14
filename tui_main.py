"""Entry point for Reachy TUI application."""

from rich.console import Console

from reachy_tui.app import ReachyTuiApp
from reachy_tui.models import load_models

COLOR_BLUE = "#7aa2f7"


def main() -> None:
    """Load models and run the Reachy TUI application."""
    console = Console()
    with console.status(f"[bold {COLOR_BLUE}]Loading models..."):
        models = load_models()
    console.print(f"[bold {COLOR_BLUE}]Models loaded!")

    app = ReachyTuiApp(models)
    app.run()


if __name__ == "__main__":
    main()
