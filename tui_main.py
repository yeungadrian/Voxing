"""Entry point for Reachy TUI application."""

from reachy_tui.app import ReachyTuiApp


def main() -> None:
    """Run the Reachy TUI application."""
    app = ReachyTuiApp()
    app.run()


if __name__ == "__main__":
    main()
