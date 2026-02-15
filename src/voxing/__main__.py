"""Entry point for Voxinging TUI application."""

from voxing.app import VoxingApp


def main() -> None:
    """Run the Voxing TUI application."""
    app = VoxingApp()
    app.run()


if __name__ == "__main__":
    main()
