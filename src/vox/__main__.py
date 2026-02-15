"""Entry point for Vox TUI application."""

from vox.app import VoxApp


def main() -> None:
    """Run the Vox TUI application."""
    app = VoxApp()
    app.run()


if __name__ == "__main__":
    main()
