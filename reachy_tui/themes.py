"""Tokyo Night color scheme for Reachy TUI."""

from textual.design import ColorSystem

# Tokyo Night color palette (from folke/tokyonight.nvim)
TOKYO_NIGHT = ColorSystem(
    primary="#7aa2f7",      # Blue (palette 4)
    secondary="#bb9af7",    # Purple (palette 5)
    accent="#7dcfff",       # Cyan (palette 6)
    background="#1a1b26",   # Dark background
    surface="#15161e",      # Darker surface (palette 0)
    panel="#414868",        # Panel (palette 8)
    success="#9ece6a",      # Green (palette 2)
    warning="#e0af68",      # Yellow (palette 3)
    error="#f7768e",        # Red (palette 1)
    dark=True,
)
