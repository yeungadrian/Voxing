"""Tokyo Night color scheme for Vox TUI."""

from textual.design import ColorSystem

# Tokyo Night Night variant - Official color palette
# Source: https://github.com/folke/tokyonight.nvim/blob/main/extras/ghostty/tokyonight_night

# Standard palette (0-8)
PALETTE_0 = "#15161e"  # Darker surface
PALETTE_1 = "#f7768e"  # Red
PALETTE_2 = "#9ece6a"  # Green
PALETTE_3 = "#e0af68"  # Yellow
PALETTE_4 = "#7aa2f7"  # Blue
PALETTE_5 = "#bb9af7"  # Purple/Magenta
PALETTE_6 = "#7dcfff"  # Cyan
PALETTE_7 = "#a9b1d6"  # Light text
PALETTE_8 = "#414868"  # Panel/border

# Bright variants (9-15)
BRIGHT_RED = "#ff899d"  # palette 9
BRIGHT_GREEN = "#9fe044"  # palette 10
BRIGHT_YELLOW = "#faba4a"  # palette 11
BRIGHT_BLUE = "#8db0ff"  # palette 12
BRIGHT_PURPLE = "#c7a9ff"  # palette 13
BRIGHT_CYAN = "#a4daff"  # palette 14
BRIGHT_WHITE = "#c0caf5"  # palette 15

# Official Tokyo Night special colors
BACKGROUND = "#1a1b26"  # Dark background
FOREGROUND = "#c0caf5"  # Main text (same as BRIGHT_WHITE)
SELECTION_BG = "#283457"  # Selection background

# Textual ColorSystem (9 semantic colors for Textual framework)
TOKYO_NIGHT = ColorSystem(
    primary=PALETTE_4,  # Blue
    secondary=PALETTE_5,  # Purple/Magenta
    accent=PALETTE_6,  # Cyan
    background=BACKGROUND,
    surface=PALETTE_0,
    panel=PALETTE_8,
    success=PALETTE_2,  # Green
    warning=PALETTE_3,  # Yellow
    error=PALETTE_1,  # Red
    dark=True,
)
