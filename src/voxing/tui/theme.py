from textual.theme import Theme

from voxing.palette import (
    BASE,
    BLUE,
    GREEN,
    MAUVE,
    RED,
    SURFACE0,
    SURFACE1,
    TEAL,
    TEXT,
    YELLOW,
)

# Semantic aliases used throughout the app — re-exported for backward compatibility
PRIMARY = BLUE
SECONDARY = MAUVE
ACCENT = TEAL
BACKGROUND = BASE
SURFACE = SURFACE0
PANEL = SURFACE1
FOREGROUND = TEXT
SUCCESS = GREEN
WARNING = YELLOW
ERROR = RED

CATPPUCCIN_MOCHA = Theme(
    name="catppuccin-mocha",
    primary=PRIMARY,
    secondary=SECONDARY,
    accent=ACCENT,
    background=BACKGROUND,
    surface=SURFACE,
    panel=PANEL,
    foreground=FOREGROUND,
    success=SUCCESS,
    warning=WARNING,
    error=ERROR,
    dark=True,
    luminosity_spread=3,
    text_alpha=0.95,
)
