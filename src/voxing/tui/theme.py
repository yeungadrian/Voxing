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

CATPPUCCIN_MOCHA = Theme(
    name="catppuccin-mocha",
    primary=BLUE,
    secondary=MAUVE,
    accent=TEAL,
    background=BASE,
    surface=SURFACE0,
    panel=SURFACE1,
    foreground=TEXT,
    success=GREEN,
    warning=YELLOW,
    error=RED,
    dark=True,
    luminosity_spread=3,
    text_alpha=0.95,
)
