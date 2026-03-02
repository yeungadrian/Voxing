from textual.theme import Theme

PRIMARY = "#89b4fa"
SECONDARY = "#cba6f7"
ACCENT = "#94e2d5"
BACKGROUND = "#1e1e2e"
SURFACE = "#313244"
PANEL = "#45475a"
FOREGROUND = "#cdd6f4"
SUCCESS = "#a6e3a1"
WARNING = "#f9e2af"
ERROR = "#f38ba8"

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
