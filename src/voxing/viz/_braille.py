"""Shared braille character rendering utilities."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from voxing.viz._protocol import BrailleGrid, ColorGrid

# Braille bit layout: each character is 2 columns × 4 rows of dots
# Left column bits:  row 0=0x01, 1=0x02, 2=0x04, 3=0x40
# Right column bits: row 0=0x08, 1=0x10, 2=0x20, 3=0x80
BRAILLE_BITS = (
    (0x01, 0x08),  # sub-row 0
    (0x02, 0x10),  # sub-row 1
    (0x04, 0x20),  # sub-row 2
    (0x40, 0x80),  # sub-row 3
)


def set_braille_pixel(
    grid: np.ndarray, x: int, y: int, width: int, height: int
) -> None:
    """Set a single braille pixel in the grid."""
    char_col = x >> 1  # x // 2
    char_row = y >> 2  # y // 4
    if 0 <= char_col < width and 0 <= char_row < height:
        sub_col = x & 1  # x % 2
        sub_row = y & 3  # y % 4
        grid[char_row, char_col] |= BRAILLE_BITS[sub_row][sub_col]


def bresenham_line(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    grid: np.ndarray,
    width: int,
    height: int,
    dot_w: int,
    dot_h: int,
    color_grid: "ColorGrid | None" = None,
    color: str | None = None,
) -> None:
    """Draw a line using Bresenham's algorithm.

    Paints braille pixels in grid. Optionally updates color_grid if provided.
    """
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        if 0 <= x0 < dot_w and 0 <= y0 < dot_h:
            set_braille_pixel(grid, x0, y0, width, height)
            if color_grid is not None and color is not None:
                cr = y0 >> 2
                cc = x0 >> 1
                if 0 <= cr < height and 0 <= cc < width:
                    color_grid[cr][cc] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
