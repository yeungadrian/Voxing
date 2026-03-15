import numpy as np

from voxing.viz._braille import BRAILLE_BITS, bresenham_line, set_braille_pixel


def test_set_braille_pixel_sets_correct_bit() -> None:
    grid = np.zeros((1, 1), dtype=np.int32)
    set_braille_pixel(grid, 0, 0, width=1, height=1)
    assert grid[0, 0] == BRAILLE_BITS[0][0]


def test_set_braille_pixel_out_of_bounds_noop() -> None:
    grid = np.zeros((2, 2), dtype=np.int32)
    set_braille_pixel(grid, -1, 0, width=2, height=2)
    set_braille_pixel(grid, 0, -1, width=2, height=2)
    set_braille_pixel(grid, 100, 0, width=2, height=2)
    set_braille_pixel(grid, 0, 100, width=2, height=2)
    assert grid.sum() == 0


def test_bresenham_horizontal_line() -> None:
    width, height = 4, 1
    dot_w, dot_h = width * 2, height * 4
    grid = np.zeros((height, width), dtype=np.int32)
    bresenham_line(0, 0, dot_w - 1, 0, grid, width, height, dot_w, dot_h)
    for col in range(width):
        assert grid[0, col] != 0


def test_bresenham_vertical_line() -> None:
    width, height = 1, 4
    dot_w, dot_h = width * 2, height * 4
    grid = np.zeros((height, width), dtype=np.int32)
    bresenham_line(0, 0, 0, dot_h - 1, grid, width, height, dot_w, dot_h)
    for row in range(height):
        assert grid[row, 0] != 0


def test_bresenham_line_with_color_grid() -> None:
    width, height = 3, 1
    dot_w, dot_h = width * 2, height * 4
    grid = np.zeros((height, width), dtype=np.int32)
    color_grid: list[list[str]] = [["" for _ in range(width)] for _ in range(height)]
    bresenham_line(
        0,
        0,
        dot_w - 1,
        0,
        grid,
        width,
        height,
        dot_w,
        dot_h,
        color_grid=color_grid,
        color="red",
    )
    painted = [color_grid[0][c] for c in range(width) if color_grid[0][c] == "red"]
    assert len(painted) == width
