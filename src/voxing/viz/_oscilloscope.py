"""Oscilloscope visualisation helpers — scrolling braille waveform trace."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from voxing.palette import (
    LAVENDER,
    MAUVE,
    OVERLAY0,
    SAPPHIRE,
    SKY,
    SURFACE1,
    SURFACE2,
    TEAL,
)
from voxing.viz._protocol import (
    NOISE_GATE,
    ROLLING_MAX_DECAY,
    VizFrame,
)

OSCILLO_WINDOW = 8000  # ~0.5s at 16kHz — enough for 3-4 periods of 100 Hz speech

# Catppuccin Mocha amplitude colour ramp: center (cool) → extremes (warm)
_AMPLITUDE_PALETTE: tuple[str, ...] = (
    TEAL,  # near center — cool, calm
    SKY,  # gentle
    SAPPHIRE,  # moderate
    LAVENDER,  # strong
    MAUVE,  # extreme — warm purple, not jarring red
)


_PHOSPHOR_COLORS: tuple[str, ...] = (SURFACE1, SURFACE2, OVERLAY0)


@dataclass
class OscilloscopeState:
    """Circular buffer for raw audio samples."""

    ring: np.ndarray
    write_pos: int = 0
    filled: int = 0
    rolling_max: float = 0.05
    trace_history: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=3))


def _build_oscillo_state(capacity: int = OSCILLO_WINDOW) -> OscilloscopeState:
    """Create an empty oscilloscope ring buffer."""
    return OscilloscopeState(ring=np.zeros(capacity, dtype=np.float32))


def _push_oscillo_samples(chunk: np.ndarray, state: OscilloscopeState) -> None:
    """Write new audio samples into the circular buffer."""
    data = chunk.astype(np.float32).ravel()
    n = len(data)
    cap = len(state.ring)
    if n >= cap:
        state.ring[:] = data[-cap:]
        state.write_pos = 0
        state.filled = cap
        return
    end = state.write_pos + n
    if end <= cap:
        state.ring[state.write_pos : end] = data
    else:
        first = cap - state.write_pos
        state.ring[state.write_pos :] = data[:first]
        state.ring[: n - first] = data[first:]
    state.write_pos = end % cap
    state.filled = min(state.filled + n, cap)


def _box_average_downsample(raw: np.ndarray, sample_count: int) -> np.ndarray:
    """Downsample by averaging groups of samples — avoids aliasing."""
    n = len(raw)
    if n >= sample_count:
        trunc = (n // sample_count) * sample_count
        return raw[:trunc].reshape(sample_count, -1).mean(axis=1).astype(np.float32)
    # Buffer shorter than display: zero-pad on the left (silence at start)
    out = np.zeros(sample_count, dtype=np.float32)
    out[-n:] = raw
    return out


def _find_trigger_offset(samples: np.ndarray) -> int:
    """Find the first rising zero-crossing in the first half of the buffer."""
    half = len(samples) // 2
    search = samples[:half]
    crossings = np.where((search[:-1] < 0) & (search[1:] >= 0))[0]
    if crossings.size == 0:
        return 0
    return int(crossings[0])


# Braille bit layout: each character is 2 columns x 4 rows of dots.
# Left column bits:  row 0=0x01, 1=0x02, 2=0x04, 3=0x40
# Right column bits: row 0=0x08, 1=0x10, 2=0x20, 3=0x80
_BRAILLE_BITS = (
    (0x01, 0x08),  # sub-row 0
    (0x02, 0x10),  # sub-row 1
    (0x04, 0x20),  # sub-row 2
    (0x40, 0x80),  # sub-row 3
)


def _set_pixel(grid: np.ndarray, x: int, y: int, width: int, height: int) -> None:
    """Set a single braille pixel in the grid."""
    char_col = x >> 1  # x // 2
    char_row = y >> 2  # y // 4
    if 0 <= char_col < width and 0 <= char_row < height:
        sub_col = x & 1  # x % 2
        sub_row = y & 3  # y % 4
        grid[char_row, char_col] |= _BRAILLE_BITS[sub_row][sub_col]


def _bresenham_rasterize(
    pixel_xs: np.ndarray,
    pixel_ys: np.ndarray,
    width: int,
    height: int,
    grid: np.ndarray,
) -> None:
    """Draw connected lines between consecutive points using Bresenham's algorithm."""
    dot_w = width * 2
    dot_h = height * 4
    n = len(pixel_xs)
    for i in range(n - 1):
        x0, y0 = int(pixel_xs[i]), int(pixel_ys[i])
        x1, y1 = int(pixel_xs[i + 1]), int(pixel_ys[i + 1])
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            if 0 <= x0 < dot_w and 0 <= y0 < dot_h:
                _set_pixel(grid, x0, y0, width, height)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy


def _render_oscillo_rows(state: OscilloscopeState, width: int, height: int) -> VizFrame:
    """Render the oscilloscope trace as connected Bresenham lines in braille.

    One sample per braille dot-column (display_count = width * 2).
    Downsamples to 2x display_count for trigger search headroom, then slices
    exactly display_count samples from the trigger point — no zero-padding.
    """
    dot_h = height * 4
    display_count = width * 2  # one sample per braille dot-column
    oversample = display_count * 2  # 2x headroom for trigger search

    grid_arr = np.zeros((height, width), dtype=np.int32)
    default_color = _AMPLITUDE_PALETTE[0]

    if state.filled == 0:
        grid = grid_arr.tolist()
        colors = [[default_color] * width for _ in range(height)]
        return VizFrame(grid=grid, colors=colors)

    cap = len(state.ring)
    available = min(state.filled, cap)
    start = (state.write_pos - available) % cap
    if start + available <= cap:
        raw = state.ring[start : start + available].copy()
    else:
        raw = np.concatenate(
            [state.ring[start:], state.ring[: (start + available) % cap]]
        )

    # Box-average downsample to oversampled buffer
    samples = _box_average_downsample(raw, oversample)
    samples = samples[::-1].copy()
    samples[np.abs(samples) < NOISE_GATE] = 0.0

    peak = float(np.max(np.abs(samples)))
    state.rolling_max = max(state.rolling_max * ROLLING_MAX_DECAY, peak, 1e-4)
    samples = samples / state.rolling_max

    # Edge trigger on oversampled buffer, then slice exactly display_count
    trigger = _find_trigger_offset(samples)
    samples = samples[trigger : trigger + display_count]
    if len(samples) < display_count:
        samples = np.pad(samples, (0, display_count - len(samples)))

    # Map to braille pixel y-coordinates
    pixel_ys = ((samples + 1.0) * 0.5 * (dot_h - 1)).astype(np.int32)
    np.clip(pixel_ys, 0, dot_h - 1, out=pixel_ys)
    pixel_xs = np.arange(display_count, dtype=np.int32)

    # Paint older phosphor traces (dim ghost trails)
    phosphor_cells: dict[tuple[int, int], str] = {}
    for trace_idx, old_ys in enumerate(state.trace_history):
        phosphor_color = _PHOSPHOR_COLORS[min(trace_idx, len(_PHOSPHOR_COLORS) - 1)]
        n = min(len(old_ys), display_count)
        old_xs = np.arange(n, dtype=np.int32)
        _bresenham_rasterize(old_xs, old_ys[:n], width, height, grid_arr)
        # Record phosphor cell locations
        for i in range(n):
            cc = int(old_xs[i]) >> 1
            cr = int(old_ys[i]) >> 2
            if 0 <= cc < width and 0 <= cr < height:
                phosphor_cells[(cr, cc)] = phosphor_color

    state.trace_history.append(pixel_ys.copy())

    # Paint current trace with Bresenham lines
    _bresenham_rasterize(pixel_xs, pixel_ys, width, height, grid_arr)

    # Amplitude-based color: distance from center → palette index
    mid_y = (dot_h - 1) / 2.0
    # Per dot-column: absolute distance from center, normalized to [0, 1]
    amp_norm = np.abs(pixel_ys.astype(np.float64) - mid_y) / mid_y
    np.clip(amp_norm, 0.0, 1.0, out=amp_norm)
    # Pair dot-columns into char columns (max of each pair)
    padded = amp_norm[:display_count]
    if display_count % 2:
        padded = np.append(padded, 0.0)
    cell_amp = np.maximum(padded[0::2], padded[1::2])[:width]

    # Build color grid
    n_colors = len(_AMPLITUDE_PALETTE)
    color_indices = np.minimum(
        (cell_amp * (n_colors - 1)).astype(np.int32), n_colors - 1
    )
    colors: list[list[str]] = []
    for _ in range(height):
        row_colors = [
            _AMPLITUDE_PALETTE[int(color_indices[col])] for col in range(width)
        ]
        colors.append(row_colors)

    # Apply phosphor colors only to cells not touched by current trace
    current_cells: set[tuple[int, int]] = set()
    for i in range(display_count):
        cc = int(pixel_xs[i]) >> 1
        cr = int(pixel_ys[i]) >> 2
        if 0 <= cc < width and 0 <= cr < height:
            current_cells.add((cr, cc))

    for (r, c), color in phosphor_cells.items():
        if (r, c) not in current_cells:
            colors[r][c] = color

    return VizFrame(grid=grid_arr.tolist(), colors=colors)


class OscilloscopeViz:
    """Oscilloscope visualizer implementing the Visualizer protocol."""

    def __init__(self, capacity: int = OSCILLO_WINDOW) -> None:
        self._state = _build_oscillo_state(capacity)

    def push(self, chunk: np.ndarray) -> None:
        """Write new audio samples into the ring buffer."""
        _push_oscillo_samples(chunk, self._state)

    def render(self, width: int, height: int) -> VizFrame:
        """Render a braille oscilloscope frame."""
        return _render_oscillo_rows(self._state, width, height)
