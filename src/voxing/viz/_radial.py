"""Radial spectrum visualiser — circular FFT frequency display."""

from dataclasses import dataclass, field

import numpy as np

from voxing.palette import (
    BLUE,
    FLAMINGO,
    LAVENDER,
    MAUVE,
    PINK,
    SAPPHIRE,
    SKY,
    SURFACE1,
    TEAL,
)
from voxing.viz._protocol import (
    NOISE_GATE,
    ROLLING_MAX_DECAY,
    VizFrame,
)

N_FFT = 512
MAX_RAYS = 128
MIN_RADIUS = 2  # dot-pixels — visible even at silence
ATTACK_ALPHA = 0.2
DECAY_ALPHA = 0.3

_RADIAL_PALETTE: tuple[str, ...] = (
    TEAL,
    SKY,
    SAPPHIRE,
    BLUE,
    LAVENDER,
    MAUVE,
    PINK,
    FLAMINGO,
)

# Braille bit layout (copied from _oscilloscope.py — small, self-contained)
_BRAILLE_BITS = (
    (0x01, 0x08),
    (0x02, 0x10),
    (0x04, 0x20),
    (0x40, 0x80),
)


def _set_pixel(grid: np.ndarray, x: int, y: int, width: int, height: int) -> None:
    """Set a single braille pixel in the grid."""
    char_col = x >> 1
    char_row = y >> 2
    if 0 <= char_col < width and 0 <= char_row < height:
        sub_col = x & 1
        sub_row = y & 3
        grid[char_row, char_col] |= _BRAILLE_BITS[sub_row][sub_col]


def _bresenham_line(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    grid: np.ndarray,
    color_grid: list[list[str]],
    width: int,
    height: int,
    dot_w: int,
    dot_h: int,
    color: str,
) -> None:
    """Draw a line using Bresenham's algorithm, painting braille pixels and colors."""
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < dot_w and 0 <= y0 < dot_h:
            _set_pixel(grid, x0, y0, width, height)
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


@dataclass
class RadialState:
    """Mutable state for the radial spectrum visualiser."""

    remainder: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    prev_amplitudes: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    per_ray_max: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    rolling_max: float = 0.05


def _push_audio(chunk: np.ndarray, state: RadialState) -> None:
    """Accumulate audio samples in the remainder buffer."""
    data = chunk.astype(np.float32).ravel()
    state.remainder = np.concatenate([state.remainder, data])
    max_keep = N_FFT * 4
    if len(state.remainder) > max_keep:
        state.remainder = state.remainder[-max_keep:]


def _compute_spectrum(state: RadialState, n_rays: int) -> np.ndarray:
    """Compute log-grouped FFT magnitudes from the remainder buffer."""
    if len(state.remainder) < N_FFT:
        return np.zeros(n_rays, dtype=np.float32)

    samples = state.remainder[-N_FFT:]
    samples = samples * np.hanning(N_FFT).astype(np.float32)
    samples[np.abs(samples) < NOISE_GATE] = 0.0

    spectrum = np.abs(np.fft.rfft(samples))
    n_bins = len(spectrum)

    # Logarithmic grouping into n_rays bins
    edges = np.logspace(0, np.log10(n_bins), n_rays + 1).astype(int)
    edges = np.clip(edges, 0, n_bins)
    amplitudes = np.zeros(n_rays, dtype=np.float32)
    for i in range(n_rays):
        lo, hi = edges[i], edges[i + 1]
        if hi > lo:
            amplitudes[i] = float(np.mean(spectrum[lo:hi]))
        elif lo < n_bins:
            amplitudes[i] = float(spectrum[lo])

    # Auto-scaling
    peak = float(np.max(amplitudes)) if amplitudes.size > 0 else 0.0
    state.rolling_max = max(state.rolling_max * ROLLING_MAX_DECAY, peak, 1e-4)
    amplitudes /= state.rolling_max

    # Per-ray normalization for visual balance across frequency bands
    if len(state.per_ray_max) != n_rays:
        state.per_ray_max = np.maximum(amplitudes, 1e-4)
    else:
        state.per_ray_max = np.maximum(
            state.per_ray_max * ROLLING_MAX_DECAY, amplitudes
        )
        state.per_ray_max = np.maximum(state.per_ray_max, 1e-4)
    amplitudes /= state.per_ray_max

    # Asymmetric EMA smoothing
    if len(state.prev_amplitudes) != n_rays:
        state.prev_amplitudes = amplitudes.copy()
    else:
        alpha = np.where(amplitudes > state.prev_amplitudes, ATTACK_ALPHA, DECAY_ALPHA)
        state.prev_amplitudes += alpha * (amplitudes - state.prev_amplitudes)
    amplitudes = state.prev_amplitudes.copy()

    # Log compression
    amplitudes = np.sqrt(np.log1p(amplitudes * 20) / np.log1p(20.0)).astype(np.float32)

    return amplitudes


def _render_radial(state: RadialState, width: int, height: int) -> VizFrame:
    """Render the radial spectrum as braille rays emanating from center."""
    dot_w = width * 2
    dot_h = height * 4
    cx = dot_w // 2
    cy = dot_h // 2

    max_radius = min(cx, cy)
    n_rays = min(max(max_radius, 8), MAX_RAYS)

    amplitudes = _compute_spectrum(state, n_rays)

    grid = np.zeros((height, width), dtype=np.int32)
    color_grid: list[list[str]] = [[SURFACE1] * width for _ in range(height)]

    # Draw mirrored rays (0-180° shows bins, 180-360° mirrors)
    for i in range(n_rays):
        amp = float(amplitudes[i])
        ray_len = MIN_RADIUS + amp * (max_radius - MIN_RADIUS)
        color = _RADIAL_PALETTE[i * len(_RADIAL_PALETTE) // n_rays]

        # Forward ray (0 → π)
        angle = np.pi * i / n_rays
        dx = np.cos(angle) * ray_len
        dy = np.sin(angle) * ray_len
        x1 = round(cx + dx)
        y1 = round(cy - dy)
        _bresenham_line(
            cx, cy, x1, y1, grid, color_grid, width, height, dot_w, dot_h, color
        )

        # Mirror ray (π → 2π)
        x1m = round(cx - dx)
        y1m = round(cy + dy)
        _bresenham_line(
            cx, cy, x1m, y1m, grid, color_grid, width, height, dot_w, dot_h, color
        )

    return VizFrame(grid=grid.tolist(), colors=color_grid)


class RadialViz:
    """Radial spectrum visualiser implementing the Visualizer protocol."""

    def __init__(self) -> None:
        self._state = RadialState()

    def push(self, chunk: np.ndarray) -> None:
        """Write new audio samples into the buffer."""
        _push_audio(chunk, self._state)

    def render(self, width: int, height: int) -> VizFrame:
        """Render a braille radial spectrum frame."""
        return _render_radial(self._state, width, height)
