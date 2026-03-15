"""Waveform visualisation helpers."""

import numpy as np

from voxing.palette import (
    BLUE,
    LAVENDER,
    MAUVE,
    OVERLAY0,
    SAPPHIRE,
    SKY,
    SURFACE1,
    SURFACE2,
)
from voxing.viz._protocol import (
    ROLLING_MAX_DECAY,
    BrailleGrid,
    ColorGrid,
    VizFrame,
)

_SUB_ROW_BITS = (0x09, 0x12, 0x24, 0xC0)  # both-column bits for sub-rows 0-3

LOG_K = 20.0
_LOG1P_K = float(np.log1p(LOG_K))
BAR_GAP = 1
MIN_AMP = 0.08
MIN_ROLLING_MAX_WAVEFORM = 0.35
WAVEFORM_NOISE_FLOOR = 0.0025
VIZ_WINDOW = 16000  # ~1 s at 16 kHz

# Catppuccin Mocha gradient: bottom (muted) → top (accent)
_WAVEFORM_GRADIENT: tuple[str, ...] = (
    SURFACE1,  # bottom / quiet
    SURFACE2,  # overlay0
    OVERLAY0,  # dim
    LAVENDER,  # bridges gray to blue
    BLUE,  # blue
    SAPPHIRE,  # sapphire
    SKY,  # sky
    MAUVE,  # top / loud
)


def peaks(buf: np.ndarray, num_bars: int) -> np.ndarray:
    """Compute per-bar RMS amplitudes and compress for display."""
    buf = buf[-VIZ_WINDOW:]
    n = len(buf)
    if n == 0:
        return np.zeros(num_bars, dtype=np.float32)
    indices = np.linspace(0, n, num_bars + 1, dtype=int)
    # Ensure no duplicate indices (can happen when num_bars > n)
    indices = np.unique(indices)
    actual_bars = len(indices) - 1
    sq_buf = buf * buf
    sq_sums = np.add.reduceat(sq_buf, indices[:-1])[:actual_bars]
    counts = np.diff(indices).astype(np.float32)
    raw = np.sqrt(sq_sums / np.maximum(counts, 1.0))
    if actual_bars < num_bars:
        raw = np.pad(raw, (0, num_bars - actual_bars))

    raw = np.maximum(raw - WAVEFORM_NOISE_FLOOR, 0.0)
    if len(raw) >= 3:
        raw = np.convolve(raw, np.array([0.2, 0.6, 0.2], dtype=np.float32), mode="same")

    compressed = np.sqrt(np.log1p(raw * LOG_K) / _LOG1P_K)
    return compressed.astype(np.float32)


def bar_columns(width: int) -> tuple[list[int | None], int]:
    """Map each column to a bar index or None for gaps; return mapping and num_bars."""
    num_bars = max(1, (width + BAR_GAP) // (1 + BAR_GAP))
    mapping: list[int | None] = [None] * width
    for b in range(num_bars):
        col = b * (1 + BAR_GAP)
        if 0 <= col < width:
            mapping[col] = b
    return mapping, num_bars


class WaveformViz:
    """Waveform visualizer implementing the Visualizer protocol."""

    def __init__(self) -> None:
        self._buf = np.empty(0, dtype=np.float32)
        self._rolling_max: float = MIN_ROLLING_MAX_WAVEFORM
        self._cached_width: int = 0
        self._cached_columns: list[int | None] = []
        self._cached_num_bars: int = 0

    def push(self, chunk: np.ndarray) -> None:
        """Append audio and trim to the visualisation window."""
        self._buf = np.concatenate([self._buf, chunk.astype(np.float32)])[-VIZ_WINDOW:]

    def render(self, width: int, height: int) -> VizFrame:
        """Render a braille waveform frame with height-gradient colours."""
        if width != self._cached_width:
            self._cached_columns, self._cached_num_bars = bar_columns(width)
            self._cached_width = width

        columns = self._cached_columns
        num_bars = self._cached_num_bars
        amps_raw = peaks(self._buf, num_bars)

        frame_max = float(amps_raw.max()) if len(amps_raw) > 0 else 0.0
        self._rolling_max = max(
            self._rolling_max * ROLLING_MAX_DECAY, frame_max, MIN_ROLLING_MAX_WAVEFORM
        )

        amps = np.maximum(amps_raw / self._rolling_max, MIN_AMP)

        total_dots = height * 4
        n_colors = len(_WAVEFORM_GRADIENT)
        grid: BrailleGrid = []
        colors: ColorGrid = []

        for y in range(height):
            row_bits: list[int] = []
            row_colors: list[str] = []
            for bar_idx in columns:
                if bar_idx is None:
                    row_bits.append(0)
                    row_colors.append(_WAVEFORM_GRADIENT[0])
                    continue
                amp = float(amps[bar_idx]) if bar_idx < len(amps) else MIN_AMP
                peak_dots = amp * total_dots
                bits = 0
                for sub in range(4):
                    dot = y * 4 + sub
                    if total_dots - 1 - dot < peak_dots:
                        bits |= _SUB_ROW_BITS[sub]
                row_bits.append(bits)
                # Pure height gradient: row position determines color
                if bits:
                    row_frac = (
                        1.0 if height <= 1 else 1.0 - y / float(max(height - 1, 1))
                    )
                    color_idx = min(int(row_frac * (n_colors - 1)), n_colors - 1)
                    row_colors.append(_WAVEFORM_GRADIENT[color_idx])
                else:
                    row_colors.append(_WAVEFORM_GRADIENT[0])
            grid.append(row_bits)
            colors.append(row_colors)

        return VizFrame(grid=grid, colors=colors)
