"""Waveform visualisation helpers."""

import numpy as np

from voxing.viz._protocol import VizFrame

VIZ_HEIGHT = 6
LOG_K = 20.0
_LOG1P_K = float(np.log1p(LOG_K))
NOISE_GATE = 0.005
BAR_GAP = 1
MIN_AMP = 0.08
ROLLING_MAX_DECAY = 0.95
VIZ_WINDOW = 16000  # ~1 s at 16 kHz
BRAILLE_BASE = 0x2800
SUB_ROW_BITS = (0x09, 0x12, 0x24, 0xC0)  # both-column bits for sub-rows 0-3


def peaks(buf: np.ndarray, num_bars: int) -> list[float]:
    """Compute per-bar peak amplitudes from recent audio samples."""
    buf = buf[-VIZ_WINDOW:]
    n = len(buf)
    if n == 0:
        return [0.0] * num_bars
    indices = np.linspace(0, n, num_bars + 1, dtype=int)
    raw = np.array(
        [
            float(np.max(np.abs(buf[indices[i] : indices[i + 1]])))
            if indices[i] < indices[i + 1]
            else 0.0
            for i in range(num_bars)
        ]
    )
    raw[raw < NOISE_GATE] = 0.0
    compressed = np.sqrt(np.log1p(raw * LOG_K) / _LOG1P_K)
    return [float(v) for v in compressed]


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
        self._rolling_max: float = MIN_AMP

    def push(self, chunk: np.ndarray) -> None:
        """Append audio and trim to the visualisation window."""
        self._buf = np.concatenate([self._buf, chunk.astype(np.float32)])[-VIZ_WINDOW:]

    def render(self, width: int, height: int) -> VizFrame:
        """Render a braille waveform frame."""
        columns, num_bars = bar_columns(width)
        amps = peaks(self._buf, num_bars)
        frame_max = max(amps) if amps else 0.0
        self._rolling_max = max(
            self._rolling_max * ROLLING_MAX_DECAY, frame_max, MIN_AMP
        )
        total_dots = height * 4
        grid: list[list[int]] = []
        for y in range(height):
            row: list[int] = []
            for bar_idx in columns:
                if bar_idx is None:
                    row.append(0)
                    continue
                amp = amps[bar_idx] if bar_idx < len(amps) else 0.0
                amp = max(amp / self._rolling_max, MIN_AMP)
                bar_dots = amp * total_dots
                bits = 0
                for sub in range(4):
                    dot = y * 4 + sub
                    if total_dots - 1 - dot < bar_dots:
                        bits |= SUB_ROW_BITS[sub]
                row.append(bits)
            grid.append(row)
        return VizFrame(grid=grid, mode="braille")
