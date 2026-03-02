"""Shared waveform visualisation constants and helpers."""

import numpy as np

VIZ_HEIGHT = 6
LOG_K = 20.0
NOISE_GATE = 0.005
NUM_BARS = 6
BAR_GAP = 1
MIN_AMP = 0.08
VIZ_WINDOW = 16000  # ~1 s at 16 kHz
BRAILLE_BASE = 0x2800
SUB_ROW_BITS = (0x09, 0x12, 0x24, 0xC0)  # both-column bits for sub-rows 0-3


def compress(peak: float) -> float:
    """Apply log compression mapping peak amplitude [0,1] -> [0,1]."""
    if peak < NOISE_GATE:
        return 0.0
    return float(np.sqrt(np.log1p(peak * LOG_K) / np.log1p(LOG_K)))


def peaks(buf: np.ndarray) -> list[float]:
    """Compute per-bar peak amplitudes from recent audio samples."""
    buf = buf[-VIZ_WINDOW:]
    n = len(buf)
    if n == 0:
        return [MIN_AMP] * NUM_BARS
    indices = np.linspace(0, n, NUM_BARS + 1, dtype=int)
    result: list[float] = []
    for i in range(NUM_BARS):
        window = buf[indices[i] : indices[i + 1]]
        peak = float(np.max(np.abs(window))) if window.size > 0 else 0.0
        result.append(max(compress(peak), MIN_AMP))
    return result


def bar_columns(width: int) -> list[int | None]:
    """Map each column to a bar index, or None for gap columns, left-aligned."""
    mapping: list[int | None] = [None] * width
    for b in range(NUM_BARS):
        col = b * (1 + BAR_GAP)
        if 0 <= col < width:
            mapping[col] = b
    return mapping
