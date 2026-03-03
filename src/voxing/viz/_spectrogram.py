"""Spectrogram visualisation helpers."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from voxing.viz._protocol import VIZ_PALETTE, VizFrame

SPEC_BANDS = 6
FREQ_LO = 80.0
FREQ_HI = 8000.0
DECAY = 0.92
NOISE_GATE = 0.005
MIN_BAND = 0.05
_LOG1P_20 = float(np.log1p(20.0))
BLOCKS = " ▁▂▃▄▅▆▇█"  # 9 levels: index 0 = silence, 8 = full


@dataclass
class SpectrogramState:
    """Pre-computed FFT artefacts and mutable normalisation state."""

    window: np.ndarray
    bin_edges: list[int]
    bands: np.ndarray
    rolling_max: float = field(default=1e-6)


def build_spec_state(sample_rate: int, chunk_samples: int) -> SpectrogramState:
    """Pre-compute all fixed FFT artefacts so the hot path is allocation-free."""
    window = np.hanning(chunk_samples).astype(np.float32)
    freqs = np.fft.rfftfreq(chunk_samples, 1.0 / sample_rate)
    edges = np.logspace(np.log10(FREQ_LO), np.log10(FREQ_HI), SPEC_BANDS + 1)
    bin_edges = [int(np.searchsorted(freqs, e)) for e in edges]
    bands = np.zeros(SPEC_BANDS, dtype=np.float32)
    return SpectrogramState(window=window, bin_edges=bin_edges, bands=bands)


def compute_column(chunk: np.ndarray, state: SpectrogramState) -> np.ndarray:
    """Return a (SPEC_BANDS,) float32 array of log-compressed band energies.

    Runs entirely before the lock is acquired — no allocation inside the lock.
    """
    mag = np.abs(np.fft.rfft(chunk * state.window))
    bands = state.bands
    for i in range(SPEC_BANDS):
        bands[i] = np.mean(mag[state.bin_edges[i] : state.bin_edges[i + 1]])
    peak = float(bands.max())
    if peak < NOISE_GATE:
        bands[:] = MIN_BAND
        return bands.copy()
    state.rolling_max = max(state.rolling_max * DECAY, peak, 1e-6)
    bands /= state.rolling_max
    bands *= 20.0
    np.log1p(bands, out=bands)
    bands /= _LOG1P_20
    np.sqrt(bands, out=bands)
    np.clip(bands, 0.0, 1.0, out=bands)
    return bands.copy()


class SpectrogramViz:
    """Spectrogram visualizer implementing the Visualizer protocol."""

    def __init__(
        self,
        sample_rate: int,
        chunk_samples: int,
        chunks_per_column: int = 1,
    ) -> None:
        self._state = build_spec_state(sample_rate, chunk_samples * chunks_per_column)
        self._columns: deque[np.ndarray] = deque(maxlen=200)
        self._pending: list[np.ndarray] = []
        self._chunks_per_column = chunks_per_column

    def push(self, chunk: np.ndarray) -> None:
        """Accumulate chunks and emit a spectrogram column when enough are gathered."""
        self._pending.append(chunk)
        if len(self._pending) >= self._chunks_per_column:
            combined = np.concatenate(self._pending)
            self._pending.clear()
            self._columns.append(compute_column(combined, self._state))

    def render(self, width: int, height: int) -> VizFrame:
        """Render a blocks-mode spectrogram frame."""
        cols = self._columns
        n = len(cols)
        start = max(0, n - width)
        bands = min(height, SPEC_BANDS)
        n_visible = n - start
        grid: list[list[int]] = []
        colors: list[list[str]] = []
        for row_idx in range(height):
            band_idx = bands - 1 - row_idx
            row: list[int] = []
            color_row: list[str] = []
            if band_idx < 0 or band_idx >= bands:
                row = [0] * width
                color_row = [VIZ_PALETTE[0]] * width
            else:
                for col_idx in range(width):
                    if col_idx < n_visible:
                        val = float(cols[n - 1 - col_idx][band_idx])
                        level = min(int(val * 8), 8)
                    else:
                        level = 0
                    row.append(level)
                    color_row.append(VIZ_PALETTE[level])
            grid.append(row)
            colors.append(color_row)
        return VizFrame(grid=grid, mode="blocks", colors=colors)
