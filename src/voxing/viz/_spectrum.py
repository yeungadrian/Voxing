"""Spectrum visualisation helpers — vertical equalizer bars with peak-hold."""

from dataclasses import dataclass

import numpy as np

from voxing.viz._protocol import VIZ_PALETTE, VizFrame

SPECTRUM_BANDS = 16
SPECTRUM_HEIGHT = 6
SPECTRUM_FREQ_LO = 60.0
SPECTRUM_FREQ_HI = 10000.0
SPECTRUM_DECAY = 0.85
SPECTRUM_PEAK_HOLD = 15
SPECTRUM_NOISE_GATE = 0.005
SPECTRUM_MIN_BAND = 0.03
_LOG1P_20 = float(np.log1p(20.0))


@dataclass
class SpectrumState:
    """Pre-computed FFT artefacts and mutable per-band state."""

    window: np.ndarray
    bin_edges: list[int]
    bands: np.ndarray
    peak_bands: np.ndarray
    peak_ttl: np.ndarray


def build_spectrum_state(sample_rate: int, chunk_samples: int) -> SpectrumState:
    """Pre-compute FFT artefacts for spectrum analysis."""
    window = np.hanning(chunk_samples).astype(np.float32)
    freqs = np.fft.rfftfreq(chunk_samples, 1.0 / sample_rate)
    edges = np.logspace(
        np.log10(SPECTRUM_FREQ_LO), np.log10(SPECTRUM_FREQ_HI), SPECTRUM_BANDS + 1
    )
    bin_edges = [int(np.searchsorted(freqs, e)) for e in edges]
    return SpectrumState(
        window=window,
        bin_edges=bin_edges,
        bands=np.zeros(SPECTRUM_BANDS, dtype=np.float32),
        peak_bands=np.zeros(SPECTRUM_BANDS, dtype=np.float32),
        peak_ttl=np.zeros(SPECTRUM_BANDS, dtype=np.int32),
    )


def compute_spectrum(chunk: np.ndarray, state: SpectrumState) -> np.ndarray:
    """Return a (SPECTRUM_BANDS,) float32 array of smoothed band magnitudes."""
    padded = chunk.astype(np.float32)
    if len(padded) < len(state.window):
        padded = np.pad(padded, (0, len(state.window) - len(padded)))
    elif len(padded) > len(state.window):
        padded = padded[: len(state.window)]

    mag = np.abs(np.fft.rfft(padded * state.window))

    raw = np.zeros(SPECTRUM_BANDS, dtype=np.float32)
    for i in range(SPECTRUM_BANDS):
        lo, hi = state.bin_edges[i], state.bin_edges[i + 1]
        if lo < hi and hi <= len(mag):
            raw[i] = np.mean(mag[lo:hi])

    peak = float(raw.max())
    if peak < SPECTRUM_NOISE_GATE:
        state.bands[:] = SPECTRUM_MIN_BAND
        state.peak_ttl[:] = np.maximum(state.peak_ttl - 1, 0)
        return state.bands.copy()

    if peak > 0:
        raw /= peak

    raw *= 20.0
    np.log1p(raw, out=raw)
    raw /= _LOG1P_20
    np.sqrt(raw, out=raw)
    np.clip(raw, 0.0, 1.0, out=raw)

    state.bands = state.bands * SPECTRUM_DECAY + raw * (1 - SPECTRUM_DECAY)

    for i in range(SPECTRUM_BANDS):
        if state.bands[i] > state.peak_bands[i]:
            state.peak_bands[i] = state.bands[i]
            state.peak_ttl[i] = SPECTRUM_PEAK_HOLD
        else:
            state.peak_ttl[i] = max(state.peak_ttl[i] - 1, 0)
            if state.peak_ttl[i] == 0:
                state.peak_bands[i] = state.bands[i]

    return state.bands.copy()


class SpectrumViz:
    """Spectrum visualizer implementing the Visualizer protocol."""

    def __init__(self, sample_rate: int, chunk_samples: int) -> None:
        self._state = build_spectrum_state(sample_rate, chunk_samples)
        self._bands = np.zeros(SPECTRUM_BANDS, dtype=np.float32)

    def push(self, chunk: np.ndarray) -> None:
        """Compute spectrum from audio chunk."""
        self._bands = compute_spectrum(chunk.astype(np.float32), self._state)

    def render(self, width: int, height: int) -> VizFrame:
        """Render a blocks-mode spectrum frame with per-band color."""
        n_bands = len(self._bands)
        bar_width = max(1, width // n_bands)
        grid: list[list[int]] = []
        colors: list[list[str]] = []
        for y in range(height):
            row: list[int] = []
            color_row: list[str] = []
            for i in range(n_bands):
                amp = float(self._bands[i])
                filled_rows = amp * height
                row_from_bottom = height - 1 - y
                if row_from_bottom < filled_rows:
                    frac = min(filled_rows - row_from_bottom, 1.0)
                    level = min(int(frac * 8), 8)
                else:
                    level = 0
                color = VIZ_PALETTE[level]
                for _ in range(bar_width):
                    row.append(level)
                    color_row.append(color)
            row = row[:width] + [0] * max(0, width - len(row))
            color_row = color_row[:width] + ["#000000"] * max(0, width - len(color_row))
            grid.append(row)
            colors.append(color_row)
        return VizFrame(grid=grid, mode="blocks", colors=colors)
