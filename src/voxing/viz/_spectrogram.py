"""Spectrogram visualisation helpers."""

from dataclasses import dataclass, field

import numpy as np

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
