"""Audio visualisation package.

Submodules:
    _waveform    — braille bar waveform helpers
    _spectrogram — rolling FFT spectrogram helpers
"""

from voxing.viz._spectrogram import (
    BLOCKS,
    DECAY,
    FREQ_HI,
    FREQ_LO,
    SPEC_BANDS,
    SpectrogramState,
    build_spec_state,
    compute_column,
)
from voxing.viz._waveform import (
    BAR_GAP,
    BRAILLE_BASE,
    LOG_K,
    MIN_AMP,
    NOISE_GATE,
    NUM_BARS,
    SUB_ROW_BITS,
    VIZ_HEIGHT,
    VIZ_WINDOW,
    bar_columns,
    peaks,
)

__all__ = [
    # waveform
    "BAR_GAP",
    "BRAILLE_BASE",
    "LOG_K",
    "MIN_AMP",
    "NOISE_GATE",
    "NUM_BARS",
    "SUB_ROW_BITS",
    "VIZ_HEIGHT",
    "VIZ_WINDOW",
    "bar_columns",
    "peaks",
    # spectrogram
    "BLOCKS",
    "DECAY",
    "FREQ_HI",
    "FREQ_LO",
    "SPEC_BANDS",
    "SpectrogramState",
    "build_spec_state",
    "compute_column",
]
