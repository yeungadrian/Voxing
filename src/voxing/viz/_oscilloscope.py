"""Oscilloscope visualisation helpers — scrolling braille waveform trace."""

from dataclasses import dataclass

import numpy as np

from voxing.palette import (
    LAVENDER,
    MAUVE,
    SAPPHIRE,
    SKY,
    TEAL,
)
from voxing.viz._braille import bresenham_line
from voxing.viz._protocol import (
    NOISE_GATE,
    ColorGrid,
    VizFrame,
)

OSCILLO_WINDOW = 4000  # ~0.25s at 16kHz — enough for 2+ periods of 100 Hz speech
OSCILLO_MIN_ROLLING_MAX = 0.05
OSCILLO_ROLLING_MAX_DECAY = 0.975
OSCILLO_LEVEL_PERCENTILE = 95.0
OSCILLO_COMPAND_GAMMA = 0.85
OSCILLO_NOISE_GATE = NOISE_GATE * 0.7

# Catppuccin Mocha amplitude colour ramp: center (cool) → extremes (warm)
_AMPLITUDE_PALETTE: tuple[str, ...] = (
    TEAL,  # near center — cool, calm
    SKY,  # gentle
    SAPPHIRE,  # moderate
    LAVENDER,  # strong
    MAUVE,  # extreme — warm purple, not jarring red
)


@dataclass
class OscilloscopeState:
    """Circular buffer for raw audio samples."""

    ring: np.ndarray
    write_pos: int = 0
    filled: int = 0
    rolling_max: float = OSCILLO_MIN_ROLLING_MAX


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


def _render_oscillo_rows(state: OscilloscopeState, width: int, height: int) -> VizFrame:
    """Render a single clean oscilloscope trace via Bresenham lines in braille."""
    dot_w = width * 2
    dot_h = height * 4
    display_count = dot_w  # one sample per braille dot-column
    oversample = display_count * 2  # 2x headroom for trigger search

    grid_arr = np.zeros((height, width), dtype=np.int32)
    default_color = _AMPLITUDE_PALETTE[0]

    if state.filled == 0:
        # Silent: draw a flat line at center via Bresenham
        mid_y = (dot_h - 1) // 2
        bresenham_line(
            0, mid_y, dot_w - 1, mid_y, grid_arr, width, height, dot_w, dot_h
        )
        return VizFrame(
            grid=grid_arr.tolist(),
            colors=[[default_color] * width for _ in range(height)],
        )

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
    samples[np.abs(samples) < OSCILLO_NOISE_GATE] = 0.0

    level = float(np.percentile(np.abs(samples), OSCILLO_LEVEL_PERCENTILE))
    state.rolling_max = max(
        state.rolling_max * OSCILLO_ROLLING_MAX_DECAY,
        level,
        OSCILLO_MIN_ROLLING_MAX,
    )
    samples = samples / state.rolling_max
    samples = np.sign(samples) * np.power(np.abs(samples), OSCILLO_COMPAND_GAMMA)
    np.clip(samples, -1.0, 1.0, out=samples)

    # Edge trigger on oversampled buffer, then slice exactly display_count
    trigger = _find_trigger_offset(samples)
    samples = samples[trigger : trigger + display_count]
    if len(samples) < display_count:
        samples = np.pad(samples, (0, display_count - len(samples)))

    # Map to braille pixel y-coordinates
    pixel_ys = ((samples + 1.0) * 0.5 * (dot_h - 1)).astype(np.int32)
    np.clip(pixel_ys, 0, dot_h - 1, out=pixel_ys)

    # Single Bresenham trace — no phosphor, no separate midline
    for i in range(display_count - 1):
        bresenham_line(
            i,
            int(pixel_ys[i]),
            i + 1,
            int(pixel_ys[i + 1]),
            grid_arr,
            width,
            height,
            dot_w,
            dot_h,
        )

    # Amplitude-based coloring: distance from center → palette index
    mid_y = (dot_h - 1) / 2.0
    amp_norm = np.abs(pixel_ys.astype(np.float64) - mid_y) / mid_y
    np.clip(amp_norm, 0.0, 1.0, out=amp_norm)
    # Pair dot-columns into char columns (max of each pair)
    if display_count % 2:
        amp_norm = np.append(amp_norm, 0.0)
    cell_amp = np.maximum(amp_norm[0::2], amp_norm[1::2])[:width]

    n_colors = len(_AMPLITUDE_PALETTE)
    color_indices = np.minimum(
        (cell_amp * (n_colors - 1)).astype(np.int32), n_colors - 1
    )
    colors: ColorGrid = []
    for _ in range(height):
        colors.append(
            [_AMPLITUDE_PALETTE[int(color_indices[col])] for col in range(width)]
        )

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
