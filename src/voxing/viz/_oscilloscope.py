"""Oscilloscope visualisation helpers — scrolling braille waveform trace."""

from dataclasses import dataclass

import numpy as np

from voxing.viz._protocol import VizFrame

OSCILLO_HEIGHT = 6
OSCILLO_NOISE_GATE = 0.005
OSCILLO_WINDOW = 4800  # ~0.3s at 16kHz
OSCILLO_ROLLING_MAX_DECAY = 0.95

BRAILLE_BASE = 0x2800
_LEFT_BITS = (0x01, 0x02, 0x04, 0x40)
_RIGHT_BITS = (0x08, 0x10, 0x20, 0x80)


@dataclass
class OscilloscopeState:
    """Circular buffer for raw audio samples."""

    ring: np.ndarray
    write_pos: int = 0
    filled: int = 0
    rolling_max: float = 0.05


def build_oscillo_state(capacity: int = OSCILLO_WINDOW) -> OscilloscopeState:
    """Create an empty oscilloscope ring buffer."""
    return OscilloscopeState(ring=np.zeros(capacity, dtype=np.float32))


def push_oscillo_samples(chunk: np.ndarray, state: OscilloscopeState) -> None:
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


def render_oscillo_rows(
    state: OscilloscopeState, width: int, height: int
) -> list[list[int]]:
    """Render the oscilloscope trace as a grid of braille offsets.

    Returns a height x width grid where each cell is a braille offset
    (add BRAILLE_BASE when converting to characters).
    Uses dual-column braille: 2 dot-columns per character cell for double
    horizontal resolution.
    """
    total_dots = height * 4
    sample_count = width * 16  # 16x oversampling for smooth continuous trace
    grid = [[0] * width for _ in range(height)]

    if state.filled == 0:
        return grid

    cap = len(state.ring)
    available = min(state.filled, cap)
    start = (state.write_pos - available) % cap
    if start + available <= cap:
        raw = state.ring[start : start + available]
    else:
        raw = np.concatenate(
            [state.ring[start:], state.ring[: (start + available) % cap]]
        )

    if len(raw) >= sample_count:
        x_old = np.linspace(0, len(raw) - 1, sample_count)
        samples = np.interp(x_old, np.arange(len(raw)), raw)
    else:
        samples = np.zeros(sample_count, dtype=np.float32)
        samples[-len(raw) :] = raw

    samples = samples[::-1]

    samples[np.abs(samples) < OSCILLO_NOISE_GATE] = 0.0

    peak = float(np.max(np.abs(samples)))
    state.rolling_max = max(state.rolling_max * OSCILLO_ROLLING_MAX_DECAY, peak, 1e-4)
    samples = samples / state.rolling_max

    dot_rows = ((samples + 1.0) * 0.5 * (total_dots - 1)).astype(int)
    np.clip(dot_rows, 0, total_dots - 1, out=dot_rows)

    for si in range(sample_count):
        dr = int(dot_rows[si])
        char_row = dr // 4
        sub_row = dr % 4
        char_col = si // 16
        is_right = (si // 8) % 2
        if 0 <= char_row < height and 0 <= char_col < width:
            if is_right:
                grid[char_row][char_col] |= _RIGHT_BITS[sub_row]
            else:
                grid[char_row][char_col] |= _LEFT_BITS[sub_row]

    for si in range(sample_count - 1):
        dr_a = int(dot_rows[si])
        dr_b = int(dot_rows[si + 1])
        if abs(dr_b - dr_a) <= 1:
            continue
        char_col = si // 16
        is_right = (si // 8) % 2
        lo, hi = (dr_a, dr_b) if dr_a < dr_b else (dr_b, dr_a)
        for dr in range(lo + 1, hi):
            cr = dr // 4
            sr = dr % 4
            if 0 <= cr < height and 0 <= char_col < width:
                if is_right:
                    grid[cr][char_col] |= _RIGHT_BITS[sr]
                else:
                    grid[cr][char_col] |= _LEFT_BITS[sr]

    return grid


class OscilloscopeViz:
    """Oscilloscope visualizer implementing the Visualizer protocol."""

    def __init__(self, capacity: int = OSCILLO_WINDOW) -> None:
        self._state = build_oscillo_state(capacity)

    def push(self, chunk: np.ndarray) -> None:
        """Write new audio samples into the ring buffer."""
        push_oscillo_samples(chunk, self._state)

    def render(self, width: int, height: int) -> VizFrame:
        """Render a braille oscilloscope frame."""
        grid = render_oscillo_rows(self._state, width, height)
        return VizFrame(grid=grid, mode="braille")
