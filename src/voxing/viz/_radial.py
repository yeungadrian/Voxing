"""Radial spectrum visualiser — circular FFT frequency display."""

import time
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
from voxing.viz._braille import bresenham_line, draw_circle
from voxing.viz._protocol import (
    MIN_ROLLING_MAX,
    NOISE_GATE,
    ColorGrid,
    VizFrame,
)

N_FFT = 512
MAX_RAYS = 360  # Maximum rays for large terminals
ATTACK_ALPHA = 0.7  # Fast attack: quickly follows rising amplitudes
DECAY_ALPHA = 0.3  # Slow decay: trails fading amplitudes for smoother visuals

# Hollow ring parameters
MIN_RAY_AMPLITUDE = 0.0  # Rays can fully collapse to center for maximum variation
ADAPTIVE_RAY_DENSITY = True  # Enable terminal-size-based ray count

# Adaptive ray density thresholds
RAY_DENSITY_SMALL = 128  # terminals < 80 cols
RAY_DENSITY_MEDIUM = 256  # terminals 80-120 cols
RAY_DENSITY_LARGE = 360  # terminals > 120 cols

# Rotation
ROTATION_SPEED = 0.5  # radians per second (~3°/sec)
COLOR_ROTATION_SPEED = 0.25  # radians per second - faster than spectrum rotation

# Visualization
VISUALIZATION_SIZE_RATIO = 0.90  # Use 90% of available terminal space
INNER_RING_RATIO = 0.50  # Inner ring at 30% of max_radius

# Power curve exponent — >1 pushes quiet bands down, preserving peak spikes
COMPRESSION_EXPONENT = 1.5
# Minimum rendered ray amplitude in [0,1] — quiet audio still shows small spikes
RAY_MIN_AMPLITUDE = 0.02
# Spectral sharpening — unsharp mask applied after bin grouping
SPECTRAL_SHARPEN_STRENGTH = 0.6  # How aggressively to boost peaks over neighbours
SPECTRAL_SHARPEN_RADIUS = 3  # Kernel half-width in rays
# Slower decay than the shared 0.95 keeps the normalization ceiling elevated longer,
# making quiet bands appear short relative to recent peaks
RADIAL_ROLLING_MAX_DECAY = 0.98

_RADIAL_PALETTE: tuple[str, ...] = (
    FLAMINGO,  # Bass - warm pink
    PINK,  # Low-bass
    MAUVE,  # Mid-bass - purple
    LAVENDER,  # Low-mids
    BLUE,  # Mids
    SAPPHIRE,  # High-mids
    SKY,  # High - cyan
    TEAL,  # Treble - cyan-green
)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert #RRGGBB hex string to (R, G, B) tuple."""
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """Convert (R, G, B) tuple to #RRGGBB hex string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _create_frequency_gradient() -> list[tuple[int, int, int]]:
    """Create warm-to-cool RGB gradient from Catppuccin palette."""
    return [_hex_to_rgb(color) for color in _RADIAL_PALETTE]


def _interpolate_color(gradient: list[tuple[int, int, int]], position: float) -> str:
    """Interpolate color at position [0,1] along gradient."""
    if not gradient:
        return SURFACE1

    n_stops = len(gradient)
    if n_stops == 1:
        return _rgb_to_hex(gradient[0])

    # Clamp position to [0, 1]
    position = max(0.0, min(1.0, position))

    # Scale position to gradient stops
    scaled_pos = position * (n_stops - 1)
    idx = int(scaled_pos)

    if idx >= n_stops - 1:
        return _rgb_to_hex(gradient[-1])

    # Linear interpolation between idx and idx+1
    t = scaled_pos - idx
    r1, g1, b1 = gradient[idx]
    r2, g2, b2 = gradient[idx + 1]

    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)

    return _rgb_to_hex((r, g, b))


def _calculate_ray_count(width: int, height: int) -> int:
    """Calculate optimal ray count based on terminal dimensions."""
    if not ADAPTIVE_RAY_DENSITY:
        return MAX_RAYS

    # Use maximum dimension as primary factor (circle size determines detail needed)
    terminal_size = max(width, height)

    if terminal_size < 80:
        ray_count = RAY_DENSITY_SMALL
    elif terminal_size < 120:
        ray_count = RAY_DENSITY_MEDIUM
    else:
        ray_count = RAY_DENSITY_LARGE

    # Clamp to MAX_RAYS
    return min(ray_count, MAX_RAYS)


@dataclass
class RadialState:
    """Mutable state for the radial spectrum visualiser."""

    remainder: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    prev_amplitudes: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    rolling_max: float = MIN_ROLLING_MAX

    # Rotation
    rotation_angle: float = 0.0
    color_rotation_angle: float = 0.0
    last_time: float = field(default_factory=lambda: time.time())

    # Cached gradient (computed once)
    color_gradient: list[tuple[int, int, int]] = field(
        default_factory=_create_frequency_gradient
    )


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
            amplitudes[i] = float(np.max(spectrum[lo:hi]))
        elif lo < n_bins:
            amplitudes[i] = float(spectrum[lo])

    # Spectral unsharp mask — boost local peaks relative to their neighbours
    kernel = np.ones(2 * SPECTRAL_SHARPEN_RADIUS + 1, dtype=np.float32)
    kernel /= kernel.sum()
    blurred = np.convolve(amplitudes, kernel, mode="same")
    amplitudes = np.clip(
        amplitudes + SPECTRAL_SHARPEN_STRENGTH * (amplitudes - blurred), 0.0, None
    )

    # Auto-scaling with a slow local decay to keep the ceiling elevated after peaks
    peak = float(np.max(amplitudes)) if amplitudes.size > 0 else 0.0
    state.rolling_max = max(state.rolling_max * RADIAL_ROLLING_MAX_DECAY, peak, 1e-4)
    amplitudes /= state.rolling_max

    # Asymmetric EMA smoothing
    if len(state.prev_amplitudes) != n_rays:
        state.prev_amplitudes = amplitudes.copy()
    else:
        alpha = np.where(amplitudes > state.prev_amplitudes, ATTACK_ALPHA, DECAY_ALPHA)
        state.prev_amplitudes += alpha * (amplitudes - state.prev_amplitudes)
    amplitudes = state.prev_amplitudes.copy()

    # Remap [0,1] → [RAY_MIN_AMPLITUDE, 1.0] so quiet audio shows small spikes
    # and loud audio still hits full height
    compressed = (
        np.clip(amplitudes, 0.0, 1.0).astype(np.float32) ** COMPRESSION_EXPONENT
    )
    return RAY_MIN_AMPLITUDE + compressed * (1.0 - RAY_MIN_AMPLITUDE)


def _render_radial(state: RadialState, width: int, height: int) -> VizFrame:
    """Render hollow ring radial spectrum with adaptive ray density."""
    # Update rotation angle based on elapsed time
    current_time = time.time()
    dt = current_time - state.last_time
    state.last_time = current_time
    state.rotation_angle = (state.rotation_angle + ROTATION_SPEED * dt) % (2 * np.pi)
    state.color_rotation_angle = (
        state.color_rotation_angle + COLOR_ROTATION_SPEED * dt
    ) % (2 * np.pi)

    # Terminal coordinates (braille: 2×4 pixels per character)
    dot_w = width * 2
    dot_h = height * 4
    cx = dot_w // 2
    cy = dot_h // 2

    # Use 90% of available radius for better screen coverage
    max_radius = int(min(cx, cy) * VISUALIZATION_SIZE_RATIO)
    inner_radius = int(max_radius * INNER_RING_RATIO)

    # ADAPTIVE: Calculate ray count based on terminal size
    n_rays = _calculate_ray_count(width, height)

    # Get full spectrum amplitudes with logarithmic frequency grouping
    amplitudes = _compute_spectrum(state, n_rays)

    # Apply minimum amplitude floor to prevent vanishing rays
    amplitudes = np.maximum(amplitudes, MIN_RAY_AMPLITUDE)

    # Initialize grids
    grid = np.zeros((height, width), dtype=np.int32)
    color_grid: ColorGrid = [[SURFACE1] * width for _ in range(height)]

    # Draw inner ring — smooth Bresenham circle at inner_radius
    ring_color_pos = state.color_rotation_angle / (2 * np.pi)
    ring_color = _interpolate_color(state.color_gradient, ring_color_pos)
    draw_circle(
        cx, cy, inner_radius, grid, width, height, dot_w, dot_h, color_grid, ring_color
    )

    # Draw rays from inner ring edge to outer amplitude edge
    for i in range(n_rays):
        amp = float(amplitudes[i])

        # Ray extends from inner_radius to amplitude-based outer radius
        outer_ray_len = inner_radius + amp * (max_radius - inner_radius)

        # Calculate angle with rotation (for spectrum positioning)
        base_angle = (2 * np.pi * i / n_rays + state.rotation_angle) % (2 * np.pi)

        # Calculate color position with independent rotation
        color_angle = (2 * np.pi * i / n_rays + state.color_rotation_angle) % (
            2 * np.pi
        )
        color_position = color_angle / (2 * np.pi)  # Normalize to [0, 1]
        color = _interpolate_color(state.color_gradient, color_position)

        cos_a = np.cos(base_angle)
        sin_a = np.sin(base_angle)

        # Starting point: inner ring circumference
        x0 = round(cx + cos_a * inner_radius)
        y0 = round(cy - sin_a * inner_radius)

        # Endpoint: based on amplitude
        x1 = round(cx + cos_a * outer_ray_len)
        y1 = round(cy - sin_a * outer_ray_len)

        # Draw ray from inner ring to outer edge
        bresenham_line(
            x0, y0, x1, y1, grid, width, height, dot_w, dot_h, color_grid, color
        )

    return VizFrame(grid=grid.tolist(), colors=color_grid)


class RadialViz:
    """Radial spectrum visualiser implementing the Visualizer protocol."""

    def __init__(self) -> None:
        """Initialize visualizer with empty state."""
        self._state = RadialState()

    def push(self, chunk: np.ndarray) -> None:
        """Write new audio samples into the buffer."""
        _push_audio(chunk, self._state)

    def render(self, width: int, height: int) -> VizFrame:
        """Render a braille radial spectrum frame."""
        return _render_radial(self._state, width, height)
