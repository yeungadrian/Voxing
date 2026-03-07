"""Unified visualisation protocol."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

# Type aliases for braille grids (internal use within viz package)
type BrailleGrid = list[list[int]]
type ColorGrid = list[list[str]]


@dataclass(frozen=True, slots=True)
class VizFrame:
    """One rendered frame of any visualisation."""

    grid: BrailleGrid  # height × width braille offsets (add BRAILLE_BASE)
    colors: ColorGrid | None = None  # per-cell hex, or None for default


class Visualizer(Protocol):
    def push(self, chunk: np.ndarray) -> None: ...
    def render(self, width: int, height: int) -> VizFrame: ...


# ── shared braille constants ──
BRAILLE_BASE = 0x2800

# ── shared audio constants ──
NOISE_GATE = 0.005
ROLLING_MAX_DECAY = 0.983
MIN_ROLLING_MAX = 0.08
