"""Unified visualisation protocol."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class VizFrame:
    """One rendered frame of any visualisation."""

    grid: list[list[int]]  # height x width, values are braille offsets (add 0x2800)
    colors: list[list[str]] | None = None  # per-cell hex, or None for default


class Visualizer(Protocol):
    def push(self, chunk: np.ndarray) -> None: ...
    def render(self, width: int, height: int) -> VizFrame: ...


# ── shared braille constants ──
BRAILLE_BASE = 0x2800

# ── shared audio constants ──
NOISE_GATE = 0.005
ROLLING_MAX_DECAY = 0.95
