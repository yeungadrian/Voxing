"""Unified visualisation protocol."""

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class VizFrame:
    """One rendered frame of any visualisation."""

    grid: list[list[int]]  # height x width
    mode: Literal["braille", "blocks"]
    # braille: values are braille offsets (add 0x2800)
    # blocks: values are 0-8 indices into BLOCKS
    colors: list[list[str]] | None = None  # per-cell hex, or None for default


class Visualizer(Protocol):
    def push(self, chunk: np.ndarray) -> None: ...
    def render(self, width: int, height: int) -> VizFrame: ...


VIZ_PALETTE: list[str] = [
    "#1e1e2e",  # 0 silence / background
    "#313244",  # 1 near-silence
    "#45475a",  # 2 very low
    "#585b70",  # 3 low
    "#6a7db5",  # 4 dim blue
    "#89b4fa",  # 5 primary blue
    "#74c7ec",  # 6 sapphire
    "#89dceb",  # 7 sky
    "#89dceb",  # 8 peak / sky (capped, avoids near-white)
]
