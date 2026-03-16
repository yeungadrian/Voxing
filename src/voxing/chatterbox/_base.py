# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class GenerationResult:
    audio: mx.array
    sample_rate: int
