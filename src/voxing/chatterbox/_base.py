from dataclasses import dataclass

import mlx.core as mx


@dataclass
class GenerationResult:
    audio: mx.array
    sample_rate: int
