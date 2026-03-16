# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class GenerationResult:
    audio: mx.array
    samples: int
    sample_rate: int
    segment_idx: int
    token_count: int
    audio_duration: str
    real_time_factor: float
    prompt: dict[str, int | float]
    audio_samples: dict[str, int | float]
    processing_time_seconds: float
    peak_memory_usage: float
    is_streaming_chunk: bool = False
    is_final_chunk: bool = False
