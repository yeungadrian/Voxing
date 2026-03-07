import inspect
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params: dict) -> "BaseModelArgs":
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def check_array_shape(arr: mx.array) -> bool:
    """Check if a 3D array has out_channels as the largest dim with equal kH/kW."""
    shape = arr.shape
    if len(shape) != 3:
        return False
    out_channels, kH, kW = shape
    return (out_channels >= kH) and (out_channels >= kW) and (kH == kW)


@dataclass
class GenerationResult:
    audio: mx.array
    samples: int
    sample_rate: int
    segment_idx: int
    token_count: int
    audio_duration: str
    real_time_factor: float
    prompt: dict
    audio_samples: dict
    processing_time_seconds: float
    peak_memory_usage: float
