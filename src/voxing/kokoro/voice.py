from typing import cast

import mlx.core as mx


def load_voice_tensor(path: str) -> mx.array:
    """Load a voice pack .safetensors file into an MLX array."""
    loaded_raw = mx.load(path)
    weights: dict[str, mx.array]
    if isinstance(loaded_raw, tuple):
        if not isinstance(loaded_raw[0], dict):
            raise TypeError(f"Unexpected voice weights format from {path}")
        weights = cast(dict[str, mx.array], loaded_raw[0])
    elif isinstance(loaded_raw, dict):
        weights = cast(dict[str, mx.array], loaded_raw)
    else:
        raise TypeError(f"Unexpected voice weights format from {path}")
    return weights["voice"]
