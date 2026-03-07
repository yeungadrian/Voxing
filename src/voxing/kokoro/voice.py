import mlx.core as mx


def load_voice_tensor(path: str) -> mx.array:
    """Load a voice pack .safetensors file into an MLX array."""
    weights = mx.load(path)
    return weights["voice"]  # ty: ignore
