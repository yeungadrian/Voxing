"""Model loading for Kokoro TTS, following the Parakeet vendoring pattern."""

import json
from pathlib import Path
from typing import cast

import mlx.core as mx
from huggingface_hub import snapshot_download

from voxing.kokoro.kokoro import Model, ModelConfig


def _resolve_kokoro_path(model_id: str) -> Path:
    """Download config + weights (excluding large voice/sample dirs)."""
    return Path(
        snapshot_download(
            model_id,
            allow_patterns=["config.json", "*.safetensors"],
            ignore_patterns=["voices/*", "samples/*"],
        )
    )


def load_model(model_id: str = "mlx-community/Kokoro-82M-bf16") -> Model:
    """Download (if needed) and load a Kokoro TTS model from HuggingFace Hub."""
    model_path = _resolve_kokoro_path(model_id)
    config = json.loads((model_path / "config.json").read_text())
    model_config = cast(ModelConfig, ModelConfig.from_dict(config))
    model = Model(model_config, repo_id=model_id)

    weight_files = list(model_path.glob("*.safetensors"))
    weights: dict[str, mx.array] = {}
    for wf in weight_files:
        loaded_raw = mx.load(str(wf))
        if isinstance(loaded_raw, tuple):
            if not isinstance(loaded_raw[0], dict):
                raise TypeError(f"Unexpected weights format from {wf}")
            loaded = cast(dict[str, mx.array], loaded_raw[0])
        elif isinstance(loaded_raw, dict):
            loaded = cast(dict[str, mx.array], loaded_raw)
        else:
            raise TypeError(f"Unexpected weights format from {wf}")
        weights.update(loaded)

    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    model.eval()
    return model
