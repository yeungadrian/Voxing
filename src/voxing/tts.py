import mlx.core as mx
import numpy as np

from voxing.chatterbox import ChatterboxTurboTTS
from voxing.chatterbox import load_model as _load_chatterbox


def load_tts(model_id: str) -> ChatterboxTurboTTS:
    """Download (if needed) and load a TTS model from HuggingFace Hub."""
    return _load_chatterbox(model_id)


def synthesize(model: ChatterboxTurboTTS, text: str) -> np.ndarray:
    """Generate speech audio from text, returning a numpy array at 24 kHz."""
    segments: list[np.ndarray] = []
    for result in model.generate(text):
        segments.append(np.array(result.audio))
    mx.clear_cache()
    return np.concatenate(segments) if segments else np.array([], dtype=np.float32)
