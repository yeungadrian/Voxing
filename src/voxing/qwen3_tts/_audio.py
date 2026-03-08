from pathlib import Path

import mlx.core as mx
import soundfile as sf
from scipy.signal import resample


def load_audio(audio: str | mx.array, sample_rate: int = 24000) -> mx.array:
    if isinstance(audio, mx.array):
        return audio

    path = Path(audio)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    samples, original_sample_rate = sf.read(path)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    if original_sample_rate != sample_rate:
        duration = samples.shape[0] / original_sample_rate
        samples = resample(samples, int(duration * sample_rate))

    return mx.array(samples, dtype=mx.float32)
