"""Test the TTS pipeline in isolation.

Usage: uv run scripts/tts.py
"""

import mlx.core as mx
import numpy as np
import sounddevice as sd

from voxing.config import Settings
from voxing.tts import load_tts, synthesize

settings = Settings()
print("Loading model...")
model = load_tts(settings.tts_model_id)
print(f"Model loaded (sample rate: {model.sample_rate} Hz).\n")

_EXAMPLES = [
    "Hello! This is a test of the text to speech system.",
    "The quick brown fox jumps over the lazy dog.",
    "How are you doing today? I hope you are having a wonderful day.",
]

for text in _EXAMPLES:
    print(f"Synthesizing: {text}")
    audio = synthesize(model, text)
    duration = len(audio) / model.sample_rate
    print(f"  Generated {duration:.2f}s of audio, playing...")
    sd.play(audio, samplerate=model.sample_rate)
    sd.wait()
    print()

del model
mx.clear_cache()
print("Done.")
