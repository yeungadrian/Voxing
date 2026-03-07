"""Smoke-test Kokoro TTS in isolation."""

import numpy as np
import sounddevice as sd

from voxing.kokoro import load_model

model = load_model("mlx-community/Kokoro-82M-bf16")

text = (
    "Hello! This is a test of the Kokoro text to speech model running on "
    "Apple Silicon with MLX."
)

for result in model.generate(text, voice="af_heart", speed=1.0):
    print(
        f"Segment {result.segment_idx}: "
        f"duration={result.audio_duration}, "
        f"rtf={result.real_time_factor}, "
        f"peak_mem={result.peak_memory_usage:.2f}GB"
    )
    audio = np.array(result.audio)
    sd.play(audio, samplerate=result.sample_rate)
    sd.wait()

del model
print("Done!")
