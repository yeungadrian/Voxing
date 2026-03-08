"""Smoke-test Qwen3-TTS in isolation."""

import time

import mlx.core as mx

from voxing.config import Settings
from voxing.tts import load_model, speak_text

settings = Settings()
model = load_model(settings.tts_model_id)

text = (
    "Hello! This is a test of newline streaming text to speech.\n"
    "The first line should start playing quickly while the next line "
    "is still being generated.\n"
    "Voxing can use this same path for low-latency spoken responses."
)

start = time.perf_counter()
first_chunk = True
for result in speak_text(
    model,
    text,
    voice=settings.tts_voice,
    speed=settings.tts_speed,
    lang_code=settings.tts_language,
    instruct=settings.tts_instruct,
    split_pattern=settings.tts_split_pattern,
    temperature=settings.tts_temperature,
    stream=settings.tts_stream,
    streaming_interval=settings.tts_streaming_interval,
    top_k=settings.tts_top_k,
    top_p=settings.tts_top_p,
    repetition_penalty=settings.tts_repetition_penalty,
):
    if first_chunk:
        ttfa = time.perf_counter() - start
        print(f"Time to first audio: {ttfa:.3f}s")
        first_chunk = False
    print(
        f"Segment {result.segment_idx}: "
        f"duration={result.audio_duration}, "
        f"rtf={result.real_time_factor}, "
        f"peak_mem={result.peak_memory_usage:.2f}GB"
    )

del model
mx.clear_cache()
print("Done!")
