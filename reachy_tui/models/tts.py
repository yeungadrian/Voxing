"""Kokoro text-to-speech wrapper."""

import asyncio
from functools import partial

import mlx.nn as nn
import numpy as np
import sounddevice as sd
from langdetect import detect

KOKORO_VOICES: dict[str, tuple[str, str]] = {
    "en": ("af_heart", "a"),
    "es": ("ef_dora", "e"),
    "fr": ("ff_siwis", "f"),
    "hi": ("hf_alpha", "h"),
    "it": ("if_sara", "i"),
    "ja": ("jf_alpha", "j"),
    "pt": ("pf_dora", "p"),
    "zh-cn": ("zf_xiaobei", "z"),
    "zh-tw": ("zf_xiaobei", "z"),
}

TTS_SAMPLE_RATE = 24000


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to prevent clipping."""
    audio = audio.flatten().astype(np.float32)
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.9
    return audio.reshape(-1, 1)


def _synthesize_and_play(tts_model: nn.Module, text: str) -> None:
    """Synthesize speech from text and play it back."""
    if not text.strip():
        return
    detected_lang = detect(text)
    voice, lang_code = KOKORO_VOICES.get(detected_lang, ("af_heart", "a"))
    with sd.OutputStream(
        samplerate=TTS_SAMPLE_RATE,
        channels=1,
        dtype=np.float32,
        blocksize=4096,
    ) as stream:
        for result in tts_model.generate(
            text=text, voice=voice, lang_code=lang_code, speed=1.0
        ):
            audio = np.array(result.audio, dtype=np.float32)
            stream.write(_normalize_audio(audio))


async def speak(tts_model: nn.Module, text: str) -> None:
    """Synthesize and play speech asynchronously."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, partial(_synthesize_and_play, tts_model, text)
    )
