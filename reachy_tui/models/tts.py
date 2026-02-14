"""Chatterbox text-to-speech wrapper."""

import asyncio
from functools import partial

import mlx.nn as nn
import numpy as np
import sounddevice as sd
from langdetect import detect

from reachy_tui.config import settings

SUPPORTED_LANGS = frozenset({
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
    "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
    "sw", "tr", "zh",
})


def _detect_lang(text: str) -> str:
    """Detect language, falling back to English if unsupported."""
    detected = detect(text)
    return detected if detected in SUPPORTED_LANGS else "en"


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
    lang_code = _detect_lang(text)
    with sd.OutputStream(
        samplerate=settings.tts_sample_rate,
        channels=1,
        dtype=np.float32,
        blocksize=4096,
    ) as stream:
        for result in tts_model.generate(text=text, lang_code=lang_code):
            audio = np.array(result.audio, dtype=np.float32)
            stream.write(_normalize_audio(audio))


async def speak(tts_model: nn.Module, text: str) -> None:
    """Synthesize and play speech asynchronously."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, partial(_synthesize_and_play, tts_model, text)
    )
