"""Chatterbox text-to-speech wrapper."""

from collections.abc import Iterator

import mlx.nn as nn
import numpy as np
from langdetect import detect

from vox.audio import play_stream
from vox.config import settings

SUPPORTED_LANGS = frozenset(
    {
        "ar",
        "da",
        "de",
        "el",
        "en",
        "es",
        "fi",
        "fr",
        "he",
        "hi",
        "it",
        "ja",
        "ko",
        "ms",
        "nl",
        "no",
        "pl",
        "pt",
        "ru",
        "sv",
        "sw",
        "tr",
        "zh",
    }
)


def _detect_lang(text: str) -> str:
    """Detect language, falling back to English if unsupported."""
    detected = detect(text)
    return detected if detected in SUPPORTED_LANGS else "en"


def _synthesize(tts_model: nn.Module, text: str) -> Iterator[np.ndarray]:
    """Yield audio chunks from the TTS model."""
    if not text.strip():
        return
    lang_code = _detect_lang(text)
    for result in tts_model.generate(text=text, lang_code=lang_code):
        yield np.array(result.audio, dtype=np.float32)


async def speak(tts_model: nn.Module, text: str) -> None:
    """Synthesize and play speech asynchronously."""
    await play_stream(_synthesize(tts_model, text), settings.tts_sample_rate)
