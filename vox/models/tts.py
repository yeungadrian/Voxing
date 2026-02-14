"""Kokoro text-to-speech wrapper."""

import asyncio
import time
from collections.abc import Callable, Iterator, Sequence
from functools import partial

import mlx.nn as nn
import numpy as np
from langdetect import detect

from vox.audio import play_stream
from vox.config import settings

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

DEFAULT_VOICE = ("af_heart", "a")


def _detect_voice(text: str) -> tuple[str, str]:
    """Detect language and return the matching Kokoro voice and lang code."""
    detected = detect(text)
    return KOKORO_VOICES.get(detected, DEFAULT_VOICE)


def _synthesize(tts_model: nn.Module, text: str) -> Iterator[np.ndarray]:
    """Yield audio chunks from the TTS model."""
    if not text.strip():
        return
    voice, lang_code = _detect_voice(text)
    for result in tts_model.generate(
        text=text, voice=voice, lang_code=lang_code, speed=1.0
    ):
        yield np.array(result.audio, dtype=np.float32)


def _synthesize_phrases(
    tts_model: nn.Module,
    phrases: Sequence[str],
) -> Iterator[np.ndarray]:
    """Yield audio chunks from a sequence of phrases."""
    for phrase in phrases:
        yield from _synthesize(tts_model, phrase)


async def speak_phrases(
    tts_model: nn.Module,
    phrases: Sequence[str],
    on_first_chunk: Callable[[], None] | None = None,
) -> float:
    """Synthesize and play a sequence of phrases. Returns synthesis time."""
    loop = asyncio.get_running_loop()

    synth_start = time.time()
    chunks = await loop.run_in_executor(
        None, partial(list, _synthesize_phrases(tts_model, phrases))
    )
    synth_time = time.time() - synth_start

    await play_stream(chunks, settings.tts_sample_rate, on_first_chunk)
    return synth_time
