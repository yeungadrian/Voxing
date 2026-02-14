"""Sounddevice audio recorder."""

import asyncio
from functools import partial

import numpy as np
import sounddevice as sd

from reachy_tui.config import settings


def _rms(audio: np.ndarray) -> float:
    """Calculate root mean square of audio signal."""
    return float(np.sqrt(np.mean(audio**2)))


def _record_blocking(
    silence_duration: float = 0.7,
    max_duration: float = 10.0,
) -> np.ndarray | None:
    """Record audio until silence is detected."""
    chunk_duration = 0.1
    chunk_samples = int(settings.audio_sample_rate * chunk_duration)
    silence_chunks_needed = int(silence_duration / chunk_duration)
    max_chunks = int(max_duration / chunk_duration)

    audio_chunks: list[np.ndarray] = []
    silence_count = 0
    recording = False

    with sd.InputStream(
        samplerate=settings.audio_sample_rate, channels=1, dtype=np.float32
    ) as stream:
        while len(audio_chunks) < max_chunks:
            chunk, _ = stream.read(chunk_samples)
            has_voice = _rms(chunk) > settings.silence_threshold

            if not recording:
                if has_voice:
                    recording = True
                    audio_chunks.append(chunk)
                    silence_count = 0
                continue

            audio_chunks.append(chunk)

            if has_voice:
                silence_count = 0
            else:
                silence_count += 1
                if silence_count >= silence_chunks_needed:
                    break

    if not audio_chunks:
        return None
    return np.concatenate(audio_chunks, axis=0).flatten()


async def record() -> np.ndarray | None:
    """Record a short voice command."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, partial(_record_blocking, silence_duration=0.7, max_duration=10.0)
    )


async def record_long() -> np.ndarray | None:
    """Record extended speech for transcription."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, partial(_record_blocking, silence_duration=3.0, max_duration=180.0)
    )
